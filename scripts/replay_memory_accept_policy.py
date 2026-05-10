import argparse
import copy
import json
import os
import re
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay memory accept policies on raw SpecEyes JSONL without rerunning models."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--benchmark", required=True, choices=["vstar", "pope", "hr"])
    parser.add_argument("--test_type", "--task", dest="test_type", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--retrieval_thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.4, 0.45, 0.5, 0.545, 0.6],
    )
    parser.add_argument(
        "--confidence_thresholds",
        nargs="+",
        type=float,
        default=[0.9, 0.92, 0.94, 0.96, 0.98],
    )
    parser.add_argument(
        "--delta_thresholds",
        nargs="+",
        type=float,
        default=[-0.02, -0.01, -0.005, 0.0, 0.005, 0.01],
        help="Minimum confidence delta required for acceptance.",
    )
    parser.add_argument(
        "--answer_change_modes",
        nargs="+",
        choices=["any", "changed", "unchanged"],
        default=["any", "changed", "unchanged"],
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of top policies to emit as replay JSONLs")
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_test_type_from_filename(path, benchmark):
    stem = os.path.splitext(os.path.basename(path))[0]
    if benchmark == "vstar":
        for name in ("direct_attributes", "relative_position"):
            if name in stem:
                return name
    if benchmark == "pope":
        for name in ("adversarial", "popular", "random"):
            if name in stem:
                return name
    if benchmark == "hr":
        for name in ("hr_bench_4k", "hr_bench_8k"):
            if name in stem:
                return name
    return None


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \n\t\r.,;:!?\"'()[]{}")


def strip_leading_option_marker(text):
    return re.sub(r"^\s*[a-f](?:[\.\):]|\s)+", "", str(text or "").strip(), flags=re.IGNORECASE)


def extract_option_letter(text):
    match = re.match(r"^\s*([a-f])(?:[\.\):]|\s|$)", str(text or "").strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def is_bare_option_prediction(text):
    return re.fullmatch(r"\s*[a-f](?:[\.\):]|\s)*", str(text or ""), flags=re.IGNORECASE) is not None


def extract_yes_no(text):
    text = normalize_text(text)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


SPATIAL_LABEL_PATTERNS = [
    ("left", [r"\bleft\b", r"\bleft side\b"]),
    ("right", [r"\bright\b", r"\bright side\b"]),
    ("above", [r"\babove\b", r"\bon top of\b", r"\bover\b"]),
    ("below", [r"\bbelow\b", r"\bunder\b", r"\bunderside\b"]),
    ("front", [r"\bin front of\b", r"\bfront of\b"]),
    ("behind", [r"\bbehind\b", r"\bin back of\b"]),
    ("inside", [r"\binside\b", r"\bwithin\b"]),
    ("outside", [r"\boutside\b", r"\boutside of\b"]),
]


def extract_spatial_label(text):
    text = normalize_text(strip_leading_option_marker(text))
    for label, patterns in SPATIAL_LABEL_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return label
    return None


def score_vstar_proxy(result, test_type):
    pred_text = result.get("pred_ans", "")
    gold_text = result.get("answer", "")

    option_letter = extract_option_letter(pred_text)
    if option_letter is not None and is_bare_option_prediction(pred_text):
        return 1.0 if option_letter == "a" else 0.0

    if test_type == "relative_position":
        gold_spatial = extract_spatial_label(gold_text)
        pred_spatial = extract_spatial_label(pred_text)
        if gold_spatial and pred_spatial:
            return 1.0 if gold_spatial == pred_spatial else 0.0

    gold_yes_no = extract_yes_no(gold_text)
    pred_yes_no = extract_yes_no(pred_text)
    if gold_yes_no and pred_yes_no:
        return 1.0 if gold_yes_no == pred_yes_no else 0.0

    gold_norm = normalize_text(strip_leading_option_marker(gold_text))
    pred_norm = normalize_text(strip_leading_option_marker(pred_text))
    if not gold_norm or not pred_norm:
        return 0.0
    return 1.0 if gold_norm == pred_norm else 0.0


def score_pope(result):
    gold = extract_yes_no(result.get("answer", ""))
    pred = extract_yes_no(result.get("pred_ans", ""))
    if not gold or not pred:
        return 0.0
    return 1.0 if gold == pred else 0.0


def score_hr_proxy(result):
    option_letter = extract_option_letter(result.get("pred_ans", ""))
    gold_letter = normalize_text(result.get("answer", ""))
    if option_letter and is_bare_option_prediction(result.get("pred_ans", "")) and gold_letter:
        return 1.0 if option_letter == gold_letter else 0.0
    return 0.0


def score_record(record, benchmark, test_type):
    result = record.get("result", {})
    if benchmark == "vstar":
        return score_vstar_proxy(result, test_type)
    if benchmark == "pope":
        return score_pope(result)
    if benchmark == "hr":
        return score_hr_proxy(result)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def extract_answer_text(text):
    text = str(text or "").strip()
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return text


def get_top1_retrieval_score(record):
    memories = record.get("retrieved_logic_memories") or []
    if not memories:
        return None
    score = memories[0].get("_retrieval_score")
    return score if isinstance(score, (int, float)) else None


def should_accept(record, rule):
    if not record.get("memory_triggered", False):
        return record.get("use_model") == "small"

    top1 = get_top1_retrieval_score(record)
    if top1 is None or top1 < rule["min_retrieval_score"]:
        return False

    mem_conf = record.get("confidence_score")
    if not isinstance(mem_conf, (int, float)) or mem_conf < rule["min_mem_conf"]:
        return False

    base_conf = record.get("base_confidence_score")
    if not isinstance(base_conf, (int, float)):
        base_conf = mem_conf
    delta_conf = mem_conf - base_conf
    if delta_conf < rule["min_delta_conf"]:
        return False

    raw_answer = extract_answer_text(record.get("base_small_answer", ""))
    mem_answer = extract_answer_text(record.get("small_answer", ""))
    changed = normalize_text(raw_answer) != normalize_text(mem_answer)
    mode = rule["answer_change_mode"]
    if mode == "changed" and not changed:
        return False
    if mode == "unchanged" and changed:
        return False

    return True


def replay_record(record, rule):
    replayed = copy.deepcopy(record)
    accepted = should_accept(record, rule)
    replayed["replay_policy"] = rule
    replayed["replay_accept_decision"] = accepted
    replayed["replay_original_use_model"] = record.get("use_model")

    if record.get("memory_triggered", False) and accepted:
        replayed["use_model"] = "small"
        replayed["memory_accept_decision"] = True
        replayed.setdefault("result", {})
        replayed["result"]["pred_ans"] = extract_answer_text(record.get("small_answer", ""))
        replayed["generated_length"] = len(re.findall(r"\w+|[^\w\s]", replayed["result"]["pred_ans"]))
    elif record.get("memory_triggered", False):
        replayed["memory_accept_decision"] = False
    return replayed


def summarise(records, rule, benchmark, test_type):
    replayed_records = [replay_record(record, rule) for record in records]
    total = 0
    small_cnt = 0
    large_cnt = 0
    triggered_cnt = 0
    accepted_triggered = 0
    changed_routes = 0
    acc_values = []

    for original, replayed in zip(records, replayed_records):
        if replayed.get("status") == "error":
            continue
        total += 1
        if replayed.get("use_model") == "small":
            small_cnt += 1
        else:
            large_cnt += 1
        if original.get("memory_triggered", False):
            triggered_cnt += 1
            if replayed.get("use_model") == "small":
                accepted_triggered += 1
        if original.get("use_model") != replayed.get("use_model"):
            changed_routes += 1
        acc_values.append(score_record(replayed, benchmark, test_type))

    return {
        "rule": rule,
        "proxy_acc": round(sum(acc_values) / len(acc_values), 6) if acc_values else None,
        "evaluated_count": total,
        "small_cnt": small_cnt,
        "large_cnt": large_cnt,
        "small_ratio": round((small_cnt / total) * 100, 4) if total else 0.0,
        "triggered_count": triggered_cnt,
        "accepted_triggered": accepted_triggered,
        "accepted_triggered_ratio": round((accepted_triggered / triggered_cnt), 6) if triggered_cnt else 0.0,
        "changed_routes": changed_routes,
        "replayed_records": replayed_records,
    }


def format_rule_tag(rule):
    retrieval = str(rule["min_retrieval_score"]).rstrip("0").rstrip(".")
    mem_conf = str(rule["min_mem_conf"]).rstrip("0").rstrip(".")
    delta = str(rule["min_delta_conf"]).rstrip("0").rstrip(".")
    delta = delta if delta else "0"
    return (
        f"chg={rule['answer_change_mode']}"
        f"_rel={retrieval}"
        f"_mconf={mem_conf}"
        f"_dconf={delta}"
    )


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_type is None:
        args.test_type = infer_test_type_from_filename(args.input_jsonl, args.benchmark)
    if args.test_type is None:
        raise ValueError("Could not infer --test_type from filename. Please pass --test_type explicitly.")

    records = load_jsonl(args.input_jsonl)
    summaries = []

    for change_mode, rel_thr, conf_thr, delta_thr in product(
        args.answer_change_modes,
        args.retrieval_thresholds,
        args.confidence_thresholds,
        args.delta_thresholds,
    ):
        rule = {
            "answer_change_mode": change_mode,
            "min_retrieval_score": rel_thr,
            "min_mem_conf": conf_thr,
            "min_delta_conf": delta_thr,
        }
        summary = summarise(records, rule, args.benchmark, args.test_type)
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["proxy_acc"],
            row["small_ratio"],
            -row["accepted_triggered"],
        ),
        reverse=True,
    )

    top_summaries = []
    for idx, summary in enumerate(summaries[: args.top_k], 1):
        tag = format_rule_tag(summary["rule"])
        output_jsonl = os.path.join(args.output_dir, f"rank{idx:02d}_{tag}.jsonl")
        save_jsonl(output_jsonl, summary["replayed_records"])
        pruned = dict(summary)
        pruned["output_jsonl"] = os.path.abspath(output_jsonl)
        del pruned["replayed_records"]
        top_summaries.append(pruned)

    full_summary = []
    for summary in summaries:
        pruned = dict(summary)
        del pruned["replayed_records"]
        full_summary.append(pruned)

    output = {
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "benchmark": args.benchmark,
        "test_type": args.test_type,
        "top_k": args.top_k,
        "top_summaries": top_summaries,
        "all_summaries": full_summary,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved policy replay summary to: {summary_path}")


if __name__ == "__main__":
    main()
