import argparse
import copy
import json
import os
import re


EPS = 1e-9
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay SpecEyes phase-II acceptance with new thresholds without rerunning models."
    )
    parser.add_argument("--input_jsonl", required=True, help="Path to a raw SpecEyes jsonl file")
    parser.add_argument("--benchmark", required=True, choices=["vstar", "pope", "hr"])
    parser.add_argument("--test_type", "--task", dest="test_type", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--original_threshold",
        type=float,
        default=None,
        help="Original raw-run acceptance threshold. If omitted, try to parse it from the filename.",
    )
    parser.add_argument(
        "--near_margin",
        type=float,
        default=0.0025,
        help="Absolute confidence margin used for near-threshold counts.",
    )
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


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


def parse_original_threshold_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"_(\d+(?:\.\d+)?)_(?:mthr=[^_]+_)?mem=", stem)
    if not match:
        return None
    return float(match.group(1))


def format_threshold(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \n\t\r.,;:!?\"'()[]{}")
    return text


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


def extract_spatial_label(text):
    text = normalize_text(strip_leading_option_marker(text))
    for label, patterns in SPATIAL_LABEL_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return label
    return None


def extract_prediction_from_small_answer(output_text):
    text = str(output_text or "").strip()
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return text


def estimate_generated_length(text):
    tokens = re.findall(r"\w+|[^\w\s]", str(text or ""))
    return len(tokens)


def get_routing_metadata(record):
    policy = record.get("memory_policy", {})
    return {
        "judge_tc": record.get("judge_tc", ""),
        "use_model": record.get("use_model", ""),
        "confidence_score": record.get("confidence_score", None),
        "trigger_threshold": record.get("memory_trigger_threshold_applied", policy.get("trigger_threshold")),
        "threshold": record.get("memory_acceptance_threshold_applied", policy.get("threshold")),
        "memory_triggered": record.get("memory_triggered", False),
        "memory_accept_decision": record.get("memory_accept_decision", False),
        "memory_task": record.get("memory_task", policy.get("task_name", "")),
        "memory_enabled": record.get("memory_enabled", policy.get("enabled", False)),
    }



def phase_two_candidate(record):
    routing = get_routing_metadata(record)
    return record.get("status") != "error" and routing["judge_tc"] == "no"


def score_pope(result):
    gold = extract_yes_no(result.get("answer", ""))
    pred = extract_yes_no(result.get("pred_ans", ""))
    if not gold or not pred:
        return 0.0
    return 1.0 if gold == pred else 0.0


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


def score_hr_proxy(result):
    option_letter = extract_option_letter(result.get("pred_ans", ""))
    gold_letter = normalize_text(result.get("answer", ""))
    if option_letter and is_bare_option_prediction(result.get("pred_ans", "")) and gold_letter:
        return 1.0 if option_letter == gold_letter else 0.0
    return 0.0


def score_record(record, benchmark, test_type):
    result = record.get("result", {})
    if benchmark == "pope":
        acc = score_pope(result)
        return {"acc": acc, "proxy_acc": acc}
    if benchmark == "vstar":
        return {"acc": None, "proxy_acc": score_vstar_proxy(result, test_type)}
    if benchmark == "hr":
        return {"acc": None, "proxy_acc": score_hr_proxy(result)}
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def build_replayed_record(record, threshold, original_threshold):
    replayed = copy.deepcopy(record)
    routing = get_routing_metadata(record)
    replayed["replay_source_use_model"] = routing["use_model"]
    replayed["replay_applied_threshold"] = threshold
    replayed["replay_original_threshold"] = original_threshold
    replayed["replay_route_changed"] = False
    replayed["replay_route_change"] = ""

    if not phase_two_candidate(record):
        return replayed

    confidence = routing["confidence_score"]
    if not isinstance(confidence, (int, float)) or confidence < 0:
        return replayed

    replay_use_model = "small" if confidence > threshold else "large"
    original_use_model = routing["use_model"]

    if original_use_model == "small" and replay_use_model == "large":
        raise ValueError(
            "Replay threshold is higher than the original run threshold, but no large fallback trace is stored "
            "for originally accepted small-model samples."
        )

    if original_use_model == "large" and replay_use_model == "small":
        small_pred = extract_prediction_from_small_answer(record.get("small_answer", ""))
        replayed["use_model"] = "small"
        replayed["generated_length"] = estimate_generated_length(record.get("small_answer", ""))
        replayed.setdefault("result", {})
        replayed["result"]["pred_ans"] = small_pred
        replayed["replay_route_changed"] = True
        replayed["replay_route_change"] = "large_to_small"

    return replayed


def summarise_threshold(records, replayed_records, threshold, benchmark, test_type, near_margin):
    acc_values = []
    proxy_values = []
    small_cnt = 0
    large_cnt = 0
    phase_two_cnt = 0
    near_threshold_cnt = 0
    small_to_large = 0
    large_to_small = 0
    reproduces_original_route = True
    reproduces_original_prediction = True

    for original, replayed in zip(records, replayed_records):
        if original.get("status") == "error":
            continue

        if replayed.get("use_model") == "small":
            small_cnt += 1
        else:
            large_cnt += 1

        original_routing = get_routing_metadata(original)
        replay_routing = get_routing_metadata(replayed)

        if phase_two_candidate(original):
            phase_two_cnt += 1
            confidence = original_routing["confidence_score"]
            if isinstance(confidence, (int, float)) and confidence >= 0:
                if abs(confidence - threshold) <= near_margin:
                    near_threshold_cnt += 1

        if original_routing["use_model"] != replay_routing["use_model"]:
            reproduces_original_route = False
            if original_routing["use_model"] == "small" and replay_routing["use_model"] == "large":
                small_to_large += 1
            elif original_routing["use_model"] == "large" and replay_routing["use_model"] == "small":
                large_to_small += 1

        original_pred = original.get("result", {}).get("pred_ans", "")
        replay_pred = replayed.get("result", {}).get("pred_ans", "")
        if normalize_text(original_pred) != normalize_text(replay_pred):
            reproduces_original_prediction = False

        score = score_record(replayed, benchmark, test_type)
        if score["acc"] is not None:
            acc_values.append(score["acc"])
        if score["proxy_acc"] is not None:
            proxy_values.append(score["proxy_acc"])

    evaluated_count = small_cnt + large_cnt
    triggered_cnt = sum(1 for record in replayed_records if get_routing_metadata(record)["memory_triggered"])
    accepted_cnt = sum(1 for record in replayed_records if get_routing_metadata(record)["memory_accept_decision"])
    summary = {
        "threshold": threshold,
        "acc": round(sum(acc_values) / len(acc_values), 6) if acc_values else None,
        "proxy_acc": round(sum(proxy_values) / len(proxy_values), 6) if proxy_values else None,
        "evaluated_count": evaluated_count,
        "small_cnt": small_cnt,
        "large_cnt": large_cnt,
        "small_ratio": round((small_cnt / evaluated_count) * 100, 4) if evaluated_count else 0.0,
        "phase_two_candidate_count": phase_two_cnt,
        "triggered_count": triggered_cnt,
        "accepted_count": accepted_cnt,
        "small_to_large": small_to_large,
        "large_to_small": large_to_small,
        "near_threshold_count": near_threshold_cnt,
        "reproduces_original_route": reproduces_original_route,
        "reproduces_original_prediction": reproduces_original_prediction,
    }
    return summary


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_type is None:
        args.test_type = infer_test_type_from_filename(args.input_jsonl, args.benchmark)
    if args.test_type is None:
        raise ValueError("Could not infer --test_type from filename. Please pass --test_type explicitly.")

    original_threshold = args.original_threshold
    if original_threshold is None:
        original_threshold = parse_original_threshold_from_filename(args.input_jsonl)
    if original_threshold is None:
        raise ValueError(
            "Could not infer the original run threshold from the filename. Please pass --original_threshold."
        )

    for threshold in args.thresholds:
        if threshold > original_threshold + EPS:
            raise ValueError(
                f"Replay threshold {threshold} is higher than the original threshold {original_threshold}. "
                "This replay only supports thresholds less than or equal to the raw run threshold."
            )

    records = load_jsonl(args.input_jsonl)
    input_stem = os.path.splitext(os.path.basename(args.input_jsonl))[0]

    threshold_summaries = []
    for threshold in args.thresholds:
        replayed_records = [
            build_replayed_record(record, threshold, original_threshold) for record in records
        ]
        output_name = f"{input_stem}_replay-thr={format_threshold(threshold)}.jsonl"
        output_path = os.path.join(args.output_dir, output_name)
        save_jsonl(output_path, replayed_records)

        threshold_summary = summarise_threshold(
            records=records,
            replayed_records=replayed_records,
            threshold=threshold,
            benchmark=args.benchmark,
            test_type=args.test_type,
            near_margin=args.near_margin,
        )
        threshold_summary["output_jsonl"] = os.path.abspath(output_path)
        threshold_summaries.append(threshold_summary)

    summary = {
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "benchmark": args.benchmark,
        "test_type": args.test_type,
        "original_threshold": original_threshold,
        "near_margin": args.near_margin,
        "threshold_summaries": threshold_summaries,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved replay summary to: {summary_path}")


if __name__ == "__main__":
    main()
