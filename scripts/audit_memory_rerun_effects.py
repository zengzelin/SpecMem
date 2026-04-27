import argparse
import json
import os
import re
from statistics import mean

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
    parser = argparse.ArgumentParser(description="Audit memory rerun effects for SpecEyes JSONL outputs.")
    parser.add_argument("--input_jsonl", required=True, help="Path to a SpecEyes JSONL file")
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs")
    parser.add_argument("--task", default="", help="Optional task label for reporting")
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



def normalize_answer(text):
    text = str(text or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \n\t\r.,;:!?\"'()[]{}")



def strip_leading_option_marker(text):
    return re.sub(r"^\s*[a-f](?:[\.\):]|\s)+", "", str(text or "").strip(), flags=re.IGNORECASE)



def extract_yes_no(text):
    text = normalize_answer(text)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None



def extract_option_letter(text):
    match = re.match(r"^\s*([a-f])(?:[\.\):]|\s|$)", str(text or "").strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()



def is_bare_option_prediction(text):
    return re.fullmatch(r"\s*[a-f](?:[\.\):]|\s)*", str(text or ""), flags=re.IGNORECASE) is not None



def extract_spatial_label(text):
    text = normalize_answer(strip_leading_option_marker(text))
    for label, patterns in SPATIAL_LABEL_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return label
    return None



def is_correct_answer(gold_text, pred_text):
    gold_spatial = extract_spatial_label(gold_text)
    pred_spatial = extract_spatial_label(pred_text)
    if gold_spatial and pred_spatial:
        return gold_spatial == pred_spatial

    gold_yes_no = extract_yes_no(gold_text)
    pred_yes_no = extract_yes_no(pred_text)
    if gold_yes_no and pred_yes_no:
        return gold_yes_no == pred_yes_no

    gold_option = extract_option_letter(gold_text)
    pred_option = extract_option_letter(pred_text)
    if gold_option and pred_option and is_bare_option_prediction(pred_text):
        return gold_option == pred_option

    gold = normalize_answer(strip_leading_option_marker(gold_text))
    pred = normalize_answer(strip_leading_option_marker(pred_text))
    return bool(gold and pred and gold == pred)



def extract_memory_answer(record):
    result_pred = str(record.get("result", {}).get("pred_ans", "")).strip()
    if result_pred:
        return result_pred

    small_answer = str(record.get("small_answer", "")).strip()
    if "<answer>" in small_answer and "</answer>" in small_answer:
        return small_answer.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return small_answer



def extract_raw_answer(record):
    raw_answer = str(record.get("base_small_answer", "")).strip()
    if "<answer>" in raw_answer and "</answer>" in raw_answer:
        return raw_answer.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return raw_answer



def summarize_group(values):
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": round(mean(clean), 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
    }



def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_jsonl(args.input_jsonl)

    triggered_rows = []
    accepted_rows = []
    corrected_rows = []
    harmed_rows = []
    changed_but_still_wrong_rows = []
    unchanged_rows = []

    tail_gain_corrected = []
    tail_gain_harmed = []
    bottom10_gain_corrected = []
    bottom10_gain_harmed = []

    total_records = 0
    phase_two_candidates = 0
    triggered_count = 0
    accepted_after_trigger = 0
    answer_changed_count = 0
    accepted_changed_count = 0

    raw_wrong_to_memory_correct = 0
    raw_correct_to_memory_wrong = 0
    raw_wrong_to_memory_wrong = 0
    raw_correct_to_memory_correct = 0

    for idx, record in enumerate(records, 1):
        if record.get("status") == "error":
            continue

        total_records += 1
        if record.get("judge_tc") == "no":
            phase_two_candidates += 1

        if not record.get("memory_triggered", False):
            continue

        triggered_count += 1

        result = record.get("result", {})
        gold_answer = result.get("answer", "")
        raw_answer = extract_raw_answer(record)
        memory_answer = extract_memory_answer(record)
        raw_correct = is_correct_answer(gold_answer, raw_answer)
        memory_correct = is_correct_answer(gold_answer, memory_answer)
        answer_changed = normalize_answer(raw_answer) != normalize_answer(memory_answer)
        accepted = bool(record.get("memory_accept_decision", False))

        row = {
            "sample_index": idx,
            "task": record.get("memory_task", args.task),
            "question": result.get("question", ""),
            "gold_answer": gold_answer,
            "raw_answer": raw_answer,
            "memory_answer": memory_answer,
            "raw_correct": raw_correct,
            "memory_correct": memory_correct,
            "answer_changed": answer_changed,
            "accepted_after_trigger": accepted,
            "use_model": record.get("use_model", ""),
            "trigger_metric": record.get("trigger_metric", "confidence_score"),
            "accept_metric": record.get("accept_metric", "confidence_score"),
            "tail_score_raw": record.get("tail_score_raw"),
            "tail_score_memory": record.get("tail_score_memory"),
            "bottom10_group_score_raw": record.get("bottom10_group_score_raw"),
            "bottom10_group_score_memory": record.get("bottom10_group_score_memory"),
            "tail_gain": record.get("tail_gain"),
            "bottom10_gain": record.get("bottom10_gain"),
            "small_answer": record.get("small_answer", ""),
            "base_small_answer": record.get("base_small_answer", ""),
        }
        triggered_rows.append(row)

        if answer_changed:
            answer_changed_count += 1

        if accepted:
            accepted_after_trigger += 1
            accepted_rows.append(row)
            if answer_changed:
                accepted_changed_count += 1

        if (not raw_correct) and memory_correct:
            raw_wrong_to_memory_correct += 1
            corrected_rows.append(row)
            tail_gain_corrected.append(record.get("tail_gain"))
            bottom10_gain_corrected.append(record.get("bottom10_gain"))
        elif raw_correct and (not memory_correct):
            raw_correct_to_memory_wrong += 1
            harmed_rows.append(row)
            tail_gain_harmed.append(record.get("tail_gain"))
            bottom10_gain_harmed.append(record.get("bottom10_gain"))
        elif (not raw_correct) and (not memory_correct):
            raw_wrong_to_memory_wrong += 1
            if answer_changed:
                changed_but_still_wrong_rows.append(row)
        elif raw_correct and memory_correct:
            raw_correct_to_memory_correct += 1

        if not answer_changed:
            unchanged_rows.append(row)

    summary = {
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "task": args.task,
        "total_records": total_records,
        "phase_two_candidates": phase_two_candidates,
        "triggered_count": triggered_count,
        "accepted_after_trigger": accepted_after_trigger,
        "accepted_after_trigger_ratio": round(accepted_after_trigger / triggered_count, 6) if triggered_count else 0.0,
        "answer_changed_count": answer_changed_count,
        "answer_changed_ratio": round(answer_changed_count / triggered_count, 6) if triggered_count else 0.0,
        "accepted_changed_count": accepted_changed_count,
        "accepted_changed_ratio": round(accepted_changed_count / accepted_after_trigger, 6) if accepted_after_trigger else 0.0,
        "raw_wrong_to_memory_correct": raw_wrong_to_memory_correct,
        "raw_correct_to_memory_wrong": raw_correct_to_memory_wrong,
        "raw_wrong_to_memory_wrong": raw_wrong_to_memory_wrong,
        "raw_correct_to_memory_correct": raw_correct_to_memory_correct,
        "corrected_count": len(corrected_rows),
        "harmed_count": len(harmed_rows),
        "changed_but_still_wrong_count": len(changed_but_still_wrong_rows),
        "unchanged_count": len(unchanged_rows),
        "tail_gain_corrected": summarize_group(tail_gain_corrected),
        "tail_gain_harmed": summarize_group(tail_gain_harmed),
        "bottom10_gain_corrected": summarize_group(bottom10_gain_corrected),
        "bottom10_gain_harmed": summarize_group(bottom10_gain_harmed),
    }

    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_jsonl(os.path.join(args.output_dir, "triggered_samples.jsonl"), triggered_rows)
    write_jsonl(os.path.join(args.output_dir, "accepted_after_trigger.jsonl"), accepted_rows)
    write_jsonl(os.path.join(args.output_dir, "corrected.jsonl"), corrected_rows)
    write_jsonl(os.path.join(args.output_dir, "harmed.jsonl"), harmed_rows)
    write_jsonl(os.path.join(args.output_dir, "changed_but_still_wrong.jsonl"), changed_but_still_wrong_rows)
    write_jsonl(os.path.join(args.output_dir, "unchanged.jsonl"), unchanged_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
