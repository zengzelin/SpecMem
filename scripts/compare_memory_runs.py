import argparse
import json
import os
import re

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


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sample_id = build_sample_id(data, line_no)
            if sample_id in records:
                raise ValueError(f"Duplicate sample_id {sample_id} in {path}")
            records[sample_id] = data
    return records


def build_sample_id(record, line_no):
    result = record.get("result", {})
    for field in ("question_id", "pid", "idx", "image"):
        value = result.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"

    image_source = str(result.get("image_source", "")).strip()
    question = str(result.get("question", "")).strip()
    if image_source:
        return f"image_source:{image_source}|question:{question}"
    if question:
        return f"question:{question}"
    return f"line:{line_no}"


def normalize_answer(text):
    text = str(text or "").strip().lower()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \n\t\r.,;:!?\"'")

    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    letter_match = re.fullmatch(r"([a-f])(?:[\.\)])?", text)
    if letter_match:
        return letter_match.group(1)

    inline_letter_match = re.search(r"\b([a-f])(?:[\.\)])\b", text)
    if inline_letter_match:
        return inline_letter_match.group(1)

    return text


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


def is_correct(record):
    result = record.get("result", {})
    gold_text = result.get("answer", "")
    pred_text = result.get("pred_ans", "")

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
    if not gold or not pred:
        return False
    return gold == pred


def extract_text(content):
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    text_chunks = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                text_chunks.append(item.strip())
        elif isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def summarize_print_messages(print_messages, max_chars=1600):
    lines = []
    for message in print_messages or []:
        role = message.get("role", "unknown")
        text = extract_text(message.get("content", ""))
        if text:
            lines.append(f"{role}: {text}")

    summary = "\n".join(lines).strip()
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."


def build_audit_record(sample_id, base_record, candidate_record):
    base_result = base_record.get("result", {})
    candidate_result = candidate_record.get("result", {})
    candidate_logic = candidate_record.get("retrieved_logic_memories", [])
    candidate_visual = candidate_record.get("retrieved_visual_memories", [])

    return {
        "sample_id": sample_id,
        "question": candidate_result.get("question", base_result.get("question", "")),
        "gold_answer": candidate_result.get("answer", base_result.get("answer", "")),
        "base_pred": base_result.get("pred_ans", ""),
        "candidate_pred": candidate_result.get("pred_ans", ""),
        "base_correct": is_correct(base_record),
        "candidate_correct": is_correct(candidate_record),
        "base_use_model": base_record.get("use_model", ""),
        "candidate_use_model": candidate_record.get("use_model", ""),
        "base_confidence_score": base_record.get("confidence_score", None),
        "candidate_confidence_score": candidate_record.get("confidence_score", None),
        "base_judge_route": base_record.get("judge_tc", ""),
        "candidate_judge_route": candidate_record.get("judge_tc", ""),
        "candidate_retrieved_logic_memories": candidate_logic,
        "candidate_retrieved_visual_memories": candidate_visual,
        "candidate_retrieved_memory_count": len(candidate_logic) + len(candidate_visual),
        "base_print_messages_summary": summarize_print_messages(base_record.get("print_messages", [])),
        "candidate_print_messages_summary": summarize_print_messages(candidate_record.get("print_messages", [])),
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare a no-memory run with a memory-enabled run.")
    parser.add_argument("--base", required=True, help="Path to the no-memory jsonl file")
    parser.add_argument("--candidate", required=True, help="Path to the memory-enabled jsonl file")
    parser.add_argument("--output_dir", required=True, help="Directory for comparison reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_records = load_jsonl(args.base)
    candidate_records = load_jsonl(args.candidate)

    shared_sample_ids = sorted(set(base_records) & set(candidate_records))
    base_only = sorted(set(base_records) - set(candidate_records))
    candidate_only = sorted(set(candidate_records) - set(base_records))

    corrected = []
    harmed = []
    retrieved_no_gain = []

    for sample_id in shared_sample_ids:
        base_record = base_records[sample_id]
        candidate_record = candidate_records[sample_id]
        audit_record = build_audit_record(sample_id, base_record, candidate_record)

        base_correct = audit_record["base_correct"]
        candidate_correct = audit_record["candidate_correct"]
        retrieved_count = audit_record["candidate_retrieved_memory_count"]

        if not base_correct and candidate_correct:
            corrected.append(audit_record)
        elif base_correct and not candidate_correct:
            harmed.append(audit_record)
        elif retrieved_count > 0:
            retrieved_no_gain.append(audit_record)

    summary = {
        "base_file": os.path.abspath(args.base),
        "candidate_file": os.path.abspath(args.candidate),
        "matched_samples": len(shared_sample_ids),
        "base_only_samples": len(base_only),
        "candidate_only_samples": len(candidate_only),
        "base_accuracy": round(
            sum(is_correct(base_records[sample_id]) for sample_id in shared_sample_ids) / len(shared_sample_ids),
            4,
        )
        if shared_sample_ids
        else 0.0,
        "candidate_accuracy": round(
            sum(is_correct(candidate_records[sample_id]) for sample_id in shared_sample_ids) / len(shared_sample_ids),
            4,
        )
        if shared_sample_ids
        else 0.0,
        "corrected_count": len(corrected),
        "harmed_count": len(harmed),
        "retrieved_no_gain_count": len(retrieved_no_gain),
        "candidate_memory_hit_count": sum(
            1
            for sample_id in shared_sample_ids
            if (
                len(candidate_records[sample_id].get("retrieved_logic_memories", []))
                + len(candidate_records[sample_id].get("retrieved_visual_memories", []))
            )
            > 0
        ),
        "base_only_sample_ids": base_only,
        "candidate_only_sample_ids": candidate_only,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    corrected_path = os.path.join(args.output_dir, "corrected.jsonl")
    harmed_path = os.path.join(args.output_dir, "harmed.jsonl")
    retrieved_no_gain_path = os.path.join(args.output_dir, "retrieved_no_gain.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(corrected_path, corrected)
    write_jsonl(harmed_path, harmed)
    write_jsonl(retrieved_no_gain_path, retrieved_no_gain)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {summary_path}")
    print(f"Saved corrected cases to: {corrected_path}")
    print(f"Saved harmed cases to: {harmed_path}")
    print(f"Saved retrieved-no-gain cases to: {retrieved_no_gain_path}")


if __name__ == "__main__":
    main()
