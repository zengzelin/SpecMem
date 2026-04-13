from __future__ import annotations


def classify_failure_type(
    question: str,
    model_output: str,
    gold_answer: str | None = None,
) -> str:
    text = f"{question}\n{model_output}".lower()

    visual_keywords = [
        "image",
        "object",
        "color",
        "position",
        "region",
        "see",
        "look",
        "visual",
        "picture",
        "chart",
        "figure",
    ]
    logical_keywords = [
        "therefore",
        "because",
        "compare",
        "calculate",
        "reason",
        "logic",
        "unit",
        "count",
    ]

    visual_hit = any(k in text for k in visual_keywords)
    logical_hit = any(k in text for k in logical_keywords)

    if visual_hit and logical_hit:
        return "mixed"
    if visual_hit:
        return "visual"
    if logical_hit:
        return "logical"
    return "logical"
