from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .store import (
    get_logic_memory_file,
    get_visual_memory_file,
    load_memories,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "question",
    "the",
    "them",
    "there",
    "this",
    "to",
    "using",
    "when",
    "whether",
    "with",
    "yes",
    "no",
}

SPATIAL_FAILURE_MODES = {
    "viewer_frame_confusion",
    "scene_layout_guess",
    "reference_object_drift",
    "size_bias_in_relation",
    "orientation_leakage",
    "descriptor_mismatch",
    "tiny_target_guessing",
}

COLOR_FAILURE_MODES = {
    "color_prior_or_bleed",
    "multi_color_mixup",
    "occluded_attribute_prior",
}

COUNT_FAILURE_MODES = {
    "count_over_under",
    "nested_scene_counting",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    return [token for token in tokens if token not in STOPWORDS]


def _overlap_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0

    q_counter = Counter(q)
    t_counter = Counter(t)

    overlap = sum(min(q_counter[k], t_counter[k]) for k in q_counter.keys())
    return overlap / max(1, len(set(q)))


def _question_profile(question: str) -> dict[str, bool]:
    tokens = set(_tokenize(question))
    return {
        "spatial_side": "left" in tokens and "right" in tokens,
        "color": "color" in tokens,
        "counting": "count" in tokens or ("how" in tokens and "many" in tokens),
    }


def _task_aware_bonus(question: str, memory: dict) -> float:
    profile = _question_profile(question)
    failure_mode = memory.get("failure_mode", "").strip().lower()
    subject = memory.get("subject", "").strip().lower()
    key_concepts = {str(item).strip().lower() for item in memory.get("key_concepts", [])}

    bonus = 0.0
    if profile["spatial_side"]:
        if "relative position" in subject:
            bonus += 0.18
        if failure_mode in SPATIAL_FAILURE_MODES:
            bonus += 0.14
        if {"left", "right"} & key_concepts:
            bonus += 0.08

    if profile["color"]:
        if "color" in subject:
            bonus += 0.18
        if failure_mode in COLOR_FAILURE_MODES:
            bonus += 0.14

    if profile["counting"]:
        if "count" in subject:
            bonus += 0.18
        if failure_mode in COUNT_FAILURE_MODES:
            bonus += 0.14

    return bonus


def retrieve_logic_memories(
    question: str,
    memory_dir: str | Path,
    top_k: int = 3,
    retrieval_style: str = "default",
) -> list[dict]:
    memory_file = get_logic_memory_file(memory_dir)
    memories = load_memories(memory_file)

    scored = []
    for memory in memories:
        text = " ".join(
            [
                memory.get("guideline", ""),
                memory.get("failure_mode", ""),
                memory.get("subject", ""),
                " ".join(memory.get("key_concepts", [])),
            ]
        )
        score = _overlap_score(question, text)
        if retrieval_style == "task_aware":
            score += _task_aware_bonus(question, memory)
        if score > 0:
            scored.append((memory, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored[:top_k]]


def retrieve_visual_memories(
    image_ref: str | None,
    question: str,
    memory_dir: str | Path,
    top_k: int = 3,
) -> list[dict]:
    memory_file = get_visual_memory_file(memory_dir)
    memories = load_memories(memory_file)

    image_name = Path(image_ref).name if image_ref else None

    exact_matches = []
    fallback_scored = []

    for memory in memories:
        source_image_path = memory.get("source_image_path", "")
        if image_name and Path(source_image_path).name == image_name:
            exact_matches.append(memory)
            continue

        text = " ".join(
            [
                memory.get("guideline", ""),
                memory.get("visual_pattern", ""),
            ]
        )
        score = _overlap_score(question, text)
        if score > 0:
            fallback_scored.append((memory, score))

    if len(exact_matches) >= top_k:
        return exact_matches[:top_k]

    fallback_scored.sort(key=lambda x: x[1], reverse=True)
    merged = exact_matches + [m for m, _ in fallback_scored]
    return merged[:top_k]


def retrieve_dual_memories(
    question: str,
    image_ref: str | None,
    memory_dir: str | Path,
    logic_top_k: int = 3,
    visual_top_k: int = 3,
    retrieval_style: str = "default",
) -> tuple[list[dict], list[dict]]:
    logic_memories = retrieve_logic_memories(
        question=question,
        memory_dir=memory_dir,
        top_k=logic_top_k,
        retrieval_style=retrieval_style,
    )
    visual_memories = retrieve_visual_memories(
        image_ref=image_ref,
        question=question,
        memory_dir=memory_dir,
        top_k=visual_top_k,
    )
    return logic_memories, visual_memories
