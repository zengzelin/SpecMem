from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .store import (
    get_logic_memory_file,
    get_visual_memory_file,
    load_memories,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _overlap_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0

    q_counter = Counter(q)
    t_counter = Counter(t)

    overlap = sum(min(q_counter[k], t_counter[k]) for k in q_counter.keys())
    return overlap / max(1, len(set(q)))


def retrieve_logic_memories(
    question: str,
    memory_dir: str | Path,
    top_k: int = 3,
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
) -> tuple[list[dict], list[dict]]:
    logic_memories = retrieve_logic_memories(
        question=question,
        memory_dir=memory_dir,
        top_k=logic_top_k,
    )
    visual_memories = retrieve_visual_memories(
        image_ref=image_ref,
        question=question,
        memory_dir=memory_dir,
        top_k=visual_top_k,
    )
    return logic_memories, visual_memories
