from __future__ import annotations

from typing import Any

from .schemas import build_logic_memory_item, build_visual_memory_item


def build_memories_from_failure(
    *,
    failure_type: str,
    source_benchmark: str,
    source_question_id: str | int | None,
    source_image_path: str | None,
    logic_guideline: str | None = None,
    visual_guideline: str | None = None,
    visual_pattern: str | None = None,
    failure_mode: str | None = None,
    subject: str | None = None,
    key_concepts: list[str] | None = None,
) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []

    if failure_type in ("logical", "mixed") and logic_guideline:
        memories.append(
            build_logic_memory_item(
                guideline=logic_guideline,
                failure_mode=failure_mode or "",
                source_benchmark=source_benchmark,
                source_question_id=source_question_id,
                subject=subject,
                key_concepts=key_concepts or [],
            )
        )

    if failure_type in ("visual", "mixed") and visual_guideline and source_image_path:
        memories.append(
            build_visual_memory_item(
                guideline=visual_guideline,
                visual_pattern=visual_pattern or "",
                source_image_path=source_image_path,
                source_benchmark=source_benchmark,
                source_question_id=source_question_id,
            )
        )

    return memories
