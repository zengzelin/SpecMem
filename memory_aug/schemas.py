from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def build_logic_memory_item(
    guideline: str,
    failure_mode: str,
    source_benchmark: str,
    source_question_id: str | int | None = None,
    subject: str | None = None,
    key_concepts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "memory_id": f"logic_{uuid4().hex[:12]}",
        "type": "logic",
        "guideline": guideline.strip(),
        "failure_mode": (failure_mode or "").strip(),
        "subject": (subject or "").strip(),
        "key_concepts": key_concepts or [],
        "source_benchmark": source_benchmark,
        "source_question_id": str(source_question_id) if source_question_id is not None else None,
        "created_at": _now_iso(),
        "usage_count": 0,
        "last_used_at": None,
    }


def build_visual_memory_item(
    guideline: str,
    visual_pattern: str,
    source_image_path: str,
    source_benchmark: str,
    source_question_id: str | int | None = None,
) -> dict[str, Any]:
    return {
        "memory_id": f"visual_{uuid4().hex[:12]}",
        "type": "visual",
        "guideline": guideline.strip(),
        "visual_pattern": (visual_pattern or "").strip(),
        "source_image_path": source_image_path,
        "source_benchmark": source_benchmark,
        "source_question_id": str(source_question_id) if source_question_id is not None else None,
        "created_at": _now_iso(),
        "usage_count": 0,
        "last_used_at": None,
    }
