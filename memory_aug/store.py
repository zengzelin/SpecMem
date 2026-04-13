from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_memories(memory_file: str | Path) -> list[dict[str, Any]]:
    path = Path(memory_file)
    if not path.exists():
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_memories(memory_file: str | Path, memories: list[dict[str, Any]]) -> None:
    path = Path(memory_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def save_memory(memory_file: str | Path, memory: dict[str, Any]) -> None:
    memories = load_memories(memory_file)
    memories.append(memory)
    save_memories(memory_file, memories)


def update_memory_usage(memory_file: str | Path, memory_id: str) -> None:
    memories = load_memories(memory_file)
    changed = False
    timestamp = datetime.utcnow().isoformat(timespec="seconds")

    for memory in memories:
        if memory.get("memory_id") == memory_id:
            memory["usage_count"] = int(memory.get("usage_count", 0)) + 1
            memory["last_used_at"] = timestamp
            changed = True
            break

    if changed:
        save_memories(memory_file, memories)


def ensure_memory_dir(memory_dir: str | Path) -> Path:
    path = Path(memory_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logic_memory_file(memory_dir: str | Path) -> Path:
    return ensure_memory_dir(memory_dir) / "logic_memories.json"


def get_visual_memory_file(memory_dir: str | Path) -> Path:
    return ensure_memory_dir(memory_dir) / "visual_memories.json"
