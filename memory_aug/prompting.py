from __future__ import annotations


def format_logic_memories(memories: list[dict]) -> str:
    if not memories:
        return "None."

    lines = []
    for idx, memory in enumerate(memories, 1):
        guideline = memory.get("guideline", "").strip()
        failure_mode = memory.get("failure_mode", "").strip()
        if failure_mode:
            lines.append(f"{idx}. [{failure_mode}] {guideline}")
        else:
            lines.append(f"{idx}. {guideline}")
    return "\n".join(lines)


def format_visual_memories(memories: list[dict]) -> str:
    if not memories:
        return "None."

    lines = []
    for idx, memory in enumerate(memories, 1):
        guideline = memory.get("guideline", "").strip()
        visual_pattern = memory.get("visual_pattern", "").strip()
        if visual_pattern:
            lines.append(f"{idx}. [pattern: {visual_pattern}] {guideline}")
        else:
            lines.append(f"{idx}. {guideline}")
    return "\n".join(lines)


def augment_small_model_prompt(
    question_prompt: str,
    visual_memories: list[dict],
    logic_memories: list[dict],
    benchmark: str,
) -> str:
    visual_block = format_visual_memories(visual_memories)
    logic_block = format_logic_memories(logic_memories)

    if benchmark == "pope":
        return f"""Question:
{question_prompt}

Relevant visual warnings from past failures:
{visual_block}

Relevant logical guidelines from past failures:
{logic_block}

Use the image and the retrieved memories to answer carefully.
Answer yes or no only.
"""

    return f"""Question:
{question_prompt}

Relevant visual warnings from past failures:
{visual_block}

Relevant logical guidelines from past failures:
{logic_block}

Use the image and the retrieved memories to answer carefully.
If a visual warning applies, avoid that mistake.
If a logical guideline applies, follow it before answering.

Answer:
"""


def augment_large_model_prompt(
    question_prompt: str,
    visual_memories: list[dict],
    logic_memories: list[dict],
    draft_answer: str | None,
    benchmark: str,
) -> str:
    visual_block = format_visual_memories(visual_memories)
    logic_block = format_logic_memories(logic_memories)
    draft_block = draft_answer.strip() if draft_answer else "None."

    return f"""Question:
{question_prompt}

Retrieved visual warnings:
{visual_block}

Retrieved logical guidelines:
{logic_block}

Smaller model draft:
{draft_block}

Now solve the problem carefully using the image and the information above.
"""
