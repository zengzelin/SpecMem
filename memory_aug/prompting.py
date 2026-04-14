from __future__ import annotations


COMPACT_SPATIAL_HINTS = {
    "viewer_frame_confusion": "Use the viewer's image perspective for left-right judgments.",
    "scene_layout_guess": "Compare the named objects directly, not the whole scene layout.",
    "descriptor_mismatch": "Match the full object descriptions before comparing positions.",
    "tiny_target_guessing": "Zoom in before judging tiny, distant, or cropped targets.",
    "reference_object_drift": "Localize the reference object first, then compare the target.",
    "size_bias_in_relation": "Use the objects' relative centers even when sizes differ a lot.",
    "orientation_leakage": "Ignore road direction, pose, and object facing direction.",
}


def format_logic_memories(memories: list[dict], bullet_style: bool = False) -> str:
    if not memories:
        return "None."

    lines = []
    for idx, memory in enumerate(memories, 1):
        guideline = memory.get("guideline", "").strip()
        failure_mode = memory.get("failure_mode", "").strip()
        prefix = "-" if bullet_style else f"{idx}."
        if failure_mode:
            lines.append(f"{prefix} [{failure_mode}] {guideline}")
        else:
            lines.append(f"{prefix} {guideline}")
    return "\n".join(lines)


def format_visual_memories(memories: list[dict], bullet_style: bool = False) -> str:
    if not memories:
        return "None."

    lines = []
    for idx, memory in enumerate(memories, 1):
        guideline = memory.get("guideline", "").strip()
        visual_pattern = memory.get("visual_pattern", "").strip()
        prefix = "-" if bullet_style else f"{idx}."
        if visual_pattern:
            lines.append(f"{prefix} [pattern: {visual_pattern}] {guideline}")
        else:
            lines.append(f"{prefix} {guideline}")
    return "\n".join(lines)


def _append_sections(question_prompt: str, sections: list[str]) -> str:
    blocks = [question_prompt.strip()]
    blocks.extend(section.strip() for section in sections if section and section.strip())
    return "\n\n".join(blocks).strip()


def _looks_like_spatial_side_question(question_prompt: str) -> bool:
    text = question_prompt.lower()
    return (
        " left or right " in text
        or " left side " in text
        or " right side " in text
        or ("left" in text and "right" in text and "side" in text)
    )


def _format_compact_logic_memories(memories: list[dict]) -> str:
    if not memories:
        return ""

    seen = set()
    lines = []
    for memory in memories:
        failure_mode = memory.get("failure_mode", "").strip()
        guideline = COMPACT_SPATIAL_HINTS.get(
            failure_mode,
            " ".join(memory.get("guideline", "").strip().split()),
        )
        if not guideline or guideline in seen:
            continue
        seen.add(guideline)
        lines.append(f"- {guideline}")
    return "\n".join(lines)


def augment_small_model_prompt(
    question_prompt: str,
    visual_memories: list[dict],
    logic_memories: list[dict],
    benchmark: str,
    prompt_style: str = "default",
) -> str:
    sections = []
    style = prompt_style or "default"
    use_compact_spatial = style == "compact_spatial" and _looks_like_spatial_side_question(question_prompt)

    if style == "no_memory":
        return question_prompt.strip()

    if use_compact_spatial and logic_memories:
        compact_logic = _format_compact_logic_memories(logic_memories)
        if compact_logic:
            sections.append(
                "Memory hints for this left-right judgment:\n"
                f"{compact_logic}"
            )
            sections.append(
                "Apply only the hints that match the visible objects. "
                "Compare the named objects directly in the image before answering."
            )
            return _append_sections(question_prompt, sections)

    if style == "empty_scaffold":
        sections.append("Relevant visual warnings from past failures:\nNone.")
        sections.append("Relevant logical guidelines from past failures:\nNone.")
    elif style == "compact_general":
        if visual_memories:
            sections.append(
                "Compact visual hints:\n"
                f"{format_visual_memories(visual_memories, bullet_style=True)}"
            )
        if logic_memories:
            sections.append(
                "Compact logical hints:\n"
                f"{format_logic_memories(logic_memories, bullet_style=True)}"
            )
    else:
        if visual_memories:
            sections.append(
                "Relevant visual warnings from past failures:\n"
                f"{format_visual_memories(visual_memories)}"
            )
        if logic_memories:
            sections.append(
                "Relevant logical guidelines from past failures:\n"
                f"{format_logic_memories(logic_memories)}"
            )

    if not sections:
        return question_prompt.strip()

    guidance = (
        "Use the image and the retrieved memories carefully. "
        "If a retrieved guideline applies, follow it before answering."
    )
    if style == "compact_general":
        guidance = "Use only the hints that clearly match the current image before answering."
    if benchmark == "pope":
        guidance += "\nAnswer yes or no only."

    sections.append(guidance)
    return _append_sections(question_prompt, sections)


def augment_large_model_prompt(
    question_prompt: str,
    visual_memories: list[dict],
    logic_memories: list[dict],
    draft_answer: str | None,
    benchmark: str,
) -> str:
    sections = []
    if visual_memories:
        sections.append(
            "Retrieved visual warnings:\n"
            f"{format_visual_memories(visual_memories)}"
        )
    if logic_memories:
        sections.append(
            "Retrieved logical guidelines:\n"
            f"{format_logic_memories(logic_memories)}"
        )
    if draft_answer and draft_answer.strip():
        sections.append(f"Smaller model draft:\n{draft_answer.strip()}")

    if not sections:
        return question_prompt.strip()

    sections.append("Now solve the problem carefully using the image and the information above.")
    return _append_sections(question_prompt, sections)
