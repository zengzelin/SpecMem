# SpecMem Coding Agent Handoff (2026-04-14)

## What has already been implemented

The current codebase has already been updated to support a first usable version of task-aware memory control and two-stage memory gating.

### Main code changes already in place

#### 1. Task-aware memory policy
File:
- `eval_code_deepeyes/SpecEyes.py`

Implemented:
- `--memory_task_policy`
- per-task override support for:
  - memory on/off
  - prompt style
  - acceptance threshold
  - logic top-k
  - trigger threshold

Related helpers already added:
- `parse_memory_task_policy(...)`
- `infer_memory_task(...)`
- `resolve_memory_policy(...)`

Current policy format:
- `task:on|off:prompt_style:accept_threshold:logic_top_k:trigger_threshold`

Example:
- `direct_attributes:on:compact_general:0.9775:1:0.96,relative_position:off`

---

#### 2. Prompt variants
File:
- `memory_aug/prompting.py`

Implemented prompt styles:
- `default`
- `compact_spatial`
- `compact_general`
- `empty_scaffold`
- `no_memory`

This is intended to support baseline/scaffold/compact/full-memory prompt ablations.

---

#### 3. Retrieval metadata
File:
- `memory_aug/retriever.py`

Retrieved memories are now enriched with metadata fields:
- `_retrieval_score`
- `_retrieval_rank`
- `_retrieval_match`

This was added for later analysis/debugging without redesigning retrieval itself.

---

#### 4. Richer per-sample JSONL outputs
File:
- `eval_code_deepeyes/SpecEyes.py`

The run outputs now include richer routing/memory metadata such as:
- `memory_task`
- `memory_policy`
- `memory_prompt_style_applied`
- `memory_mode_applied`
- `memory_trigger_threshold_applied`
- `memory_acceptance_threshold_applied`
- `memory_triggered`
- `memory_accept_decision`
- `base_confidence_score`
- `base_small_answer`
- retrieval counts

This is meant to support later corrected/harmed analysis and calibration analysis.

---

#### 5. Two-stage small-model flow
File:
- `eval_code_deepeyes/SpecEyes.py`

The current small-model branch is no longer a simple one-pass memory prompt.

It now follows this structure:
1. run the small model with the raw prompt
2. get `base_confidence_score`
3. decide whether to trigger memory using `trigger_threshold`
4. if triggered, rerun the small model with the memory-augmented prompt
5. use the acceptance threshold to decide small vs large

This is the current intended flow.

---

#### 6. Analysis scripts updated for richer schema
Files:
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`

Implemented:
- compatibility with richer routing fields
- awareness of trigger/accept metadata
- additional trigger/accept fields in compare outputs

This is not yet a fully general dual-threshold offline replay framework, but it is already usable for basic analysis.

---

## What has been verified already

The following has already been checked:

1. Python syntax compile passes for:
- `eval_code_deepeyes/SpecEyes.py`
- `memory_aug/prompting.py`
- `memory_aug/retriever.py`
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`

2. `scripts/replay_memory_threshold.py` was smoke-tested on synthetic richer-schema JSONL and produced summaries including:
- `triggered_count`
- `accepted_count`

---

## What is likely still imperfect / what should be reviewed carefully

The current code should be treated as a **first working pass**, not as final polished infrastructure.

Please review carefully:

1. Whether the two-stage gating flow in `eval_code_deepeyes/SpecEyes.py` is logically consistent in all branches.
2. Whether `memory_triggered` and `memory_accept_decision` are always written correctly.
3. Whether the rerun-with-memory path preserves the intended sample metadata.
4. Whether output records remain backward-compatible enough for existing analysis scripts.
5. Whether any field naming or result-builder logic should be cleaned up slightly without broad refactoring.

Important: do **not** redesign the whole system unless necessary.
The goal is to make the current implementation correct and ready for experiments, not to start a large rewrite.

---

## Recommended next task for the coding agent

The next coding agent should do a focused **review + stabilization pass** on the current implementation.

### Scope
Work inside:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Focus files:
- `eval_code_deepeyes/SpecEyes.py`
- `memory_aug/prompting.py`
- `memory_aug/retriever.py`
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`

### Goal
Make sure the current task-aware memory control and two-stage gating implementation is reliable enough to start new experiments.

### What to do
1. Review the recently added memory policy and two-stage gating logic.
2. Fix any obvious bugs or inconsistent branches.
3. Add only minimal necessary cleanup if it improves correctness or stability.
4. Keep output JSONL schema usable for the existing analysis workflow.
5. Preserve the current design direction:
   - task-aware memory control
   - raw-first then memory-on-trigger
   - accept-on-threshold
   - richer routing/memory logging

### What not to do
- Do not redesign retrieval.
- Do not expand to large new features.
- Do not refactor the whole pipeline.
- Do not remove current logging fields unless clearly broken.

### Expected deliverables
At the end, report:
1. which files were modified
2. what bugs or issues were found
3. what was fixed
4. what verification was run
5. whether the code is ready for the next experiment batch

---

## One prompt to give the next coding agent

Use the following prompt directly:

---

You are working in:
- `/home/zelin/SpecMem`

Please take over the current task-aware memory control and two-stage memory gating implementation.

The following has already been added and should be treated as the intended design direction, not discarded:
- per-task memory policy in `eval_code_deepeyes/SpecEyes.py`
- prompt variants in `memory_aug/prompting.py`
- retrieval score metadata in `memory_aug/retriever.py`
- richer JSONL routing/memory fields
- a raw-first, memory-on-trigger, accept-on-threshold small-model flow
- richer-schema compatibility updates in `scripts/replay_memory_threshold.py` and `scripts/compare_memory_runs.py`

Your job is **not** to redesign the system.
Your job is to do a focused **review + stabilization pass** so the code is reliable enough to start the next experiment batch.

Please review and, if needed, fix only the following files:
- `eval_code_deepeyes/SpecEyes.py`
- `memory_aug/prompting.py`
- `memory_aug/retriever.py`
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`

Priority:
1. verify the two-stage gating flow is logically correct in all branches
2. verify `memory_triggered` / `memory_accept_decision` / threshold fields are written consistently
3. verify rerun-with-memory behavior preserves the right metadata
4. verify analysis scripts remain compatible with the new richer schema
5. run minimal verification and fix any obvious bugs

Constraints:
- do not do a large refactor
- do not redesign retrieval
- do not broaden scope beyond making the current implementation correct and usable

At the end, report:
- files changed
- issues found
- fixes made
- checks run
- whether the repo is ready for the next experiment batch

---

## Final note

If the next agent finds only small issues, it should fix them and stop.
If it finds one or two critical logic bugs in the two-stage flow, it should fix those first before doing any further cleanup.
The target outcome is simple:

> make the current implementation trustworthy enough that new experiments can be launched immediately afterward.
