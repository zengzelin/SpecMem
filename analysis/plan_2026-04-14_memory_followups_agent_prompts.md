# Ready-to-Send Coding-Agent Prompts for SpecMem Follow-ups

This document contains copy-pasteable prompts for coding agents.

Each prompt is scoped to a concrete implementation issue and references the relevant local files.

---

## Prompt 1 — Implement per-task memory policy controls

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Implement per-task memory policy controls for the DeepEyes evaluation path so memory behavior can differ between task families such as `direct_attributes` and `relative_position`.

Why:
Current experiments show memory can help `direct_attributes` but harm `relative_position`. We need selective memory activation instead of one global memory policy.

Primary file to modify:
- `eval_code_deepeyes/SpecEyes.py`

Secondary file if needed:
- `scripts/specmem.py`

Please do the following:
1. Add support for per-task overrides for at least:
   - memory enabled/disabled
   - memory threshold
   - top-k
2. Keep global defaults working when no per-task override is provided.
3. Use a CLI/config interface that is easy to run from the command line. Prefer simple CLI strings over heavy config redesign.
4. Ensure the resolved per-task policy is recorded in output JSONL for each sample.
5. Ensure output filenames/tags are distinguishable enough for these runs.

Suggested interface shape:
- `--memory_task_policy direct_attributes:on,relative_position:off`
- `--memory_task_thresholds direct_attributes:0.98,relative_position:0.97`
- `--memory_task_topk direct_attributes:1,relative_position:1`

Requirements:
- Do not redesign the whole config system.
- Prefer the smallest implementation that makes the experiments runnable.
- Preserve existing behavior when the new flags are not set.

Acceptance criteria:
- One run can enable memory for `direct_attributes` and disable it for `relative_position`
- JSONL output includes the resolved per-task memory settings
- Existing runs still work without the new flags

When done, report:
- which functions/files changed
- the exact new CLI flags
- one example command line for a gated run

---

## Prompt 2 — Implement dual-threshold memory gating

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Split the current single memory threshold into two explicit decisions:
1. whether to trigger memory
2. whether to accept the memory-augmented small-model answer

Why:
Current experiments suggest the main bottleneck is calibration/routing disruption, especially on `relative_position`. A single threshold is too coarse.

Primary files to modify:
- `eval_code_deepeyes/SpecEyes.py`
- `scripts/replay_memory_threshold.py`

Please do the following:
1. Add two explicit knobs:
   - `--memory_trigger_threshold`
   - `--memory_accept_threshold`
2. Update the evaluation flow so these two decisions are separate.
3. Log both decisions in output JSONL.
4. Update replay logic so replay can test different trigger/accept threshold pairs without rerunning models.
5. Add useful summary counts in replay output.

Per-sample JSONL should log at least:
- `memory_trigger_threshold`
- `memory_accept_threshold`
- `memory_triggered`
- `memory_accept_decision`
- final route / final model used

Replay summary should include at least:
- triggered count
- accepted-after-memory count
- large_to_small count
- near-threshold counts if available

Requirements:
- Keep implementation minimal and practical.
- Do not add a learned router.
- Prefer deterministic rule-based behavior only.
- Preserve old behavior if possible when only one threshold is provided, but do not overcomplicate compatibility.

Acceptance criteria:
- A raw run can be replayed over multiple trigger/accept threshold pairs
- JSONL clearly shows trigger vs accept behavior
- Replay summary is more informative than the current single-threshold version

When done, report:
- which routing logic changed
- which new fields are written per sample
- an example replay command using two thresholds

---

## Prompt 3 — Add prompt-mode variants for memory ablations

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Add prompt-mode variants so we can compare:
- `no_memory`
- `empty_memory_scaffold`
- `short_memory`
- `full_memory`

Why:
We need to determine whether current failures are caused by real memory content, prompt structure, or prompt verbosity.

Primary files to modify:
- `memory_aug/prompting.py`
- `eval_code_deepeyes/SpecEyes.py`

Please do the following:
1. Add a clean prompt-mode switch that can be selected from CLI/config.
2. Implement the four modes above.
3. Keep current behavior as `full_memory`.
4. Log the resolved prompt mode in output JSONL.
5. Keep the prompting code readable and avoid spreading string-assembly logic all over the codebase.

Definitions:
- `no_memory`: no prompt augmentation
- `empty_memory_scaffold`: same memory section headers/instructions, but no actual retrieved memory content
- `short_memory`: compressed memory bullets or one-line hints only
- `full_memory`: current detailed memory prompt

Requirements:
- Prefer a small extension of existing prompt helper functions.
- Do not duplicate large amounts of prompt code if avoidable.
- Preserve current default behavior when no new prompt mode is specified.

Acceptance criteria:
- All four modes can be selected without code edits
- JSONL records which prompt mode was used
- Existing full-memory behavior still works

When done, report:
- the new CLI/config knob name
- how each mode is implemented
- an example command line for each mode

---

## Prompt 4 — Add richer per-sample routing and memory logging

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Make the evaluation JSONL rich enough to diagnose corrected, harmed, and unchanged examples without manual reconstruction.

Why:
Current threshold and memory analyses are partly blind because key routing and memory metadata are missing from the raw run outputs.

Primary file to modify:
- `eval_code_deepeyes/SpecEyes.py`

Secondary consumers to keep in mind:
- `scripts/compare_memory_runs.py`
- `scripts/replay_memory_threshold.py`

Please add per-sample fields for at least:
- task family / test type
- memory enabled
- memory mode
- prompt mode / prompt variant
- trigger threshold
- accept threshold
- trigger decision
- accept decision
- retrieved memory IDs
- retrieved logic memories
- retrieved visual memories
- retrieval scores if available
- confidence score
- final model used (`small` / `large`)
- whether the final answer came from the memory-augmented small branch or from the large branch

Requirements:
- Prefer adding structured fields to existing JSONL rows.
- Do not create a separate logging system.
- Do not remove existing fields that current scripts rely on.

Acceptance criteria:
- A single run file contains enough metadata to understand routing and memory usage per sample
- Existing analysis scripts are not broken by the new fields

When done, report:
- the full list of newly added fields
- any field naming conventions chosen
- whether any analysis script updates are still needed downstream

---

## Prompt 5 — Extend replay tooling for calibration analysis

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Upgrade replay analysis so it supports the new calibration experiments and produces more informative summaries.

Primary file to modify:
- `scripts/replay_memory_threshold.py`

Why:
Replay is the fastest way to explore threshold behavior, but it currently reflects only the older single-threshold design.

Please do the following:
1. Support dual-threshold replay if dual-threshold logging is available.
2. Add summary outputs for trigger count vs accept count.
3. Report routing changes more explicitly.
4. If practical, add task-family-aware summary slices.

Requirements:
- Keep the script usable from CLI.
- Avoid turning it into a large framework.
- Reuse existing helpers where reasonable.

Acceptance criteria:
- Replay can evaluate multiple trigger/accept pairs from one raw run
- Output summaries are clearly more informative for calibration debugging

When done, report:
- the new CLI syntax
- what assumptions replay now makes about input JSONL fields
- an example replay command

---

## Prompt 6 — Extend corrected/harmed comparison tooling

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Improve corrected/harmed comparison outputs so they are more useful for manual analysis and taxonomy building.

Primary file to modify:
- `scripts/compare_memory_runs.py`

Why:
The current project narrative already relies on corrected/harmed examples. We need richer comparison outputs with routing and memory metadata.

Please do the following:
1. Extend the audit record with routing and memory metadata if present.
2. Add grouped summaries by task family.
3. Add grouped summaries by prompt mode and by whether memory actually triggered, if those fields are available.
4. Keep output easy to read and easy to annotate manually.

Suggested additional fields:
- task type
- prompt mode
- trigger decision
- accept decision
- retrieved memory IDs
- retrieval scores if available
- final route metadata

Acceptance criteria:
- comparison output clearly shows where gains and harms cluster
- corrected/harmed exports are sufficient for manual auditing without re-opening raw JSONL repeatedly

When done, report:
- new output files or summaries added
- which new metadata fields are consumed if present

---

## Prompt 7 — Add manual annotation export support

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Make it easy for humans to annotate corrected/harmed examples with error taxonomy labels.

Primary file to modify:
- `scripts/compare_memory_runs.py`

Optional fallback:
- create a small helper under `scripts/` only if needed

Please do the following:
1. Add an export format suitable for manual annotation.
2. Prefer CSV or JSONL with flat fields.
3. Include enough context to label examples without opening many additional files.

Suggested fields:
- `sample_id`
- `task_type`
- `question`
- `gold_answer`
- `base_pred`
- `candidate_pred`
- `base_correct`
- `candidate_correct`
- `candidate_retrieved_memory_count`
- `candidate_retrieved_logic_memories`
- `error_taxonomy_label`
- `human_notes`

Acceptance criteria:
- a human can open the export and start labeling immediately
- corrected and harmed examples are both supported

When done, report:
- the export file format
- the fields included
- the command used to generate it

---

## Prompt 8 — Expose retrieval scores and stable memory IDs

You are working in:
- `/Users/bytedance/Desktop/repos/specmem_fresh`

Goal:
Expose retrieval scores and stable memory IDs for debugging, without redesigning retrieval.

Primary files to modify:
- `memory_aug/retriever.py`
- possibly `memory_aug/store.py`
- possibly `memory_aug/schemas.py`

Why:
We need more retrieval transparency in logs and comparison outputs.

Please do the following:
1. Ensure each retrieved memory has a stable ID.
2. Return retrieval scores where practical.
3. Preserve compatibility with the existing prompting path.
4. Make it easy for evaluation code to log the extra retrieval metadata.

Requirements:
- Do not redesign the retrieval algorithm.
- Keep returned structures simple and serializable.

Acceptance criteria:
- evaluation JSONL can include memory IDs and retrieval scores
- prompting code still works with the returned memory objects

When done, report:
- exact returned fields added to retrieved memory objects
- any compatibility considerations for existing code

---

## Recommended rollout order

If you want to assign these to coding agents in order, use:

1. Prompt 1 — per-task policy controls
2. Prompt 2 — dual-threshold gating
3. Prompt 3 — prompt-mode variants
4. Prompt 4 — richer logging
5. Prompt 5 — replay tooling
6. Prompt 6 — comparison tooling
7. Prompt 7 — annotation export
8. Prompt 8 — retrieval transparency
