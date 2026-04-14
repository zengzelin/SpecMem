# SpecMem Follow-up Implementation Issues for Coding Agents

This document breaks the current follow-up plan into concrete implementation issues.

Each issue is written so that a coding agent can pick it up with minimal extra context.

---

## Issue 1 — Add per-task memory policy controls

### Objective
Allow memory behavior to vary by task family instead of using one global setting for all tasks.

### Why this matters
Current experiments show:
- `direct_attributes` can benefit from memory
- `relative_position` can degrade under the same memory settings

So memory should be configurable per task family.

### Primary files
- `eval_code_deepeyes/SpecEyes.py`
- optionally `scripts/specmem.py` if it is used as the launcher wrapper

### Likely code locations
- argument parsing: `eval_code_deepeyes/SpecEyes.py:33`
- memory tag generation: `eval_code_deepeyes/SpecEyes.py:89`
- threshold selection logic: `eval_code_deepeyes/SpecEyes.py:107`
- main evaluation loop in `SpecEyes.py` where `test_type` is already available per sample

### Required changes
Add support for per-task overrides for at least:
- memory enabled/disabled
- memory threshold
- top-k
- prompt mode (if cheap to support here)

### Suggested interface options
Prefer one of these:

#### Option A: dedicated CLI fields
- `--memory_task_policy direct_attributes:on,relative_position:off`
- `--memory_task_thresholds direct_attributes:0.98,relative_position:0.97`
- `--memory_task_topk direct_attributes:1,relative_position:1`

#### Option B: JSON string or config blob
- `--memory_task_policy_json '{"direct_attributes":{"enabled":true},"relative_position":{"enabled":false}}'`

Option A is simpler for immediate use.

### Output changes
For each sample, output JSONL should record the effective resolved policy, for example:
- `task_type`
- `memory_policy_enabled`
- `memory_policy_threshold`
- `memory_policy_top_k`

### Acceptance criteria
- Can run memory enabled on `direct_attributes` and disabled on `relative_position` in one command/config
- Output JSONL clearly shows the resolved per-task policy
- Output filename/tag includes enough info to distinguish per-task policy runs

### Nice-to-have
- sensible fallback to global defaults when a task has no override

---

## Issue 2 — Split single memory threshold into dual-threshold gating

### Objective
Separate:
1. the decision to trigger memory
2. the decision to accept the memory-augmented small-model answer

### Why this matters
Current single-threshold logic mixes two different behaviors.
This likely contributes to routing/calibration collapse on `relative_position`.

### Primary files
- `eval_code_deepeyes/SpecEyes.py`
- `scripts/replay_memory_threshold.py`

### Likely code locations
- threshold parsing and resolution: `eval_code_deepeyes/SpecEyes.py:41`, `eval_code_deepeyes/SpecEyes.py:107`
- routing logic in the main inference loop
- replay logic: `scripts/replay_memory_threshold.py:198`

### Required changes
Introduce two explicit knobs:
- `memory_trigger_threshold`
- `memory_accept_threshold`

Behavior should look like:
1. determine whether memory should be applied
2. if memory is applied and a small-model answer is produced, decide whether to accept it or escalate

### Suggested interface
Add CLI args such as:
- `--memory_trigger_threshold`
- `--memory_accept_threshold`

Keep old `--memory_score_threshold` only if backward compatibility is cheap; otherwise replace it.

### Output changes
For each sample, log:
- `memory_trigger_threshold`
- `memory_accept_threshold`
- `memory_triggered` (bool)
- `memory_accept_decision` (bool or enum)
- `route_before_accept`
- `final_use_model`

### Replay tool changes
`scripts/replay_memory_threshold.py` should support replaying dual-threshold decisions without rerunning the models.
Add summary counts for:
- triggered count
- accepted-after-memory count
- large_to_small changes
- trigger-only changes
- accept-only changes

### Acceptance criteria
- Can run and log with separate trigger and accept thresholds
- Replay script can evaluate different trigger/accept pairs from one raw run
- Summary output clearly separates trigger behavior from accept behavior

---

## Issue 3 — Add prompt-mode variants for memory ablations

### Objective
Support clean prompt ablations without editing code each time.

### Why this matters
We need to separate:
- no-memory baseline
- prompt scaffold effects
- concise memory content effects
- full memory content effects

### Primary files
- `memory_aug/prompting.py`
- `eval_code_deepeyes/SpecEyes.py`

### Likely code locations
- `augment_small_model_prompt()`: `memory_aug/prompting.py:80`
- `augment_large_model_prompt()`: `memory_aug/prompting.py:128`
- CLI parsing for prompt style/mode: `eval_code_deepeyes/SpecEyes.py:58`, `eval_code_deepeyes/SpecEyes.py:61`

### Required prompt modes
Implement at least:
- `full_memory`
- `short_memory`
- `empty_memory_scaffold`
- `no_memory`

### Definitions
- `full_memory`: current default behavior
- `short_memory`: one-line or bullet-style compressed memory hints
- `empty_memory_scaffold`: same scaffold headers/instructions, but no actual retrieved memory content
- `no_memory`: no prompt augmentation

### Suggested interface
Either:
- extend `--memory_prompt_style`

or better:
- add `--memory_prompt_mode_variant` with choices above

### Output changes
Log the resolved prompt mode in per-sample JSONL, e.g.:
- `memory_prompt_variant`

### Acceptance criteria
- All four prompt variants can be selected from CLI/config
- Prompt builder functions remain readable and do not fork into unmaintainable branches
- Output JSONL captures which prompt mode was used

### Nice-to-have
- allow task-specific prompt variants later without redesign

---

## Issue 4 — Make low-noise `k=1` a clean first-class setting

### Objective
Make `k=1` the easiest default for the next experiment round and ensure its behavior is clearly logged.

### Why this matters
Current evidence does not show a clear gain from larger `k`, while higher `k` likely increases prompt noise.

### Primary files
- `eval_code_deepeyes/SpecEyes.py`
- `memory_aug/retriever.py`

### Required changes
- ensure `logic_top_k=1` is easy to set and shows up clearly in run tags/logs
- if per-task overrides are added, allow `k=1` by task family
- return/log retrieval scores if practical

### Output changes
Log at least:
- `logic_top_k`
- `visual_top_k`
- number of retrieved memories actually inserted

### Acceptance criteria
- easy to launch `k=1` runs without code edits
- run outputs clearly distinguish `k=1` from older settings

### Important note
This is a cleanup/support issue, not a major retrieval redesign.

---

## Issue 5 — Add richer per-sample routing and memory logs

### Objective
Make it possible to debug why a sample was corrected, harmed, or unaffected.

### Why this matters
Threshold tuning is currently partly blind because the raw JSONL does not expose enough routing metadata.

### Primary file
- `eval_code_deepeyes/SpecEyes.py`

### Secondary consumers
- `scripts/compare_memory_runs.py`
- `scripts/replay_memory_threshold.py`

### Required fields to log per sample
At minimum:
- `task_type`
- `memory_enabled`
- `memory_mode`
- `memory_prompt_variant`
- `memory_trigger_threshold`
- `memory_accept_threshold`
- `memory_triggered`
- `memory_accept_decision`
- `retrieved_logic_memories`
- `retrieved_visual_memories`
- retrieved memory IDs
- retrieval scores if available
- `confidence_score`
- `use_model`
- whether final answer came from memory-augmented small branch or large branch

### Acceptance criteria
- corrected/harmed analysis can be done from one run file without additional manual reconstruction
- replay and compare scripts can consume the new fields directly

### Nice-to-have
- a compact log mode and a verbose log mode

---

## Issue 6 — Extend replay tooling for calibration analysis

### Objective
Upgrade replay analysis so threshold studies can be run faster and with more diagnostic value.

### Why this matters
Replay is currently the fastest way to test threshold ideas, but it still reflects the old single-threshold logic.

### Primary file
- `scripts/replay_memory_threshold.py`

### Required changes
Add support for:
- dual-threshold replay
- reporting trigger/accept counts separately
- reporting near-threshold examples separately for trigger and accept
- optional breakdown by task family

### Suggested outputs
For each replay setting, report:
- all_acc / proxy_acc
- small_ratio
- triggered_count
- accepted_count
- large_to_small count
- small_to_large count where meaningful
- near-trigger-threshold count
- near-accept-threshold count

### Acceptance criteria
- Can replay multiple trigger/accept pairs from one raw run
- Output makes it easy to identify calibration-sensitive regions

---

## Issue 7 — Extend corrected/harmed comparison tooling

### Objective
Make comparison outputs more useful for manual auditing and error taxonomy.

### Why this matters
Current report conclusions already depend on corrected/harmed examples, so this analysis path should be stronger.

### Primary file
- `scripts/compare_memory_runs.py`

### Likely code locations
- audit record builder: `scripts/compare_memory_runs.py:166`
- corrected/harmed extraction: `scripts/compare_memory_runs.py:216`

### Required changes
Add to audit records:
- task family
- prompt mode
- trigger/accept decisions
- retrieved memory IDs
- retrieval scores if available
- final route metadata

Also add grouped summaries such as:
- corrected/harmed counts by task family
- corrected/harmed counts by prompt mode
- corrected/harmed counts by whether memory actually triggered

### Acceptance criteria
- comparison output is enough to support manual failure taxonomy labeling
- grouped summary highlights where gains and harms concentrate

---

## Issue 8 — Add support for manual error taxonomy exports

### Objective
Make it easy to annotate corrected/harmed examples with human labels.

### Why this matters
A useful next-step narrative depends on identifying whether memory helps object-centric tasks and hurts spatial-reference tasks.

### Primary file
- `scripts/compare_memory_runs.py`

### Optional secondary location
- a new lightweight helper under `scripts/` if needed

### Required changes
Export CSV or JSONL with fields prepared for annotation, for example:
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

### Acceptance criteria
- human reviewers can label corrected/harmed examples without manually reconstructing context from raw JSONL

---

## Issue 9 — Expose retrieval scores and stable memory IDs for debugging

### Objective
Improve retrieval transparency without redesigning retrieval itself.

### Why this matters
We need to know not just which memories were returned, but how strongly they matched.

### Primary files
- `memory_aug/retriever.py`
- possibly `memory_aug/store.py`
- possibly `memory_aug/schemas.py`

### Required changes
- return stable memory IDs in a guaranteed way
- optionally return retrieval scores with each selected memory
- preserve compatibility with current prompt builder expectations

### Suggested output shape
For example, each retrieved memory item could include:
- `memory_id`
- `score`
- `guideline`
- existing metadata fields

### Acceptance criteria
- retrieval diagnostics can be surfaced in JSONL logs
- no large redesign of retrieval logic is required

---

## Issue 10 — Keep retrieval redesign explicitly out of scope for this round

### Objective
Prevent scope drift.

### Why this matters
Current evidence does not justify spending the next cycle on:
- major task-aware retrieval redesign
- larger top-k exploration
- visual/dual-memory expansion as the main branch

### Action for coding agents
Do **not** start by redesigning retrieval scoring unless another issue explicitly requires a small compatibility update.

### Acceptance criteria
- implementation effort stays focused on selective activation, calibration, prompt ablations, and logging

---

## Suggested implementation sequence

### Sequence A — Minimum viable infrastructure
1. Issue 1 — per-task memory policy
2. Issue 2 — dual-threshold gating
3. Issue 3 — prompt-mode variants
4. Issue 5 — richer logging

### Sequence B — Fast analysis support
5. Issue 6 — replay tooling
6. Issue 7 — comparison tooling
7. Issue 8 — annotation export

### Sequence C — Light retrieval transparency only
8. Issue 9 — retrieval scores + stable IDs

---

## Definition of done for the next coding round

The next coding round is successful if the repo can support these experiments cleanly without code editing between runs:

1. `direct_attributes` memory on / `relative_position` memory off
2. single-threshold vs dual-threshold routing
3. `no_memory` vs `empty_memory_scaffold` vs `short_memory` vs `full_memory`
4. corrected/harmed export with routing metadata
5. threshold replay with richer calibration summaries
