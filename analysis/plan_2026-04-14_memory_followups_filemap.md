# SpecMem Follow-up File Map for Coding Agents

This document maps the follow-up plan to the most relevant files in `specmem_fresh`.

---

## 1. Files to modify first

## 1.1 `eval_code_deepeyes/SpecEyes.py`

### Why this file matters
This is the main integration point for:
- CLI/config parsing
- memory flags
- memory thresholds
- output file naming
- evaluation-time routing flow

### Relevant sections
- argument parsing: `eval_code_deepeyes/SpecEyes.py:33`
- memory defaults and tags: `eval_code_deepeyes/SpecEyes.py:83`
- acceptance threshold logic: `eval_code_deepeyes/SpecEyes.py:107`

### This file should be modified for
1. per-task memory enable/disable
2. per-task threshold override
3. separate trigger vs accept thresholds
4. prompt-mode selection (`no_memory`, `empty_memory_scaffold`, `short_memory`, `full_memory`)
5. richer per-sample logging fields
6. output filename tags for new ablations

### Suggested first-pass edits
- extend CLI args/config surface first
- then thread new options into the inference/evaluation loop
- then add JSONL logging for trigger/accept decisions

---

## 1.2 `memory_aug/prompting.py`

### Why this file matters
This file currently controls how retrieved memories are injected into prompts.

### Relevant sections
- logic formatting: `memory_aug/prompting.py:15`
- visual formatting: `memory_aug/prompting.py:30`
- small-model augmentation: `memory_aug/prompting.py:80`
- large-model augmentation: `memory_aug/prompting.py:128`

### This file should be modified for
1. `empty_memory_scaffold`
2. `short_memory`
3. keeping `full_memory` as current default
4. potentially task-family-specific prompt styles
5. reducing verbosity for spatial tasks

### Suggested first-pass edits
Add a prompt-mode switch in `augment_small_model_prompt()` and, if needed, `augment_large_model_prompt()`.

Recommended modes:
- `full_memory`
- `short_memory`
- `empty_memory_scaffold`
- `no_memory` (or handle upstream by skipping augmentation)

---

## 1.3 `memory_aug/retriever.py`

### Why this file matters
This file controls retrieval and current task-aware bonuses.

### Relevant sections
- task-aware bonus logic: `memory_aug/retriever.py:94`
- logic retrieval: `memory_aug/retriever.py:124`
- dual retrieval: `memory_aug/retriever.py:191`

### This file should be modified for
1. returning retrieval scores in addition to memory entries
2. enabling lower-noise `k=1` defaults cleanly
3. optional future filters such as score-gap filtering or confidence-aware filtering

### What not to do first
Do **not** make this the primary focus for large algorithmic changes yet.

### Suggested first-pass edits
- expose retrieval scores in the returned structure or add an optional debug mode
- preserve current behavior by default if compatibility is needed

---

## 1.4 `scripts/replay_memory_threshold.py`

### Why this file matters
This is already the main threshold replay / calibration analysis tool.

### Relevant sections
- CLI definition: `scripts/replay_memory_threshold.py:21`
- route replay logic: `scripts/replay_memory_threshold.py:198`
- threshold summary: `scripts/replay_memory_threshold.py:234`

### This file should be modified for
1. dual-threshold replay
2. reporting trigger-vs-accept decision changes separately
3. logging more diagnostics around near-threshold examples
4. supporting per-task threshold policy experiments

### Suggested first-pass edits
- extend replay model from single threshold to two thresholds
- add summary counters for:
  - memory triggered count
  - memory accepted count
  - trigger-only changes
  - accept-only changes

---

## 1.5 `scripts/compare_memory_runs.py`

### Why this file matters
This is already the most useful comparison script for corrected/harmed analysis.

### Relevant sections
- sample matching: `scripts/compare_memory_runs.py:18`
- audit record construction: `scripts/compare_memory_runs.py:166`
- corrected/harmed extraction: `scripts/compare_memory_runs.py:216`

### This file should be modified for
1. exporting more routing metadata
2. grouping corrected/harmed examples by task family
3. adding placeholders/fields for manual error taxonomy
4. supporting comparison across prompt modes and gating policies

### Suggested first-pass edits
- extend `build_audit_record()`
- include new fields such as:
  - memory prompt mode
  - trigger decision
  - accept decision
  - retrieved memory IDs
  - threshold settings

---

## 2. Files that are likely involved next

## 2.1 `memory_aug/store.py`

### Why it may matter
If retrieval output needs richer metadata (memory IDs, scores, source info), storage/loading code may need light extension.

### Likely usage
- expose stable IDs
- preserve metadata needed in logs

---

## 2.2 `memory_aug/schemas.py`

### Why it may matter
If new memory metadata fields are introduced, schema definitions may need updating.

### Likely usage
- retrieval score field
- task-family field
- abbreviated guideline field for short prompt mode

---

## 2.3 `scripts/specmem.py`

### Why it may matter
If this script is your main launcher/wrapper, it should expose the new experiment knobs cleanly.

### Likely usage
- easier batch launching of prompt-mode and gating ablations
- standardizing new CLI surfaces

---

## 2.4 `vis/vis_ablation_thres.py`

### Why it may matter
If threshold experiments remain central, this plotting script should be updated for new gating variants.

### Likely usage
- visualize single-threshold vs dual-threshold results
- compare task-family-specific thresholds

---

## 3. Lower priority files for now

## 3.1 `eval_code_thyme/SpecEyes.py`

### Why lower priority
Current experimental narrative is built around the DeepEyes path, not Thyme.

### Recommendation
Do not duplicate changes here until the DeepEyes branch is stable and useful.

---

## 3.2 `judge_code/*.py`

### Why lower priority
Current problem is not primarily in official judging logic.

### Recommendation
Only touch judge code if new outputs require compatibility changes.

---

## 3.3 visual/dual-memory-specific files

### Why lower priority
Current evidence does not support spending major engineering effort here before logic-memory calibration is fixed.

---

## 4. Suggested implementation order

### Step 1
Modify `eval_code_deepeyes/SpecEyes.py`

Add:
- per-task memory policy knobs
- dual-threshold knobs
- prompt-mode knobs
- richer output logging

### Step 2
Modify `memory_aug/prompting.py`

Add:
- `empty_memory_scaffold`
- `short_memory`
- cleaner prompt switching

### Step 3
Modify `scripts/replay_memory_threshold.py`

Add:
- dual-threshold replay support
- richer routing summaries

### Step 4
Modify `scripts/compare_memory_runs.py`

Add:
- new audit fields
- task-family grouping
- better corrected/harmed exports

### Step 5
Only then touch `memory_aug/retriever.py` if needed

Add only:
- retrieval score exposure
- light filtering hooks

Do not start with a major retrieval redesign.

---

## 5. Concrete mapping from plan items to files

### Plan item: per-task gating
Primary file:
- `eval_code_deepeyes/SpecEyes.py`

Possible secondary file:
- `scripts/specmem.py`

### Plan item: dual-threshold routing
Primary files:
- `eval_code_deepeyes/SpecEyes.py`
- `scripts/replay_memory_threshold.py`

### Plan item: prompt-mode ablation
Primary files:
- `memory_aug/prompting.py`
- `eval_code_deepeyes/SpecEyes.py`

### Plan item: richer run logging
Primary file:
- `eval_code_deepeyes/SpecEyes.py`

Secondary analysis consumers:
- `scripts/compare_memory_runs.py`
- `scripts/replay_memory_threshold.py`

### Plan item: corrected/harmed taxonomy support
Primary file:
- `scripts/compare_memory_runs.py`

### Plan item: retrieval debug metadata
Primary files:
- `memory_aug/retriever.py`
- possibly `memory_aug/store.py`

---

## 6. Practical recommendation for the next coding pass

If an agent is going to start coding immediately, the highest-value first target is:

1. `eval_code_deepeyes/SpecEyes.py`
2. `memory_aug/prompting.py`
3. `scripts/replay_memory_threshold.py`
4. `scripts/compare_memory_runs.py`

This sequence aligns with the main current bottleneck:
- not “better retrieval ideas first”
- but “better control over when memory is injected and trusted”
