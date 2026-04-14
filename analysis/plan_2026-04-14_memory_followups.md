# SpecMem / SpecEyes Memory Follow-up Plan (2026-04-14)

## 1. Purpose

This document summarizes the current experimental bottlenecks in the SpecMem-for-SpecEyes integration and lays out a concrete follow-up plan for future coding/experiment agents.

It is meant to answer four questions:

1. What have we already established from the current runs?
2. What are the main unresolved technical problems?
3. Which hypotheses are worth testing next?
4. What code and experiment changes should be prioritized?

---

## 2. Current status summary

### 2.1 What is already supported by current results

#### A. Memory is not universally useless
On `vstar_direct_attributes`, logic-memory already shows a real positive signal under official judge:

- baseline (`mem=off`): `90.43`
- `logic-small-k1 @ mthr=0.98`: `92.17`
- `logic-small-k2 @ mthr=0.98`: `92.17`

Relevant files:
- `judge_results_deepeyes/SpecEyes_vstar_stage1/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mem=off_acc.jsonl`
- `judge_results_deepeyes/SpecEyes_vstar_stage16_raw/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k1_acc.jsonl`
- `judge_results_deepeyes/SpecEyes_vstar_stage16_raw/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_acc.jsonl`

This means the current memory integration can already produce net gains on at least one task family.

#### B. Relative-position is the main failure case
On `vstar_relative_position`, raw memory hurts performance:

- baseline (`mem=off`): `86.84`
- `logic-small-k2 @ mthr=0.98`: `82.89`

But tuning the memory threshold can recover the score back to baseline:

- tuned setting (older k3 run): `86.84`

Relevant files:
- `judge_results_deepeyes/SpecEyes_vstar_stage1/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mem=off_acc.jsonl`
- `judge_results_deepeyes/SpecEyes_vstar_stage16_raw/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_acc.jsonl`
- `judge_results_deepeyes/SpecEyes_vstar_stage15_raw/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.9725_mem=logic-small-k3_acc.jsonl`

This strongly suggests that the major issue is not “memory never helps”, but “memory currently disrupts routing/calibration on some task types”.

#### C. Task-aware retrieval is not currently the bottleneck
Current stage16 results show no meaningful difference between default retrieval and the current task-aware variant.

This means task-aware retrieval should not be the main experimental narrative right now.

---

## 3. Main technical problems

## 3.1 Problem 1: Memory gains are task-dependent, not stable across task families

Observed pattern:
- `direct_attributes`: positive gain
- `relative_position`: raw degradation, tuned recovery only

Interpretation:
- Memory is currently beneficial for some object-/attribute-centric reasoning tasks.
- Memory is currently unreliable for spatial relation tasks.

Implication:
- We should stop treating memory as a universally-on module.
- The next design should explicitly support selective activation.

---

## 3.2 Problem 2: The most immediate issue is calibration / routing disruption

The strongest current evidence is the shift in `small_ratio`.

Example:
- `relative_position` baseline small ratio: `75.0`
- `relative_position` raw memory small ratio: `39.47`
- `relative_position` tuned memory small ratio: `69.74`

Interpretation:
- Memory is changing the decision boundary of whether the smaller model answer is accepted.
- Even if retrieved memory is not completely irrelevant, its presence changes the confidence dynamics.
- This causes too many cases to fall into the larger-model path or otherwise damages the routing behavior.

Implication:
- The next main line should focus on calibration-aware memory activation, not just retrieval tuning.

---

## 3.3 Problem 3: Retrieval quality may still matter, but current evidence points first to prompt-side noise and triggering logic

We do **not** yet have enough evidence to say retrieval quality is irrelevant.

However, the currently strongest signals are:
- threshold tuning matters more than current retrieval variant changes
- `k=1` and `k=2` are already tied on direct_attributes
- task-aware retrieval has no visible gain

Interpretation:
- The main bottleneck is likely not “we need more sophisticated retrieval logic immediately”.
- It is more likely that the current memory payload is too noisy or too aggressively activated.

Implication:
- Prioritize lower-noise memory usage before investing in more advanced retrieval policies.

---

## 3.4 Problem 4: Current evidence is still small-sample and should be presented carefully

The current direct_attributes gain is only about +1.74 points on 115 examples.
This is promising, but should be described as:
- a consistent positive signal
- not yet a universally stable conclusion

Implication:
- Future experiments should preserve honesty about sample size.
- We should expand only after the triggering/calibration logic is cleaner.

---

## 4. Recommended main research direction

### Calibration-aware selective memory activation

This should be the central design direction.

Core idea:
- Memory should not be applied uniformly.
- The system should decide **when** memory is likely to help.
- The system should also decide **whether** to trust the memory-augmented answer.

This is a better main direction than:
- adding more task-aware retrieval rules
- increasing top-k
- expanding to more memory variants too early

A good internal summary is:

> The current challenge is not only retrieval relevance, but memory-trigger and acceptance calibration.

---

## 5. Detailed follow-up plan

## 5.1 Priority 1 — Fix activation/calibration before expanding retrieval complexity

### 5.1.1 Introduce selective memory activation by task family

#### Goal
Avoid applying memory to task families where it is currently harmful.

#### Minimal experiment
Implement a simple task-family gate:
- enable memory by default for `direct_attributes`
- disable memory by default for `relative_position`, or apply a stricter trigger rule

#### Why this matters
This tests whether the current gains can be preserved while avoiding known failure modes.

#### Suggested implementation options
- Hard-coded task whitelist/blacklist in the routing path
- Per-task config in YAML/CLI args
- Per-task memory threshold override

#### Expected signal
- Aggregate score should improve relative to “uniform memory on”.
- `relative_position` should stop paying the current penalty.

---

### 5.1.2 Replace a single memory threshold with two-stage gating

#### Goal
Separate the decision of “should I consult memory?” from “should I trust the memory-augmented answer?”.

#### Proposed thresholds
1. `memory_trigger_threshold`
   - controls whether memory retrieval/augmentation is activated
2. `memory_accept_threshold`
   - controls whether the small-model answer after memory should be accepted or escalated

#### Why this matters
A single threshold currently mixes two different decisions:
- memory usage
- answer acceptance

Those should be decoupled.

#### Expected benefit
- Better control of over-triggering on relative_position
- Better preservation of gains on direct_attributes

#### Suggested code changes
- Add separate config fields in the argument parser / YAML config
- Log both decisions in output JSONL for later diagnosis
- Preserve backward compatibility only if cheap; otherwise prefer direct replacement

---

### 5.1.3 Add uncertainty-triggered memory heuristics

#### Goal
Only activate memory on examples where it is likely to help.

#### Possible trigger features
- small model confidence / reliability score
- margin between yes/no routing decision
- whether zoom tool was used
- answer format instability
- known task family

#### Minimal version
Start with simple heuristics instead of training a learned gate.

Example:
- for direct_attributes: use memory when confidence is below threshold but not extremely low
- for relative_position: use memory only in a narrow uncertainty band, or do not use it by default

#### Why this matters
Current evidence suggests all-example memory activation is too blunt.

---

## 5.2 Priority 2 — Reduce prompt-side memory noise

### 5.2.1 Fix `k=1` as the default low-noise baseline

#### Current evidence
- `direct_attributes`: `k=1` and `k=2` tie
- no evidence that larger `k` helps enough to justify added noise

#### Recommendation
Adopt `k=1` as the default baseline for the next round.

#### Why this matters
This reduces prompt length and lowers the chance that memory text distorts the original reasoning.

---

### 5.2.2 Compare short-memory vs full-memory formatting

#### Goal
Test whether relpos degradation is caused by memory content length/format, not only trigger policy.

#### Minimal ablation
Run:
1. no memory
2. empty memory scaffold
3. short memory (one-line guideline only)
4. full memory (current format)

#### Why this matters
This separates:
- prompt scaffold effects
- memory content effects
- memory verbosity effects

#### Expected outcome
If short memory works better than full memory, then prompt payload size/verbosity is a major issue.

---

### 5.2.3 Add an empty-memory scaffold control

#### Goal
Determine whether simply changing the prompt template already shifts routing behavior.

#### Minimal test
Use the exact same prompt structure as the memory setting, but provide no actual memory content.

#### Why this matters
If empty scaffold already hurts relpos, then prompt structure itself is part of the failure mode.

---

## 5.3 Priority 3 — Build error taxonomy before scaling up

### 5.3.1 Manually inspect corrected and harmed examples

#### Goal
Understand exactly which error types memory fixes and which error types it causes.

#### Suggested labels
- counting confusion
- attribute ambiguity
- left/right confusion
- reference object confusion
- occlusion/clutter
- memory contradicts current image
- memory changes confidence without improving reasoning

#### Why this matters
This will sharpen the research story.

A likely emerging narrative is:
- memory helps object-centric attribute questions
- memory hurts egocentric or reference-sensitive spatial judgments unless tightly gated

---

### 5.3.2 Inspect whether harmed relative_position cases cluster by sub-type

#### Goal
Find out whether a narrow sub-family is driving the relpos failures.

#### Possible subtypes
- left/right relations
- front/behind
- near/far
- ambiguous anchor entity
- multiple similar objects in scene

#### Why this matters
If the harms cluster tightly, we may want subtask-specific handling instead of treating all relpos questions the same.

---

## 5.4 Priority 4 — Only after the above, revisit retrieval improvements

### 5.4.1 Revisit task-aware retrieval later, not now

Current evidence does not justify task-aware retrieval as the next main branch.

It can be revisited only after:
- selective activation is in place
- prompt payload is cleaner
- error taxonomy is understood

---

### 5.4.2 Revisit visual/dual memory only after logic-memory is well-calibrated

Current evidence is not yet strong enough to justify spending major effort on:
- visual memory
- dual memory
- more elaborate retrieval pipelines

The logic-memory path should be made reliable first.

---

## 6. Recommended experiment matrix

## Phase A — Fast confirmation experiments

### A1. Selective activation baseline
Run:
- baseline (`mem=off`)
- direct_attributes + `k=1` + default threshold
- relative_position + `k=1` + default threshold
- direct_attributes + `k=1` + tuned threshold
- relative_position + `k=1` + tuned threshold
- task-family gated memory (direct on, relpos off or stricter)

#### Main question
Does selective activation outperform uniform memory-on?

---

### A2. Prompt payload ablation
Run on both direct_attributes and relative_position:
- no memory
- empty scaffold
- short memory
- full memory

#### Main question
Is the current failure caused by memory content, prompt structure, or verbosity?

---

### A3. Acceptance calibration ablation
Run:
- single-threshold baseline
- dual-threshold gating

Track:
- all_acc
- small_ratio
- large_acc
- count of corrected/harmed examples

#### Main question
Can we preserve direct gains while stopping relpos collapse?

---

## Phase B — Diagnosis experiments

### B1. Corrected/harmed case audit
For each shortlisted run, export corrected and harmed examples and annotate them manually.

#### Deliverable
A compact table of example IDs and failure labels.

---

### B2. Routing diagnostics
Add richer logging for each sample:
- whether memory was triggered
- retrieved memory IDs
- retrieval scores
- trigger score / confidence
- acceptance score
- whether final answer came from small or large model

#### Why this matters
Without this, threshold tuning remains blind.

---

## Phase C — Only if A/B succeed

### C1. Smarter retrieval revisits
Possible later directions:
- task-aware retrieval v2
- memory filtering by confidence or similarity gap
- different memory template by task family

### C2. Visual/dual memory revisits
Only revisit if:
- logic-memory path is stable
- calibration issues are understood
- there is a clear hypothesis for why extra modalities should help

---

## 7. Concrete implementation tasks for coding agents

### Task 1 — Add per-task memory config override
Implement support for:
- per-task enable/disable
- per-task threshold override
- per-task top-k override

Preferred interface:
- YAML config fields or CLI flags
- task name → memory policy mapping

---

### Task 2 — Add dual-threshold memory gating
Implement separate parameters for:
- triggering memory
- accepting memory-augmented small answers

Also record both decisions in output JSONL.

---

### Task 3 — Add prompt-mode variants
Implement prompt modes for:
- `no_memory`
- `empty_memory_scaffold`
- `short_memory`
- `full_memory`

This should be selectable via CLI/config without touching code each time.

---

### Task 4 — Improve logging for analysis
For every sample, log:
- task family
- memory enabled or not
- retrieved memory IDs
- retrieval scores
- chosen prompt mode
- trigger threshold(s)
- whether small answer was accepted
- whether final answer came from small or large branch

---

### Task 5 — Add analysis helpers for corrected/harmed diff
Create or extend analysis scripts to automatically compare two runs and export:
- corrected examples
- harmed examples
- unchanged examples
- summary counts by task family

---

## 8. What should *not* be the next main focus

The following directions should be deprioritized for now:

1. Making task-aware retrieval the main story
2. Increasing top-k beyond the current low-noise range
3. Investing heavily in visual memory before logic-memory is calibrated
4. Expanding benchmark coverage before activation/calibration behavior is understood

Reason:
- current evidence does not indicate these are the dominant bottlenecks
- they are likely to create more complexity before the current failure mode is isolated

---

## 9. Proposed working narrative for the next round

A strong internal framing is:

> Retrieval-only memory can provide real gains in SpecEyes, but the gains are task-dependent. The current bottleneck is not simply retrieval relevance; it is calibration-aware activation and acceptance. Memory should be treated as a selective module, not a universally-on augmentation.

A more concise version:

> The next phase should optimize **when to use memory**, not just **how to retrieve memory**.

---

## 10. Immediate next-step checklist

Recommended order:

1. Fix the next baseline to `logic-memory + k=1`
2. Add per-task memory on/off or per-task threshold override
3. Implement dual-threshold gating
4. Add prompt-mode ablations (`empty`, `short`, `full`)
5. Add richer JSONL logging
6. Run corrected/harmed manual audit
7. Only then consider revisiting task-aware / visual / dual memory

---

## 11. Final recommendation

If only one main line is pursued next, it should be:

### `calibration-aware selective memory activation`

This is the most evidence-supported direction based on current runs, and it is the most likely route to turning the current task-dependent results into a robust method story.
