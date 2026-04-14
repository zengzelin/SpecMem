# Coding Agent Execution Checklist for SpecMem Follow-ups

This checklist is the short operational version of `analysis/plan_2026-04-14_memory_followups.md`.

---

## Goal
Turn the current task-dependent memory results into a cleaner, testable system by focusing on:

1. selective memory activation
2. calibration-aware routing
3. lower-noise prompt formats
4. better per-sample logging and comparison tooling

---

## Priority order

### P0. Keep the current narrative straight
- Do **not** assume memory is universally beneficial.
- Do **not** make task-aware retrieval the main branch yet.
- Treat `direct_attributes` as a positive signal.
- Treat `relative_position` as a calibration failure case.

---

## P1. Implement selective memory activation

### Task 1
Add per-task memory policy controls.

Required capabilities:
- enable/disable memory per task family
- override memory threshold per task family
- override top-k per task family

Minimum target task families:
- `direct_attributes`
- `relative_position`

Expected outcome:
- easy experiments like “memory on for direct, off for relpos”

---

## P2. Implement dual-threshold gating

### Task 2
Split current memory behavior into two decisions:

1. `memory_trigger_threshold`
   - whether to use memory at all
2. `memory_accept_threshold`
   - whether to accept the memory-augmented small-model answer

Required outputs per sample:
- whether memory was triggered
- trigger score / trigger decision
- accept score / accept decision
- whether final answer came from small or large model

Expected outcome:
- less routing collapse on `relative_position`

---

## P3. Add prompt-mode ablations

### Task 3
Support the following prompt modes via CLI/config:
- `no_memory`
- `empty_memory_scaffold`
- `short_memory`
- `full_memory`

Definitions:
- `no_memory`: current baseline
- `empty_memory_scaffold`: same prompt structure, but no actual memory content
- `short_memory`: one-line or bullet-style memory hints only
- `full_memory`: current default memory prompt

Expected outcome:
- separate prompt-structure effects from actual memory-content effects

---

## P4. Reduce prompt noise

### Task 4
Make `k=1` the default low-noise memory baseline for next experiments.

Do not prioritize:
- increasing top-k
- complex retrieval variants

Expected outcome:
- shorter prompts
- lower prompt-side interference

---

## P5. Improve run logging

### Task 5
Add richer JSONL logging for every evaluated sample.

Must log:
- task family / test type
- memory enabled or disabled
- memory mode
- prompt mode
- retrieved memory IDs
- retrieval scores if available
- trigger threshold(s)
- acceptance threshold(s)
- trigger decision
- acceptance decision
- final branch (`small` / `large`)
- confidence score

Expected outcome:
- threshold tuning and routing analysis become inspectable

---

## P6. Strengthen analysis utilities

### Task 6
Extend comparison scripts to export:
- corrected examples
- harmed examples
- retrieved-but-no-gain examples
- summary counts by task family

### Task 7
Add support for manual audit tables with fields like:
- error type
- likely failure mode
- whether memory changed confidence only
- whether memory contradicted visible evidence

Expected outcome:
- easy creation of corrected/harmed error taxonomies

---

## P7. Run the next minimum experiment set

### Experiment Set A: selective activation
Run:
- baseline (`mem=off`)
- direct + `k=1`
- relpos + `k=1`
- direct + tuned threshold
- relpos + tuned threshold
- direct-on / relpos-off gated setting

Question:
- does selective activation outperform uniform memory-on?

### Experiment Set B: prompt ablation
Run on both task families:
- `no_memory`
- `empty_memory_scaffold`
- `short_memory`
- `full_memory`

Question:
- is the failure caused by scaffold, verbosity, or real memory content?

### Experiment Set C: dual-threshold gating
Run:
- single-threshold baseline
- dual-threshold routing

Track:
- all_acc
- small_ratio
- corrected_count
- harmed_count

Question:
- can direct gains be preserved without relpos collapse?

---

## P8. Deprioritized for now
Do not make these the next main branch:
- task-aware retrieval as headline method
- bigger top-k search
- visual memory as main focus
- dual memory as main focus
- broad benchmark expansion before routing is understood

---

## Suggested success criteria

A follow-up iteration is successful if it achieves at least one of these:

1. keeps `direct_attributes` above baseline while preventing `relative_position` degradation
2. shows that selective activation beats uniform memory-on
3. shows that prompt structure / verbosity explains part of the relpos failure
4. produces enough logging to explain why corrected and harmed samples happen

---

## Final main takeaway for the coding agent

Optimize **when memory is used** and **when it is trusted** before optimizing more advanced retrieval variants.
