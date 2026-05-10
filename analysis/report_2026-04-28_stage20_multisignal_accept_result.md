# Stage20 Multisignal Accept Result

## Run

- Date: 2026-04-28
- Benchmark: `vstar`
- Task: `direct_attributes`
- Output dir: `/home/zelin/SpecMem/eval_results_deepeyes/SpecEyes_vstar_stage20_da_multisignal_accept`
- Judge dir: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage20_da_multisignal_accept`

## Online Config

- `memory_prompt_style=answer_focus`
- `logic_top_k=1`
- `trigger_metric=confidence_score`
- `accept_metric=confidence_score`
- `memory_accept_policy=triggered_multisignal`
- `memory_accept_min_confidence=0.9`
- `memory_accept_min_delta_conf=-0.02`
- `memory_accept_min_retrieval_score=0.0`
- `memory_accept_answer_change=changed`

## Stabilization Fixes Needed Before Run

- Added Phase-I router OOM retry with recursive micro-batching.
- Added realtime `max_memory` allocation so small and large models can load across partially occupied GPUs.
- Added missing `normalize_answer()` used by triggered multisignal acceptance.

## Headline Metrics

| Run | all_acc | small_cnt | small_ratio | large_cnt |
|---|---:|---:|---:|---:|
| stage19 conf->conf | 90.4348 | 67 | 58.2609% | 48 |
| stage20 online multisignal | 90.4348 | 70 | 60.8696% | 45 |
| replay shortlist rank01 | 90.4348 | 71 | 61.7391% | 44 |
| replay shortlist rank02 | 90.4348 | 70 | 60.8696% | 45 |

## Interpretation

- Online stage20 preserved `all_acc` exactly.
- Online stage20 increased `small_cnt` by `+3` and reduced `large_cnt` by `-3` versus stage19.
- Online stage20 matched the replay-validated `60.8696%` small-ratio point.
- Stage19 vs stage20 changed `use_model` on `5` samples:
  - `4` samples changed from `large -> small`
  - `1` sample changed from `small -> large`
- Final predictions did not change between stage19 and stage20 on this task, so the gain here is routing efficiency rather than accuracy lift.

## Trigger Audit

Source: `/home/zelin/SpecMem/analysis/audit_stage20_da_multisignal_accept/summary.json`

- `phase_two_candidates=106`
- `triggered_count=26`
- `accepted_after_trigger=4`
- `accepted_after_trigger_ratio=15.3846%`
- `answer_changed_count=16`
- `accepted_changed_count=4`
- `corrected_count=12`
- `harmed_count=3`

## Takeaway

- The memory line is not yet improving `all_acc` on this task.
- But the new triggered multisignal control is now an actually working online mechanism:
  - it is stable enough to run end-to-end,
  - it preserves accuracy,
  - and it reduces fallback frequency.
- This is the first clean sign that memory can help as a control signal even when it does not yet help as a direct answer improver.
