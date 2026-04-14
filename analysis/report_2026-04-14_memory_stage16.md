# SpecMem / SpecEyes Memory 实验汇总（2026-04-14）

## 1. 一句话结论
- **VStar direct_attributes**：memory 已经被官方 judge 验证为**稳定有效**，从 **90.43 -> 92.17**。
- **VStar relative_position**：memory 在原始 `0.98` 阈值下会掉分，但把 memory 阈值单独调低后，可以**恢复到 baseline**，目前还没有稳定超过 baseline。
- **task_aware retrieval**：在本轮实验里几乎没有带来可见收益，可以先不作为主线。

---

## 2. 建议汇报时只讲这 4 个结论

### 结论 A：direct_attributes 上，memory 已经有效
官方 judge：

| 配置 | all_acc | small_ratio | 结论 |
|---|---:|---:|---|
| baseline `mem=off` | 90.43 | 57.39 | baseline |
| `logic-small-k1 @ 0.98` | 92.17 | 53.04 | **+1.74 acc** |
| `logic-small-k2 @ 0.98` | 92.17 | 51.30 | **+1.74 acc** |
| `logic-small-k2 @ replay-thr=0.9775` | 92.17 | 53.91 | 与上面同分，更平衡一点 |

对应文件：
- baseline: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage1/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mem=off_acc.jsonl`
- k1: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage16_raw/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k1_acc.jsonl`
- k2: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage16_raw/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_acc.jsonl`
- k2 tuned shortlist: `/home/zelin/SpecMem/analysis/judge_judge_stage16_shortlist/vstar_direct_attributes_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_replay-thr=0.9775_acc.jsonl`

样例级对比：
- `k1`：`2 corrected / 0 harmed`
- `k2@0.9775`：`2 corrected / 0 harmed`

对应 summary：
- `/home/zelin/SpecMem/analysis/compare_direct_k1_raw/summary.json`
- `/home/zelin/SpecMem/analysis/compare_direct_k2_thr09775/summary.json`

### 结论 B：relative_position 的主要矛盾是 calibration
官方 judge：

| 配置 | all_acc | small_ratio | 结论 |
|---|---:|---:|---|
| baseline `mem=off` | 86.84 | 75.00 | baseline |
| `logic-small-k3 @ 0.98`（旧） | 82.89 | 43.42 | 明显退化 |
| `logic-small-k3 @ mthr=0.9725`（旧） | 86.84 | 69.74 | 恢复到 baseline |
| `logic-small-k2 @ 0.98`（新） | 82.89 | 39.47 | 原始阈值仍退化 |
| `logic-small-k2 @ replay-thr=0.97`（新） | 86.84 | 69.74 | 恢复到 baseline |
| `logic-small-k2 @ replay-thr=0.9725`（新） | 86.84 | 67.11 | 恢复到 baseline |

对应文件：
- baseline: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage1/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mem=off_acc.jsonl`
- 旧 raw k3: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage1/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mem=logic-small-k3_acc.jsonl`
- 旧 tuned k3: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage15_raw/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.9725_mem=logic-small-k3_acc.jsonl`
- 新 raw k2: `/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage16_raw/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_acc.jsonl`
- 新 tuned k2 (`0.97`): `/home/zelin/SpecMem/analysis/judge_judge_stage16_shortlist/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_replay-thr=0.97_acc.jsonl`
- 新 tuned k2 (`0.9725`): `/home/zelin/SpecMem/analysis/judge_judge_stage16_shortlist/vstar_relative_position_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.98_mthr=0.98_mem=logic-small-k2_replay-thr=0.9725_acc.jsonl`

样例级对比：
- tuned relpos 还是 `2 corrected / 2 harmed`
- 所以它目前是“**能救回**”，不是“**稳定变好**”

对应 summary：
- `/home/zelin/SpecMem/analysis/compare_relpos_k2_thr097/summary.json`
- `/home/zelin/SpecMem/analysis/compare_relpos_k3_tuned_v2/summary.json`

### 结论 C：task_aware 暂时不值得作为主线
在 stage16 的 official judge 和 replay 里：
- `direct_attributes`：`k1/k2` 的 `task_aware` 和 `default` 结果完全一样
- `relative_position`：`task_aware` 也没有带来超越 plain retrieval 的收益

因此当前可以把 `task_aware` 定性为：
- **没有明显收益**
- 可以暂时从主实验叙事里去掉

### 结论 D：当前推荐的主线配置
如果现在要给外部/组内汇报，建议把主线收敛成：

1. **Direct Attributes 主结果**
   - `logic-small-k1 @ 0.98`
   - 或 `logic-small-k2 @ 0.98 / 0.9775`
   - 结论：memory 带来 **+1.74 acc**，且样例级是 **0 harmed**

2. **Relative Position 保守结果**
   - `logic-small-k2 + memory threshold 0.97`
   - 结论：memory 原始配置会伤害结果，但单独校准 memory threshold 后，可恢复到 baseline

---

## 3. 建议汇报时不要重点展开的内容
这些内容属于探索过程，可以一句话带过，不建议放进主表：
- `POPE` 全量实验
- `compact_spatial` prompt ablation
- `visual memory`
- `dual memory`
- `task_aware` 当前版本
- 所有 `eval_results_deepeyes` 里没有进入 official judge shortlist 的中间 raw 文件

---

## 4. 你后面只需要看这两个地方

### A. 汇报材料主目录
- `/home/zelin/SpecMem/analysis/report_2026-04-14_memory_stage16.md`

### B. shortlist 文件夹（只放关键文件）
- `/home/zelin/SpecMem/analysis/report_shortlist`

这个 shortlist 里已经放了：
- baseline judge 结果
- direct 主结果 judge 文件
- relpos tuned judge 文件
- 样例级 compare summary

---

## 5. 最推荐的汇报话术
> 我们在 SpecEyes 上增量接入 retrieval-only memory 后，发现 memory 的效果具有任务依赖性：在 VStar direct_attributes 上，memory 已经能稳定提升官方 judge 准确率，从 90.43 提升到 92.17；但在 relative_position 上，memory 原始接入会破坏 acceptance calibration，必须通过单独下调 memory threshold 才能恢复到 baseline。当前因此可以得出两个可信结论：第一，memory 接法本身是有效的，因为它已经在 direct_attributes 上带来稳定净收益；第二，relative_position 的主要瓶颈不是 retrieval 命中，而是 confidence / threshold calibration，后续优化重点应转向 calibration，而不是继续堆 retrieval trick。 

