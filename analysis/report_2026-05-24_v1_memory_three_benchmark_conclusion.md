# V1 Memory 跨三 Benchmark 结论总结（2026-05-24）

## 1. 这份结论在总结什么

这份文档只总结最初版 `v1 memory` 的效果，不混入后续 `stage17~23` 的 trigger / accept / selector 优化。

这里的 `v1 memory` 指的是：

- 在 `SpecEyes` 的 small-model answer 阶段前先检索 memory
- 把 memory 直接拼进 small model prompt
- 让 small model 带着 memory 重答

当前对应的主配置是：

- `--memory_enable`
- `--memory_mode logic_only`
- `--memory_prompt_mode small_only`
- `--memory_prompt_style default`
- `logic_top_k=1`

需要注意：

- `pope` 和 `hr` 这次重新补跑的是统一的 `k1`
- `vstar` 使用的是历史 `stage1` 结果，其中：
  - `direct_attributes` 代表结果是 `k3`
  - `relative_position` 同时保留了 `k1` 和 `k3`

---

## 2. 对比口径

三类对比对象分别是：

- `DeepEyes baseline`：只跑大模型，不走 speculative small branch
- `SpecEyes mem=off`：原始 `SpecEyes`，走 small/large 路由，但不加 memory
- `v1 memory`：原始 memory 注入版

需要特别说明：

- `vstar stage1` 同时有 `DeepEyes baseline`、`SpecEyes mem=off`、`v1 memory`
- 这次新补跑的 `pope` 和 `hr` 只有 `DeepEyes baseline` 与 `v1 memory`
- 因此 `pope` / `hr` 没有同批次 `SpecEyes mem=off` 数字，这一列记为 `-`

---

## 3. 数据集总表

| Benchmark | 数据集 | v1 配置 | DeepEyes baseline | SpecEyes mem=off | v1 memory | 相对 mem=off | 相对 baseline | small_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vstar` | `direct_attributes` | `k3` | 90.43 | 90.43 | 92.17 | +1.74 | +1.74 | 50.43% |
| `vstar` | `relative_position` | `k1` | 82.89 | 86.84 | 84.21 | -2.63 | +1.32 | 40.79% |
| `vstar` | `relative_position` | `k3` | 82.89 | 86.84 | 82.89 | -3.95 | +0.00 | 43.42% |
| `pope` | `adversarial` | `k1` | 78.80 | - | 80.87 | - | +2.07 | 27.20% |
| `pope` | `popular` | `k1` | 82.17 | - | 84.23 | - | +2.07 | 27.77% |
| `pope` | `random` | `k1` | 89.30 | - | 90.00 | - | +0.70 | 24.30% |
| `hr` | `hr_bench_4k` | `k1` | 76.01 | - | 71.36 | - | -4.65 | 82.41% |
| `hr` | `hr_bench_8k` | `k1` | 71.27 | - | 67.42 | - | -3.85 | 82.96% |

---

## 4. 结果怎么解读

### 4.1 VStar

`vstar` 的结论是“局部有效，但不稳定”。

| 数据集 | 现象 | 结论 |
|---|---|---|
| `direct_attributes` | `92.17 > 90.43` | memory 在属性类任务上有明确正信号 |
| `relative_position (k1)` | 比 `mem=off` 低 `2.63` | memory 没有保住原始 `SpecEyes` 的相对位置能力 |
| `relative_position (k3)` | 直接掉到 `82.89` | top-k 增大没有救回来，反而更差 |

因此：

> `vstar` 不能支持“v1 memory 普遍有效”，只能支持“它在 direct_attributes 上曾经有效”。 

---

### 4.2 POPE

`pope` 的结论是“这版 v1 接法在三个 split 上都稳定增益”。

| 数据集 | baseline | v1 memory | delta |
|---|---:|---:|---:|
| `adversarial` | 78.80 | 80.87 | +2.07 |
| `popular` | 82.17 | 84.23 | +2.07 |
| `random` | 89.30 | 90.00 | +0.70 |

同时，这三组的 `small_ratio` 只有大约 `24% ~ 28%`。

这说明：

- memory 没有把大量样本都强行留在 small branch
- 它只影响了一部分样本
- 这部分样本总体上带来了净收益

因此：

> `pope` 是当前最支持这条 memory 线的 benchmark。 

---

### 4.3 HR

`hr` 的结论是“这版 v1 接法明显退步”。

| 数据集 | baseline | v1 memory | delta |
|---|---:|---:|---:|
| `hr_bench_4k` | 76.01 | 71.36 | -4.65 |
| `hr_bench_8k` | 71.27 | 67.42 | -3.85 |

同时，`hr` 的 `small_ratio` 高达 `82%+`。

这说明：

- memory 版在 `hr` 上把过多样本留在了 small branch
- 这些 small answer 的可靠性并没有被保住
- 最终整体精度被明显拖低

因此：

> `hr` 是当前最明确的反例，说明这版 v1 memory 不能直接跨 benchmark 复用。 

---

## 5. 总结结论

如果只保留一句话：

> 最初版 `v1 memory` 在 `pope` 上稳定有效，在 `vstar` 上只有 `direct_attributes` 有局部正信号，在 `hr` 上明显无效。 

如果展开成三条：

1. `v1 memory` 不是完全没用，因为它在 `pope` 三个 split 上都提升，在 `vstar direct_attributes` 上也提升。
2. `v1 memory` 不是稳定方法，因为它在 `vstar relative_position` 上不能超过 `SpecEyes mem=off`，在 `hr` 上还出现了明显退化。
3. 当前最合理的判断不是“memory 没价值”，而是“这版最原始的 memory 注入方式只对部分任务有效，不能作为统一默认方案”。 

---

## 6. 当前最实用的判断

从“要不要继续沿这条线做实验”的角度看，当前最实用的判断是：

- 如果目的是证明“memory 接法本身是否可能有价值”，答案是 `有`，因为 `pope` 和 `vstar direct_attributes` 都给出了正信号。
- 如果目的是证明“这版 v1 是否已经可以作为通用方案”，答案是 `不可以`，因为 `hr` 和 `vstar relative_position` 都否定了这一点。
- 因此后续若要继续做 memory，最值得优先围绕 `pope` 这类 object-existence / hallucination-sensitive 面去推进，而不是再把当前 `v1` 直接平推到所有 benchmark。 

---

## 7. 对应结果文件

### VStar

- [SpecEyes_vstar_stage1](/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_vstar_stage1)

### POPE

- [SpecEyes_pope_v1_baseline](/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_pope_v1_baseline)
- [SpecEyes_pope_v1_memory](/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_pope_v1_memory)

### HR

- [SpecEyes_hr_v1_baseline](/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_hr_v1_baseline)
- [SpecEyes_hr_v1_memory](/home/zelin/SpecMem/judge_results_deepeyes/SpecEyes_hr_v1_memory)
