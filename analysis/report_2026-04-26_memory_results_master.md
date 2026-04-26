# SpecEyes Memory 实验总报告与后续修改方向（2026-04-26）

## 1. 一句话总结

当前这条 memory 线已经完成了第一轮系统接入和实验验证，但结果说明：

- **memory 不是完全没用**，因为它在 `direct_attributes` 上曾经带来过稳定正收益；
- **但这条线目前还没有被做成稳定可用的方法**，因为 `stage17` 的新路由没有保住这部分收益；
- **真正的主瓶颈已经不是 retrieval 是否接入，而是 confidence / separability score 不适合当前 routing**；
- 尤其在 `relative_position` 上，当前 score 会把大量错误的 left/right 判断打成高分，导致 **不触发 memory，也不 fallback**。

因此，下一轮主线不应该继续放在 retrieval / task-aware / 再扫阈值，而应该转到：

> **重做 routing score：从全局单标量升级成局部窗口化 + 尾段/答案中心化 + trigger/accept 分离的 confidence system。**

---

## 2. 当前工作已经完成到什么程度

代码层面，第一轮系统改造已经基本完成：

1. task-aware memory policy 已经接入；
2. 两阶段 gating（raw first -> memory rerun on trigger -> accept/fallback）已经接入；
3. richer JSONL logging 已经接入；
4. replay / compare 脚本已经能读取 richer schema。

所以现在不是“代码还没实现”，而是：

- 框架已经在；
- memory 已经能跑；
- 路由也已经能跑；
- 但这版路由分数没有把系统真正带到更好的实验结果上。

---

## 3. 已有实验结果总表

下面只保留和当前判断最相关的核心结果。

### 3.1 Direct Attributes

| 阶段 | 配置 | all_acc | small_ratio | small_cnt | large_cnt | 解释 |
|---|---|---:|---:|---:|---:|---|
| baseline | `mem=off` | 90.43 | 57.39 | 66 | 49 | baseline |
| stage16 | `logic-small-k1, mthr=0.98` | 92.17 | 53.04 | 61 | 54 | 有明确提升 |
| stage16 | `logic-small-k2, mthr=0.98` | 92.17 | 51.30 | 59 | 56 | 有明确提升 |
| stage16 | `k1/k2 + task_aware` | 92.17 | 53.04 / 51.30 | 61 / 59 | 54 / 56 | 与 plain 一样 |
| stage17 | `k1, acc=0.9775, trig=0.97` | 90.43 | 60.87 | 70 | 45 | 回到 baseline |
| stage17 | `k1, acc=0.98, always-memory` | 92.17 | 53.04 | 61 | 54 | 仍能维持 stage16 收益 |
| stage17 | `k1, acc=0.98, trig=0.97` | 90.43 | 57.39 | 66 | 49 | 回到 baseline |
| stage17 | `k2, acc=0.9775, trig=0.97` | 90.43 | 60.87 | 70 | 45 | 回到 baseline |

### 3.2 Relative Position

| 阶段 | 配置 | all_acc | small_ratio | small_cnt | large_cnt | 解释 |
|---|---|---:|---:|---:|---:|---|
| baseline | `mem=off` | 86.84 | 75.00 | 57 | 19 | baseline |
| stage15 | `k3, mthr=0.9725` | 86.84 | 69.74 | 53 | 23 | 只能恢复到 baseline |
| stage15 | `k3, compact_spatial` | 81.58 | 76.32 | 58 | 18 | 明显退化 |
| stage15 | `k1, compact_spatial, task_aware` | 82.89 | 78.95 | 60 | 16 | 明显退化 |
| stage16 | `k1, task_aware, mthr=0.98` | 84.21 | 40.79 | 31 | 45 | 仍低于 baseline |
| stage16 | `k2, mthr=0.98` | 82.89 | 39.47 | 30 | 46 | 明显退化 |
| stage16 | `k2, task_aware, mthr=0.98` | 82.89 | 39.47 | 30 | 46 | 与 plain 一样 |
| stage16 | `k3, task_aware, mthr=0.98` | 82.89 | 44.74 | 34 | 42 | 明显退化 |
| stage17 | `k1, acc=0.97, trig=0.95` | 84.21 | 94.74 | 72 | 4 | 没有优化，路由明显失真 |
| stage17 | `k1, acc=0.97, trig=0.97` | 84.21 | 94.74 | 72 | 4 | 与上面一样 |
| stage17 | `k2, acc=0.9725, trig=0.95` | 84.21 | 92.11 | 70 | 6 | 没有优化，路由明显失真 |
| stage17 | `k2, acc=0.97, trig=0.95` | 84.21 | 94.74 | 72 | 4 | 没有优化，路由明显失真 |

---

## 4. 当前结果能支持什么结论

### 4.1 可以确认：memory 不是完全没用

`direct_attributes` 上已经有过：
- baseline `90.43`
- memory `92.17`

所以当前不能得出“memory 完全无效”的结论。

更准确的结论是：

> **memory 在某些 object-/attribute-centric 任务上是有正信号的。**

这条结论依然成立。

---

### 4.2 也可以确认：memory 目前还不稳定

如果这条线真的已经成熟，那么 stage17 在引入更细路由后，应该：
- 至少保住 direct 上已有收益；
- 至少不要让 relpos 路由更失真。

但实际结果不是这样：
- direct 的触发式配置又回到了 baseline；
- relpos 继续失败，而且 small_ratio 进一步异常升高。

所以当前更准确的判断是：

> **memory 这条线目前还是“有局部有效信号的实验性模块”，不是“已经稳定成立的方法”。**

---

### 4.3 task-aware retrieval 不是主瓶颈

当前 stage16 / stage17 的结果都说明：
- `task_aware` 和 plain retrieval 差异很小；
- 没有形成主收益来源；
- 不能解释当前系统为什么没优化成功。

所以这条线现在不该作为主叙事。

---

### 4.4 relative_position 的主问题是 routing score，而不是 retrieval 本身

这点是当前最重要的结论。

在 relpos 的 stage17 eval 里，能直接看到大量样本属于：
- 答案错了；
- 但 `confidence_score` 仍然很高；
- `memory_triggered = false`；
- `memory_accept_decision = true`；
- 最终 `use_model = small`。

典型例子包括：
- wheelchair vs lamp post：gold=left, pred=right, score=0.9786
- blue truck vs white vehicle：gold=right, pred=left, score=0.9759
- baby carriage vs cone：gold=right, pred=left, score=0.9807
- red chair vs road：gold=right, pred=left, score=0.9844

这说明：

> **当前 separability / confidence score 对 relpos 的 left/right 错答几乎没有识别力。**

因此当前主问题不是：
- retrieval 没命中；
- trigger 阈值差一点；
- prompt 再改一行就好了；

而是：

> **score 本身没有把真正不可靠的 small answer 标出来。**

---

## 5. 当前系统失败在什么地方

### 5.1 失败点一：全局单标量 score 太扁平

当前路由还是高度依赖一个全局 `confidence_score`。

这会带来三个问题：
1. 局部错误会被平均掉；
2. 最终答案段没有被单独强调；
3. 同一个分数被拿去同时做 trigger / accept / fallback，职责过重。

对 relpos 这种任务，这个问题尤其严重。

---

### 5.2 失败点二：当前 score 不够答案中心化

当前多模态问题里，真正决定 correctness 的往往是：
- 最终 option token
- yes/no token
- 最后一行短答案
- left/right / above/below 这类最后几 token 的决策

如果让所有 token 同权，前面的解释、模板、无关文本都会把真正关键的答案信号稀释掉。

---

### 5.3 失败点三：两阶段 gating 已经接上，但 score 没有为两阶段服务

现在系统已经能：
- raw first
- trigger memory rerun
- accept / fallback

但问题是：
- trigger 用的还是当前不合适的 score；
- accept 也还是围绕这个 score；
- 所以两阶段结构虽然在，决策依据却不对。

这也是为什么 stage17 的 relpos 会出现：
- trigger 几乎为 0；
- 但错误 small 仍然大量被接受。

---

## 6. 因此，下一步的主线应该是什么

当前最推荐的主线不是：
- 再做 retrieval trick
- 再扫更多 top-k
- 再加更多 task-aware retrieval 规则
- 再尝试更多 prompt 变体

而应该是：

## 主线：重做 routing score

也就是：

1. 从单个全局 scalar 变成 **score profile**；
2. 从 trace-level confidence 变成 **answer-centric confidence**；
3. 让 trigger 和 accept 使用不同 metric；
4. 显式比较 raw vs memory 的 score 改善幅度。

---

## 7. 下一轮代码应该怎么修改

下面是明确的代码方向，不是泛泛建议。

### 7.1 必须新增的分数

下一轮建议优先新增以下几类 score：

1. `tail_score`
   - 只看最后一段 token 的 score
   - 适合衡量最终答案段是否稳定

2. `lowest_group_score`
   - 把 token 序列切成窗口，取最差窗口
   - 适合发现局部崩点

3. `bottom10_group_score`
   - 对最差 10% 窗口取平均
   - 比 `lowest_group` 更稳，更适合做 trigger

4. `answer_span_score`
   - 只对最终答案区域算 score
   - 对 relpos / yes-no / option 类任务尤其重要

---

### 7.2 trigger 和 accept 要分开

下一轮默认建议：

- `trigger_metric = bottom10_group_score`
- `accept_metric = tail_score`

直觉是：
- trigger 的目标是发现“局部脆弱但可能可修复”的样本；
- accept 的目标是判断“最终答案段是否足够可靠”。

这比继续让单个 global scalar 同时承担两种决策更合理。

---

### 7.3 要加入 raw vs memory 的 delta logging

当前只有：
- `base_confidence_score`
- `confidence_score`

这远远不够。

下一轮应该新增：
- `tail_score_raw`
- `tail_score_memory`
- `bottom10_score_raw`
- `bottom10_score_memory`
- `answer_score_raw`
- `answer_score_memory`

以及：
- `tail_gain`
- `bottom10_gain`
- `answer_gain`

这样后面才能判断：
- memory 是否真的修复了关键区域；
- 还是只是重写了一遍回答；
- corrected / harmed 到底和哪些局部 score 变化有关。

---

### 7.4 relpos 要优先试 answer-focused score

当前 relpos 的失败高度集中在：
- left/right
- above/below
- closer/farther

这些任务最终 correctness 就在最后那几个 token。

因此 relpos 下一轮最值得优先试的是：
- final option token score
- final short tail score
- answer span score

而不是继续看全序列平均 separability。

---

## 8. 优先修改哪些文件

### `eval_code_deepeyes/SpecEyes.py`

这是主修改文件，需要重点改：

1. `_run_small_model_pass(...)`
   - 现在这里生成后只给单个 `confidence_score`
   - 下一轮要改成先算 score profile，再输出多个 summary

2. `should_trigger_memory(...)`
   - 支持 `trigger_metric` 配置
   - 默认改为 `bottom10_group_score`

3. `_process_batch_small_model_once(...)`
   - 让 rerun / accept 逻辑使用新的 metric
   - 同时记录 raw vs memory 的多种 score

4. `build_result_record(...)`
   - 把新的 score profile 和 gain 都写进 JSONL

### `scripts/replay_memory_threshold.py`

下一轮应该支持：
- 按不同 metric replay
- 至少支持 `tail_score`、`bottom10_group_score`
- 输出对应 metric 下的 triggered / accepted / small ratio

### `scripts/compare_memory_runs.py`

下一轮应该增加：
- raw vs memory profile 对比
- `tail_gain`
- `bottom10_gain`
- `answer_gain`

方便直接定位 corrected / harmed 的原因。

---

## 9. 暂时不建议优先做什么

下一轮不建议优先主攻：

1. 更多 task-aware retrieval 规则
2. 更多 top-k sweep
3. 更多 prompt variant 当主线
4. visual memory / dual memory 扩展

原因很简单：

> 当前不是 retrieval 命中不到，而是 score 不知道哪些样本错得离谱。

在这个问题没解决之前，继续加 retrieval 复杂度很可能只是把系统弄得更乱。

---

## 10. 推荐的下一轮实验顺序

### 第一轮：只改 score，不动 retrieval

新增：
- `tail_score`
- `lowest_group_score`
- `bottom10_group_score`

默认试：
- `trigger = bottom10_group`
- `accept = tail`

观察：
- judge acc
- small ratio
- corrected / harmed
- 高置信错答是否减少

### 第二轮：只在 relpos 上加 answer-focused score

测试：
- answer span score
- final option token score
- short tail score

目标：
- 专门降低 relpos 里的高置信错答

### 第三轮：分析 memory gain

重点看：
- `tail_gain`
- `answer_gain`
- corrected / harmed 的 profile 变化

目标：
- 判断 memory 是否真的在修关键区域

---

## 11. 最终建议

当前最准确的工作判断是：

> Memory 线已经完成了第一轮系统接入，但 stage17 说明主瓶颈不在 retrieval，而在 routing score。下一轮应该停止把主要精力放在 retrieval / task-aware / top-k 上，转而把 confidence 从“全局单标量”升级成“局部窗口化 + 尾段 + 答案中心化 + memory-gain”的 score system。

如果只保留一个最小可执行建议，就是：

> **先新增 `bottom10_group_score`、`tail_score` 和 `answer_span_score`；默认用 `bottom10_group` 触发 memory，用 `tail` 决定 small accept，再用 `gain` 判断 memory 是否真的修复了关键区域。**
