# SpecEyes Stage18 Rerun Audit Report（2026-04-27）

## 1. 一句话结论

这轮 rerun audit 把 stage18 的“最好组”和“最差组”都拆开看之后，结论比只看总表更明确：

> 当前 memory rerun 不是完全没用，而是**确实会修掉一部分错答**；但现有 accept/fallback 规则既不能稳定保留这些修复，也无法阻止大量“没修好甚至变差”的 rerun 留在 small branch。

因此，当前主问题不是：
- trigger metric 还不够多
- score 还没扫够

而是：
- **acceptance calibration 不成立**
- 并且 `DA` 和 `RP` 的 rerun 行为机制明显不同

---

## 2. 本次审计对象

本次选了 stage18 里两类最有代表性的配置做 audit：

### 2.1 baseline-best 代表组
- `trigger=confidence_score, accept=confidence_score`
- 这组在 stage18 中等价于回到 baseline

审计文件：
- `analysis/audit_stage18_da_conf_conf`
- `analysis/audit_stage18_rp_conf_conf`

### 2.2 最差退化代表组
- `trigger=confidence_score, accept=tail_score`
- 这组在 `DA` 上退化最明显

审计文件：
- `analysis/audit_stage18_da_conf_tail`
- `analysis/audit_stage18_rp_conf_tail`

---

## 3. 审计结果总表

| 配置 | Task | triggered | accepted_after_trigger | answer_changed | corrected | harmed | 核心现象 |
|---|---|---:|---:|---:|---:|---:|---|
| conf -> conf | DA | 26 | 0 | 16 | 12 | 3 | rerun 有修复，但一个都没被 accept |
| conf -> conf | RP | 3 | 0 | 1 | 1 | 0 | rerun 样本极少，也一个都没被 accept |
| conf -> tail | DA | 26 | 25 | 6 | 1 | 4 | accept 过松，几乎全收，导致明显退化 |
| conf -> tail | RP | 3 | 3 | 0 | 0 | 0 | rerun 基本不改答案，但全部被保留在 small |

---

## 4. 关键发现

### 4.1 baseline-best 组“看起来最好”，其实是因为 rerun 被全部拒绝了

#### DA: `trigger=confidence_score, accept=confidence_score`
- triggered = `26`
- accepted_after_trigger = `0`
- raw_wrong_to_memory_correct = `12`
- raw_correct_to_memory_wrong = `3`

这说明：
- rerun 并不是没效果；
- 在这 26 个触发样本里，其实有 **12 个样本从错变对**；
- 但 accept 规则把它们**全部拒绝**了；
- 所以系统最终表现回到 baseline，不是因为 routing 成功，而是因为 rerun 结果根本没进入最终答案。

更准确地说：

> baseline-best 组的“稳定”并不代表 calibration 成功，而是代表当前 accept 规则过于保守，宁可错过真实修复，也不让 rerun 生效。

#### RP: `trigger=confidence_score, accept=confidence_score`
- triggered = `3`
- accepted_after_trigger = `0`
- corrected = `1`

RP 也有同样现象，只是触发样本极少。

---

### 4.2 最差组的问题不是 rerun 改得太多，而是 accept 太松

#### DA: `trigger=confidence_score, accept=tail_score`
- triggered = `26`
- accepted_after_trigger = `25`
- answer_changed = `6`
- corrected = `1`
- harmed = `4`
- raw_wrong_to_memory_wrong = `15`
- unchanged = `20`

这说明：
- rerun 被触发后，大多数样本根本没有被真正修好；
- 只有 `1` 个样本从错变对；
- 却有 `25` 个被 accept；
- 其中大量样本属于：
  - 仍然错误
  - 甚至答案都没变
  - 却被保留在 small branch

最关键的是：
- `answer_changed_count = 6`
- 但 `accepted_after_trigger = 25`

也就是说：

> accept=tail_score 并不是“更会接受真正修复后的答案”，而是“几乎把所有 rerun 样本都放过去了”，包括大量没变、没修好、仍然错误的样本。

这正是 `DA` 从 `90.43` 掉到 `76.52` 的机制解释。

#### RP: `trigger=confidence_score, accept=tail_score`
- triggered = `3`
- accepted_after_trigger = `3`
- answer_changed = `0`
- corrected = `0`
- harmed = `0`
- raw_wrong_to_memory_wrong = `1`
- raw_correct_to_memory_correct = `2`

RP 上更极端：
- rerun 后答案根本没变；
- 但还是全部 accept；
- 说明 tail-based accept 在 RP 上几乎没有任何真实筛选能力。

---

### 4.3 rerun 确实可能修复答案，但当前系统没有把这些修复稳定留下来

这次 audit 最重要的发现之一是：

#### 在 DA 的 baseline-best 组里
- corrected = `12`
- harmed = `3`

也就是说，rerun 候选中真实修复样本其实不少。

所以现在不能说：

> rerun 本身没用

更准确地说是：

> rerun 候选里同时混有“真的修复”和“没有修复甚至变差”的样本，而当前 accept 规则既分不出两者，也没有把真实修复稳定保住。

这说明后续研究不能只看 rerun 有没有用，而要看：
- 哪些 rerun 是 genuinely corrected
- 哪些 rerun 只是 high-confidence wrong answer

---

### 4.4 gain 目前没有表现出足够的 corrected / harmed 可分性

在 `DA conf->conf` 中：
- corrected 的 `tail_gain mean = -0.000069`
- harmed 的 `tail_gain mean = -0.000190`
- corrected 的 `bottom10_gain mean = -0.003741`
- harmed 的 `bottom10_gain mean = -0.003843`

在 `DA conf->tail` 中：
- corrected 的 `tail_gain mean = -0.000154`
- harmed 的 `tail_gain mean = -0.000169`
- corrected 的 `bottom10_gain mean = -0.005947`
- harmed 的 `bottom10_gain mean = -0.004859`

这些数值说明：
- 当前 `tail_gain` / `bottom10_gain` 的量级非常小；
- corrected 和 harmed 的均值并没有拉开；
- 甚至多数 gain 还是负的；
- 因此现阶段它们**还不能直接当成可靠 accept 信号**。

更严谨地说：

> 当前新增的 local/tail score 还没有表现出足够的 answer-faithfulness，至少在 stage18 的这组 rerun 样本中，gain 不能有效区分 corrected vs harmed。

---

## 5. 当前最重要的机制判断

基于这次 audit，我认为当前系统失败机制已经比 stage18 summary 更清楚：

### 5.1 失败机制一：保守 accept 会错过真实修复

`accept=confidence_score` 的问题不是“它稳”，而是：
- 它把 rerun 修好的样本也一起拒掉了；
- 因此 memory 无法转化成最终收益。

### 5.2 失败机制二：宽松 accept 会放进大量错误 rerun

`accept=tail_score` 的问题不是“更激进”，而是：
- 它没有筛掉错误样本；
- 大量 rerun 结果并未修好，却被保留在 small。

### 5.3 失败机制三：RP 和 DA 的 rerun 行为明显不同

- `DA`：rerun 触发较多，也确实存在不少 corrected 样本
- `RP`：rerun 触发很少，而且经常不改变答案

这意味着：

> `DA` 和 `RP` 很可能不应该共享同一套 rerun / accept 逻辑。

---

## 6. 下一步最合理的研究结论

### 6.1 不应该继续做什么

现在不建议继续主攻：
- 更多 `trigger x accept` 组合 sweep
- 更多 threshold sweep
- 继续扩大 routing metric 搜索空间

因为 stage18 + 本次 audit 已经说明：
- 主要矛盾不是“还没试够 metric”
- 而是“acceptance 机制没法稳定识别哪些 rerun 真的值得留下”

### 6.2 下一步应该做什么

下一步最合理的主线是：

#### 方向 A：研究 rerun 本身的答案质量，而不是继续扫 accept
重点看：
- rerun 的 prompt 是否让模型更容易改写但不更准确
- rerun 是否应该强制输出更短、更答案中心化的回答
- 是否应该让 rerun 只输出 option / yes-no / spatial label，而不是完整生成

#### 方向 B：做 task-specific rerun policy
基于当前证据：
- `DA` 可以继续研究 rerun，因为存在较明显 corrected 样本
- `RP` 应优先收缩 rerun，或直接先禁用 rerun，再做专门设计

#### 方向 C：重新定义 accept 信号
当前 accept 不应继续只靠：
- `confidence_score`
- `tail_score`
- `bottom10_group_score`

下一步更合理的是研究：
- final answer token / option token 的一致性
- raw answer vs rerun answer 的显式比较
- 是否引入“只有当 rerun 明显修正 raw 错误时才 accept”的规则

---

## 7. 当前最值得执行的最小下一步

如果只做一个最小、最稳的下一步，我建议是：

> **不要再扫更多 routing score；改做 rerun answer format / rerun prompt audit，尤其是在 `direct_attributes` 上，尝试让 rerun 直接输出更短、更答案中心化的最终答案，再重新评估 corrected / harmed。**

如果只保留工程动作，可以写成：

1. 针对 `direct_attributes` 新增一个更短的 rerun prompt 版本
   - 强制只输出最终选项或极短答案
2. 暂时不要动 retrieval
3. 暂时不要再扫 16 组 metric
4. 对新 rerun prompt 只做小规模对照：
   - corrected/harmed
   - accepted_after_trigger
   - answer_changed rate

---

## 8. 最终结论

当前可以比较确定地说：

> stage18 的失败并不是因为 memory rerun 完全无效，而是因为 rerun 候选里既有真实修复，也有大量无效甚至有害改写；现有 accept/fallback 规则既不能稳定接受前者，也不能有效过滤后者。`DA` 和 `RP` 的机制又不相同，因此下一步主线应该从“继续扫 score”转向“分析并改造 rerun answer behavior”，并逐步过渡到 task-specific rerun / accept policy。
