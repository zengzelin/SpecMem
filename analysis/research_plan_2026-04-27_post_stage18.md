# SpecEyes Post-Stage18 研究计划（2026-04-27）

## 1. 结论先行

基于 `analysis/report_2026-04-27_stage18_metric_sweep_summary.md`，当前最重要的判断是：

- `stage18` 已经把 16 组 `trigger_metric x accept_metric` 组合完整扫过；
- 最好的结果仍然只是复现 baseline，而不是超过 baseline；
- 一旦放松 accept，`small_ratio` 上升、`DA/RP` 一起退化；
- 因此，**当前继续扩大 routing metric sweep 并不合理**。

更严谨地说：

> 当前失败并不是“metric 还不够多”，而是“memory rerun 产出的候选答案本身没有被稳定地区分为可接受 vs 不可接受”，也就是 accept/fallback calibration 还不成立。

所以，下一步不应再把主线放在：
- 更多 trigger/accept metric 的笛卡尔积搜索；
- 再扫更多阈值；
- 把现有 score 再做少量变体后继续全量 sweep。

这些方向在 stage18 后的边际信息价值已经明显下降。

---

## 2. 对当前“下一步实验计划”的批判性判断

从 `eval_results_deepeyes/` 当前目录结构看，已有实验主线包括：

- stage1 baseline
- stage15 / stage16 memory 阈值与 retrieval 配置实验
- stage17 trigger/accept 路由实验
- stage18 16 组 metric sweep

这说明“routing metric 空间是否试得不够”这个问题，当前已经基本回答过了。

### 2.1 不合理的计划方向

如果下一步还想继续做下面这些事情，我认为都不够合理：

1. **继续扩大 metric sweep**
   - stage18 已经说明：只要 accept 不收紧在 `confidence_score`，系统就会稳定退化；
   - 继续换 metric 组合，大概率只是重复验证同一个结论。

2. **继续主攻 retrieval / top-k / task-aware retrieval**
   - 现有结果并不支持 retrieval 是主瓶颈；
   - stage16 / stage17 / stage18 的主要失败都发生在 acceptance 端，而不是 retrieval 端。

3. **继续只看 aggregate acc / small_ratio 做扫参**
   - 这会继续浪费实验成本；
   - 现在更缺的是“错误机制分析”，而不是“更多平均结果表”。

4. **把 stage16 的 92.17 当成已经稳定可复现的现象**
   - 当前证据只能说明它是一个真实正信号；
   - 但还不能说明它是一个稳定方法。

### 2.2 相对合理的方向

下一步合理的方向应该从“继续搜索配置”切换到“解释为什么 accept 失败”。

也就是说，研究重点要从：
- `哪个 metric 更好？`

转成：
- `为什么 memory rerun 的 accepted samples 经常是错的？`
- `为什么 baseline-best 组合里 triggered>0 但 accepted_after_trigger=0？`
- `为什么一旦 accept 放宽，错误 small 样本大量残留？`

---

## 3. 新的研究主线：从“metric 搜索”转向“acceptance 失效机理分析”

### 核心假设

当前更可能的真实问题是下面三者之一，而不是 score 维度不够多：

1. **memory rerun 本身没有稳定改进答案质量**
   - rerun 只是改写了一遍回答；
   - 并没有把错答修成对答。

2. **rerun 的局部分数升高和最终 correctness 没有稳定对应关系**
   - tail / local-window 变高了；
   - 但答案仍然错。

3. **存在任务异质性，统一 accept 规则不成立**
   - `direct_attributes` 和 `relative_position` 的 rerun 行为本质不同；
   - 同一套 accept 逻辑不应该跨任务共享。

这三个假设，如果不先验证，再继续扫 metric 只会重复消耗算力。

---

## 4. 下一轮研究目标

下一轮不再追求“立刻找到更高 acc 的新配置”，而是先完成三个更基础的问题回答：

### 目标 1：判断 memory rerun 是否真的修正了答案

重点不是看 rerun 之后 score 有没有变化，而是看：
- rerun 前错、rerun 后对：有多少
- rerun 前对、rerun 后错：有多少
- rerun 前后答案没变：有多少
- rerun 前后答案变了但还是错：有多少

如果发现“rerun 后被 accept 的样本，多数只是答案变了但没变对”，那主问题就不是 metric，而是 rerun 质量本身。

### 目标 2：判断 gain 是否和 correctness 有关联

已经新增了：
- `tail_gain`
- `bottom10_gain`

下一步应该直接问：
- corrected 样本的 `tail_gain` 是否显著更高？
- harmed 样本是否也同样会出现高 `tail_gain`？
- `bottom10_gain` 是否能区分 corrected vs harmed？

如果 gain 不能区分 corrected / harmed，就说明当前新增 score 仍不够 answer-faithful。

### 目标 3：判断 task-specific acceptance 是否必要

不是先上 task-specific override，而是先验证是否必要：
- `direct_attributes` 的 corrected / harmed 分布和 `relative_position` 是否显著不同？
- 某个任务上 rerun 有修复作用，另一个任务上基本只有干扰？

如果答案是“不同任务的 rerun 行为机制不同”，那下一步才合理进入 task-specific acceptance。

---

## 5. 新的实验计划

下面按优先级给出新的研究顺序。

### Phase A：停止继续做大规模 metric sweep，先做 rerun mechanism audit

#### 要做什么
基于 stage18 已经跑出的 JSONL，不新增大规模模型实验，先做离线分析。

#### 重点分析问题
1. 对每个 triggered 样本，统计：
   - raw answer
   - memory answer
   - 最终是否 accept
   - raw 是否正确
   - memory 是否正确
   - accept 后最终是否正确

2. 按样本分成四类：
   - corrected
n   - harmed
   - changed_but_still_wrong
   - unchanged

3. 看这些类别在 `DA` / `RP` 上的占比。

#### 为什么先做这个
因为 stage18 已经证明“平均指标 sweep 不再给新信息”，而这一轮 audit 才能告诉我们：
- 问题是 rerun 质量；
- 还是 accept 规则；
- 还是任务异质性。

---

### Phase B：只做小规模分析脚本增强，不先做大范围模型改造

#### 建议新增的分析输出
建议新增一个轻量分析脚本，专门做 triggered 样本机制分析，例如：
- `scripts/audit_memory_rerun_effects.py`

最少输出：
- total triggered
- total accepted_after_trigger
- raw_correct -> memory_correct
- raw_correct -> memory_wrong
- raw_wrong -> memory_correct
- raw_wrong -> memory_wrong
- answer_changed rate
- accepted_changed rate
- corrected/harmed 的 gain 分布摘要

#### 为什么合理
这比再开新一轮模型 sweep 更便宜，也更可能回答真正问题。

---

### Phase C：如果 audit 证明 rerun 质量不稳定，再决定是否改 prompt / memory scaffold

如果 Phase A/B 的结果显示：
- rerun 很少把错答修成对答；
- rerun 大量把原本可 fallback 的样本改成“错但高分”；

那下一步应该优先改的是：
- rerun prompt 的结构
- memory 引导方式
- 是否要求 rerun 输出更短、更答案中心化

而不是继续改 acceptance metric。

这一步才是 prompt 层面的合理切入点。

---

### Phase D：只有在 audit 显示 task 差异明显时，才进入 task-specific acceptance

如果 audit 结果显示：
- `direct_attributes` 上 rerun 有一定修复率；
- `relative_position` 上 rerun 基本没有修复率，反而常引入错误接受；

那么才合理进入下一步：
- `direct_attributes` 保留 memory rerun
- `relative_position` 暂时禁用 rerun，或提高 accept 要求

也就是：
> task-specific acceptance / rerun policy 应该建立在 audit 证据上，而不是先验猜测上。

---

## 6. 不建议的下一轮代码主线

当前不建议直接把主线放在下面这些改动上：

### 6.1 不建议立刻再加更多 score 类型
例如：
- 更多 window size
- 更多 bottom-k 比例
- 更多 tail 长度组合

理由：
- stage18 已经说明当前问题不是“还没搜够”；
- 如果 rerun 候选本身质量不稳定，再细的 score 也只是给错误答案打分。

### 6.2 不建议立刻做更大规模 threshold sweep
理由同上，更多 sweep 只会重复验证“放宽 accept 会退化”。

### 6.3 不建议重新把主线转回 retrieval
除非 audit 明确证明：
- triggered 样本里 retrieval 几乎总是无关；
- memory 内容本身没有帮助。

在此之前，不应把 retrieval 当作主要解释。

---

## 7. 新研究计划对应的最小代码工作

### 需要优先新增
建议优先增加一份离线分析脚本，而不是再改主推理流程：

- `scripts/audit_memory_rerun_effects.py`

输入：
- baseline jsonl
- candidate jsonl
- task name

输出：
- triggered/accepted 总表
- raw->memory correctness transition 统计
- answer changed vs unchanged 统计
- corrected/harmed 样本清单
- gain 分布摘要

### 当前已有代码足以支撑的分析字段
现在已经有：
- `base_small_answer`
- `small_answer`
- `memory_triggered`
- `memory_accept_decision`
- `tail_score_raw`
- `tail_score_memory`
- `bottom10_group_score_raw`
- `bottom10_group_score_memory`
- `tail_gain`
- `bottom10_gain`

因此当前完全可以先做机制分析，而不需要再大改主流程。

---

## 8. 建议保留的一句对外口径

如果现在需要一个严格、不过度乐观的对外说法，建议用：

> stage18 已经完成了 score-routing 的完整组合实验，结果表明当前没有任何一组 trigger/accept metric 超过 baseline。现阶段问题不再是“还没试够 routing score”，而是 memory rerun 的 accept/fallback calibration 仍不成立。下一步应从继续 sweep 配置转向分析 rerun 何时真正修正答案、何时只是把错误答案以更高分留下来。

---

## 9. 最终建议

新的研究主线应该是：

> **停止继续扩大 metric sweep；先做 triggered sample 的 rerun mechanism audit，验证 memory rerun 到底是在修答案，还是只是在制造“错但高分”的接受样本；再根据 audit 结果决定是否改 prompt、做 task-specific acceptance，或彻底收缩某些任务的 memory rerun。**

如果只保留一个最小可执行建议，就是：

> **下一轮先不要再扫更多 routing score 组合，而是新增一个 `audit_memory_rerun_effects.py`，系统统计 raw→memory 的 correctness transition、answer_changed rate、以及 corrected/harmed 的 `tail_gain` / `bottom10_gain` 分布。**
