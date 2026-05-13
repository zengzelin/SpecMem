# SpecMem / SpecEyes 优化尝试与结果总汇报（讨论版，2026-04-27）

## 1. 这份汇报的目的

这份文档的目标不是再展开某一轮局部实验，而是把当前 `specmem_fresh` 里已经做过的优化尝试、对应结果、哪些方向已经基本证伪、哪些方向仍然保留局部信号，统一整理成一份便于后续讨论的总汇报。

当前最重要的背景是：
- 这条 memory 线已经做了多轮接入与优化；
- 其中很多尝试都没有带来稳定收益，甚至会稳定退化；
- 因此现在最需要的是**把哪些方向已经不值得继续主攻说清楚**，以及**把后续讨论真正应该聚焦的问题收敛出来**。

---

## 2. 一句话总判断

当前最准确的总体判断是：

> **memory 不是完全没用，但到目前为止还没有被做成一个稳定有效的方法。**

更具体一点：
- 在 `direct_attributes` 上，memory 曾经出现过真实正收益；
- 但后续 routing / score / acceptance 设计没有把这部分收益稳定保住；
- 在 `relative_position` 上，多轮尝试都没有形成有效提升；
- 到 `stage18` 为止，继续扩大 routing score 组合搜索已经基本证明无效；
- 当前主问题已经不是“memory 有没有接进去”，而是：
  - **哪些 rerun 值得保留**
  - **哪些 rerun 只是把错误答案以更高分留下来**
  - **accept/fallback calibration 为什么始终不成立**

---

## 3. 到目前为止做过哪些大类优化

### 3.1 memory 系统接入与基础设施改造
这一部分已经基本完成，主要包括：

- task-aware memory policy
- two-stage gating
  - raw first
  - memory rerun on trigger
  - accept / fallback
- richer JSONL logging
- replay / compare 脚本兼容 richer schema
- retrieval metadata logging
- prompt variants

这部分工作的重要意义在于：
- 现在不是“系统还没搭起来”；
- 而是“系统已经能跑，但路由和接受机制没有把它变成稳定方法”。

也就是说，当前失败不能再归因为：
- memory 根本没接上
- 数据没记录下来
- 分析脚本不支持

这些基础工作已经不是主瓶颈。

---

### 3.2 retrieval 相关尝试
做过的方向包括：
- plain retrieval
- task-aware retrieval
- logic top-k 不同配置
- retrieval metadata 保留与分析

#### 当前结论
这些方向**不是当前主瓶颈**。

证据是：
- stage16 / stage17 里 task-aware 和 plain 差异很小；
- 没有形成主要收益来源；
- 无法解释为什么系统整体没有优化成功。

所以当前不应继续把主叙事放在：
- retrieval 再细化一点
- task-aware 规则再加几条
- top-k 再扫更大范围

这些方向的信息增量已经很低。

---

### 3.3 prompt 相关尝试
已经接入和试过的 prompt 侧变化包括：
- `default`
- `compact_spatial`
- `compact_general`
- `empty_scaffold`
- `no_memory`
- 更新到最近，又新增了 `answer_focus`

#### 当前结论
prompt 变体本身还不能算成功主线。

尤其在 `relative_position` 上：
- `compact_spatial` 等尝试并没有带来稳定提升；
- 某些配置甚至明显退化。

不过，prompt 方向和 retrieval 不同：
- retrieval 目前更像“已经证伪为非主瓶颈”；
- prompt 还不能完全放弃，因为 rerun 的答案行为本身仍可能是问题来源。

所以更准确地说：
- **“继续把 prompt variant 当大范围 sweep 主线”不合理**；
- **“针对 rerun answer behavior 做更有针对性的 prompt 改造”仍值得继续**。

---

### 3.4 routing / score 相关尝试
这是中后期最核心的优化主线。

做过的方向包括：
- 从单次 memory prompt 改成 two-stage gating
- trigger threshold / accept threshold 分离
- task-aware memory policy
- richer routing logging
- score profile 化尝试
- 新增：
  - `tail_score`
  - `lowest_group_score`
  - `bottom10_group_score`
- trigger/accept 使用不同 metric
- stage18 做 16 组 `trigger_metric x accept_metric` sweep

这条线原本是因为一个明确判断而建立起来的：
- 当前全局单标量 `confidence_score` 太扁平；
- answer-centric / local-window confidence 可能更适合 routing。

#### 当前结论
这条线**目前没有得到正结果验证**。

更具体地说：
- 新增 score profile 并没有带来超过 baseline 的配置；
- stage18 完整 sweep 之后，最好也只是回到 baseline；
- 一旦 accept 放宽，就会稳定退化。

所以当前不能再说：
- “还没试够 routing score”
- “再换几个 trigger/accept 组合也许就好了”

stage18 基本已经把这个空间扫到足够说明问题的程度了。

---

## 4. 当前最关键的实验结果怎么总结

### 4.1 direct_attributes：存在真实正信号，但不稳定
这是整条 memory 线里最值得保留的正面信息。

历史最强结果：
- baseline：`90.43`
- stage16 best：`92.17`

这说明：
- memory 在 `direct_attributes` 上**不是假信号**；
- 它确实有可能带来真实收益。

但是问题在于：
- stage17 没有保住这个收益；
- stage18 也没有把这个收益重新找回来；
- 所以当前仍不能把它当成一个稳定方法，只能当成“局部有效信号”。

### 4.2 relative_position：多轮尝试基本无效
这是整条线最负面的部分。

观察到的现象包括：
- baseline 就不低；
- memory 配置大多数不升反降；
- 某些 routing 配置会让 `small_ratio` 异常升高；
- 但准确率并没有提高，反而退化。

这说明：
- `relative_position` 不是当前 memory 线最适合优先推进的任务；
- 它的失败不是简单改一个阈值就能解决的。

当前更合理的做法是：
- 把 `RP` 当作需要专门设计的难例任务；
- 而不是继续和 `DA` 共用同一套 rerun / accept 逻辑。

---

## 5. 为什么 stage17 / stage18 都没有真正优化成功

### 5.1 stage17 的问题：新路由没有保住 direct 的收益
stage17 的现象很明确：
- always-memory 还能维持 `92.17`
- 触发式 routing 却大多回到 baseline `90.43`

这说明：
- memory 本身不一定无用；
- 真正失败的是 routing score / trigger / accept 这层决策逻辑。

### 5.2 stage18 的问题：完整 metric sweep 也找不到有效组合
stage18 的结论更硬：
- 最好组仍然只是 baseline；
- 没有任何一组 score 组合超过 baseline；
- 一旦 accept 放宽，small_ratio 上升，DA/RP 一起退化。

这说明：
- 当前问题不是“组合还没试够”；
- 而是当前 accept/fallback 的依据根本没有抓住什么叫“值得保留的 rerun”。

---

## 6. rerun audit 带来的新认识

这是最近最有价值的一步，因为它把“平均结果不好”进一步拆成了机制层面的解释。

### 6.1 rerun 不是完全没用
在 `DA` 的 baseline-best 组（`conf -> conf`）里：
- triggered = `26`
- corrected = `12`
- harmed = `3`
- accepted_after_trigger = `0`

这说明：
- rerun 候选中确实存在大量真实修复；
- 但 accept 规则把它们全部拒掉了。

也就是说：
> 当前系统并不是 rerun 本身毫无价值，而是 accept 太保守，无法把真实修复转化成最终收益。

### 6.2 宽松 accept 会把大量错误 rerun 放过去
在 `DA` 的最差组（`conf -> tail`）里：
- triggered = `26`
- accepted_after_trigger = `25`
- corrected = `1`
- harmed = `4`
- 大量样本仍然 wrong，甚至答案没变

这说明：
- tail-based accept 在当前实现下几乎没有可靠筛选能力；
- 它不是“更聪明地接受修复后的答案”，而是“几乎把所有 rerun 都放过了”。

### 6.3 DA 和 RP 的 rerun 行为机制不同
- `DA`：存在不少 corrected rerun
- `RP`：rerun 很少真正改变答案，很多时候几乎没起作用

所以当前非常不适合继续把 DA 和 RP 当成一个统一任务处理。

---

## 7. 到目前为止，哪些方向基本可以判定“没效果”

下面这些方向，至少在当前证据下，已经不适合继续当成主线：

### 7.1 继续扩大 trigger/accept metric sweep
理由：
- stage18 已经完整扫过 16 组组合；
- 没有超 baseline；
- 再扩大搜索空间边际信息价值很低。

### 7.2 继续大范围 threshold sweep
理由：
- 当前已知现象很稳定：accept 一松就退化；
- 再扫更多阈值大概率只是重复验证同一个现象。

### 7.3 把 retrieval / task-aware retrieval 当主瓶颈继续深挖
理由：
- 现有结果不支持 retrieval 是当前主矛盾；
- 即便 retrieval 有些局部帮助，也解释不了 acceptance 失效。

### 7.4 把更多 prompt variant 当主线 sweep
理由：
- 大范围 prompt sweep 已经没有很强信息增量；
- 现在更值得做的是“针对 rerun answer behavior 的定向 prompt 改造”，而不是继续盲扫变体。

---

## 8. 到目前为止，哪些方向仍然值得保留

### 8.1 保留 direct_attributes 这条 memory 线
因为它确实出现过真实正收益。

### 8.2 保留对 rerun 机制本身的研究
因为 rerun audit 表明：
- rerun 候选里有真实修复；
- 问题在于如何识别并保住它们。

### 8.3 保留 task-specific 策略这个思路
因为 DA 和 RP 的行为明显不同。

### 8.4 保留“更短、更答案中心化 rerun”这个新方向
因为当前已知问题之一就是：
- rerun 可能在做大量改写；
- 但这些改写不一定让最终答案更准确；
- 因此需要把 rerun 的输出变得更 answer-centric，而不是更长、更解释化。

---

## 9. 当前最值得后续讨论的核心问题

后续讨论不应该再围绕：
- “再加什么 metric”
- “再扫几个阈值”
- “retrieval top-k 要不要再改”

而应该集中在下面三个问题：

### 问题 1：rerun 到底在修什么？
- 它是真的在修最终答案，还是只是在改写表述？

### 问题 2：accept 应该依据什么？
- 当前的 score / gain 都没有足够区分力；
- 那 accept 的信号到底应该是什么？
- 是否应该直接比较 raw answer vs rerun answer，而不是只看 rerun 自身分数？

### 问题 3：DA 和 RP 是否应该彻底拆开策略？
- 从现有证据看，这个答案越来越像“是”。

---

## 10. 当前最合理的下一步方向

基于现在所有结果，我认为下一步最合理的方向是：

> **停止继续扩大 routing score 搜索；转向分析并改造 rerun answer behavior，优先在 `direct_attributes` 上尝试更短、更答案中心化的 rerun prompt，再用 corrected / harmed / accepted_after_trigger 等机制指标评估。**

这也是为什么最近已经开始推进：
- `answer_focus` prompt style
- `audit_memory_rerun_effects.py`
- `direct_attributes` 的 stage19 answer-focus rerun 计划与脚本

---

## 11. 最终总结

如果只保留一个便于讨论的总结版本，可以直接说：

> 到目前为止，SpecMem / SpecEyes 的 memory 优化已经做了多轮系统接入、retrieval 调整、routing score 改造和完整的 stage18 metric sweep。结论是：memory 不是完全没用，因为它曾在 `direct_attributes` 上带来过从 `90.43` 到 `92.17` 的真实提升；但后续大多数优化尝试都没有把这部分收益稳定保住，很多尝试甚至没有效果或会稳定退化。特别是 stage18 已经说明，继续扩大 trigger/accept metric 组合搜索没有意义。当前真正的问题不是“还没试够不同 score”，而是 rerun 候选里既有真实修复，也有大量无效甚至有害改写，而 accept/fallback 机制既保不住前者，也过滤不掉后者。后续讨论应从继续 sweep 配置转向分析 rerun 何时真正修正答案、何时只是把错误答案以更高分留在 small branch，并逐步过渡到 task-specific rerun / accept 策略。**
