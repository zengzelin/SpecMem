# SpecEyes Memory 下一轮执行计划（2026-04-27）

## 1. 背景与当前判断

截至 `2026-04-27`，当前这条 `memory` 线已经有两个相对明确的事实：

1. `memory` 不是完全无效  
   在更早阶段，`VStar direct_attributes` 曾从 `90.43` 提升到 `92.17`。

2. 现有两阶段 gating + score routing 还不稳定  
   `stage18` 的 `16` 组完整 score sweep 没有找到优于 baseline 的组合；  
   只要放宽 `accept_metric`，`small_ratio` 就上升，准确率就下降。

因此，下一轮不应继续在：

- `routing score` 大扫
- `threshold / trigger / accept` 大扫
- `k1/k2/k3` 微调
- prompt wording 小修补

这些已经高度重复的轴上继续消耗时间。

下一轮应该转向：

> **减少无关 memory 进入 prompt，改进 accept 机制，验证更接近 case-based 的 memory 内容。**

---

## 2. 总目标

下一轮的目标不是“让 memory 在所有任务上都有效”，而是：

1. 先在 `VStar direct_attributes` 上做出一个稳定点
2. 证明 memory 的收益来自更合理的控制逻辑，而不是偶然阈值
3. 把实验主线从“重复扫分数”切换到“真正新的机制验证”

---

## 3. 本轮先冻结的方向

以下方向先不再作为主线投入：

| 方向 | 原因 |
|---|---|
| 再扫 `routing score` | `stage18` 已经完整做过，收益很低 |
| 再扫 `threshold / trigger / accept` | `stage17 + stage18` 已经覆盖大量组合 |
| 再扫 `k1/k2/k3` | 如果 memory 内容不变，继续扫 `k` 价值不高 |
| 继续在 `RP` 上硬塞文本 memory | `RP` 更像视觉定位问题，不像规则回忆问题 |
| 继续做 prompt wording 微调 | `compact_spatial` 一类实验已经说明这条线很难解决主矛盾 |

---

## 4. 下一轮只保留三条真正新的实验主线

| 优先级 | 主线 | 核心假设 | 与之前实验关系 |
|---|---|---|---|
| `P0` | `retrieval relevance gate` | 很多 harmed case 来自弱相关 memory 被硬塞进 prompt | 新的，不重复 |
| `P1` | `delta / multi-signal accept` | rerun 后“更自信”不等于“更正确” | 新的，不重复 |
| `P2` | `case-based memory bank` | 抽象 guideline 太泛，真实纠错案例更有帮助 | 新的，不重复 |

---

## 5. 主验证面与任务范围

### 5.1 主验证面

下一轮主验证面固定为：

- `VStar direct_attributes`

原因：

- 这是目前唯一出现过明确正收益的任务面
- 更适合检验 textual / logic memory 是否有真实帮助

### 5.2 辅助验证面

辅助验证面为：

- `VStar relative_position`

但处理策略调整为：

- 不再作为当前主 KPI
- 只做少量监控
- 不再为了 `RP` 继续大规模扫 memory 组合

### 5.3 对 RP 的当前结论

当前默认判断：

> `relative_position` 的主要问题更像视觉定位与局部观察问题，  
> 不是通过追加通用文本 memory 就能解决的问题。

---

## 6. 分阶段执行计划

## Phase 0：收敛实验面

### 目标

把下一轮主线明确收缩到：

- `DA-only memory`

### 操作

1. 后续新实验默认只在 `direct_attributes` 上开 memory
2. `relative_position` 先不继续作为主实验面
3. 保留现有对照组：
   - `SpecEyes baseline`
   - `stage16 best`
   - `stage18 best`

### 通过标准

- 所有后续实验先只问一个问题：  
  **能不能在 DA 上稳定不差于 baseline，并减少 harmed case**

---

## Phase 1：先做 retrieval relevance gate

### 目标

减少“弱相关 memory 污染 prompt”的情况。

### 核心思路

当前系统的问题之一是：

- 只要样本触发 memory，就会把检索到的 memory 追加到 prompt
- 但这些 memory 并不一定与当前样本真的相关

所以这一阶段增加：

- `retrieval score threshold`
- 或者更简单的 `top1 relevance gate`

只有当检索结果足够相关时，才允许 rerun-with-memory。

### 本阶段不改动

- 不改 retrieval backend
- 不改整体两阶段框架
- 不改 memory bank schema
- 不引入更复杂模型

### 推荐实验组

| 实验编号 | 配置 |
|---|---|
| `A1` | 当前 `stage18 best` 作为 control |
| `A2` | `DA-only + retrieval gate (strict)` |
| `A3` | `DA-only + retrieval gate (medium)` |
| `A4` | `DA-only + retrieval gate + top1-only` |

### 验收标准

1. `DA all_acc >= baseline 90.43`
2. `corrected > harmed`
3. `small_ratio` 不要异常上升
4. 即使 `triggered` 数量下降，只要质量更高，仍视为正向结果

---

## Phase 2：改 accept 逻辑，不再只看绝对分数

### 目标

避免“memory rerun 后更自信，但其实更错”的问题。

### 核心思路

当前 accept 过于依赖：

- `memory_score > threshold`

但实验已经说明：

- 更高的分数不等于更高的正确率

所以下一步应把 accept 改为多信号判断，至少引入：

- `base_confidence_score`
- `memory_confidence_score`
- `delta = memory_score - base_score`
- `retrieval_top1_score`
- `answer_changed`

### 推荐先试的规则

| 实验编号 | accept 规则 |
|---|---|
| `B1` | 旧规则：只看 `memory_score > thr` |
| `B2` | `memory_score > thr` 且 `delta > min_delta` |
| `B3` | `memory_score > thr` 且 `retrieval_top1 > min_rel` |
| `B4` | `memory_score > thr` 且 `delta > min_delta` 且 `answer_changed=true` |
| `B5` | `memory_score > thr` 且 `delta > min_delta` 且 `retrieval_top1 > min_rel` |

### 本阶段原则

- 先用手工规则
- 不引入学习器
- 不做大规模超参搜索

### 验收标准

1. 至少不差于 `Phase 1` 最优点
2. `accepted_after_trigger` 不能再大规模带来 harmed cases
3. 如果 accept 一放开就掉分，该规则立即淘汰

---

## Phase 3：把 memory 内容从 guideline 改成 case-based

### 目标

验证当前 memory 失败的原因，是否来自内容形态过于抽象。

### 当前问题

现有 memory bank 主要是：

- 一句 guideline
- failure_mode
- subject
- key_concepts

这类 memory 很容易变成：

- 通用建议
- prompt 噪声
- 模型先验暗示

而不是真正能纠错的“可复用经验”。

### 新方向

把 memory 改成：

- 更接近真实纠错案例的短 case memory

建议先做小规模：

- `20-50` 条
- 只覆盖 `direct_attributes`

### case-based memory 可以包含

1. 题目类型
2. 常见误判模式
3. 正确判断原则
4. 一条很短的修正示例

### 原则

- 尽量复用当前 schema
- 不做大改接口
- 先验证形态，不先追求规模

### 验收标准

1. 是否比抽象 guideline memory 更稳定
2. 是否减少 harmed case
3. 是否能在 DA 上重新接近或超过 `stage16 best`

---

## 7. 每轮实验固定要看的指标

后续所有实验统一关注以下指标：

| 指标 | 用途 |
|---|---|
| `all_acc` | 主指标 |
| `small_ratio` | 检查 routing 是否失真 |
| `triggered_cnt` | 检查 gate 是否过宽 |
| `accepted_after_trigger` | 检查 memory rerun 真正被采纳的数量 |
| `corrected_cnt` | 检查 memory 是否真的在救样本 |
| `harmed_cnt` | 检查副作用 |
| `retrieval hit quality` | 检查检索相关性 |
| `answer_changed` 比例 | 检查 rerun 是否真的在改变决策 |

---

## 8. 停止条件

满足任一条，就先停止继续扩展这条线：

| 条件 | 含义 |
|---|---|
| `DA` 连续几轮都不超过 baseline | 当前 memory 形态可能不成立 |
| `corrected <= harmed` | memory 质量没有真正起来 |
| 只能靠极端压低 fallback 才“看起来有效” | 说明不是稳健收益 |
| `RP` 持续拖累整体判断 | 应彻底从 memory 主线剥离 |

---

## 9. 推荐执行顺序

建议严格按下面顺序推进：

1. 冻结 `RP`，主线切到 `DA-only`
2. 先做 `retrieval relevance gate`
3. 在 gate 最优点上做 `delta / multi-signal accept`
4. 如果前两步有正信号，再做 `case-based memory bank`
5. 只有 DA 这条线稳定后，才决定是否回头处理 `RP`

---

## 10. 最短汇报版

如果需要向老师或组里做一句话式汇报，可以直接使用：

> 下一轮不再继续扫 routing score、阈值或 k，而是切到三个真正新的方向：  
> 先只在 `VStar direct_attributes` 上验证 `retrieval relevance gate`，  
> 再做 `delta / multi-signal accept`，  
> 最后验证 `case-based memory bank` 是否比当前抽象 guideline 更稳。  
> `relative_position` 暂时不再作为 memory 主验证面。
