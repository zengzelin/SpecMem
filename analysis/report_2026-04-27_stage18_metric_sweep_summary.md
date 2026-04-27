# SpecEyes Memory 当前结果整理（2026-04-27）

## 1. 当前一句话结论

- `stage18` 的 `16` 组 routing score 组合已经全部跑完，`RP` 缺失结果也已经在 `2026-04-27` 上午补齐。
- 这轮 sweep 的正式结论很明确：
  - **最好的配置仍然只是复现原始 SpecEyes baseline**
  - **没有任何一组 score 组合超过 baseline**
  - **一旦放宽 accept score，准确率会稳定下降**
- 因此当前主问题仍然是：
  - **memory accept / fallback 的 calibration 不成立**
  - 不是“还没试够不同 routing score”

---

## 2. 当前最该汇报的三条结论

### 2.1 stage18 没有找到优于 baseline 的 routing score

在 `VStar direct_attributes` 和 `VStar relative_position` 上，完整比较 `16` 组 `trigger_metric x accept_metric` 后：

- 最优组为：
  - `trigger=confidence_score, accept=confidence_score`
  - `trigger=tail_score, accept=confidence_score`
  - `trigger=lowest_group_score, accept=confidence_score`
  - `trigger=bottom10_group_score, accept=confidence_score`
- 这四组的结果完全相同，本质上都只是回到了 baseline：
  - `DA = 90.43`
  - `RP = 86.84`

这说明：

> 当前只要 `accept_metric` 仍然收紧在 `confidence_score`，系统最后行为就会退回到原始 SpecEyes 的路由方式。

---

### 2.2 accept score 一旦放宽，系统会稳定退化

只要把 `accept_metric` 换成：

- `tail_score`
- `lowest_group_score`
- `bottom10_group_score`

就会看到两个稳定现象：

- `small_ratio` 大幅上升
- `DA / RP` 准确率一起下降

最差组是：

- `trigger=confidence_score, accept=tail_score`
- `DA = 76.52`
- `RP = 82.89`

这说明：

> 当前 memory rerun 产生的答案，并没有被新的 accept score 可靠地区分出来；
> 放宽 acceptance 只是在让更多本该 fallback 的样本错误地留在 small branch。

---

### 2.3 memory 仍然有“局部有效”信号，但还不是稳定方法

历史上最强的正结果仍然来自更早阶段：

- `Stage16 direct_attributes best = 92.17`
- baseline 只有 `90.43`

所以我们不能说：

> memory 完全没用

更准确的说法是：

> memory 在 `direct_attributes` 上出现过真实正收益，但当前这套两阶段 gating + 新 score 体系，还没把这部分收益稳定保住。

---

## 3. 现在最值得对外展示的核心对照表

### 3.1 原始 baseline、历史最好点、当前最好点

| 配置 | Direct Attributes | Relative Position | 备注 |
|---|---:|---:|---|
| SpecEyes baseline | `90.43` | `86.84` | 当前稳定参照系 |
| Stage16 best memory | `92.17` | `84.21` | 只在 DA 上有明确正收益 |
| Stage18 best score sweep | `90.43` | `86.84` | 只是复现 baseline |

补充路由占比：

| 配置 | DA small_ratio | RP small_ratio |
|---|---:|---:|
| SpecEyes baseline | `57.39%` | `75.00%` |
| Stage16 best memory | `53.04%` | `40.79%` |
| Stage18 best score sweep | `57.39%` | `75.00%` |

---

### 3.2 stage18 的 16 组正式结果

| trigger_metric | accept_metric | DA acc | DA small_ratio | RP acc | RP small_ratio | 结论 |
|---|---|---:|---:|---:|---:|---|
| `confidence_score` | `confidence_score` | `90.43` | `57.39%` | `86.84` | `75.00%` | baseline |
| `tail_score` | `confidence_score` | `90.43` | `57.39%` | `86.84` | `75.00%` | baseline |
| `lowest_group_score` | `confidence_score` | `90.43` | `57.39%` | `86.84` | `75.00%` | baseline |
| `bottom10_group_score` | `confidence_score` | `90.43` | `57.39%` | `86.84` | `75.00%` | baseline |
| `confidence_score` | `lowest_group_score` | `84.35` | `80.00%` | `82.89` | `98.68%` | 退化 |
| `confidence_score` | `bottom10_group_score` | `84.35` | `80.00%` | `82.89` | `98.68%` | 退化 |
| `tail_score` | `lowest_group_score` | `81.74` | `87.83%` | `82.89` | `98.68%` | 退化 |
| `tail_score` | `bottom10_group_score` | `81.74` | `87.83%` | `82.89` | `98.68%` | 退化 |
| `lowest_group_score` | `lowest_group_score` | `81.74` | `87.83%` | `82.89` | `98.68%` | 退化 |
| `lowest_group_score` | `bottom10_group_score` | `81.74` | `87.83%` | `82.89` | `98.68%` | 退化 |
| `bottom10_group_score` | `lowest_group_score` | `81.74` | `87.83%` | `82.89` | `98.68%` | 退化 |
| `bottom10_group_score` | `bottom10_group_score` | `81.74` | `87.83%` | `82.89` | `98.68%` | 退化 |
| `tail_score` | `tail_score` | `79.13` | `92.17%` | `82.89` | `98.68%` | 明显退化 |
| `lowest_group_score` | `tail_score` | `79.13` | `91.30%` | `82.89` | `98.68%` | 明显退化 |
| `bottom10_group_score` | `tail_score` | `79.13` | `91.30%` | `82.89` | `98.68%` | 明显退化 |
| `confidence_score` | `tail_score` | `76.52` | `91.30%` | `82.89` | `98.68%` | 最差 |

---

## 4. 为什么 stage18 会这样

从 raw JSONL 的路由统计看，当前最关键的现象是：

### 4.1 最好组里，memory 虽然会触发，但没有真正被接受

以当前最好组 `trigger=confidence_score, accept=confidence_score` 为例：

- `DA`: `triggered=26`，但 `accepted_after_trigger=0`
- `RP`: `triggered=3`，但 `accepted_after_trigger=0`

这说明：

> 这组看起来“最好”，不是因为 memory 真起作用了，
> 而是因为 memory rerun 最终一个都没被 accept，系统实际上退回到了原始 baseline 行为。

---

### 4.2 退化组里，问题是 accept 放得太松

比如最差组 `trigger=confidence_score, accept=tail_score`：

- `DA`: `triggered=26`
- 其中 `25` 个被接受
- 最终 `DA` 从 `90.43` 掉到 `76.52`

这说明：

> 当前 tail / local-group 类 accept score，会把很多错误的 memory rerun 答案误判成“可以留在 small branch”，导致 fallback 失效。

---

## 5. 当前最稳妥的汇报口径

如果现在需要给老师或组里做简洁汇报，建议直接用下面这段：

> 我们已经完成了 memory 两阶段 gating 版本的完整 `16` 组 routing score sweep，并补齐了所有缺失结果。  
> 当前结论是：没有任何 score 组合优于原始 SpecEyes baseline；只要放宽 memory acceptance，small branch 占比就会上升且准确率下降。  
> 因此 memory 这条线的主要矛盾已经不是 retrieval 接得够不够，而是 acceptance / fallback calibration 还不成立。  
> 不过 memory 不是完全无效，因为在更早阶段它曾在 `direct_attributes` 上把准确率从 `90.43` 提到 `92.17`。当前问题是这部分收益还没有被稳定保住。

---

## 6. 当前状态结论

- `stage18` 结果现在已经完整，可以作为正式结论使用。
- 这轮实验支持我们继续围绕：
  - acceptance calibration
  - 哪些任务该触发 memory
  - 哪些任务不该强行 rerun
  来迭代。
- 这轮实验**不支持**继续单纯扩大 routing score 类型搜索空间。
