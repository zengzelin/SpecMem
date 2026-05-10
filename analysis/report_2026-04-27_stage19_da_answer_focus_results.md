\# Stage19 Direct Attributes Answer-Focus Rerun 结果整理（2026-04-27）

## 1. 这轮实验在验证什么

这轮 `stage19` 不再继续扫新的 routing score，而是专门验证：

> **把 memory rerun 改成更短、更答案中心化的 `answer_focus` 风格后，是否能让 rerun 更“干净”。**

当前重点不是追求立即大幅提升总准确率，而是看：

- 是否减少错误 accept
- 是否减少 `changed_but_still_wrong`
- 是否保留真实 corrected 样本
- 是否比上一轮 `confidence -> tail` 更不容易“几乎全收”

---

## 2. 本轮实际运行的两组实验

都在 `VStar direct_attributes` 上运行，均使用：

- `memory_prompt_style=answer_focus`
- `logic_top_k=1`
- `memory_trigger_threshold=0.97`
- `memory_score_threshold=0.98`
- `trigger_metric=confidence_score`

只改 `accept_metric`：

| 组别 | 配置 | 目的 |
|---|---|---|
| 主组 | `confidence -> confidence` | 看 answer-focus 在保守 accept 下是否更干净 |
| 对照组 | `confidence -> tail` | 看 answer-focus 是否能改善激进 accept 的劣化问题 |

---

## 3. Judge 结果

### 3.1 本轮两组结果

| 组别 | all_acc | small_ratio | small_cnt | large_cnt |
|---|---:|---:|---:|---:|
| `answer_focus + conf->conf` | `90.43` | `58.26%` | `67` | `48` |
| `answer_focus + conf->tail` | `77.39` | `92.17%` | `106` | `9` |

### 3.2 与关键历史结果对照

| 配置 | all_acc | small_ratio | 说明 |
|---|---:|---:|---|
| SpecEyes baseline | `90.43` | `57.39%` | 当前稳态 baseline |
| stage18 `conf->conf` | `90.43` | `57.39%` | 不接受 memory rerun，回到 baseline |
| stage18 `conf->tail` | `76.52` | `91.30%` | 激进 accept 明显崩掉 |
| stage19 `answer_focus + conf->conf` | `90.43` | `58.26%` | 仍是 baseline 水平 |
| stage19 `answer_focus + conf->tail` | `77.39` | `92.17%` | 比 stage18 `conf->tail` 略好，但仍明显差 |

---

## 4. Rerun Audit 结果

### 4.1 `answer_focus + conf->conf`

来源：

- `analysis/audit_stage19_da_answer_focus_conf_accept/summary.json`

核心指标：

| 指标 | 数值 |
|---|---:|
| `triggered_count` | `26` |
| `accepted_after_trigger` | `1` |
| `accepted_after_trigger_ratio` | `3.85%` |
| `corrected_count` | `12` |
| `harmed_count` | `3` |
| `changed_but_still_wrong_count` | `1` |
| `answer_changed_count` | `16` |

解释：

- rerun 仍然能产生真实 corrected 样本
- 但保守 accept 几乎全部拒绝了这些 rerun
- 最终系统行为仍接近 baseline

与 `stage18 conf->conf` 对比：

| 指标 | stage18 | stage19 |
|---|---:|---:|
| `accepted_after_trigger` | `0` | `1` |
| `corrected_count` | `12` | `12` |
| `harmed_count` | `3` | `3` |
| `changed_but_still_wrong_count` | `1` | `1` |

结论：

> `answer_focus` 没有让保守 accept 变得真正“会收正确 rerun”，只是把 `0 accepted` 变成了 `1 accepted`。

---

### 4.2 `answer_focus + conf->tail`

来源：

- `analysis/audit_stage19_da_answer_focus_tail_accept/summary.json`

核心指标：

| 指标 | 数值 |
|---|---:|
| `triggered_count` | `26` |
| `accepted_after_trigger` | `26` |
| `accepted_after_trigger_ratio` | `100%` |
| `corrected_count` | `1` |
| `harmed_count` | `3` |
| `changed_but_still_wrong_count` | `1` |
| `answer_changed_count` | `5` |

与 `stage18 conf->tail` 对比：

| 指标 | stage18 | stage19 |
|---|---:|---:|
| `accepted_after_trigger` | `25` | `26` |
| `corrected_count` | `1` | `1` |
| `harmed_count` | `4` | `3` |
| `changed_but_still_wrong_count` | `1` | `1` |
| `all_acc` | `76.52` | `77.39` |

结论：

> `answer_focus` 对激进 accept 有轻微帮助，但帮助很有限。  
> 它没有改变“conf->tail 会把几乎所有 triggered rerun 都收下”的根本问题。

---

## 5. 当前最重要的结论

### 5.1 可以确认的正面信号

- `answer_focus + conf->tail` 比 `stage18 conf->tail` 略有改善：
  - `all_acc`: `76.52 -> 77.39`
  - `harmed_count`: `4 -> 3`

这说明：

> rerun 输出形态确实会影响 memory 行为，方向不是完全没信号。

### 5.2 仍然没有解决的核心问题

- 在保守 accept 下：
  - corrected 仍然存在
  - 但几乎收不进来
- 在激进 accept 下：
  - 几乎全收的问题依旧存在
  - 总准确率仍然明显崩掉

这说明：

> 单靠把 rerun 改成 `answer_focus`，还不足以让 accept / fallback 机制真正成立。

---

## 6. 现阶段最稳妥的判断

这轮 `stage19` 支持下面这个判断：

> `answer_focus` 是一个有轻微信号的改动，但它不是决定性修复。  
> 当前主矛盾仍然是：  
> **哪些 rerun 应该被 accept，accept 逻辑如何利用更多信号，而不是只靠单一 score。**

因此下一步更值得继续的方向仍然是：

1. `retrieval relevance gate`
2. `delta / multi-signal accept`
3. task-specific memory policy

而不是继续只改 rerun 文案本身。
