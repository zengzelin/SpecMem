# SpecMem / SpecEyes 优化尝试全量交接总结（重写版，便于接手）

## 1. 这份文档是给谁看的

这份文档是给“没有全程参与这轮实验、但现在需要接手”的人看的。

目标不是只给出一句“memory 没有效果”，而是把下面四件事讲清楚：

| 需要讲清楚的事 | 这份文档怎么回答 |
|---|---|
| 原始 `SpecEyes` / `SpecMem` 流程是什么 | 第 2 节 |
| memory 具体插在代码哪一层 | 第 3 节 |
| 每一轮优化相比上一轮改了什么 | 第 4 节和第 5 节 |
| 每轮改动对应的结果和结论是什么 | 第 4 节、第 5 节、第 6 节 |

如果只想快速接手，建议阅读顺序是：

1. 第 2 节：先建立系统全貌  
2. 第 3 节：看代码入口在哪  
3. 第 4 节：看每个 stage 改了什么、结果如何  
4. 第 6 节：直接看目前已经比较确定的结论  

---

## 2. 原始系统和 memory 版系统到底是什么关系

### 2.1 原始 `SpecEyes` 是什么流程

原始 `SpecEyes` 不是“所有样本都直接走小模型”，而是一个两阶段 speculative 流程：

| 阶段 | 逻辑 | 代码位置 |
|---|---|---|
| Phase-I router | 先判断这个样本是否适合走 small branch | `eval_code_deepeyes/SpecEyes.py` 里的 `process_test_type()` |
| Phase-II answer | 如果 Phase-I 输出 `no`，就走 small model；如果输出 `yes`，就直接走 large model | `process_test_type()` |
| small fallback | small model 会输出答案和置信度；如果 small answer 不够可靠，就 fallback 到 large model | `process_test_type()` 和 small-branch score 判断 |

这里最重要的一点是：

> `SpecEyes` 的核心不是“只靠 small model”，而是“先用 router 选哪些样本值得尝试 small，再决定是否 fallback 到 large”。

### 2.2 最早的 `SpecMem` 是怎么插入 memory 的

最早期的 `SpecMem` 可以理解为：

> 在 `SpecEyes` 的 small-model answer 阶段前，把检索到的 memory 作为额外文本提示拼进 small model prompt。

也就是说，最早的 memory 不是独立模块去改 router，而是：

1. 检索 logic / visual memory
2. 把 memory 拼进 small model 的 question prompt
3. 让 small model 带着这段 memory 重新回答

这条线最开始主要做的是：

| 早期尝试 | 实质 |
|---|---|
| `logic_top_k` 改成 `k1/k2/k3` | 控制注入多少条 logic memory |
| `task_aware retrieval` | 对 spatial / color / counting 题给不同 memory bonus |
| `compact_spatial` 等 prompt | 改 memory 拼接方式 |

### 2.3 后面做的“优化”本质上是在修哪一层

后续 stage17 之后的大部分优化，重点已经不是“有没有把 memory 接进 prompt”，而是在修下面这件事：

> 如果 small model 的原始回答先出来了，什么时候应该触发 memory rerun？  
> 触发后 rerun 的答案什么时候应该保留？  
> 什么时候应该直接 fallback 到 large model？

所以，后面真正的主线是：

| 主线 | 实质 |
|---|---|
| trigger 设计 | 什么样的 small answer 才值得用 memory 再试一次 |
| accept 设计 | rerun with memory 之后，什么样的 rerun 值得被保留 |
| fallback 设计 | 哪些 rerun 不值得保留，应该回退到大模型 |

---

## 3. 代码对应关系：到底改了哪些文件，分别负责什么

这一节是给接手人最快建立代码地图用的。

### 3.1 主流程文件

| 文件 | 关键函数 / 参数 | 作用 | 这轮实验主要改的东西 |
|---|---|---|---|
| `eval_code_deepeyes/SpecEyes.py` | `parse_arguments()` | 所有 memory / score / routing CLI 开关入口 | 新增了 `memory_score_threshold`、`memory_trigger_threshold`、`trigger_metric`、`accept_metric`、`memory_retrieval_min_score`、`memory_accept_policy`、`memory_selector_*` 等 |
| `eval_code_deepeyes/SpecEyes.py` | `resolve_memory_policy()` | 按 task 决定 memory 是否启用、prompt style、threshold 等 | 接入 task-aware memory policy |
| `eval_code_deepeyes/SpecEyes.py` | `_process_batch_small_model_once()` | small branch 的核心逻辑 | 先 raw pass，再按条件决定是否 rerun with memory |
| `eval_code_deepeyes/SpecEyes.py` | `should_trigger_memory()` | 决定是否触发 rerun | 用 `trigger_metric` 和 `memory_trigger_threshold` 判断 |
| `eval_code_deepeyes/SpecEyes.py` | `evaluate_memory_retrieval_gate()` | 判断 retrieval 是否过门槛 | stage21 的 retrieval gate 在这里 |
| `eval_code_deepeyes/SpecEyes.py` | `should_accept_small_result()` | 决定 rerun 答案是否保留 | stage20 / stage22 的 accept policy 在这里 |
| `eval_code_deepeyes/SpecEyes.py` | `run_selector_for_item()` | 在 base answer 和 memory answer 之间做 A/B selector | stage23 的 selector 路线在这里 |
| `eval_code_deepeyes/SpecEyes.py` | `build_result_record()` | 把路由、trigger、score profile、retrieval 等写进 jsonl | richer schema 的核心写出点 |

### 3.2 prompt 和 retrieval 文件

| 文件 | 关键函数 | 作用 | 本轮实验对应内容 |
|---|---|---|---|
| `memory_aug/prompting.py` | `augment_small_model_prompt()` | 把 memory section 拼到原始 question prompt 后面 | `default`、`compact_spatial`、`compact_general`、`answer_focus`、`empty_scaffold`、`no_memory` 都在这里 |
| `memory_aug/retriever.py` | `retrieve_logic_memories()` | 本地 lexical overlap 检索 logic memory | `logic_top_k`、`task_aware bonus`、retrieval metadata 都在这里 |
| `memory_aug/retriever.py` | `retrieve_visual_memories()` | visual memory 检索 | 目前保留接口，但不是主要有效来源 |
| `memory_aug/retriever.py` | `_attach_retrieval_metadata()` | 把 `_retrieval_score/_rank/_match` 写到 memory 条目里 | stage21 以及审计分析要用 |

### 3.3 分析和离线回放文件

| 文件 | 作用 | 什么时候用 |
|---|---|---|
| `scripts/replay_memory_threshold.py` | 不重跑模型，只离线重放不同 acceptance threshold 的最终路由 | 用来排查“阈值是不是主要矛盾” |
| `scripts/compare_memory_runs.py` | 对比 base run 和 memory run，输出 corrected / harmed / retrieved-no-gain | 看 memory 到底修对了哪些样本、害了哪些样本 |
| `scripts/audit_memory_rerun_effects.py` | 对单个 memory run 做 rerun 审计 | stage18 之后非常关键 |

### 3.4 shell 脚本状态

当前仓库里保留下来的代表性脚本主要有：

| 脚本 | 作用 |
|---|---|
| `scripts/run_score_routing_stage18.sh` | stage18 单组 score 配置运行 |
| `scripts/run_score_metric_sweep_stage18_seq.sh` | stage18 16 组组合 sweep |
| `scripts/run_stage19_da_answer_focus.sh` | stage19 answer-focus rerun |
| `scripts/run_stage21_da_retrieval_gate.sh` | stage21 retrieval gate |
| `scripts/run_stage23_selector_ab.sh` | stage23 selector A/B |

需要特别说明：

> `stage20` 和 `stage22` 的结果目录、judge 结果、analysis 结论都在，但仓库里没有保留一个单独命名的 shell 脚本；  
> 这两轮的“实验定义”主要由 `SpecEyes.py` 的参数组合和结果目录名体现。

---

## 4. 所有优化尝试的时间线总表

下面这张表是最重要的总览表。

### 4.1 阶段总表

| 阶段 | 相比上一阶段改了什么 | 主要代码 / 参数 | 代表结果 | 结论 |
|---|---|---|---|---|
| baseline | 原始 `SpecEyes`，无 memory | `--memory_enable` 关闭 | `DA=90.43`，`RP=86.84` | 参照系 |
| stage15 | 先试 memory prompt 变体，重点看 `RP` | `compact_spatial`、`task_aware`、`k3/k1` | `RP k3 default=86.84`，`compact_spatial=81.58` | prompt 改造没有救 `RP` |
| stage16 | 继续原始 memory 注入，但系统性看 `k1/k2/task_aware` | `logic_top_k=1/2`，task-aware retrieval | `DA=92.17`，`RP<=84.21` | `DA` 有真实正信号，`RP` 没有 |
| stage17 | 引入真正的 two-stage memory gating：raw first，再决定是否 rerun with memory | `memory_trigger_threshold`、`memory_score_threshold`，`_process_batch_small_model_once()` | `DA triggered` 掉回 `90.43`，`always-memory` 仍有 `92.17` | memory 可能有用，但 trigger/accept 逻辑没保住收益 |
| stage18 | 把单一 `confidence_score` 扩成 score profile，并做 `trigger_metric x accept_metric` 全 sweep | `tail_score`、`lowest_group_score`、`bottom10_group_score`，`run_score_routing_stage18.sh` | 最好仍是 baseline，最差 `DA=76.52` | 扩大 score 组合搜索无效 |
| stage19 | 改 memory rerun prompt，让输出更短、更 answer-centric | `memory_prompt_style=answer_focus` | `conf->conf=90.43`，`conf->tail=77.39` | prompt 有轻微信号，但不是根修复 |
| stage20 | accept 从单阈值改成多信号规则 | `memory_accept_policy=triggered_multisignal` | `DA=90.43`，但 `small_ratio` 从 `58.26%` 到 `60.87%` | 第一次得到“稳定但不涨点”的控制增益 |
| stage21 | 在 trigger 后增加 retrieval relevance gate | `memory_retrieval_min_score=0.45/0.50` | `DA` 仍 `90.43`，`small_ratio` 小降到 `60.0%` | retrieval gate 只是卫生措施，不是主收益来源 |
| stage22 | 收紧 delta-conf accept | `memory_accept_min_delta_conf=0/-0.01` | `DA` 仍 `90.43`，`small_ratio=59.13%/60.0%` | 更保守，但仍不带来精度收益 |
| stage23 | 不再靠阈值 accept，而是显式让 selector 在 base / memory 之间二选一 | `memory_accept_policy=selector_ab`，`memory_selector_model`，`memory_selector_base_action` | 4 组都 `90.43`，但 `small_ratio` 塌到 `1.74%-4.35%` | 强负信号，基本可判定不值得继续 |

### 4.2 目前最该记住的三条事实

| 事实 | 证据 |
|---|---|
| memory 不是完全没用 | `stage16 direct_attributes` 从 baseline `90.43` 到 `92.17` |
| 但 memory 没被做成稳定方法 | `stage17` 一旦改成 trigger 式 rerun，`DA` 大多掉回 `90.43` |
| 后续大部分优化都只是在“更稳地回到 baseline”，没有重新超过 `92.17` | `stage18-23` 全部没超过 baseline `90.43`，更没超过 `92.17` |

---

## 5. 分阶段展开：每轮到底改了什么、结果是什么、该怎么理解

## 5.1 baseline / stage15 / stage16：先确认 memory 有没有正信号

### baseline

baseline 是原始 `SpecEyes` 参照系：

| 任务 | all_acc | small_ratio |
|---|---:|---:|
| `VStar direct_attributes` | `90.43` | `57.39%` |
| `VStar relative_position` | `86.84` | `75.00%` |

### stage15：先试 `relative_position`

这一轮的思路是：

> 先不动系统主流程，只改 memory prompt 和 retrieval 配置，看 `RP` 能不能先被救回来。

代表配置和结果：

| 配置 | all_acc | small_ratio | 结论 |
|---|---:|---:|---|
| `k3 default memory` | `86.84` | `69.74%` | 只能回到 baseline |
| `k3 + compact_spatial` | `81.58` | `76.32%` | 明显退化 |
| `k1 + compact_spatial + task_aware` | `82.89` | `78.95%` | 仍退化 |

这一轮的结论很简单：

> 单靠 spatial prompt 工程和 task-aware retrieval，救不回 `relative_position`。

### stage16：确认 `direct_attributes` 上有过真实提升

这一轮的关键价值是：

> 证明 memory 至少在一个任务上不是假信号。

代表结果：

| 配置 | all_acc | small_ratio | 结论 |
|---|---:|---:|---|
| `DA k1` | `92.17` | `53.04%` | 明确优于 baseline |
| `DA k2` | `92.17` | `51.30%` | 同样优于 baseline |
| `DA k1 + task_aware` | `92.17` | `53.04%` | 与 plain 基本一样 |
| `DA k2 + task_aware` | `92.17` | `51.30%` | 与 plain 基本一样 |
| `RP k1 + task_aware` | `84.21` | `40.79%` | 低于 baseline |
| `RP k2 / k3` | `82.89~84.21` | `39.47%~44.74%` | 低于 baseline |

这一轮留下的结论是：

1. `direct_attributes` 上 memory 确实出现过真实正收益。  
2. `task_aware retrieval` 不是主收益来源，因为开和不开几乎一样。  
3. `relative_position` 依然是明显失败面。  

---

## 5.2 stage17：从“总是带 memory”切到“先 raw，再条件 rerun with memory”

### 这一轮到底改了什么

这是第一个真正的结构性改动。

相比 stage16，stage17 不再默认 small branch 一开始就带着 memory 生成，而是改成：

1. small model 先跑一次 **raw pass**
2. 先拿到 `base_output_text`、`base_confidence_score`、`base_score_profile`
3. 再根据 trigger 条件决定是否 rerun with memory
4. rerun 之后再决定保留 small answer 还是 fallback 到 large

对应代码就是：

| 代码 | 作用 |
|---|---|
| `SpecEyes.py::_process_batch_small_model_once()` | 先 `use_memory=False` 跑 base pass，再决定是否 `use_memory=True` rerun |
| `SpecEyes.py::should_trigger_memory()` | 用 `trigger_metric + memory_trigger_threshold` 决定要不要 rerun |
| `SpecEyes.py::get_acceptance_threshold()` | memory-on 时允许 acceptance threshold 与 baseline threshold 分离 |

这轮新增的关键参数是：

| 参数 | 作用 |
|---|---|
| `--memory_score_threshold` | memory rerun 后是否 accept 的阈值 |
| `--memory_trigger_threshold` | 是否触发 memory rerun 的阈值 |
| `--trigger_metric` | 用哪个分数做 trigger |
| `--accept_metric` | 用哪个分数做 accept |

### 为什么这轮重要

因为它第一次把“memory 有用吗”拆成了两个问题：

1. 哪些样本值得用 memory 再试一次  
2. rerun 后哪些答案值得留下来  

### 代表结果

`direct_attributes`：

| 配置 | all_acc | small_ratio | 解释 |
|---|---:|---:|---|
| `k1, always-memory` | `92.17` | `53.04%` | 仍保留 stage16 收益 |
| `k1, acc=0.9775, trig=0.97` | `90.43` | `60.87%` | 掉回 baseline |
| `k1, acc=0.98, trig=0.97` | `90.43` | `57.39%` | 掉回 baseline |
| `k2, acc=0.9775, trig=0.97` | `90.43` | `60.87%` | 掉回 baseline |

`relative_position`：

| 配置 | all_acc | small_ratio | 解释 |
|---|---:|---:|---|
| `k1, acc=0.97, trig=0.95` | `84.21` | `94.74%` | 路由明显失真 |
| `k1, acc=0.97, trig=0.97` | `84.21` | `94.74%` | 几乎一样 |
| `k2, acc=0.9725, trig=0.95` | `84.21` | `92.11%` | 仍失败 |

### 这一轮得出的核心结论

> memory 本身可能不是无效的，因为 always-memory 还能维持 `92.17`；  
> 真正失败的是“什么时候 rerun / 什么时候 accept”这套 routing 和 acceptance 逻辑。

---

## 5.3 stage18：把单一 `confidence_score` 扩成 score profile，并做 16 组 sweep

### 这一轮到底改了什么

stage17 之后的怀疑是：

> 不是 memory 无用，而是原来的 `confidence_score` 太粗糙，不能分出“值得 rerun / 值得 accept”的样本。

所以这一轮新增了三个局部分数：

| 分数 | 含义 |
|---|---|
| `tail_score` | 看答案尾部 token 的稳定性 |
| `lowest_group_score` | 看最差局部窗口 |
| `bottom10_group_score` | 看最差 10% 局部窗口 |

这些分数是在 `_run_small_model_pass()` 里由 `score_profile` 一起产出的。

然后通过：

- `scripts/run_score_routing_stage18.sh`
- `scripts/run_score_metric_sweep_stage18_seq.sh`

系统性跑了 `4 x 4 = 16` 组：

- `trigger_metric ∈ {confidence, tail, lowest_group, bottom10_group}`
- `accept_metric ∈ {confidence, tail, lowest_group, bottom10_group}`

### 代表结果

最好组全部只是 baseline：

| trigger_metric | accept_metric | DA acc | RP acc | 结论 |
|---|---|---:|---:|---|
| `confidence_score` | `confidence_score` | `90.43` | `86.84` | baseline |
| `tail_score` | `confidence_score` | `90.43` | `86.84` | baseline |
| `lowest_group_score` | `confidence_score` | `90.43` | `86.84` | baseline |
| `bottom10_group_score` | `confidence_score` | `90.43` | `86.84` | baseline |

最差组：

| trigger_metric | accept_metric | DA acc | RP acc | 结论 |
|---|---|---:|---:|---|
| `confidence_score` | `tail_score` | `76.52` | `82.89` | 最差 |

### 这一轮真正告诉我们的事情

这轮实验不是“没有结果”，而是给了一个很明确的负结论：

> 继续扩大 `trigger_metric x accept_metric` 搜索空间，已经没有明显价值。

因为：

1. 最好配置仍然只是回到 baseline  
2. 一旦 accept 放松，`small_ratio` 会明显上升，但准确率会下降  
3. 所以问题不是“还没试够 metric”，而是当前 acceptance 信号本身没有抓住“哪些 rerun 值得保留”  

---

## 5.4 stage19：改 prompt，不再追求长回答，改成 `answer_focus`

### 这一轮到底改了什么

这一轮没有再扫新的 routing score，而是怀疑：

> memory rerun 之所以不稳定，可能不是分数不行，而是 rerun 输出本身太长、太散、不够 answer-centric。

所以在 `memory_aug/prompting.py::augment_small_model_prompt()` 里新增了：

- `prompt_style = answer_focus`

它做的事是：

1. 仍然在原 question prompt 后追加 memory section  
2. 但额外加一句明确指令：  
   “Re-evaluate only the final answer. Keep the response minimal and answer-centric.”  
3. 对多选题尽量只输出“最佳选项 + 对应答案文本”  

### 这一轮跑了什么

`scripts/run_stage19_da_answer_focus.sh`

只在 `VStar direct_attributes` 上测试两组：

| 组别 | trigger_metric | accept_metric |
|---|---|---|
| 主组 | `confidence_score` | `confidence_score` |
| 对照组 | `confidence_score` | `tail_score` |

### 结果

| 配置 | all_acc | small_ratio | 结论 |
|---|---:|---:|---|
| `answer_focus + conf->conf` | `90.43` | `58.26%` | 仍然只是 baseline |
| `answer_focus + conf->tail` | `77.39` | `92.17%` | 比 stage18 最差组略好，但仍明显差 |

对应 audit：

| 配置 | triggered | accepted_after_trigger | corrected | harmed |
|---|---:|---:|---:|---:|
| `conf->conf` | `26` | `1` | `12` | `3` |
| `conf->tail` | `26` | `26` | `1` | `3` |

### 这一轮的结论

这轮留下的是一个“弱正信号”：

> rerun 输出形态确实会影响效果，但单靠 prompt 改成 answer-focus，不足以把 accept/fallback 做对。

换句话说：

- 它不是完全没用  
- 但它不是决定性修复  

---

## 5.5 stage20：把 accept 从“单阈值”改成“多信号规则”

### 这一轮到底改了什么

stage19 之后的判断是：

> 单靠一个 accept 分数，不足以决定 rerun 是否值得保留。  
> 应该显式引入更多约束条件。

于是新增了：

- `memory_accept_policy=triggered_multisignal`

这个逻辑在 `SpecEyes.py::should_accept_small_result()` 中实现。

相对 stage19 的变化是：

| 约束 | 参数 | 含义 |
|---|---|---|
| rerun 自身最小置信度 | `memory_accept_min_confidence=0.9` | 太低分的 rerun 直接不收 |
| rerun 比 raw 至少不能差太多 | `memory_accept_min_delta_conf=-0.02` | `(memory_conf - base_conf)` 不能低于阈值 |
| retrieval 至少得有一点相关性 | `memory_accept_min_retrieval_score=0.0` | 这一轮先不收紧，但接口已经接入 |
| rerun 最好真的改了答案 | `memory_accept_answer_change=changed` | 只收“答案有变化”的 rerun |

这轮的实验定义保存在：

- `analysis/report_2026-04-28_stage20_multisignal_accept_result.md`
- 结果目录 `eval_results_deepeyes/SpecEyes_vstar_stage20_da_multisignal_accept`

### 结果

| 配置 | all_acc | small_ratio | small_cnt | large_cnt |
|---|---:|---:|---:|---:|
| stage19 `answer_focus + conf->conf` | `90.43` | `58.26%` | `67` | `48` |
| stage20 `triggered_multisignal` | `90.43` | `60.87%` | `70` | `45` |

对应 audit：

| 指标 | 数值 |
|---|---:|
| `triggered_count` | `26` |
| `accepted_after_trigger` | `4` |
| `corrected_count` | `12` |
| `harmed_count` | `3` |
| `accepted_changed_count` | `4` |

### 这一轮的结论

这轮是后半段实验里最重要的一轮，因为它第一次给出：

> 精度不涨，但控制逻辑变稳了。

更具体地说：

1. `all_acc` 没有退化  
2. `small_ratio` 上升，说明少量样本不再无谓 fallback 到 large  
3. rerun 不再是“几乎全拒”或“几乎全收”这两种极端  

所以 stage20 的意义不是“把 memory 做成提升精度的方法”，而是：

> 第一次把 memory 做成了一个相对可控的 speculative control signal。

---

## 5.6 stage21：增加 retrieval relevance gate

### 这一轮到底改了什么

stage20 虽然更稳了，但一个自然问题是：

> 触发 rerun 后，如果检索到的 memory 根本不相关，是不是应该直接别 rerun？

因此新增了：

- `memory_retrieval_min_score`

对应代码：

| 代码 | 作用 |
|---|---|
| `SpecEyes.py::evaluate_memory_retrieval_gate()` | 先看 top-1 retrieval score 是否过门槛，不达标则不触发 rerun |
| `memory_aug/retriever.py::_attach_retrieval_metadata()` | 给每条 memory 记录 `_retrieval_score`、`_retrieval_rank`、`_retrieval_match` |

实验脚本：

- `scripts/run_stage21_da_retrieval_gate.sh`

### 跑了什么

在 stage20 基础上只改：

| 配置 | 变化 |
|---|---|
| `rmin=0.45` | top-1 retrieval score < 0.45 不 rerun |
| `rmin=0.50` | top-1 retrieval score < 0.50 不 rerun |

### 结果

| 配置 | all_acc | small_ratio | small_cnt |
|---|---:|---:|---:|
| stage20 | `90.43` | `60.87%` | `70` |
| stage21 `rmin=0.45` | `90.43` | `60.00%` | `69` |
| stage21 `rmin=0.50` | `90.43` | `60.00%` | `69` |

对应 audit：

| 配置 | triggered | accepted_after_trigger | corrected | harmed |
|---|---:|---:|---:|---:|
| `rmin=0.45` | `23` | `3` | `11` | `3` |
| `rmin=0.50` | `17` | `3` | `9` | `2` |

### 这一轮的结论

这一轮的价值主要是“收紧脏 rerun”，不是“带来收益”。

换句话说：

> retrieval gate 是一个合理的 hygiene 约束，但它不是当前主增益来源。

---

## 5.7 stage22：继续收紧 delta-conf accept

### 这一轮到底改了什么

stage21 之后继续追问：

> 如果 rerun 的置信度没有明显比 raw 好，是否应该更严格地拒绝它？

所以这一轮在 stage20 / stage21 的基础上继续收紧：

- `memory_accept_min_delta_conf`

主要测试：

| 配置 | 含义 |
|---|---|
| `delta=0` | rerun 的置信度至少不能低于 raw |
| `delta=-0.01` | 比 stage20 的 `-0.02` 更严格一些 |

实验定义主要体现在：

- 结果目录 `SpecEyes_vstar_stage22_da_delta_0`
- 结果目录 `SpecEyes_vstar_stage22_da_delta_m001`
- 对应 audit 目录

### 结果

| 配置 | all_acc | small_ratio | small_cnt |
|---|---:|---:|---:|
| stage20 | `90.43` | `60.87%` | `70` |
| stage22 `delta=0` | `90.43` | `59.13%` | `68` |
| stage22 `delta=-0.01` | `90.43` | `60.00%` | `69` |

对应 audit：

| 配置 | triggered | accepted_after_trigger | corrected | harmed |
|---|---:|---:|---:|---:|
| `delta=0` | `26` | `2` | `12` | `3` |
| `delta=-0.01` | `26` | `3` | `12` | `3` |

### 这一轮的结论

这轮进一步确认了一件事：

> 更严格的 delta-conf 只是在更保守地回退，不会自动把 memory 变成增益。

所以它仍然属于“控制细化”，不是“找到新收益机制”。

---

## 5.8 stage23：不再相信阈值，改成 selector 在 base / memory 之间显式二选一

### 这一轮到底改了什么

stage20-22 都没有涨点之后，最后尝试了一条更激进的思路：

> 不再靠阈值判断 memory rerun 值不值得保留，  
> 直接再让一个模型看 `base answer` 和 `memory answer`，在二者之间显式选择。

对应代码：

| 代码 | 作用 |
|---|---|
| `SpecEyes.py::run_selector_for_item()` | 构造 A/B selector prompt，让 selector 模型在 base / memory 间二选一 |
| `SpecEyes.py::should_run_selector()` | 只有 base 和 rerun 答案不同才运行 selector |
| `SpecEyes.py::memory_selector_model` | selector 用 large 还是 small |
| `SpecEyes.py::memory_selector_base_action` | 当 selector 选 base 时，是直接 keep base，还是 fallback 到 large |

实验脚本：

- `scripts/run_stage23_selector_ab.sh`

### 跑了哪 4 组

| 组别 | selector model | base action |
|---|---|---|
| `large_keep` | `large` | `keep_base` |
| `large_fallback` | `large` | `fallback_large` |
| `small_keep` | `small` | `keep_base` |
| `small_fallback` | `small` | `fallback_large` |

### 正式 judge 结果

| 配置 | all_acc | small_ratio | small_cnt | large_cnt |
|---|---:|---:|---:|---:|
| `large_keep` | `90.43` | `4.35%` | `5` | `110` |
| `large_fallback` | `90.43` | `2.61%` | `3` | `112` |
| `small_keep` | `90.43` | `4.35%` | `5` | `110` |
| `small_fallback` | `90.43` | `1.74%` | `2` | `113` |

离线对 stage20 的对比：

| 配置 | route_changed vs stage20 | pred_changed vs stage20 | 结论 |
|---|---:|---:|---|
| `large_keep` | `67` | `5` | 大幅改路由，几乎不改最终答案 |
| `large_fallback` | `67` | `4` | 同上 |
| `small_keep` | `67` | `6` | 同上 |
| `small_fallback` | `68` | `4` | 同上 |

对应 audit：

| 配置 | triggered | accepted_after_trigger | corrected | harmed |
|---|---:|---:|---:|---:|
| `large_keep` | `26` | `5` | `12` | `3` |
| `large_fallback` | `26` | `3` | `12` | `3` |
| `small_keep` | `26` | `5` | `11` | `2` |
| `small_fallback` | `26` | `2` | `12` | `3` |

### 这一轮的结论

这轮可以非常明确地下结论：

> `selector_ab` 这条线是强负信号。

原因不是“没变化”，而是：

1. 它强烈改变了 speculative routing  
2. 几乎把 small branch 打没了  
3. 但最终精度没有变好  

所以 stage23 不是“没想清楚，还可以再调”，而是：

> 这条 selector 方向基本已经说明不值得继续主攻。

---

## 6. 现在可以比较确定的结论

### 6.1 哪些事情已经基本可以确定

| 结论 | 支撑证据 |
|---|---|
| memory 不是完全没用 | `stage16 direct_attributes` 到过 `92.17` |
| memory 还没有被做成稳定有效的方法 | stage17 之后所有“更正式的控制版”都没重新超过 baseline，更没超过 `92.17` |
| `relative_position` 不是当前最适合继续主攻的验证面 | stage15-18 多轮都没带来稳定收益 |
| `task_aware retrieval` 不是主瓶颈 | stage16 plain 和 task-aware 基本一样 |
| 单纯继续扫 `trigger_metric x accept_metric` 没有价值 | stage18 已经做了完整 16 组 sweep |
| prompt 不是完全无关，但不是根因 | `answer_focus` 只有轻微信号，没有根修复 |
| stage20 的 `triggered_multisignal` 是后期最有保留价值的控制版本 | 不涨精度，但更稳、更少无谓 fallback |
| stage23 selector 路线基本可判负面 | 4 组正式 judge 全部没有收益，small_ratio 严重塌缩 |

### 6.2 当前最合理的总体判断

到目前为止，最准确的总判断不是：

- “memory 完全没用”

而是：

> memory 在 `direct_attributes` 上曾经出现过真实正信号；  
> 但把这部分信号稳定地保存在 `SpecEyes` 的 speculative routing 里，这件事目前没有成功。  

更具体一点说：

1. 早期 memory 注入曾经在 `DA` 上有效  
2. 后期为了把它做成更干净、更可解释的系统，引入了 trigger / accept / rerun / selector 等控制逻辑  
3. 这些控制逻辑到目前为止都没有把收益稳定保住  
4. 最好结果是“更稳地回到 baseline”，不是“稳定超过 baseline”  

---

## 7. 对接手人的直接建议：哪些方向不要再重复，哪些资产要保留

### 7.1 不建议再重复主攻的方向

| 方向 | 原因 |
|---|---|
| 再做更大规模的 `trigger_metric x accept_metric` sweep | stage18 已经足够说明问题 |
| 再做更大规模 threshold sweep | 现象已经很稳定，信息增量低 |
| 把 `task_aware retrieval` 当主突破口 | 历史结果不支持 |
| 继续深挖 `relative_position` 作为主验证面 | 失败信号过强，ROI 太低 |
| 继续推进 selector A/B 路线 | stage23 已经给出强负结论 |

### 7.2 需要保留的资产

| 资产 | 为什么要保留 |
|---|---|
| two-stage memory gating 主流程 | 它让实验从“黑箱 prompt trick”变成了可解释流程 |
| richer JSONL schema | 后续诊断最依赖它 |
| `answer_focus` prompt style | 虽然不够，但至少比盲目长回答更合理 |
| `triggered_multisignal` accept policy | 这是后期最稳的一条控制线 |
| retrieval metadata logging | stage21 以后所有分析都依赖 `_retrieval_score` |
| `replay_memory_threshold.py` | 做离线 calibration 很方便 |
| `compare_memory_runs.py` / rerun audit | 看 corrected / harmed 非常关键 |

---

## 8. 证据目录索引

如果要进一步追证据，建议按下面的路径看：

| 类型 | 路径 |
|---|---|
| 总报告（早期结论） | `analysis/report_2026-04-26_memory_results_master.md` |
| stage18 总结 | `analysis/report_2026-04-27_stage18_metric_sweep_summary.md` |
| stage18 rerun audit | `analysis/report_2026-04-27_stage18_rerun_audit.md` |
| stage19 总结 | `analysis/report_2026-04-27_stage19_da_answer_focus_results.md` |
| stage20 总结 | `analysis/report_2026-04-28_stage20_multisignal_accept_result.md` |
| stage18 DA audit | `analysis/audit_stage18_da_conf_conf`、`analysis/audit_stage18_da_conf_tail` |
| stage18 RP audit | `analysis/audit_stage18_rp_conf_conf`、`analysis/audit_stage18_rp_conf_tail` |
| stage19 audit | `analysis/audit_stage19_da_answer_focus_conf_accept`、`analysis/audit_stage19_da_answer_focus_tail_accept` |
| stage20 audit | `analysis/audit_stage20_da_multisignal_accept` |
| stage21 audit | `analysis/audit_stage21_da_retrieval_gate_rmin045`、`analysis/audit_stage21_da_retrieval_gate_rmin050` |
| stage22 audit | `analysis/audit_stage22_da_delta_0`、`analysis/audit_stage22_da_delta_m001` |
| stage23 audit | `analysis/SpecEyes_vstar_stage23_selector_*_audit` |

---

## 9. 最后一句话总结

如果必须把所有尝试压缩成一句交接结论，那就是：

> 这条 memory 线已经证明“在 `direct_attributes` 上有过真实正信号”，但也已经证明“仅靠目前这套 trigger / accept / selector 设计，还不能把它做成稳定有效的方法”；  
> 真正值得保留的是这套可审计的控制框架和分析工具，而不是把 stage18-23 再原样多跑几轮。
