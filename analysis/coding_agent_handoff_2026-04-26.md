# SpecEyes Coding Agent Handoff (2026-04-26)

## 1. 当前项目状态

当前代码库已经完成了第一轮 memory 系统接入，主要包括：

- task-aware memory policy
- two-stage gating（raw first -> memory rerun on trigger -> accept/fallback）
- richer per-sample JSONL logging
- replay / compare 脚本对 richer schema 的兼容

也就是说，当前不是“memory 还没接进去”，而是：

> memory 已经能跑，但当前 routing score 没有把它做成稳定有效的方法。

---

## 2. 已经实现了什么

### 2.1 Task-aware memory policy
主要文件：
- `eval_code_deepeyes/SpecEyes.py`

当前已经支持：
- `--memory_task_policy`
- 按任务覆盖：
  - memory on/off
  - prompt style
  - acceptance threshold
  - logic top-k
  - trigger threshold

相关 helper 已加入：
- `parse_memory_task_policy(...)`
- `infer_memory_task(...)`
- `resolve_memory_policy(...)`

---

### 2.2 Prompt variants
主要文件：
- `memory_aug/prompting.py`

当前已有：
- `default`
- `compact_spatial`
- `compact_general`
- `empty_scaffold`
- `no_memory`

---

### 2.3 Retrieval metadata
主要文件：
- `memory_aug/retriever.py`

当前 retrieval 返回的 memory 已包含：
- `_retrieval_score`
- `_retrieval_rank`
- `_retrieval_match`

---

### 2.4 Richer per-sample JSONL outputs
主要文件：
- `eval_code_deepeyes/SpecEyes.py`

当前输出已包含：
- `memory_task`
- `memory_policy`
- `memory_prompt_style_applied`
- `memory_mode_applied`
- `memory_trigger_threshold_applied`
- `memory_acceptance_threshold_applied`
- `memory_triggered`
- `memory_accept_decision`
- `base_confidence_score`
- `base_small_answer`
- retrieval counts

---

### 2.5 Two-stage small-model flow
主要文件：
- `eval_code_deepeyes/SpecEyes.py`

当前 small-model path 已经不是单次 memory prompt，而是：
1. 先跑 raw prompt
2. 得到 `base_confidence_score`
3. 判断是否触发 memory
4. 若触发，则 rerun memory-augmented prompt
5. 根据 accept threshold 决定 small vs large

---

### 2.6 Analysis scripts updated
主要文件：
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`

当前已经支持：
- richer routing 字段
- trigger / accept 元数据
- compare 输出里附加 trigger / accept 字段

---

## 3. 当前实验结果怎么定性

### 3.1 可信结论 1：memory 不是完全没用
在 `direct_attributes` 上，stage16 曾出现稳定提升：

- baseline `mem=off`: `90.43`
- `logic-small-k1 @ mthr=0.98`: `92.17`
- `logic-small-k2 @ mthr=0.98`: `92.17`

因此当前不能说 memory 无效。

更准确地说：

> memory 在 `direct_attributes` 这类 object-/attribute-centric 任务上有正信号。

---

### 3.2 可信结论 2：memory 目前还不稳定
在 stage17 里，一旦切到新的 trigger/accept 路由：

- `direct_attributes` 的大多数触发式配置又回到了 baseline `90.43`
- 只有 always-memory 版本仍维持 `92.17`

因此当前不能说这条线已经稳定成立。

更准确地说：

> memory 是“有局部有效信号的实验性模块”，不是“已经稳定成立的方法”。

---

### 3.3 可信结论 3：task-aware retrieval 不是当前主瓶颈
当前 stage16 / stage17 结果都说明：
- task-aware 与 plain retrieval 差异很小
- 没有形成主要收益来源

所以不应该把 task-aware retrieval 当作当前主线。

---

### 3.4 可信结论 4：relative_position 的主问题是 routing score
这是当前最重要的结论。

在 relpos 的 stage17 eval 中，大量错误样本属于：
- 答案错了
- `confidence_score` 却很高（常见在 `0.975 ~ 0.985+`）
- `memory_triggered = false`
- `memory_accept_decision = true`
- 最终 `use_model = small`

典型错误包括：
- left/right 错误
- above/below 错误
- closer/farther 错误

因此当前主问题不是 retrieval 没命中，而是：

> 当前 separability / confidence score 无法识别真正不可靠的 relpos small answer。

---

## 4. 为什么 stage17 没有真正优化成功

### 4.1 Direct 没保住之前的收益
stage17 表明：
- always-memory 还能维持 `92.17`
- 触发式 routing 却大多掉回 `90.43`

这说明：
- memory 本身不一定没用；
- 问题在于新 trigger / accept score 没有把有用样本稳定地留下来。

### 4.2 Relpos 的路由明显失真
代表性 relpos 配置里：
- `small_ratio` 升到 `92% ~ 95%`
- `large_cnt` 只剩 `4 ~ 6`
- eval 中 `memory_triggered` 甚至可能是 `0`

这不是“路由更自信”，而是：
- 错样本也继续停在 small
- 该触发 memory 的样本没被识别
- fallback 几乎失效

---

## 5. 当前系统的主瓶颈

当前主瓶颈不是：
- retrieval 是否命中
- task-aware retrieval 是否更复杂
- top-k 再扫一轮
- 再加一点 prompt trick

当前主瓶颈是：

### 5.1 全局单标量 score 太扁平
- 局部错误被平均掉
- 最终答案段没有被强调
- trigger / accept / fallback 都挤在一个 score 上

### 5.2 score 不够答案中心化
当前多模态 QA 真正决定 correctness 的常常是：
- 最终 option token
- yes/no token
- 最后一行答案
- left/right / above/below 这种最后几 token

如果所有 token 同权，真正关键的答案信号会被稀释。

### 5.3 两阶段结构已在，但分数没为两阶段服务
系统现在已经能：
- raw first
- trigger memory rerun
- accept / fallback

但 trigger 和 accept 仍然围绕当前不合适的 score 做判断，因此结构有了，决策依据却不对。

---

## 6. 当前最推荐的下一步主线

不要继续把主要精力放在：
- retrieval tricks
- task-aware retrieval
- 更多 top-k sweep
- 更多 prompt 变体

而应该转到：

# 主线：重做 routing score

具体来说：
1. 从单个全局 scalar 变成 **score profile**
2. 从 trace-level confidence 变成 **answer-centric confidence**
3. 让 trigger 和 accept 使用不同 metric
4. 显式比较 raw vs memory 的 score 改善幅度

---

## 7. 下一轮代码应该怎么改

### 7.1 必须新增的 score
建议优先新增：

1. `tail_score`
   - 只看最后一段 token 的 score
   - 用来判断最终答案段是否稳定

2. `lowest_group_score`
   - 把 token 序列切成窗口，取最差窗口
   - 用来发现局部崩点

3. `bottom10_group_score`
   - 对最差 10% 窗口取平均
   - 比 `lowest_group` 更稳，更适合做 trigger

4. `answer_span_score`
   - 只对最终答案区域算 score
   - 对 relpos / yes-no / option 类任务特别重要

---

### 7.2 trigger 和 accept 要分开
推荐下一轮默认试：

- `trigger_metric = bottom10_group_score`
- `accept_metric = tail_score`

理由：
- trigger 的目标是发现“局部脆弱但可能可修复”的样本；
- accept 的目标是判断“最终答案段是否足够可靠”。

---

### 7.3 必须加入 raw vs memory 的 delta logging
当前只有：
- `base_confidence_score`
- `confidence_score`

这不够。

建议新增：
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

这样才能判断：
- memory 是否真的修复了关键区域；
- 还是只是改写了一遍回答。

---

### 7.4 relpos 要优先做 answer-focused score
relpos 目前大量错误集中在：
- left/right
- above/below
- closer/farther

这些任务的 correctness 高度依赖最后短答案。

因此 relpos 下一轮优先试：
- final option token score
- short tail score
- answer span score

不要继续只看全序列平均 separability。

---

## 8. 建议优先修改的文件

### `eval_code_deepeyes/SpecEyes.py`
主修改文件，重点改：

1. `_run_small_model_pass(...)`
   - 生成后不再只返回单个 `confidence_score`
   - 改为先保留 score profile，再产出多个 summary

2. `should_trigger_memory(...)`
   - 支持 `trigger_metric`
   - 默认从单一 global score 改成 `bottom10_group_score`

3. `_process_batch_small_model_once(...)`
   - 让 rerun / accept 逻辑使用新的 metric
   - 同时记录 raw vs memory 的多种 score

4. `build_result_record(...)`
   - 把新的 score profile 和 gain 写进 JSONL

### `scripts/replay_memory_threshold.py`
建议补充：
- 按不同 metric replay
- 至少支持 `tail_score`、`bottom10_group_score`
- 输出对应 metric 的 triggered / accepted / small ratio

### `scripts/compare_memory_runs.py`
建议补充：
- raw vs memory profile 对比
- `tail_gain`
- `bottom10_gain`
- `answer_gain`

---

## 9. 暂时不建议优先做的方向

当前不建议优先作为主线：

1. 更多 task-aware retrieval 规则
2. 更多 top-k sweep
3. 更多 prompt variant 作为主线
4. visual memory / dual memory 扩展

理由：

> 当前不是 retrieval 命不中，而是 score 不知道哪些样本错得离谱。

---

## 10. 推荐的下一轮实验顺序

### 第一轮
只改 score，不动 retrieval：
- 新增 `tail_score`
- 新增 `lowest_group_score`
- 新增 `bottom10_group_score`

默认试：
- `trigger = bottom10_group`
- `accept = tail`

重点看：
- judge acc
- small ratio
- corrected / harmed
- 高置信错答是否减少

### 第二轮
只在 `relative_position` 上试：
- `answer_span_score`
- final option token score
- short tail score

### 第三轮
再分析：
- `tail_gain`
- `answer_gain`
- corrected / harmed 的 score 变化

---

## 11. 对下一位 coding agent 的要求

### 目标
不是重写整个系统，而是：

> **把当前 routing score 从“单个全局 scalar”升级成“局部窗口化 + 尾段 + 答案中心化 + memory-gain”的 score system。**

### 具体要求
1. 不要 redesign retrieval
2. 不要扩 scope 到 visual/dual memory 主线
3. 不要把重点放在更多 task-aware retrieval trick 上
4. 只做与 score、routing、logging、analysis 直接相关的必要修改

### 交付时必须报告
1. 修改了哪些文件
2. 新增了哪些 score / logging 字段
3. trigger / accept 如何改的
4. 跑了哪些验证
5. 结果是否比当前 stage17 更好

---

## 12. 2026-04-26 追加更新：本次已经落地的代码修改

本次已经开始按“重做 routing score”主线改代码，且以下内容已经落地：

### 12.1 已修改文件
- `eval_code_deepeyes/utils.py`
- `eval_code_deepeyes/SpecEyes.py`
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`
- `scripts/run_score_routing_stage18.sh`（新增）

### 12.2 已新增 / 已接入的 score 与路由能力
在 `eval_code_deepeyes/utils.py` 中：
- 新增 `build_score_profile(...)`
- 保留 `answer_separability(...)` 兼容旧逻辑

当前 score profile 已支持：
- `confidence_score`
- `tail_score`
- `lowest_group_score`
- `bottom10_group_score`

在 `eval_code_deepeyes/SpecEyes.py` 中：
- 新增 CLI 参数：
  - `--trigger_metric`
  - `--accept_metric`
  - `--score_group_size`
  - `--score_tail_length`
- `should_trigger_memory(...)` 已改为读取 score profile
- `_run_small_model_pass(...)` 已返回：
  - `confidence_score`
  - `score_profile`
- small 分支 accept/fallback 已改为使用 `accept_metric`
- rerun 前后的 profile 已开始记录到结果中

### 12.3 当前 JSONL 已新增字段
`eval_code_deepeyes/SpecEyes.py` 当前已写出：
- `trigger_metric`
- `accept_metric`
- `score_profile_raw`
- `score_profile_memory`
- `tail_score_raw`
- `tail_score_memory`
- `lowest_group_score_raw`
- `lowest_group_score_memory`
- `bottom10_group_score_raw`
- `bottom10_group_score_memory`
- `tail_gain`
- `bottom10_gain`

旧字段仍保留：
- `confidence_score`
- `base_confidence_score`
- `memory_triggered`
- `memory_accept_decision`

### 12.4 当前 analysis 脚本已完成的适配
`scripts/replay_memory_threshold.py` 已支持：
- `--metric confidence_score|tail_score|lowest_group_score|bottom10_group_score`
- 缺失新字段时回退到 `confidence_score`

`scripts/compare_memory_runs.py` 已支持导出：
- `candidate_trigger_metric`
- `candidate_accept_metric`
- `candidate_tail_score_raw`
- `candidate_tail_score_memory`
- `candidate_bottom10_group_score_raw`
- `candidate_bottom10_group_score_memory`
- `candidate_tail_gain`
- `candidate_bottom10_gain`

### 12.5 当前默认实验脚本
新增：
- `scripts/run_score_routing_stage18.sh`

默认实验主线：
- `trigger_metric=bottom10_group_score`
- `accept_metric=tail_score`
- 跑两个任务：
  - `direct_attributes`
  - `relative_position`

直接运行：
```bash
bash scripts/run_score_routing_stage18.sh
```

如需覆盖参数，可直接用环境变量，例如：
```bash
TRIGGER_METRIC=bottom10_group_score \
ACCEPT_METRIC=tail_score \
MEMORY_TRIGGER_THRESHOLD=0.97 \
MEMORY_SCORE_THRESHOLD=0.98 \
LOGIC_TOP_K=1 \
RUN_JUDGES=1 \
bash scripts/run_score_routing_stage18.sh
```

### 12.6 当前还未完成 / 下一位 coding agent 应继续的部分
1. 跑一次真实 stage18 eval，确认新字段都已写进 JSONL
2. 检查 `relative_position` 是否减少高置信错答停留在 small 的情况
3. 检查 `direct_attributes` 是否恢复或保住 `92.17` 的收益
4. 若 `tail_score + bottom10_group_score` 仍不足，再继续补：
   - `answer_span_score`
   - `answer_gain`
5. 根据 stage18 实验结果，再决定是否把 task-level metric override 加进 policy

### 12.7 当前已经做过的本地验证
已通过语法检查：
- `python3 -m py_compile eval_code_deepeyes/SpecEyes.py eval_code_deepeyes/utils.py`
- `python3 -m py_compile scripts/replay_memory_threshold.py`
- `python3 -m py_compile scripts/compare_memory_runs.py`
- `bash -n scripts/run_score_routing_stage18.sh`

## 13. 一段可以直接给下一个 coding agent 的 prompt

你在：
- `/Users/bytedance/Desktop/repos/specmem_fresh`

请接手当前 SpecEyes 的 memory 路由改造工作。

当前代码已经完成：
- task-aware memory policy
- two-stage gating
- richer JSONL logging
- replay / compare 对 richer schema 的兼容

当前实验结论：
- memory 在 `direct_attributes` 上曾经有正收益（90.43 -> 92.17）
- 但 `stage17` 的新路由没有保住这部分收益
- `relative_position` 仍然失败，主因不是 retrieval，而是当前 confidence / separability score 会把大量错误 left/right 判断打成高分，导致不触发 memory，也不 fallback

你的任务不是 redesign retrieval，而是：
- 重做 routing score
- 把当前单个全局 `confidence_score` 扩展成 score profile
- 至少新增：
  - `tail_score`
  - `lowest_group_score`
  - `bottom10_group_score`
- 如果实现成本可控，再加：
  - `answer_span_score`
- 默认试：
  - `trigger_metric = bottom10_group_score`
  - `accept_metric = tail_score`
- 同时新增 raw vs memory 的 delta logging：
  - `tail_gain`
  - `bottom10_gain`
  - `answer_gain`（若实现了 answer score）

重点修改文件：
- `eval_code_deepeyes/SpecEyes.py`
- `scripts/replay_memory_threshold.py`
- `scripts/compare_memory_runs.py`

约束：
- 不要大改 retrieval
- 不要扩到 visual/dual memory 主线
- 不要做大范围无关重构

最后请报告：
- 改了哪些文件
- 新增了哪些 score 和日志字段
- 跑了哪些验证
- 是否缓解了 relpos 的高置信错答问题
- 是否保住或恢复了 direct_attributes 的收益
