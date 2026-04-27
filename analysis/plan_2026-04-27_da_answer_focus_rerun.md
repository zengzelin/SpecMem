# Direct Attributes Answer-Focused Rerun 实验计划（2026-04-27）

## 1. 为什么要做这轮实验

基于：
- `analysis/report_2026-04-27_stage18_rerun_audit.md`
- `analysis/research_plan_2026-04-27_post_stage18.md`

当前最关键的结论是：
- `direct_attributes` 的 memory rerun 里存在真实 corrected 样本；
- 但当前 accept/fallback 机制既保不住这些 corrected，也放不过滤掉大量无效 rerun；
- 继续扩大 score sweep 已经不合理。

因此这轮实验不再主攻新 metric，而是改 rerun 本身的输出形态：

> 让 rerun 更短、更答案中心化，减少“长回答改写但不更准确”的情况。

---

## 2. 这次代码改了什么

### 2.1 新增 prompt style
文件：
- `memory_aug/prompting.py`
- `eval_code_deepeyes/SpecEyes.py`

新增：
- `memory_prompt_style=answer_focus`

这个 style 的目标是：
- 保留 memory hints
- 但把 rerun 指令改成只关注最终答案
- 尽量减少长解释
- 对多选题强调“只输出最终选项和对应答案文本”

---

## 3. 这轮实验要验证什么

重点不是平均 acc 是否立刻大涨，而是先回答下面几个问题：

1. `direct_attributes` 上，answer-focused rerun 是否会减少：
   - accepted_after_trigger 里的错误样本
   - changed_but_still_wrong

2. 它是否会提高：
   - corrected_count
   - accepted_after_trigger 中真正 corrected 的占比

3. 和上一轮 `confidence -> tail` 相比，是否能减少“几乎全收”的错误 accept 行为

---

## 4. 建议运行方式

### 4.1 直接运行单组 direct_attributes
```bash
python3 eval_code_deepeyes/SpecEyes.py \
  --large_model_path ChenShawn/DeepEyes-7B \
  --small_model_path Qwen/Qwen3-VL-2B-Instruct \
  --benchmark vstar \
  --test_type direct_attributes \
  --output_path eval_results_deepeyes/SpecEyes_vstar_stage19_da_answer_focus \
  --batch_size 6 \
  --K 64 \
  --score_threshold 0.98 \
  --memory_enable \
  --memory_mode logic_only \
  --memory_prompt_mode small_only \
  --memory_prompt_style answer_focus \
  --logic_top_k 1 \
  --mode min \
  --memory_score_threshold 0.98 \
  --memory_trigger_threshold 0.97 \
  --trigger_metric confidence_score \
  --accept_metric confidence_score
```

### 4.2 如果要跑一组“更激进 accept”的对照
```bash
python3 eval_code_deepeyes/SpecEyes.py \
  --large_model_path ChenShawn/DeepEyes-7B \
  --small_model_path Qwen/Qwen3-VL-2B-Instruct \
  --benchmark vstar \
  --test_type direct_attributes \
  --output_path eval_results_deepeyes/SpecEyes_vstar_stage19_da_answer_focus_tail_accept \
  --batch_size 6 \
  --K 64 \
  --score_threshold 0.98 \
  --memory_enable \
  --memory_mode logic_only \
  --memory_prompt_mode small_only \
  --memory_prompt_style answer_focus \
  --logic_top_k 1 \
  --mode min \
  --memory_score_threshold 0.98 \
  --memory_trigger_threshold 0.97 \
  --trigger_metric confidence_score \
  --accept_metric tail_score
```

---

## 5. 跑完后先看什么

### 5.1 常规结果
看：
- `all_acc`
- `small_ratio`
- `small_cnt`
- `large_cnt`

### 5.2 更关键的是 rerun audit
跑完后，必须继续用：
- `scripts/audit_memory_rerun_effects.py`

示例：
```bash
python3 scripts/audit_memory_rerun_effects.py \
  --input_jsonl "<stage19 direct_attributes jsonl 路径>" \
  --output_dir analysis/audit_stage19_da_answer_focus \
  --task direct_attributes
```

重点看：
- `triggered_count`
- `accepted_after_trigger`
- `corrected_count`
- `harmed_count`
- `changed_but_still_wrong_count`
- `accepted_after_trigger` 里真正 corrected 的比例

---

## 6. 如何判断这轮实验是否有效

### 成功的最低标准
不是必须立刻超过 `92.17`，而是先满足下面任意两条：

1. `accepted_after_trigger` 不再像 `conf->tail` 那样几乎全收
2. `corrected_count` 高于 stage18 的 `conf->tail`
3. `harmed_count` 低于 stage18 的 `conf->tail`
4. `changed_but_still_wrong_count` 明显下降
5. 最终 acc 不再像 `76.52` 那样明显崩掉

### 如果无效，意味着什么
如果 answer-focused rerun 仍然：
- corrected 很少
- harmed 不降
- accepted wrong 仍多

那下一步应优先考虑：
- rerun 只输出 option token / yes-no token
- 或直接改成 task-specific policy，而不是继续细调 accept metric

---

## 7. 下一位 coding agent 需要知道的事

这轮实验的目标不是继续扫分数，而是验证：

> **把 rerun 变得更短、更答案中心化，是否能让 memory correction 更干净，从而让 accept/fallback 有机会成立。**

如果这轮仍失败，就说明问题更可能在：
- rerun 质量本身
- 或任务层面的策略分裂

而不是 score 选择。
