#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zelin/miniconda3/envs/speceyes/bin/python}"
JUDGE_PYTHON_BIN="${JUDGE_PYTHON_BIN:-/home/zelin/miniconda3/envs/judge_env/bin/python}"

LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/mnt/zelin/DeepEyes-7B}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/mnt/public_models/Qwen/Qwen3-VL-2B-Instruct}"
BENCHMARK="${BENCHMARK:-vstar}"
TEST_TYPE="${TEST_TYPE:-direct_attributes}"
VSTAR_PATH="${VSTAR_PATH:-/mnt/zelin/vstar}"
MEMORY_DIR="${MEMORY_DIR:-$ROOT_DIR/memory_data/$BENCHMARK}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/eval_results_deepeyes}"
BATCH_SIZE="${BATCH_SIZE:-6}"
K="${K:-64}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.98}"
MEMORY_SCORE_THRESHOLD="${MEMORY_SCORE_THRESHOLD:-0.98}"
MEMORY_TRIGGER_THRESHOLD="${MEMORY_TRIGGER_THRESHOLD:-0.97}"
TRIGGER_METRIC="${TRIGGER_METRIC:-confidence_score}"
ACCEPT_METRIC="${ACCEPT_METRIC:-confidence_score}"
SCORE_GROUP_SIZE="${SCORE_GROUP_SIZE:-4}"
SCORE_TAIL_LENGTH="${SCORE_TAIL_LENGTH:-8}"
LOGIC_TOP_K="${LOGIC_TOP_K:-1}"
MEMORY_PROMPT_STYLE="${MEMORY_PROMPT_STYLE:-answer_focus}"
MEMORY_RETRIEVAL_MIN_SCORE="${MEMORY_RETRIEVAL_MIN_SCORE:-0.45}"
MEMORY_ACCEPT_POLICY="${MEMORY_ACCEPT_POLICY:-triggered_multisignal}"
MEMORY_ACCEPT_MIN_CONFIDENCE="${MEMORY_ACCEPT_MIN_CONFIDENCE:-0.9}"
MEMORY_ACCEPT_MIN_DELTA_CONF="${MEMORY_ACCEPT_MIN_DELTA_CONF:--0.02}"
MEMORY_ACCEPT_MIN_RETRIEVAL_SCORE="${MEMORY_ACCEPT_MIN_RETRIEVAL_SCORE:-0.0}"
MEMORY_ACCEPT_ANSWER_CHANGE="${MEMORY_ACCEPT_ANSWER_CHANGE:-changed}"
COMMON_OUTPUT_TAG="${COMMON_OUTPUT_TAG:-SpecEyes_vstar_stage21_da_retrieval_gate}"
RUN_JUDGES="${RUN_JUDGES:-0}"
RUN_AUDIT="${RUN_AUDIT:-0}"
JUDGE_API_URL="${JUDGE_API_URL:-http://127.0.0.1:23333/v1}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-qwen72b}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$ROOT_DIR/analysis/audit_stage21_da_retrieval_gate}"

OUTPUT_PATH="$OUTPUT_ROOT/$COMMON_OUTPUT_TAG"
mkdir -p "$OUTPUT_PATH"

"$PYTHON_BIN" "$ROOT_DIR/eval_code_deepeyes/SpecEyes.py" \
  --large_model_path "$LARGE_MODEL_PATH" \
  --small_model_path "$SMALL_MODEL_PATH" \
  --benchmark "$BENCHMARK" \
  --test_type "$TEST_TYPE" \
  --vstar_path "$VSTAR_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size "$BATCH_SIZE" \
  --K "$K" \
  --score_threshold "$SCORE_THRESHOLD" \
  --memory_enable \
  --memory_dir "$MEMORY_DIR" \
  --memory_mode logic_only \
  --memory_prompt_mode small_only \
  --memory_prompt_style "$MEMORY_PROMPT_STYLE" \
  --logic_top_k "$LOGIC_TOP_K" \
  --mode min \
  --memory_score_threshold "$MEMORY_SCORE_THRESHOLD" \
  --memory_trigger_threshold "$MEMORY_TRIGGER_THRESHOLD" \
  --trigger_metric "$TRIGGER_METRIC" \
  --accept_metric "$ACCEPT_METRIC" \
  --score_group_size "$SCORE_GROUP_SIZE" \
  --score_tail_length "$SCORE_TAIL_LENGTH" \
  --memory_retrieval_min_score "$MEMORY_RETRIEVAL_MIN_SCORE" \
  --memory_accept_policy "$MEMORY_ACCEPT_POLICY" \
  --memory_accept_min_confidence "$MEMORY_ACCEPT_MIN_CONFIDENCE" \
  --memory_accept_min_delta_conf "$MEMORY_ACCEPT_MIN_DELTA_CONF" \
  --memory_accept_min_retrieval_score "$MEMORY_ACCEPT_MIN_RETRIEVAL_SCORE" \
  --memory_accept_answer_change "$MEMORY_ACCEPT_ANSWER_CHANGE"

if [[ "$RUN_JUDGES" == "1" ]]; then
  "$JUDGE_PYTHON_BIN" "$ROOT_DIR/judge_code/judge_vstar.py" \
    --api_url "$JUDGE_API_URL" \
    --eval_model_name "$JUDGE_MODEL_NAME" \
    --input_folder "$OUTPUT_PATH"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  INPUT_JSONL="$(find "$OUTPUT_PATH" -maxdepth 1 -type f -name "*.jsonl" | head -n 1)"
  if [[ -z "$INPUT_JSONL" ]]; then
    echo "No JSONL found under $OUTPUT_PATH for audit." >&2
    exit 1
  fi

  mkdir -p "$AUDIT_OUTPUT_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/audit_memory_rerun_effects.py" \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$AUDIT_OUTPUT_DIR" \
    --task "$TEST_TYPE"
fi
