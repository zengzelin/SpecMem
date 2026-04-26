#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-ChenShawn/DeepEyes-7B}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
BENCHMARK="${BENCHMARK:-vstar}"
MEMORY_DIR="${MEMORY_DIR:-$ROOT_DIR/memory_data/$BENCHMARK}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/eval_results_deepeyes}"
JUDGE_INPUT_ROOT="${JUDGE_INPUT_ROOT:-$ROOT_DIR/judge_results_deepeyes}"
BATCH_SIZE="${BATCH_SIZE:-6}"
K="${K:-64}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.98}"
MEMORY_SCORE_THRESHOLD="${MEMORY_SCORE_THRESHOLD:-0.98}"
MEMORY_TRIGGER_THRESHOLD="${MEMORY_TRIGGER_THRESHOLD:-0.97}"
TRIGGER_METRIC="${TRIGGER_METRIC:-bottom10_group_score}"
ACCEPT_METRIC="${ACCEPT_METRIC:-tail_score}"
SCORE_GROUP_SIZE="${SCORE_GROUP_SIZE:-4}"
SCORE_TAIL_LENGTH="${SCORE_TAIL_LENGTH:-8}"
LOGIC_TOP_K="${LOGIC_TOP_K:-1}"
RUN_JUDGES="${RUN_JUDGES:-0}"
COMMON_OUTPUT_TAG="${COMMON_OUTPUT_TAG:-SpecEyes_vstar_stage18_score_routing}"

run_eval() {
  local test_type="$1"
  local output_path="$OUTPUT_ROOT/$2"

  mkdir -p "$output_path"

  "$PYTHON_BIN" "$ROOT_DIR/eval_code_deepeyes/SpecEyes.py" \
    --large_model_path "$LARGE_MODEL_PATH" \
    --small_model_path "$SMALL_MODEL_PATH" \
    --benchmark "$BENCHMARK" \
    --test_type "$test_type" \
    --output_path "$output_path" \
    --batch_size "$BATCH_SIZE" \
    --K "$K" \
    --score_threshold "$SCORE_THRESHOLD" \
    --memory_enable \
    --memory_dir "$MEMORY_DIR" \
    --memory_mode logic_only \
    --memory_prompt_mode small_only \
    --memory_prompt_style default \
    --logic_top_k "$LOGIC_TOP_K" \
    --mode min \
    --memory_score_threshold "$MEMORY_SCORE_THRESHOLD" \
    --memory_trigger_threshold "$MEMORY_TRIGGER_THRESHOLD" \
    --trigger_metric "$TRIGGER_METRIC" \
    --accept_metric "$ACCEPT_METRIC" \
    --score_group_size "$SCORE_GROUP_SIZE" \
    --score_tail_length "$SCORE_TAIL_LENGTH"

  if [[ "$RUN_JUDGES" == "1" ]]; then
    bash "$ROOT_DIR/scripts/run_judges.sh" "$output_path" "$JUDGE_INPUT_ROOT"
  fi
}

run_eval "direct_attributes" "${COMMON_OUTPUT_TAG}_da"
run_eval "relative_position" "${COMMON_OUTPUT_TAG}_rp"
