#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-ChenShawn/DeepEyes-7B}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
BENCHMARK="${BENCHMARK:-vstar}"
TEST_TYPE="${TEST_TYPE:-direct_attributes}"
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
COMMON_OUTPUT_TAG="${COMMON_OUTPUT_TAG:-SpecEyes_vstar_stage19_da_answer_focus}"
RUN_JUDGES="${RUN_JUDGES:-0}"
RUN_AUDIT="${RUN_AUDIT:-0}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$ROOT_DIR/analysis/audit_stage19_da_answer_focus}"

OUTPUT_PATH="$OUTPUT_ROOT/$COMMON_OUTPUT_TAG"
mkdir -p "$OUTPUT_PATH"

"$PYTHON_BIN" "$ROOT_DIR/eval_code_deepeyes/SpecEyes.py" \
  --large_model_path "$LARGE_MODEL_PATH" \
  --small_model_path "$SMALL_MODEL_PATH" \
  --benchmark "$BENCHMARK" \
  --test_type "$TEST_TYPE" \
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
  --score_tail_length "$SCORE_TAIL_LENGTH"

if [[ "$RUN_JUDGES" == "1" ]]; then
  bash "$ROOT_DIR/scripts/run_judges.sh" "$OUTPUT_PATH"
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
