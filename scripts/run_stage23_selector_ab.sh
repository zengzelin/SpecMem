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
OUTPUT_TAG="${OUTPUT_TAG:-SpecEyes_vstar_stage23_selector_ab}"
BATCH_SIZE="${BATCH_SIZE:-6}"
K="${K:-64}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.98}"
MEMORY_SCORE_THRESHOLD="${MEMORY_SCORE_THRESHOLD:-0.98}"
MEMORY_TRIGGER_THRESHOLD="${MEMORY_TRIGGER_THRESHOLD:-0.97}"
TRIGGER_METRIC="${TRIGGER_METRIC:-confidence_score}"
ACCEPT_METRIC="${ACCEPT_METRIC:-confidence_score}"
LOGIC_TOP_K="${LOGIC_TOP_K:-1}"
MEMORY_PROMPT_STYLE="${MEMORY_PROMPT_STYLE:-answer_focus}"
MEMORY_SELECTOR_MODEL="${MEMORY_SELECTOR_MODEL:-large}"
MEMORY_SELECTOR_BASE_ACTION="${MEMORY_SELECTOR_BASE_ACTION:-keep_base}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
JUDGE_API_URL="${JUDGE_API_URL:-http://127.0.0.1:23333/v1}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-qwen72b}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$ROOT_DIR/analysis/${OUTPUT_TAG}_audit}"

OUTPUT_PATH="$OUTPUT_ROOT/$OUTPUT_TAG"
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
  --memory_accept_policy selector_ab \
  --memory_selector_model "$MEMORY_SELECTOR_MODEL" \
  --memory_selector_base_action "$MEMORY_SELECTOR_BASE_ACTION"

if [[ "$RUN_JUDGE" == "1" ]]; then
  "$JUDGE_PYTHON_BIN" "$ROOT_DIR/judge_code/judge_vstar.py" \
    --api_url "$JUDGE_API_URL" \
    --eval_model_name "$JUDGE_MODEL_NAME" \
    --input_folder "$OUTPUT_PATH"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  INPUT_JSONL="$(find "$OUTPUT_PATH" -maxdepth 1 -type f -name "*.jsonl" | head -n 1)"
  mkdir -p "$AUDIT_OUTPUT_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/audit_memory_rerun_effects.py" \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$AUDIT_OUTPUT_DIR" \
    --task "$TEST_TYPE"
fi
