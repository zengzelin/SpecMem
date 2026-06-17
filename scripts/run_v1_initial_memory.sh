#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zelin/miniconda3/envs/speceyes/bin/python}"

LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/mnt/zelin/DeepEyes-7B}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/mnt/public_models/Qwen/Qwen3-VL-2B-Instruct}"
BENCHMARK="${BENCHMARK:-vstar}"
TEST_TYPE="${TEST_TYPE:-all}"
RUN_MODE="${RUN_MODE:-memory}"   # baseline or memory
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/eval_results_deepeyes}"
OUTPUT_TAG="${OUTPUT_TAG:-SpecEyes_${BENCHMARK}_v1_${RUN_MODE}}"
BATCH_SIZE="${BATCH_SIZE:-6}"
K="${K:-64}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.98}"
LOGIC_TOP_K="${LOGIC_TOP_K:-1}"
MEMORY_DIR="${MEMORY_DIR:-$ROOT_DIR/memory_data/$BENCHMARK}"
HRBENCH_PATH="${HRBENCH_PATH:-/mnt/zelin/hr_bench}"
POPE_PATH="${POPE_PATH:-/mnt/zelin/pope}"
VSTAR_PATH="${VSTAR_PATH:-/mnt/zelin/vstar}"

OUTPUT_PATH="$OUTPUT_ROOT/$OUTPUT_TAG"
mkdir -p "$OUTPUT_PATH"

COMMON_ARGS=(
  --large_model_path "$LARGE_MODEL_PATH"
  --small_model_path "$SMALL_MODEL_PATH"
  --benchmark "$BENCHMARK"
  --test_type "$TEST_TYPE"
  --output_path "$OUTPUT_PATH"
  --batch_size "$BATCH_SIZE"
  --K "$K"
  --score_threshold "$SCORE_THRESHOLD"
  --mode min
  --vstar_path "$VSTAR_PATH"
  --hrbench_path "$HRBENCH_PATH"
  --pope_path "$POPE_PATH"
)

if [[ "$RUN_MODE" == "baseline" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/eval_code_deepeyes/SpecEyes.py" \
    "${COMMON_ARGS[@]}" \
    --baseline
else
  "$PYTHON_BIN" "$ROOT_DIR/eval_code_deepeyes/SpecEyes.py" \
    "${COMMON_ARGS[@]}" \
    --memory_enable \
    --memory_dir "$MEMORY_DIR" \
    --memory_mode logic_only \
    --memory_prompt_mode small_only \
    --memory_prompt_style default \
    --logic_top_k "$LOGIC_TOP_K"
fi
