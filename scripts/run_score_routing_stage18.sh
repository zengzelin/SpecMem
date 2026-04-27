#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zelin/miniconda3/envs/speceyes/bin/python}"

LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/mnt/zelin/DeepEyes-7B}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/mnt/public_models/Qwen/Qwen3-VL-2B-Instruct}"
BENCHMARK="${BENCHMARK:-vstar}"
VSTAR_PATH="${VSTAR_PATH:-/mnt/zelin/vstar}"
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
GPU_DEVICES="${GPU_DEVICES:-0,1,3,4}"
HF_HOME="${HF_HOME:-/mnt2/zelin_cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [[ "${#GPU_LIST[@]}" -lt 2 ]]; then
  echo "GPU_DEVICES must provide at least 2 GPU ids for the two stage18 tasks." >&2
  exit 1
fi

select_target_gpus() {
  local allowed_csv
  local selected
  allowed_csv="$(IFS=','; echo "${GPU_LIST[*]}")"
  selected="$(
    ALLOWED_GPU_CSV="$allowed_csv" python3 - <<'PY'
import os
import subprocess

allowed = {int(x) for x in os.environ["ALLOWED_GPU_CSV"].split(",") if x}
rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
gpus = []
for line in rows.strip().splitlines():
    idx_str, free_str = [part.strip() for part in line.split(",")]
    idx = int(idx_str)
    if idx in allowed:
        gpus.append((int(free_str), idx))
gpus.sort(reverse=True)
print(",".join(str(idx) for _, idx in gpus[:2]))
PY
  )"
  if [[ -z "$selected" ]]; then
    echo "Failed to select GPUs from: $GPU_DEVICES" >&2
    exit 1
  fi
  IFS=',' read -r -a SELECTED_GPUS <<< "$selected"
  if [[ "${#SELECTED_GPUS[@]}" -lt 2 ]]; then
    echo "Need at least 2 selected GPUs, got: $selected" >&2
    exit 1
  fi
}

run_eval() {
  local test_type="$1"
  local output_path="$OUTPUT_ROOT/$2"
  local gpu_id="$3"
  local log_path="$ROOT_DIR/logs/$2_gpu${gpu_id}.log"

  mkdir -p "$output_path"
  mkdir -p "$ROOT_DIR/logs"

  echo "[$(date '+%F %T')] Launching $test_type on GPU $gpu_id -> $output_path"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" "$ROOT_DIR/eval_code_deepeyes/SpecEyes.py" \
      --large_model_path "$LARGE_MODEL_PATH" \
      --small_model_path "$SMALL_MODEL_PATH" \
      --benchmark "$BENCHMARK" \
      --test_type "$test_type" \
      --vstar_path "$VSTAR_PATH" \
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
      --score_tail_length "$SCORE_TAIL_LENGTH" \
      > "$log_path" 2>&1

  if [[ "$RUN_JUDGES" == "1" ]]; then
    bash "$ROOT_DIR/scripts/run_judges.sh" "$output_path" "$JUDGE_INPUT_ROOT"
  fi
}

pids=()
select_target_gpus
echo "[$(date '+%F %T')] Selected GPUs for stage18: ${SELECTED_GPUS[*]} (from $GPU_DEVICES)"
run_eval "direct_attributes" "${COMMON_OUTPUT_TAG}_da" "${SELECTED_GPUS[0]}" &
pids+=("$!")
run_eval "relative_position" "${COMMON_OUTPUT_TAG}_rp" "${SELECTED_GPUS[1]}" &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid"
done
