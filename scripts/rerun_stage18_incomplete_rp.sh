#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zelin/miniconda3/envs/speceyes/bin/python}"
JUDGE_PYTHON="${JUDGE_PYTHON:-/home/zelin/miniconda3/envs/judge_env/bin/python}"
API_URL="${API_URL:-http://127.0.0.1:23333/v1}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen72b}"
HF_HOME="${HF_HOME:-/mnt2/zelin_cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
VSTAR_PATH="${VSTAR_PATH:-/mnt/zelin/vstar}"
LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/mnt/zelin/DeepEyes-7B}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/mnt/public_models/Qwen/Qwen3-VL-2B-Instruct}"
MEMORY_DIR="${MEMORY_DIR:-$ROOT_DIR/memory_data/vstar}"
LOG_ROOT="${ROOT_DIR}/logs/stage18_rp_retry"
GPU_A="${GPU_A:-3}"
GPU_B="${GPU_B:-4}"

mkdir -p "${LOG_ROOT}"
: > "${LOG_ROOT}/driver.log"

run_one() {
  local trigger_metric="$1"
  local accept_metric="$2"
  local gpu_id="$3"
  local tag="SpecEyes_vstar_stage18_metric_sweep_seq_trig=${trigger_metric}_accept=${accept_metric}_rp"
  local output_dir="${ROOT_DIR}/eval_results_deepeyes/${tag}"
  local judge_dir="${ROOT_DIR}/judge_results_deepeyes/${tag}"
  local log_path="${LOG_ROOT}/trig=${trigger_metric}_accept=${accept_metric}_gpu${gpu_id}.log"

  rm -rf "${output_dir}" "${judge_dir}"

  echo "[$(date '+%F %T')] START trig=${trigger_metric} accept=${accept_metric} gpu=${gpu_id}" | tee -a "${LOG_ROOT}/driver.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  HF_HOME="${HF_HOME}" \
  HF_HUB_CACHE="${HF_HUB_CACHE}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" "${ROOT_DIR}/eval_code_deepeyes/SpecEyes.py" \
    --large_model_path "${LARGE_MODEL_PATH}" \
    --small_model_path "${SMALL_MODEL_PATH}" \
    --benchmark vstar \
    --test_type relative_position \
    --vstar_path "${VSTAR_PATH}" \
    --output_path "${output_dir}" \
    --batch_size 6 \
    --K 64 \
    --score_threshold 0.98 \
    --memory_enable \
    --memory_dir "${MEMORY_DIR}" \
    --memory_mode logic_only \
    --memory_prompt_mode small_only \
    --memory_prompt_style default \
    --logic_top_k 1 \
    --mode min \
    --memory_score_threshold 0.98 \
    --memory_trigger_threshold 0.97 \
    --trigger_metric "${trigger_metric}" \
    --accept_metric "${accept_metric}" \
    --score_group_size 4 \
    --score_tail_length 8 \
    > "${log_path}" 2>&1

  echo "[$(date '+%F %T')] JUDGE trig=${trigger_metric} accept=${accept_metric}" | tee -a "${LOG_ROOT}/driver.log"
  "${JUDGE_PYTHON}" "${ROOT_DIR}/judge_code/judge_vstar.py" \
    --api_url "${API_URL}" \
    --eval_model_name "${EVAL_MODEL_NAME}" \
    --input_folder "${output_dir}" >> "${log_path}" 2>&1

  local raw_file
  local line_count=0
  raw_file="$(find "${output_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name 'latency_summary.json' | head -n 1)"
  if [[ -n "${raw_file}" ]]; then
    line_count="$(wc -l < "${raw_file}")"
  fi
  echo "[$(date '+%F %T')] DONE trig=${trigger_metric} accept=${accept_metric} lines=${line_count}" | tee -a "${LOG_ROOT}/driver.log"
}

COMBOS=(
  "tail_score confidence_score"
  "tail_score lowest_group_score"
  "tail_score bottom10_group_score"
  "lowest_group_score confidence_score"
  "lowest_group_score tail_score"
  "lowest_group_score lowest_group_score"
  "lowest_group_score bottom10_group_score"
  "bottom10_group_score confidence_score"
)

idx=0
while [[ "${idx}" -lt "${#COMBOS[@]}" ]]; do
  read -r trig_a accept_a <<< "${COMBOS[$idx]}"
  run_one "${trig_a}" "${accept_a}" "${GPU_A}" &
  pid_a=$!

  pid_b=""
  if [[ $((idx + 1)) -lt "${#COMBOS[@]}" ]]; then
    read -r trig_b accept_b <<< "${COMBOS[$((idx + 1))]}"
    run_one "${trig_b}" "${accept_b}" "${GPU_B}" &
    pid_b=$!
  fi

  wait "${pid_a}"
  if [[ -n "${pid_b}" ]]; then
    wait "${pid_b}"
  fi
  idx=$((idx + 2))
done

echo "[$(date '+%F %T')] stage18 RP retry finished" | tee -a "${LOG_ROOT}/driver.log"
