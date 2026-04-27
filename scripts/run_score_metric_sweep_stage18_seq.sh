#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAGE18_SCRIPT="${ROOT_DIR}/scripts/run_score_routing_stage18.sh"
JUDGE_SCRIPT="${ROOT_DIR}/judge_code/judge_vstar.py"
JUDGE_PYTHON="${JUDGE_PYTHON:-/home/zelin/miniconda3/envs/judge_env/bin/python}"
GPU_DEVICES="${GPU_DEVICES:-3,4}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-SpecEyes_vstar_stage18_metric_sweep_seq}"
API_URL="${API_URL:-http://127.0.0.1:23333/v1}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen72b}"
RUN_JUDGE="${RUN_JUDGE:-1}"
LOG_DIR="${ROOT_DIR}/logs/stage18_metric_sweep_seq"

TRIGGER_METRICS=(
  confidence_score
  tail_score
  lowest_group_score
  bottom10_group_score
)
ACCEPT_METRICS=(
  confidence_score
  tail_score
  lowest_group_score
  bottom10_group_score
)

mkdir -p "${LOG_DIR}"

run_combo() {
  local trigger_metric="$1"
  local accept_metric="$2"
  local combo_tag="trig=${trigger_metric}_accept=${accept_metric}"
  local output_tag="${OUTPUT_PREFIX}_${combo_tag}"
  local eval_da_dir="${ROOT_DIR}/eval_results_deepeyes/${output_tag}_da"
  local eval_rp_dir="${ROOT_DIR}/eval_results_deepeyes/${output_tag}_rp"
  local judge_da_dir="${ROOT_DIR}/judge_results_deepeyes/${output_tag}_da"
  local judge_rp_dir="${ROOT_DIR}/judge_results_deepeyes/${output_tag}_rp"
  local log_path="${LOG_DIR}/${combo_tag}.log"

  rm -rf "${eval_da_dir}" "${eval_rp_dir}" "${judge_da_dir}" "${judge_rp_dir}"

  echo "[$(date '+%F %T')] Running ${combo_tag} on GPUs ${GPU_DEVICES}" | tee "${log_path}"
  (
    cd "${ROOT_DIR}"
    TRIGGER_METRIC="${trigger_metric}" \
    ACCEPT_METRIC="${accept_metric}" \
    COMMON_OUTPUT_TAG="${output_tag}" \
    GPU_DEVICES="${GPU_DEVICES}" \
    bash "${STAGE18_SCRIPT}"
  ) >> "${log_path}" 2>&1

  if [[ "${RUN_JUDGE}" == "1" ]]; then
    echo "[$(date '+%F %T')] Judging ${combo_tag} direct_attributes" | tee -a "${log_path}"
    "${JUDGE_PYTHON}" "${JUDGE_SCRIPT}" \
      --api_url "${API_URL}" \
      --eval_model_name "${EVAL_MODEL_NAME}" \
      --input_folder "${eval_da_dir}" >> "${log_path}" 2>&1

    echo "[$(date '+%F %T')] Judging ${combo_tag} relative_position" | tee -a "${log_path}"
    "${JUDGE_PYTHON}" "${JUDGE_SCRIPT}" \
      --api_url "${API_URL}" \
      --eval_model_name "${EVAL_MODEL_NAME}" \
      --input_folder "${eval_rp_dir}" >> "${log_path}" 2>&1
  fi
}

for trigger_metric in "${TRIGGER_METRICS[@]}"; do
  for accept_metric in "${ACCEPT_METRICS[@]}"; do
    run_combo "${trigger_metric}" "${accept_metric}"
  done
done

echo "[$(date '+%F %T')] Stage18 sequential metric sweep finished."
