#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAGE18_SCRIPT="${ROOT_DIR}/scripts/run_score_routing_stage18.sh"
LOG_DIR="${ROOT_DIR}/logs/stage18_metric_sweep"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-SpecEyes_vstar_stage18_metric_sweep}"
GPU_GROUP_A="${GPU_GROUP_A:-0,1}"
GPU_GROUP_B="${GPU_GROUP_B:-3,4}"

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

combos=()
for trigger_metric in "${TRIGGER_METRICS[@]}"; do
  for accept_metric in "${ACCEPT_METRICS[@]}"; do
    combos+=("${trigger_metric}|${accept_metric}")
  done
done

launch_combo() {
  local combo="$1"
  local gpu_group="$2"
  local trigger_metric="${combo%%|*}"
  local accept_metric="${combo##*|}"
  local combo_tag="trig=${trigger_metric}_accept=${accept_metric}"
  local output_tag="${OUTPUT_PREFIX}_${combo_tag}"
  local log_path="${LOG_DIR}/${combo_tag}.log"

  echo "[$(date '+%F %T')] Launch ${combo_tag} on GPUs ${gpu_group}"
  (
    cd "${ROOT_DIR}"
    TRIGGER_METRIC="${trigger_metric}" \
    ACCEPT_METRIC="${accept_metric}" \
    COMMON_OUTPUT_TAG="${output_tag}" \
    GPU_DEVICES="${gpu_group}" \
    bash "${STAGE18_SCRIPT}"
  ) > "${log_path}" 2>&1
}

idx=0
while [[ "${idx}" -lt "${#combos[@]}" ]]; do
  combo_a="${combos[$idx]}"
  combo_b=""
  if [[ $((idx + 1)) -lt "${#combos[@]}" ]]; then
    combo_b="${combos[$((idx + 1))]}"
  fi

  launch_combo "${combo_a}" "${GPU_GROUP_A}" &
  pid_a=$!

  if [[ -n "${combo_b}" ]]; then
    launch_combo "${combo_b}" "${GPU_GROUP_B}" &
    pid_b=$!
  else
    pid_b=""
  fi

  wait "${pid_a}"
  if [[ -n "${pid_b}" ]]; then
    wait "${pid_b}"
  fi

  idx=$((idx + 2))
done

echo "[$(date '+%F %T')] Stage18 metric sweep finished."
