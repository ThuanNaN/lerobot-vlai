#!/usr/bin/env bash
# Why does HuggingFaceVLA/smolvla_libero score 66.5 on LIBERO when the SmolVLA paper
# (arXiv:2506.01844) reports 87.3 for the 0.45B model?
#
# Prime suspect: that checkpoint's config.json sets `n_action_steps: 1`, while LeRobot's
# SmolVLA default is 50 (= chunk_size). At 1, the policy predicts a 50-step action chunk
# and executes only its first action before re-planning -- 50x the forward passes, and a
# different control regime from the chunked execution SmolVLA was trained for.
#
# This sweeps n_action_steps on ONE suite (libero_spatial, 10 tasks x 10 episodes = 100
# episodes) with native English instructions, and compares against the paper's 90 for
# that suite. Cheap enough to answer the question before committing to a full 400-episode
# rerun.
#
# Stakes: if a corrected n_action_steps recovers ~90, then the project's baseline row
# ("EN 66.5 -> VI 0.0") was measured in a degraded regime. The 0.0 would still be 0.0,
# but the VI leg must be re-measured too before that number is used as evidence for the
# language cliff.
#
# Waits for ladder_driver.sh to exit so it never contends with Phase 1a -- an NCCL
# collective timeout already killed one stage-1 run under GPU contention on this host.
#
# Usage:
#   setsid nohup ./vlai-experiments/vi-instructions/probe_n_action_steps.sh \
#     > outputs/probe_nas/driver.stdout 2>&1 < /dev/null &
set -uo pipefail

REPO="/home/thuandn/Repository/lerobot"
OUT="${REPO}/outputs/probe_nas"
LOG="${OUT}/probe.log"
CKPT="HuggingFaceVLA/smolvla_libero"
SUITE="libero_spatial"
PAPER_SCORE=90
STEPS_LIST=(1 10 25 50)
NEED_MIB=9000

mkdir -p "${OUT}"
cd "${REPO}" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "=== n_action_steps probe: ${CKPT} on ${SUITE} (paper reports ${PAPER_SCORE}) ==="

# --- wait for Phase 1a to finish ---------------------------------------------------
while pgrep -f "ladder_driver.sh" > /dev/null 2>&1; do
  log "waiting: ladder_driver.sh still running"
  sleep 600
done
log "ladder driver finished"

pick_gpu() {
  local idx used total
  while IFS=', ' read -r idx used total; do
    if (( total - used > NEED_MIB )); then echo "${idx}"; return 0; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
  return 1
}

gpu=""
while [[ -z "${gpu}" ]]; do
  gpu="$(pick_gpu | head -1)"
  [[ -z "${gpu}" ]] && { log "waiting for a free GPU"; sleep 300; }
done
log "using GPU ${gpu}"

for n in "${STEPS_LIST[@]}"; do
  dir="${OUT}/nas_${n}"
  if [[ -f "${dir}/eval_info.json" ]]; then log "n_action_steps=${n}: already done -- skipping"; continue; fi
  [[ -d "${dir}" ]] && mv "${dir}" "${dir}.stale-$(date +%Y%m%d%H%M%S)"

  log "eval n_action_steps=${n} ..."
  env CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_GL="${MUJOCO_GL:-egl}" \
    uv run lerobot-eval \
      --policy.path="${CKPT}" \
      --policy.n_action_steps="${n}" \
      --env.type=libero \
      --env.task="${SUITE}" \
      --eval.batch_size=1 \
      --eval.n_episodes=10 \
      --env.max_parallel_tasks=1 \
      --output_dir="${dir}" \
    > "${OUT}/eval_nas_${n}.log" 2>&1

  if [[ -f "${dir}/eval_info.json" ]]; then
    sc=$(uv run python -c "
import json; print(json.load(open('${dir}/eval_info.json'))['overall']['pc_success'])")
    log "n_action_steps=${n}: ${sc} (paper ${PAPER_SCORE} on ${SUITE})"
  else
    log "n_action_steps=${n}: FAILED, see ${OUT}/eval_nas_${n}.log"
  fi
done

log "=== summary: ${SUITE}, native English instructions ==="
log "  paper (SmolVLA 0.45B): ${PAPER_SCORE}"
for n in "${STEPS_LIST[@]}"; do
  f="${OUT}/nas_${n}/eval_info.json"
  v=$(uv run python -c "
import json,sys
try: print(json.load(open('${f}'))['overall']['pc_success'])
except Exception: print('n/a')" 2>/dev/null)
  log "  n_action_steps=${n}: ${v}"
done
