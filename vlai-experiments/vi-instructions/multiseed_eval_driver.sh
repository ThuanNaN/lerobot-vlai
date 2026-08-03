#!/usr/bin/env bash
# Replacement eval driver for the seed-2000/3000 full-FT multiseed run.
#
# Why this exists: outputs/ft_multiseed_pipeline/driver.sh tested completion with
# `ckpt_ok "${train_dir}/checkpoints/last/pretrained_model"`, but `checkpoints/last`
# is created at the FIRST save (SAVE_FREQ=10000), not the last. Every job was declared
# done at ~step 10000, so all 6 were launched onto 3 GPUs at once and an eval ran
# against a 20k checkpoint labelled as final. That driver was killed on 2026-08-03;
# its 6 training jobs were left running and are unharmed.
#
# This driver fixes the detection with two independent conditions, both required:
#   1. checkpoints/050000/pretrained_model/model.safetensors exists (explicit step
#      directory -- `last` is never consulted);
#   2. no lerobot-train process for that job's repo_id is alive.
#
# Evals run one at a time on whichever GPU has room, so they never contend with the
# training jobs that are still finishing.
#
# Usage (detached, survives SSH loss):
#   setsid nohup ./vlai-experiments/vi-instructions/multiseed_eval_driver.sh \
#     > outputs/ft_multiseed_pipeline/eval_driver.stdout 2>&1 &
set -uo pipefail

REPO="/home/thuandn/Repository/lerobot"
OUT="${REPO}/outputs/ft_multiseed_pipeline"
LOG="${OUT}/eval_driver.log"
STEPS=50000
NEED_MIB=9000
DEADLINE=$(( $(date +%s) + 30*3600 ))

mkdir -p "${OUT}"
cd "${REPO}" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# name|repo_id_suffix|train_dir|eval_dir
JOBS=(
  "smolvla-ft-seed2000|ft_smolvla_seed2000|train_vi_ft_smolvla_seed2000|eval_vi_ft_smolvla_seed2000"
  "smolvla-ft-seed3000|ft_smolvla_seed3000|train_vi_ft_smolvla_seed3000|eval_vi_ft_smolvla_seed3000"
  "smolvla-vi-ft-seed2000|ft_vi_seed2000|train_vi_ft_vi_seed2000|eval_vi_ft_vi_seed2000"
  "smolvla-vi-ft-seed3000|ft_vi_seed3000|train_vi_ft_vi_seed3000|eval_vi_ft_vi_seed3000"
  "vocab-only-ft-seed2000|ft_vocab_only_seed2000|train_vi_vocab_only_ft_seed2000|eval_vi_vocab_only_ft_seed2000"
  "vocab-only-ft-seed3000|ft_vocab_only_seed3000|train_vi_vocab_only_ft_seed3000|eval_vi_vocab_only_ft_seed3000"
)

final_ckpt() { echo "${REPO}/outputs/$1/checkpoints/$(printf '%06d' "${STEPS}")/pretrained_model"; }

training_alive() {  # $1 = repo_id suffix
  pgrep -af "lerobot-train" 2>/dev/null | grep -q "libero-vi-$1"
}

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=', ' read -r idx used total; do
    if (( total - used > NEED_MIB )); then echo "${idx}"; return 0; fi
  done
}

log "=== multiseed eval driver start (${#JOBS[@]} jobs, final step ${STEPS}) ==="

declare -A DONE
completed=0

while (( completed < ${#JOBS[@]} )); do
  if (( $(date +%s) > DEADLINE )); then log "ERROR: deadline exceeded, giving up"; exit 1; fi

  for entry in "${JOBS[@]}"; do
    IFS='|' read -r name suffix train_dir eval_dir <<< "${entry}"
    [[ -n "${DONE[$name]:-}" ]] && continue

    ckpt="$(final_ckpt "${train_dir}")"
    [[ -f "${ckpt}/model.safetensors" ]] || continue
    if training_alive "${suffix}"; then
      log "${name}: final checkpoint present but training still alive -- waiting"
      continue
    fi

    # Gate: the stale-tokenizer bug that has twice produced a confident, silent 0.0%.
    if ! uv run python -c "
import sys
sys.path.insert(0, 'vlai-experiments/vi-instructions')
from validate_backbone import assert_policy_tokenizer_matches
assert_policy_tokenizer_matches('${ckpt}')
" >>"${LOG}" 2>&1; then
      log "FAILED ${name}: tokenizer_name mismatch in policy_preprocessor.json -- skipping"
      DONE[$name]=skipped
      completed=$((completed + 1))
      continue
    fi

    gpu="$(pick_gpu | head -1)"
    if [[ -z "${gpu}" ]]; then log "${name}: ready but no GPU with ${NEED_MIB}MiB free"; continue; fi

    target="${REPO}/outputs/${eval_dir}"
    if [[ -d "${target}" ]]; then
      stale="${target}.stale-$(date +%Y%m%d%H%M%S)"
      log "${name}: moving pre-existing ${eval_dir} aside -> $(basename "${stale}")"
      mv "${target}" "${stale}"
    fi

    log "eval ${name} on GPU ${gpu} <- ${ckpt}"
    env CUDA_VISIBLE_DEVICES="${gpu}" ./run_eval_vi.sh "${ckpt}" "${target}" \
      > "${OUT}/eval_${suffix}.log" 2>&1
    if [[ -f "${target}/eval_info.json" ]]; then
      log "DONE ${name}"
      DONE[$name]=ok
    else
      log "ERROR: eval ${name} produced no eval_info.json (see ${OUT}/eval_${suffix}.log)"
      DONE[$name]=failed
    fi
    completed=$((completed + 1))
  done

  (( completed < ${#JOBS[@]} )) && sleep 300
done

ARGS=()
for entry in "${JOBS[@]}"; do
  IFS='|' read -r name suffix train_dir eval_dir <<< "${entry}"
  info="${REPO}/outputs/${eval_dir}/eval_info.json"
  [[ -f "${info}" ]] && ARGS+=("--run" "${name}=${info}")
done

if (( ${#ARGS[@]} >= 2 )); then
  log "writing comparison table -> ${OUT}/comparison_seed2000_3000.md"
  uv run python vlai-experiments/vi-instructions/compare_eval.py "${ARGS[@]}" \
    > "${OUT}/comparison_seed2000_3000.md" 2>>"${LOG}"
fi

log "=== all ${#JOBS[@]} jobs resolved ==="
