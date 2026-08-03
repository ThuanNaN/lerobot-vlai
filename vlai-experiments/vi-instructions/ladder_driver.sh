#!/usr/bin/env bash
# Backbone language-cliff ladder driver: a GPU-count-agnostic job queue.
#
# Each job = train SmolVLA 50k full-FT on one backbone, then eval EN and VI.
#
# Three things it does that outputs/ft_multiseed_pipeline/driver.sh got wrong:
#   * completion requires the explicit checkpoints/050000 directory AND the training
#     process being gone -- `checkpoints/last` appears at the FIRST save and is never
#     consulted here;
#   * every backbone passes validate_backbone.py before any GPU time is committed;
#   * the trained checkpoint's policy_preprocessor.json is checked against its
#     config.json before eval, catching the stale-tokenizer bug that twice produced a
#     confident, silent 0.0%.
#
# Usage:
#   ./vlai-experiments/vi-instructions/ladder_driver.sh <phase> [seed]
#     phase: 1a | 1b | 2 | vi     seed: default 1000
#
# Launch detached so it survives SSH loss:
#   mkdir -p outputs/ladder_pipeline
#   setsid nohup ./vlai-experiments/vi-instructions/ladder_driver.sh 1a \
#     > outputs/ladder_pipeline/driver.stdout 2>&1 < /dev/null &
set -uo pipefail

REPO="/home/thuandn/Repository/lerobot"
PHASE="${1:?Usage: ladder_driver.sh <1a|1b|2|vi> [seed]}"
SEED="${2:-1000}"
OUT="${REPO}/outputs/ladder_pipeline"
LOG="${OUT}/driver_phase${PHASE}_seed${SEED}.log"
STEPS=50000
BATCH_SIZE=16
NEED_MIB=9000
# The pre-flight gate runs on CPU by default: it is cheap, and it must not contend
# for VRAM with training jobs that may still be finishing when this driver starts.
VALIDATE_DEVICE="${VALIDATE_DEVICE:-cpu}"
EN_DATASET="HuggingFaceVLA/libero"
VI_DATASET="VLAIResearchLab/lerobot_libero_vi"
DEADLINE=$(( $(date +%s) + 120*3600 ))

mkdir -p "${OUT}"
cd "${REPO}" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# point|backbone|expected_vocab
BACKBONES=(
  "stock|HuggingFaceTB/SmolVLM2-500M-Video-Instruct|49280"
  "d0|${REPO}/outputs/backbones/vi_dose_0|57344"
  "d10|${REPO}/outputs/backbones/vi_dose_10|57344"
  "d25|${REPO}/outputs/backbones/vi_dose_25|57344"
  "d50|${REPO}/outputs/backbones/vi_dose_50|57344"
  "d100|thuanan/SmolVLM2-500M-vi-stage1|57344"
  "s256m|HuggingFaceTB/SmolVLM2-256M-Video-Instruct|49280"
  "s2200m|HuggingFaceTB/SmolVLM2-2.2B-Instruct|49280"
)

backbone_for() {
  local want="$1" point path vocab
  for entry in "${BACKBONES[@]}"; do
    IFS='|' read -r point path vocab <<< "${entry}"
    if [[ "${point}" == "${want}" ]]; then echo "${path}|${vocab}"; return 0; fi
  done
  return 1
}

case "${PHASE}" in
  1a|1b) POINTS=(stock d0 d10 d25 d50 d100); LANG=en; DATASET="${EN_DATASET}" ;;
  2)     POINTS=(s256m s2200m);              LANG=en; DATASET="${EN_DATASET}" ;;
  vi)    POINTS=(d10 d25 d50);               LANG=vi; DATASET="${VI_DATASET}" ;;
  *)     echo "unknown phase: ${PHASE}" >&2; exit 1 ;;
esac

SUFFIX=""
[[ "${SEED}" != "1000" ]] && SUFFIX="_seed${SEED}"

tag_for() { echo "ladder_$1_${LANG}${SUFFIX}"; }
final_ckpt() {
  echo "${REPO}/outputs/train_$(tag_for "$1")/checkpoints/$(printf '%06d' "${STEPS}")/pretrained_model"
}
training_alive() { pgrep -af "lerobot-train" 2>/dev/null | grep -q "libero-vi-$(tag_for "$1")"; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=', ' read -r idx used total; do
    if (( total - used > NEED_MIB )); then echo "${idx}"; return 0; fi
  done
}

log "=== ladder driver: phase=${PHASE} seed=${SEED} points=${POINTS[*]} train-lang=${LANG} ==="

# --- Gate: validate every backbone before committing any GPU time ------------------
for point in "${POINTS[@]}"; do
  IFS='|' read -r path vocab <<< "$(backbone_for "${point}")"
  log "validating ${point} (${path}, vocab=${vocab})"
  if ! uv run python vlai-experiments/vi-instructions/validate_backbone.py \
        "${path}" --expected-vocab "${vocab}" --max-lines 200 \
        --device "${VALIDATE_DEVICE}" \
        --report "${OUT}/validate_${point}.json" >>"${LOG}" 2>&1; then
    log "FATAL: ${point} failed validation -- aborting before any training"
    exit 1
  fi
done
log "all ${#POINTS[@]} backbones validated"

# --- Queue ------------------------------------------------------------------------
declare -A LAUNCHED RESOLVED
next=0
resolved=0

while (( resolved < ${#POINTS[@]} )); do
  if (( $(date +%s) > DEADLINE )); then log "ERROR: deadline exceeded"; exit 1; fi

  # Launch queued jobs onto free GPUs.
  while (( next < ${#POINTS[@]} )); do
    gpu="$(pick_gpu | head -1)"
    [[ -z "${gpu}" ]] && break

    point="${POINTS[$next]}"
    IFS='|' read -r path vocab <<< "$(backbone_for "${point}")"
    tag="$(tag_for "${point}")"
    log "launch ${tag} on GPU ${gpu} (backbone=${path})"
    # Do NOT mkdir the output dir: lerobot-train's cfg.validate() requires it to not
    # already exist when resume=False, and fails with FileExistsError otherwise.
    setsid nohup env CUDA_VISIBLE_DEVICES="${gpu}" SKIP_SYNC=1 \
        VLM_MODEL="${path}" DATASET_REPO="${DATASET}" \
        RUN_TAG="${tag}" OUTPUT_DIR="${REPO}/outputs/train_${tag}/" SEED="${SEED}" \
        STEPS="${STEPS}" BATCH_SIZE="${BATCH_SIZE}" TRAIN_EXPERT_ONLY=false \
        SAVE_FREQ=10000 ENV_EVAL_FREQ=10000 LOG_FREQ=250 \
        ./run_vi.sh > "${OUT}/train_${tag}.log" 2>&1 < /dev/null &
    LAUNCHED[$point]=1
    next=$((next + 1))
    sleep 180   # let the job claim its VRAM before pick_gpu is polled again
  done

  sleep 300

  # Resolve finished jobs and run their evals.
  for point in "${POINTS[@]}"; do
    [[ -z "${LAUNCHED[$point]:-}" ]] && continue
    [[ -n "${RESOLVED[$point]:-}" ]] && continue

    tag="$(tag_for "${point}")"
    ckpt="$(final_ckpt "${point}")"
    train_log="${OUT}/train_${tag}.log"

    if [[ ! -f "${ckpt}/model.safetensors" ]]; then
      if ! training_alive "${point}" && grep -qiE "Traceback|CUDA out of memory" "${train_log}" 2>/dev/null; then
        log "FAILED ${tag}: training died (see ${train_log})"
        RESOLVED[$point]=failed
        resolved=$((resolved + 1))
      fi
      continue
    fi
    training_alive "${point}" && continue

    if ! uv run python -c "
import sys
sys.path.insert(0, 'vlai-experiments/vi-instructions')
from validate_backbone import assert_policy_tokenizer_matches
assert_policy_tokenizer_matches('${ckpt}')
" >>"${LOG}" 2>&1; then
      log "FAILED ${tag}: stale tokenizer_name in policy_preprocessor.json -- skipping eval"
      RESOLVED[$point]=bad_tokenizer
      resolved=$((resolved + 1))
      continue
    fi

    for eval_lang in en vi; do
      eval_dir="${REPO}/outputs/eval_${tag}_${eval_lang}"
      gpu="$(pick_gpu | head -1)"
      if [[ -z "${gpu}" ]]; then log "${tag}: eval ${eval_lang} waiting for a free GPU"; break; fi
      [[ -d "${eval_dir}" ]] && mv "${eval_dir}" "${eval_dir}.stale-$(date +%Y%m%d%H%M%S)"
      log "eval ${tag} (${eval_lang}) on GPU ${gpu}"
      env CUDA_VISIBLE_DEVICES="${gpu}" "./run_eval_${eval_lang}.sh" "${ckpt}" "${eval_dir}" \
        > "${OUT}/eval_${tag}_${eval_lang}.log" 2>&1 ||
        log "  eval ${eval_lang} FAILED for ${tag}"
    done

    if [[ -f "${REPO}/outputs/eval_${tag}_en/eval_info.json" &&
          -f "${REPO}/outputs/eval_${tag}_vi/eval_info.json" ]]; then
      log "DONE ${tag}"
      RESOLVED[$point]=ok
      resolved=$((resolved + 1))
    fi
  done
done

log "=== phase ${PHASE} seed ${SEED} complete: ${resolved}/${#POINTS[@]} resolved ==="
for point in "${POINTS[@]}"; do log "  ${point}: ${RESOLVED[$point]:-unresolved}"; done
