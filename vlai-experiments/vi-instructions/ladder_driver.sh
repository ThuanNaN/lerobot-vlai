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

# LADDER_POINTS restricts this invocation to a subset, so rungs whose backbone already
# exists can be started while the remaining backbones are still being built.
if [[ -n "${LADDER_POINTS:-}" ]]; then
  read -r -a POINTS <<< "${LADDER_POINTS}"
fi

SUFFIX=""
[[ "${SEED}" != "1000" ]] && SUFFIX="_seed${SEED}"

tag_for() { echo "ladder_$1_${LANG}${SUFFIX}"; }
final_ckpt() {
  echo "${REPO}/outputs/train_$(tag_for "$1")/checkpoints/$(printf '%06d' "${STEPS}")/pretrained_model"
}
training_alive() { pgrep -af "lerobot-train" 2>/dev/null | grep -q "libero-vi-$(tag_for "$1")"; }

# One rung per GPU. A free-VRAM threshold alone is the wrong model here: SmolVLA needs
# only ~6.3 GB, so a 24 GB card still looks "free" right after a job claims it and the
# queue stacks every rung onto the same GPU. GPU_TAKEN records what this driver has
# placed, and is cleared when a rung finishes its evals.
declare -A GPU_TAKEN

pick_gpu() {
  local idx used total
  while IFS=', ' read -r idx used total; do
    [[ -n "${GPU_TAKEN[$idx]:-}" ]] && continue
    if (( total - used > NEED_MIB )); then echo "${idx}"; return 0; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
  return 1
}

point_complete() {  # final checkpoint plus both eval_info.json already on disk
  local tag; tag="$(tag_for "$1")"
  [[ -f "$(final_ckpt "$1")/model.safetensors" &&
     -f "${REPO}/outputs/eval_${tag}_en/eval_info.json" &&
     -f "${REPO}/outputs/eval_${tag}_vi/eval_info.json" ]]
}

log "=== ladder driver: phase=${PHASE} seed=${SEED} points=${POINTS[*]} train-lang=${LANG} ==="

# Idempotency: drop rungs already finished by an earlier invocation, so this driver can
# be restarted after a failure, or run first on a subset and later on the rest, without
# redoing ~10h of training per rung.
REMAINING=()
for point in "${POINTS[@]}"; do
  if point_complete "${point}"; then
    log "${point}: already complete (checkpoint + both evals) -- skipping"
  else
    REMAINING+=("${point}")
  fi
done
POINTS=("${REMAINING[@]}")
if (( ${#POINTS[@]} == 0 )); then
  log "=== nothing to do: every requested rung is already complete ==="
  exit 0
fi
log "rungs to run: ${POINTS[*]}"

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
declare -A LAUNCHED RESOLVED GPU_OF
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
    # DRY_RUN exists because this driver launches training with `setsid nohup`: killing
    # the driver (timeout, Ctrl-C) does NOT kill jobs it already started, so testing the
    # queue logic for real leaves orphaned 10-hour runs behind. Learned the hard way.
    if [[ -n "${DRY_RUN:-}" ]]; then
      log "  DRY_RUN: not launching"
      # Still claim the card, so a dry run exercises the real placement logic. Without
      # this the dry run happily reports every rung on the same GPU and hides the very
      # bug it is meant to catch.
      LAUNCHED[$point]=1; RESOLVED[$point]=dry_run; GPU_TAKEN[$gpu]=1; GPU_OF[$point]="${gpu}"
      next=$((next + 1)); resolved=$((resolved + 1))
      continue
    fi
    # Do NOT mkdir the output dir: lerobot-train's cfg.validate() requires it to not
    # already exist when resume=False, and fails with FileExistsError otherwise.
    setsid nohup env CUDA_VISIBLE_DEVICES="${gpu}" SKIP_SYNC=1 \
        VLM_MODEL="${path}" DATASET_REPO="${DATASET}" \
        RUN_TAG="${tag}" OUTPUT_DIR="${REPO}/outputs/train_${tag}/" SEED="${SEED}" \
        STEPS="${STEPS}" BATCH_SIZE="${BATCH_SIZE}" TRAIN_EXPERT_ONLY=false \
        SAVE_FREQ=10000 ENV_EVAL_FREQ=10000 LOG_FREQ=250 \
        ./run_vi.sh > "${OUT}/train_${tag}.log" 2>&1 < /dev/null &
    LAUNCHED[$point]=1
    GPU_TAKEN[$gpu]=1
    GPU_OF[$point]="${gpu}"
    next=$((next + 1))

    # Wait until the job has actually claimed VRAM before polling pick_gpu again.
    # A fixed sleep is not enough: SmolVLA spends several minutes indexing the dataset
    # and loading the backbone before it touches the GPU, so a short guard lets the next
    # iteration see the same card as free and stack every rung onto one GPU. That is
    # invisible until the other GPUs free up and nothing migrates to them.
    baseline=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
    for _ in $(seq 1 60); do   # up to 20 minutes
      sleep 20
      current=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
      if (( current - baseline > 2000 )); then
        log "  ${tag} has claimed GPU ${gpu} (${baseline} -> ${current} MiB)"
        break
      fi
      if ! training_alive "${point}"; then
        log "  WARNING: ${tag} exited before claiming GPU ${gpu} -- see ${OUT}/train_${tag}.log"
        break
      fi
    done
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
        unset "GPU_TAKEN[${GPU_OF[$point]}]"
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
      unset "GPU_TAKEN[${GPU_OF[$point]}]"
      continue
    fi

    # Evaluate on the GPU this rung trained on: its training just exited, so the card is
    # free, and it is still marked taken so the queue will not put another rung there.
    gpu="${GPU_OF[$point]}"
    for eval_lang in en vi; do
      eval_dir="${REPO}/outputs/eval_${tag}_${eval_lang}"
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
      unset "GPU_TAKEN[${gpu}]"   # release the card for the next queued rung
    fi
  done
done

log "=== phase ${PHASE} seed ${SEED} complete: ${resolved}/${#POINTS[@]} resolved ==="
for point in "${POINTS[@]}"; do log "  ${point}: ${RESOLVED[$point]:-unresolved}"; done
