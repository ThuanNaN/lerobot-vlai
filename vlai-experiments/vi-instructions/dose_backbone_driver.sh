#!/usr/bin/env bash
# Build the d10 / d25 / d50 dose backbones: stage-1 continued-pretraining at 10%, 25%
# and 50% of the Vietnamese mixture, then merge each adapter into a standalone backbone
# and validate it.
#
# The three stage-1 runs each use all 3 GPUs via DDP, so they run SEQUENTIALLY.
# Measured full-mixture run: 1436 steps / 39,939s = 11.1h, so expect roughly
#   d10 ~144 steps ~1.1h   d25 ~359 steps ~2.8h   d50 ~718 steps ~5.6h   => ~9.5h total.
#
# Three facts this script exists to encode (each cost a failed or wasted run to learn):
#   * stage-1 must run under conda env `smol` (torch 2.13 + transformers 4.50 + peft
#     0.19.1). A plain shell picks up base conda's python 3.13 and dies in torchrun.
#   * merging must also run under `smol` -- lerobot's venv has no peft, and must not be
#     `uv sync`ed while other jobs depend on it.
#   * THE PUBLISHED d100 USED EFFECTIVE BATCH 128, NOT train_gpus.sh's DEFAULT 48.
#     Read from checkpoints/vietnamese_stage1_3gpu_v2/training_args.bin:
#     per_device_train_batch_size=8, gradient_accumulation_steps=8, world_size=2
#     (despite the "3gpu" directory name). Both the model card and train_gpus.sh's own
#     comment claim 4 x 3 GPU x 4 = 48, and both are wrong for this checkpoint.
#     Training the dose rungs at 48 while d100 sits at 128 would vary optimizer batch
#     size alongside dose -- exactly the confound the d0 anchor exists to avoid. The
#     settings below reproduce 8 x 2 x 8 = 128 and are NOT to be "optimised" to use the
#     third GPU: 3 GPUs cannot factor 128 with integer accumulation.
#
# Usage (detached, survives SSH loss):
#   setsid nohup ./vlai-experiments/vi-instructions/dose_backbone_driver.sh \
#     > outputs/backbones/dose_driver.stdout 2>&1 < /dev/null &
set -uo pipefail

REPO="/home/thuandn/Repository/lerobot"
SMOLLM="${HOME}/Repository/smollm-vi"
SMOL_PY="${HOME}/miniconda3/envs/smol/bin/python"
OUT="${REPO}/outputs/backbones"
LOG="${OUT}/dose_driver.log"
DOSES=(10 25 50)
# Reproduce the published d100's optimizer configuration exactly -- see header.
NUM_GPUS=2
PER_DEVICE_BATCH=8
GRAD_ACCUM=8

mkdir -p "${OUT}"
cd "${REPO}" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

log "=== dose backbone driver start (doses: ${DOSES[*]}) ==="

for pct in "${DOSES[@]}"; do
  mixture="${OUT}/mixtures/dose_${pct}/mixture_dose.yaml"
  ckpt_dir="${SMOLLM}/checkpoints/vietnamese_stage1_dose_${pct}"
  backbone_dir="${OUT}/vi_dose_${pct}"

  if [[ -f "${backbone_dir}/build_metadata.json" ]]; then
    log "d${pct}: backbone already built -- skipping"
    continue
  fi

  if [[ ! -f "${mixture}" ]]; then
    log "FATAL: missing ${mixture} (run dose_mixture.py first)"
    exit 1
  fi

  # --- stage-1 -------------------------------------------------------------------
  if [[ -f "${ckpt_dir}/adapter_model.safetensors" ]]; then
    log "d${pct}: stage-1 checkpoint already present -- skipping training"
  else
    log "d${pct}: stage-1 starting (mixture=${mixture})"
    (
      conda activate smol || exit 1
      cd "${SMOLLM}" || exit 1
      MIXTURE_TEMPLATE="${mixture}" \
      OUTPUT_DIR="${ckpt_dir}" \
      RUN_NAME="vietnamese_stage1_dose_${pct}" \
      NUM_GPUS="${NUM_GPUS}" \
      PER_DEVICE_BATCH="${PER_DEVICE_BATCH}" \
      GRAD_ACCUM="${GRAD_ACCUM}" \
        ./vision/experiments/pretraining/vietnamese/train_gpus.sh
    ) > "${OUT}/stage1_dose_${pct}.log" 2>&1
    rc=$?
    if (( rc != 0 )) || [[ ! -f "${ckpt_dir}/adapter_model.safetensors" ]]; then
      log "FATAL: d${pct} stage-1 failed (rc=${rc}), see ${OUT}/stage1_dose_${pct}.log"
      exit 1
    fi
    steps=$("${SMOL_PY}" -c "
import json; print(json.load(open('${ckpt_dir}/trainer_state.json'))['global_step'])" 2>/dev/null)
    log "d${pct}: stage-1 done, global_step=${steps}"
  fi

  # --- merge ---------------------------------------------------------------------
  log "d${pct}: merging adapter -> ${backbone_dir}"
  PYTHONPATH="${REPO}/vlai-experiments/vi-instructions" "${SMOL_PY}" \
    "${REPO}/vlai-experiments/vi-instructions/merge_stage1_adapter.py" \
    --adapter-dir "${ckpt_dir}" --out-dir "${backbone_dir}" \
    >> "${OUT}/stage1_dose_${pct}.log" 2>&1
  if [[ ! -f "${backbone_dir}/build_metadata.json" ]]; then
    log "FATAL: d${pct} merge failed, see ${OUT}/stage1_dose_${pct}.log"
    exit 1
  fi

  # --- validate ------------------------------------------------------------------
  if uv run python "${REPO}/vlai-experiments/vi-instructions/validate_backbone.py" \
       "${backbone_dir}" --expected-vocab 57344 --device cuda --max-lines 400 \
       --report "${backbone_dir}/validation.json" >> "${LOG}" 2>&1; then
    bpc=$(uv run python -c "
import json; print(json.load(open('${backbone_dir}/validation.json'))['bits_per_character'])")
    log "d${pct}: DONE, ${bpc} bits/char"
  else
    log "FATAL: d${pct} failed validation"
    exit 1
  fi
done

log "=== all doses built ==="
log "BPC ladder (expect a decreasing trend from d0 4.4098 to d100 3.3183):"
for name in vi_dose_0 vi_dose_10 vi_dose_25 vi_dose_50; do
  v=$(uv run python -c "
import json,sys
try: print(json.load(open('${OUT}/${name}/validation.json'))['bits_per_character'])
except Exception: print('n/a')" 2>/dev/null)
  log "  ${name}: ${v}"
done
