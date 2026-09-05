#!/usr/bin/env bash
set -euo pipefail

# Stage 1 baseline: train SmolVLA on HuggingFaceVLA/libero (English instructions).
# Benchmark the resulting checkpoint separately with run_eval.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

HF_USER="${HF_USER:?Set HF_USER (in .env or env) to your Hugging Face username}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-1}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--wandb.enable=true)
fi

uv sync --locked --extra smolvla --extra libero

# https://huggingface.co/docs/lerobot/multi_gpu_training
TRAIN_CMD=(uv run lerobot-train)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  TRAIN_CMD=(uv run accelerate launch --multi_gpu --num_processes="${NUM_GPUS}" "$(uv run which lerobot-train)")
fi

"${TRAIN_CMD[@]}" \
  --policy.type=smolvla \
  --policy.repo_id="${HF_USER}/libero-vlai" \
  --policy.load_vlm_weights=true \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_decay_lr=1e-4 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task="${TASK_SUITE}" \
  --output_dir=./outputs/train/ \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=2000 \
  "${WANDB_ARGS[@]}"
