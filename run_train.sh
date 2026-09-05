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
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_GPUS="${NUM_GPUS:-1}"
# Dedicated output dir so this run never collides with the many other
# experiment outputs already under ./outputs/ (lerobot-train hard-fails if
# output_dir already exists and resume=false).
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/repro_paper_100k}"

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
  --output_dir="${OUTPUT_DIR}/train" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=2000 \
  "${WANDB_ARGS[@]}"

# Benchmark: full LIBERO protocol (4 suites x 10 episodes = 400 episodes)
CHECKPOINT_PATH="${OUTPUT_DIR}/train/checkpoints/last/pretrained_model"

# --policy.n_action_steps=1: the paper's simulation inference protocol resamples
# a new action after every executed action, not after the full 50-action chunk
# (section 4.3: "In simulation, we perform inference by sampling new observations
# and predicting a new action after each executed action."). The trained policy
# still predicts full chunk_size=50 chunks; this only changes how many of each
# chunk's actions get executed before replanning.
uv run lerobot-eval \
  --policy.path="${CHECKPOINT_PATH}" \
  --policy.n_action_steps=1 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --output_dir="${OUTPUT_DIR}/eval"
