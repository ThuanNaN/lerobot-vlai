#!/usr/bin/env bash
set -euo pipefail

# Stage 1 benchmark: eval a checkpoint against the full LIBERO protocol
# (4 suites x 10 episodes = 400 episodes) with the original English instructions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

CHECKPOINT_PATH="${1:?Usage: run_eval.sh <checkpoint_path> <output_dir>}"
OUTPUT_DIR="${2:?Usage: run_eval.sh <checkpoint_path> <output_dir>}"

uv sync --locked --extra smolvla --extra libero

uv run lerobot-eval \
  --policy.path="${CHECKPOINT_PATH}" \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --output_dir="${OUTPUT_DIR}"


uv run lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir=./outputs/evals/smolvla_base
