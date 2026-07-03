#!/usr/bin/env bash
set -euo pipefail

# Stage 3 eval: run a checkpoint against LIBERO with Vietnamese task instructions
# substituted in via the Stage 1b override mechanism (envs/libero.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

CHECKPOINT_PATH="${1:?Usage: run_eval_vi.sh <checkpoint_path> <output_dir>}"
OUTPUT_DIR="${2:?Usage: run_eval_vi.sh <checkpoint_path> <output_dir>}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

uv run lerobot-eval \
  --policy.path="${CHECKPOINT_PATH}" \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.task_language_overrides_path="${SCRIPT_DIR}/vlai-experiments/vi-instructions/data/eval_overrides.json" \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --output_dir="${OUTPUT_DIR}"
