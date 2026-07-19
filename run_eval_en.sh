#!/usr/bin/env bash
set -euo pipefail

# English-instruction counterpart to run_eval_vi.sh: runs a checkpoint against
# LIBERO with the benchmark's native English task descriptions (no language
# override), so the two scripts give a same-checkpoint EN vs VI comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

CHECKPOINT_PATH="${1:?Usage: run_eval_en.sh <checkpoint_path> <output_dir> [rename_map_json]}"
OUTPUT_DIR="${2:?Usage: run_eval_en.sh <checkpoint_path> <output_dir> [rename_map_json]}"
RENAME_MAP="${3:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

extra_args=()
if [[ -n "${RENAME_MAP}" ]]; then
  extra_args+=(--rename_map="${RENAME_MAP}")
fi

uv run lerobot-eval \
  --policy.path="${CHECKPOINT_PATH}" \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  "${extra_args[@]}" \
  --output_dir="${OUTPUT_DIR}"
