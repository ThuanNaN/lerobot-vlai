#!/usr/bin/env bash
set -euo pipefail

# SmolVLA + LIBERO Vietnamese-instruction LoRA finetune (Stage 2 of
# docs/superpowers/specs/2026-07-02-smolvla-vietnamese-instructions-design.md)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

HF_USER="${HF_USER:?Set HF_USER (in .env or env) to your Hugging Face username}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--wandb.enable=true)
fi

# SmolVLA's default LoRA targets (_get_default_peft_targets in modeling_smolvla.py:495-500)
# only adapt the action expert's q/v projections. For Vietnamese instruction comprehension we
# also need the VLM's own text-model attention layers.
TARGET_MODULES='(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.vlm_with_expert\.vlm\.model\.text_model\.layers\..*\.self_attn\.(q|k|v|o)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'

uv sync --locked --extra smolvla --extra libero

uv run lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="${HF_USER}/libero-vi" \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=false \
  --dataset.repo_id="${HF_USER}/libero-vi" \
  --dataset.root="${HOME}/.cache/huggingface/lerobot/${HF_USER}/libero-vi" \
  --peft.method_type=LORA \
  --peft.r="${LORA_R:-16}" \
  --peft.lora_alpha="${LORA_ALPHA:-32}" \
  --peft.target_modules="${TARGET_MODULES}" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/train_vi/ \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000 \
  "${WANDB_ARGS[@]}"
