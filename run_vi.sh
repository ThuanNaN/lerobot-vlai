#!/usr/bin/env bash
set -euo pipefail

# SmolVLA + LIBERO Vietnamese-instruction training (Stage 2). Recipe: frozen VLM
# backbone + train the action expert from scratch (train_expert_only=true, no PEFT).
# The A/B experiment swaps only VLM_MODEL — see
# docs/superpowers/runbooks/2026-07-14-smolvla-vs-smolvla-vi-backbone.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

HF_USER="${HF_USER:?Set HF_USER (in .env or env) to your Hugging Face username}"
# Vietnamese LIBERO dataset (task_index-aligned; fixed on the Hub). Override with
# DATASET_REPO if you use your own fork.
DATASET_REPO="${DATASET_REPO:-VLAIResearchLab/lerobot_libero_vi}"
# VLM backbone swapped by the A/B experiment (see docs/.../runbook). Default arm =
# stock English SmolVLM2; set VLM_MODEL=thuanan/SmolVLM2-500M-vi-stage1 for the VN arm.
VLM_MODEL="${VLM_MODEL:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
# Per-arm output + run tag so the two arms never clobber each other. Seed held equal
# across arms so the only difference is the backbone.
RUN_TAG="${RUN_TAG:-smolvla}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/train_vi_${RUN_TAG}/}"
SEED="${SEED:-1000}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--wandb.enable=true)
fi

uv sync --locked --extra smolvla --extra libero

# Recipe: freeze the whole VLM backbone (train_expert_only=true) and train only the
# action expert (~100M of 450M params). This isolates the backbone as the single A/B
# variable — a frozen VN backbone's representations vs a frozen English one's — and
# needs no PEFT / pretrained_path. lerobot-train hard-fails PEFT without a
# pretrained_path (pretrained.py:_validate_peft_config), which is why LoRA-from-scratch
# is not used here.
uv run lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="${HF_USER}/libero-vi-${RUN_TAG}" \
  --policy.vlm_model_name="${VLM_MODEL}" \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.root="${HOME}/.cache/huggingface/lerobot/${DATASET_REPO}" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="${OUTPUT_DIR}" \
  --seed="${SEED}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000 \
  "${WANDB_ARGS[@]}"
