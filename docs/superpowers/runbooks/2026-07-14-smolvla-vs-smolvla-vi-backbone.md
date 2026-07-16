# Runbook — SmolVLA vs SmolVLA-VI backbone on Vietnamese LIBERO

**Date:** 2026-07-14 (recipe redesigned 2026-07-17) · **Owner:** thuanan · **Compute:** 1 GPU (sequential arms)

## Research question

Does swapping the VLM backbone from the stock English-centric SmolVLM2-500M to a
Vietnamese continued-pretrained backbone (`thuanan/SmolVLM2-500M-vi-stage1`) improve
**Vietnamese-instruction** task success on LIBERO, all else held equal?

- **H1 (backbone helps):** `smolvla-vi` beats `smolvla` on the VI-instruction eval, especially
  on longer / more compositional suites (`libero_goal`, `libero_10`) where language
  comprehension matters most.
- **H0 (null):** no significant success-rate delta → a frozen VN backbone's representations are
  no more useful to the action expert than the frozen English one's for these tasks.

## Recipe — frozen backbone, train the action expert (no PEFT)

Both arms use the **identical** recipe (parametrized `run_vi.sh`); only `VLM_MODEL` differs.

The VLM backbone is **frozen** (`--policy.train_expert_only=true`) and only the action expert
(~100M of 450M params) is trained from scratch. This is the cleanest test of the research
question: with the VLM held fixed in both arms, the eval measures the **quality of the
backbone's frozen representations** of Vietnamese text — not how fast a backbone can adapt.

> **Why not LoRA?** The earlier draft of this runbook paired `--policy.type=smolvla` (fresh
> init) with `--peft.*`. `lerobot-train` **hard-fails** that combination —
> `PreTrainedPolicy._validate_peft_config` (`src/lerobot/policies/pretrained.py`) raises
> *"Training from scratch using PEFT is unlikely to yield good results. Supply a
> `policy.pretrained_path`."* PEFT requires a pretrained checkpoint to adapt. The only stock
> checkpoint (`lerobot/smolvla_base`) embeds the **English** VLM, so there is no symmetric
> starting point for the VN arm. Freezing the VLM and training the expert sidesteps PEFT
> entirely and keeps the two arms perfectly symmetric. *(Verified 2026-07-17: this recipe
> reaches "End of training" with `num_learnable_params=99,880,992 / 450,046,176` — i.e. only
> the expert is trainable.)*

| Held constant across arms | Value |
|---|---|
| Dataset | `VLAIResearchLab/lerobot_libero_vi` (task_index-aligned, fixed & verified 40/40) |
| Recipe | frozen VLM (`train_expert_only=true`), train action expert, **no PEFT** |
| Init | fresh SmolVLA (random action expert) + `load_vlm_weights=true` |
| Seed | `1000` (both arms) → identical action-expert init; only the backbone differs |
| Steps / batch | `STEPS=8000`, `BATCH_SIZE=16` |
| Eval | full LIBERO, 4 suites × 10 ep = 400, VI instructions via `eval_overrides.json` |

| Independent variable | Arm A `smolvla` | Arm B `smolvla-vi` |
|---|---|---|
| `VLM_MODEL` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `thuanan/SmolVLM2-500M-vi-stage1` |
| Tokenizer / vocab | 49 280 | 57 344 (VN-extended) |
| Output dir | `outputs/train_vi_smolvla/` | `outputs/train_vi_vi/` |

> Swapping `vlm_model_name` swaps the **whole VLM incl. its tokenizer/processor** (loaded from
> the same `model_id`), so each arm conditions on VI text through its own tokenizer. That is the
> effect under test — do **not** try to share one tokenizer across arms.

## Prerequisites

- [x] Dataset present locally: `~/.cache/huggingface/lerobot/VLAIResearchLab/lerobot_libero_vi`
      (33 GB, 384 files, verified — see "Verify dataset" below).
- [ ] Extras installed: `run_vi.sh` runs `uv sync --locked --extra smolvla --extra libero`
      (no `peft` extra needed — this recipe does not use PEFT).
- [ ] GPU visible; `MUJOCO_GL=egl` (set by the scripts). Pin an idle GPU with
      `CUDA_VISIBLE_DEVICES=<n>` if the box is shared.
- [ ] `.env` has `HF_USER` (and optionally `WANDB_API_KEY` to log curves).
- [ ] Backbone repos reachable: both `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and
      `thuanan/SmolVLM2-500M-vi-stage1` (the checkpoint config re-references its backbone at
      eval time, so the VN repo must stay accessible).

### Verify dataset (already done — re-run if the cache is touched)

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('VLAIResearchLab/lerobot_libero_vi',
                    root='$HOME/.cache/huggingface/lerobot/VLAIResearchLab/lerobot_libero_vi')
print('episodes:', ds.num_episodes, 'frames:', ds.num_frames)
print('sample task:', ds[0]['task'])   # must be Vietnamese, matching the trajectory
"
```
Expected (confirmed 2026-07-17): `episodes: 1693 frames: 273465`, and
`ds[0]['task'] == 'đặt cốc trắng lên đĩa bên trái và đặt cốc vàng trắng lên đĩa bên phải'`
(task_index 0, aligned to the original English source-of-truth ordering).

> Pass `root=` explicitly. Without it, `LeRobotDataset` contacts the Hub to resolve a revision
> and can fail even though the data is fully cached locally.

## Step 1 — Smoke test both arms (cheap, catches wiring bugs)

```bash
# Arm A (stock English backbone)
CUDA_VISIBLE_DEVICES=<idle-gpu> STEPS=10 BATCH_SIZE=2 RUN_TAG=smoke_en ./run_vi.sh

# Arm B (Vietnamese backbone) — first run also downloads ~1 GB backbone weights
CUDA_VISIBLE_DEVICES=<idle-gpu> STEPS=10 BATCH_SIZE=2 RUN_TAG=smoke_vi \
  VLM_MODEL=thuanan/SmolVLM2-500M-vi-stage1 ./run_vi.sh
```
Expected: each completes and writes
`outputs/train_vi_<tag>/checkpoints/last/pretrained_model/{config.json,model.safetensors}`.
Confirm Arm B's `config.json` shows `"vlm_model_name": "thuanan/SmolVLM2-500M-vi-stage1"`, and
that the training log reports `num_learnable_params` ≈ 100M (expert only, VLM frozen).

> Each smoke run pushes a policy repo to `${HF_USER}/libero-vi-<tag>` at startup (via
> `--policy.repo_id`). Delete the throwaway `libero-vi-smoke_*` repos afterward if you don't
> want them lingering on the Hub.

## Step 2 — Full training, one arm at a time (1 GPU)

```bash
# Arm A — smolvla (baseline)
CUDA_VISIBLE_DEVICES=<idle-gpu> RUN_TAG=smolvla ./run_vi.sh

# Arm B — smolvla-vi
CUDA_VISIBLE_DEVICES=<idle-gpu> RUN_TAG=vi VLM_MODEL=thuanan/SmolVLM2-500M-vi-stage1 ./run_vi.sh
```
Seed is pinned to `1000` in both, so the action-expert init is identical; the frozen backbone is
the only difference. Checkpoints land in `outputs/train_vi_smolvla/` and `outputs/train_vi_vi/`.

## Step 3 — Full VI eval for each checkpoint (400 episodes each)

`run_eval_vi.sh` substitutes Vietnamese instructions via
`vlai-experiments/vi-instructions/data/eval_overrides.json` (matches `tasks_vi.csv`), so the
policy is scored on VI language it never saw verbatim at train time.

```bash
./run_eval_vi.sh outputs/train_vi_smolvla/checkpoints/last/pretrained_model outputs/eval_vi_smolvla
./run_eval_vi.sh outputs/train_vi_vi/checkpoints/last/pretrained_model      outputs/eval_vi_vi
```
Each writes `eval_info.json` with per-suite and aggregate success rates. `--policy.path` reloads
each checkpoint with its own saved backbone — no need to set `VLM_MODEL` at eval.

## Step 4 — Compare

```bash
uv run python vlai-experiments/vi-instructions/compare_eval.py \
  --run smolvla=outputs/eval_vi_smolvla/eval_info.json \
  --run smolvla-vi=outputs/eval_vi_vi/eval_info.json
```
Prints a per-suite + overall `pc_success` markdown table, one column per `--run`. Read the
`smolvla-vi − smolvla` delta per suite off the table.

## Decision criteria

- **Primary:** overall VI success rate (mean over 400 episodes), Arm B vs Arm A.
- **Secondary:** per-suite deltas — H1 predicts the largest gain on `libero_goal` / `libero_10`.
- With `eval.n_episodes=10` per task, treat single-digit-percent deltas as noise. A credible
  effect is a consistent gap across suites, not one suite alone.
- If inconclusive, escalate (below) before drawing a conclusion.

## Threats to validity / caveats

1. **Expert trained from scratch, backbone frozen** may cap absolute success low in both arms.
   The *delta* can still be informative, but if both arms sit near-zero the test is underpowered
   → escalate to a full finetune (unfreeze the VLM). This is the main power risk of freezing.
2. **Single seed.** One run per arm. If the delta is small, repeat with 2–3 seeds
   (`SEED=1000,2000,3000`) before believing it.
3. **Vocab/tokenizer differs by construction** — intended (it is *how* a VN backbone helps),
   not a confound to remove. Arm B has a strictly larger embedding table; the comparison answers
   "does this VN backbone help," not "does VN pretraining alone help holding vocab fixed."
4. **Backbone availability at eval:** Arm B's checkpoint config points at
   `thuanan/SmolVLM2-500M-vi-stage1`; keep it reachable/cached.

## Escalation path (only if Step 4 is inconclusive)

1. Multi-seed (2–3) per arm; report mean ± range.
2. **Full finetune:** set `--policy.train_expert_only=false` (unfreeze the VLM) and raise `STEPS`
   to 50–100k for a stronger signal — still swapping only `VLM_MODEL`. Watch for OOM at
   `BATCH_SIZE=16` on a 24 GB GPU (all ~450M params become trainable); drop batch + add grad
   accumulation if needed. Note this shifts the question from "are the frozen VN representations
   better" to "does starting from a VN backbone finetune better," which mixes representation
   quality with adaptability.
3. Add a third arm: stock backbone but VN-vocab-extended-only (isolates vocab from VN
   pretraining) if the vocab confound (caveat 3) matters to the claim.

## Artifacts to keep

- `outputs/train_vi_smolvla/` and `outputs/train_vi_vi/` checkpoints + `config.json`.
- `outputs/eval_vi_smolvla/eval_info.json`, `outputs/eval_vi_vi/eval_info.json`.
- The comparison table (Step 4) + W&B run URLs if enabled.
