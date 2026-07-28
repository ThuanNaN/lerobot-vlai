# Arm C — Vocab-Only Backbone Ablation — Design

## Goal

Isolate the **vocab-size confound** from the **language-pretraining effect** in the
SmolVLA EN-vs-VI backbone A/B experiment. The VI backbone
(`thuanan/SmolVLM2-500M-vi-stage1`) differs from the stock EN backbone
(`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) in two ways simultaneously: it has a
larger vocabulary (57,344 vs 49,280 tokens) _and_ it was continued-pretrained on
Vietnamese multimodal data. The existing frozen-backbone result (Arm A `smolvla` 29.5%
vs Arm B `smolvla-vi` 20.2%, gap −9.3, see
[`outputs/report_smolvla_en_vs_vi.md`](../../../outputs/report_smolvla_en_vs_vi.md))
cannot currently distinguish which of these two factors drives the VI arm's
underperformance.

This design adds a third arm ("Arm C") — a backbone with the VI vocabulary but with
**zero Vietnamese language-pretraining signal** — to answer: does the frozen-backbone
gap come from the larger embedding table being harder for the action expert to
condition on, or from the pretrained VI backbone's representations genuinely mattering
(good or bad)?

Related context: [[project-vi-ab-pipeline-running]], [[smolvla-hybrid-backbone-gotchas]],
runbook [`docs/superpowers/runbooks/2026-07-14-smolvla-vs-smolvla-vi-backbone.md`](../runbooks/2026-07-14-smolvla-vs-smolvla-vi-backbone.md).

## Current state

- 4-stage A/B (frozen-8K, full-FT-10K premature, full-FT-50K) + hybrid-backbone +
  external baseline experiments are all DONE and written up in
  `outputs/report_smolvla_en_vs_vi.md`.
- `outputs/eda_libero_suites.md` root-caused the `libero_spatial` weakness to
  action-expert grasp/place-timing robustness, not a language/vocab issue — but that
  finding is about a _different_ suite-level pattern, not this vocab-vs-pretraining
  question.
- `thuanan/SmolVLM2-500M-vi-stage1`'s README (HF Hub) confirms: 8,064 new Vietnamese
  BPE tokens were mean-initialized, then trained **jointly** with LoRA adapters
  (r=32, α=64, attention+MLP projections) on a Vietnamese multimodal mixture — vocab
  expansion and language-pretraining happened in one inseparable step. No standalone
  "vocab-expanded-but-untrained" checkpoint exists publicly for this model.
- Both existing arms use `run_vi.sh` with `train_expert_only=true` (frozen VLM, train
  only the ~100M-param action expert), `SEED=1000`, `STEPS=8000`, `BATCH_SIZE=16`.

## Non-goals

- Full-finetune (50K-step) variant of Arm C — deferred; the confound matters most at
  frozen-8K, where the gap is negative and unexplained. Full-FT already shows VI
  winning, so isolating the confound there is lower priority.
- Multi-seed reruns of any arm — a separate, independently-flagged reviewer concern
  (statistical significance), out of scope for this design.
- Quantifying the grasp/place-timing hypothesis with motion metrics — a separate
  reviewer concern, out of scope here.
- Retraining or otherwise modifying `thuanan/SmolVLM2-500M-vi-stage1` itself.

## Build method — Arm C backbone

1. Load the stock EN backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
   (`AutoModelForImageTextToText`, `AutoProcessor`).
2. Load the tokenizer/processor from `thuanan/SmolVLM2-500M-vi-stage1` (57,344-token
   vocab, includes the 8,064 Vietnamese BPE tokens).
3. `resize_token_embeddings(57344, mean_resizing=True)` on the EN model's
   `embed_tokens` and `lm_head`. This does not copy a single mean vector into all
   8,064 new rows; it samples each new row independently from a multivariate normal
   distribution fitted to the existing 49,280 EN embedding vectors' mean and
   covariance, so the new rows are distinct from each other (verified empirically:
   new-row std ≈0.06, non-zero variance across rows) — matching the original stage-1
   init method, but with **no further training** (no LoRA, no Vietnamese data) on top.
   The new rows still carry zero genuine Vietnamese semantic content.
4. Leave every other weight (attention, MLP, vision encoder, connector) exactly as the
   stock EN checkpoint — untouched.
5. Save as a new standalone checkpoint: `outputs/backbones/smolvlm2_vi_vocab_only/`
   (local; push to Hub only if useful for reuse — not required for this experiment).
6. Sanity-check before training: confirm `embed_tokens.weight.shape[0] == 57344`,
   `lm_head.weight.shape[0] == 57344`, and that the processor's tokenizer round-trips
   a Vietnamese LIBERO instruction into the expected extended-vocab token ids.

New script: `vlai-experiments/vi-instructions/build_vocab_only_backbone.py`.

> This is a different construction than the SmolVLA hybrid-checkpoint method in
> [[smolvla-hybrid-backbone-gotchas]] (that swaps `vlm_with_expert.vlm.*` inside an
> already-built SmolVLA policy). Here we build a standalone VLM backbone _before_ any
> SmolVLA-specific construction, to be passed as `VLM_MODEL` into the normal
> `run_vi.sh` training entry point — SmolVLA's own `load_vlm_weights=True` path handles
> wrapping it into a fresh policy.

## Recipe & scope

Reuse `run_vi.sh` unchanged, matching Arm A/B exactly except `VLM_MODEL`:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> RUN_TAG=vocab_only \
  VLM_MODEL=outputs/backbones/smolvlm2_vi_vocab_only ./run_vi.sh
```

- `train_expert_only=true` (frozen VLM, train action expert only)
- `SEED=1000` (identical action-expert init to Arm A/B)
- `STEPS=8000`, `BATCH_SIZE=16`
- Only frozen-8K — no full-finetune variant for Arm C (see Non-goals)
- Output: `outputs/train_vi_vocab_only/checkpoints/last/pretrained_model/`

## Eval plan

1. **Primary — VI instructions** (the main question): `run_eval_vi.sh` against the
   trained checkpoint, full 400 episodes (4 suites × 10 tasks × 10 episodes), same
   `eval_overrides.json` as Arm B.
   ```bash
   ./run_eval_vi.sh outputs/train_vi_vocab_only/checkpoints/last/pretrained_model outputs/eval_vi_vocab_only
   ```
2. **Secondary — EN sanity check** (approved, optional but cheap: no extra training,
   one more 400-episode eval pass on the same checkpoint): does mean-init vocab
   expansion alone degrade the model's original English-instruction performance?
   ```bash
   ./run_eval_en.sh outputs/train_vi_vocab_only/checkpoints/last/pretrained_model outputs/eval_en_vocab_only
   ```

## Decision criteria

Extend `vlai-experiments/vi-instructions/compare_eval.py`'s `--run` mechanism to a
3-way comparison:

```bash
uv run python vlai-experiments/vi-instructions/compare_eval.py \
  --run smolvla=outputs/eval_vi_smolvla/eval_info.json \
  --run smolvla-vi=outputs/eval_vi_vi/eval_info.json \
  --run vocab-only=outputs/eval_vi_vocab_only/eval_info.json
```

Read overall `pc_success` plus per-suite, with particular attention to
`libero_spatial` (largest existing EN−VI gap at frozen-8K: −13.0).

- **Arm C ≈ Arm B (~20%, within noise)** → the frozen-8K gap is a vocab-size artifact:
  a larger embedding table alone makes the action expert's conditioning harder,
  regardless of whether the extra tokens carry learned Vietnamese semantics.
- **Arm C ≪ Arm B (collapses toward 0%)** → confirms the VI-pretrained backbone's
  _learned_ representations (not merely having the right vocab) are what let Arm B
  reach 20.2% — consistent with the earlier hybrid zero-shot finding that vocab
  correctness alone was insufficient for cross-lingual transfer.
- Anything in between: report as inconclusive at n=1 seed / 10 episodes-per-task
  (same statistical caveat that already applies to Arms A/B); do not over-interpret a
  single-digit-point difference.

Treat this as one additional row/section in `outputs/report_smolvla_en_vs_vi.md`
(not a new standalone report), since it extends the same frozen-8K comparison already
documented there.

## Artifacts to keep

- `vlai-experiments/vi-instructions/build_vocab_only_backbone.py` (new build script)
- `outputs/backbones/smolvlm2_vi_vocab_only/` (Arm C backbone checkpoint)
- `outputs/train_vi_vocab_only/` (trained action-expert checkpoint + config)
- `outputs/eval_vi_vocab_only/eval_info.json` (primary VI eval)
- `outputs/eval_en_vocab_only/eval_info.json` (secondary EN sanity eval)
- Updated 3-way comparison table, folded into `outputs/report_smolvla_en_vs_vi.md`
