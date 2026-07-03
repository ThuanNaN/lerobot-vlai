# SmolVLA Vietnamese Instruction-Following — Design

## Goal

Adapt the SmolVLA + LIBERO training pipeline (`run.sh`) so the robot correctly executes
tasks when given **Vietnamese** instructions instead of English ones. Language is
**input-only**: the model must understand a Vietnamese task command and condition action
generation on it correctly. No Vietnamese text generation is required at inference.

Related project context: [[project-vqa-vlm-tieng-viet]] (the broader VQA/VLM-tiếng-Việt
research track this project sits under — see `vlai-docs/`, especially Bài 08, 09, 12, 14, 15
which cover general concepts this design applies concretely).

## Current state

- `run.sh` trains `smolvla` on `HuggingFaceVLA/libero` (English task strings) and evaluates
  with `lerobot-eval` across all 4 LIBERO suites (`libero_spatial`, `libero_object`,
  `libero_goal`, `libero_10`), 10 episodes each.
- Default backbone: `vlm_model_name = HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  (`src/lerobot/policies/smolvla/configuration_smolvla.py:84`). Its LLM decoder (SmolLM2) is
  trained predominantly on English corpora — Vietnamese support is not something to assume
  either way; it must be measured (Stage 0).
- `vlm_model_name` is loaded via `AutoModelForImageTextToText` /
  `SmolVLMForConditionalGeneration` in `smolvlm_with_expert.py:90-116`, and the LoRA "expert"
  construction assumes a specific config shape (`config.text_config.hidden_size`,
  `num_key_value_heads`, `head_dim`, `attention_bias`, `text_model.layers`). This constrains
  backbone swaps to architecturally-compatible (Idefics3/SmolVLM-style) checkpoints — see
  Stage 4.
- SmolVLA's default PEFT targets (`_get_default_peft_targets`,
  `modeling_smolvla.py:495-500`) only adapt the **action expert's** q/v projections, not the
  VLM's language layers. This is insufficient for a language-adaptation goal (Stage 2).
- **The `lerobot-eval` LIBERO rollout is decoupled from `tasks.parquet` entirely.** Confirmed:
  `rollout()` in `scripts/lerobot_eval.py:263-271` sets `observation["task"]` from
  `env.call("task_description")`, which reads `LiberoEnv.task_description` — set in
  `envs/libero.py:172` as `task.language`, sourced directly from the upstream `libero`
  package's own English benchmark suite definitions (`libero.libero.benchmark`), never from
  our dataset. This is the exact same `task` channel the policy processor consumes at train
  time (`render_messages_processor.py:66` and friends), so **translating the training
  dataset alone has no effect on what instruction the policy sees during eval** — eval will
  keep showing English regardless. There is currently no override knob (`LiberoEnv.gym_kwargs`
  only forwards `obs_type`/`render_mode`/observation size/`task_ids`). See Stage 1b.

## Non-goals

- Vietnamese text generation / VQA output (separate track, see `vlai-docs/10-15`).
- New tokenizer vocabulary or vocab-extension pretraining — not attempted unless Stage 0
  diagnostics show it's actually the bottleneck (it is explicitly the last-resort escalation
  in Stage 4, not a starting assumption).
- Real-robot data collection — this design stays entirely within LIBERO sim, reusing existing
  episodes/actions/video.

## Stage 0 — Diagnostics (no training)

Two cheap, forward-pass-only checks against the current backbone, run before any training:

1. **Tokenizer fertility**: tokenize a sample of Vietnamese text (draft-translated LIBERO
   task strings) with the backbone's `AutoProcessor` tokenizer; compare tokens/word against
   the equivalent English strings. Establishes whether `tokenizer_max_length=48`
   (`configuration_smolvla.py:60`) has enough headroom for Vietnamese phrasing.
2. **Zero-shot semantic sanity check**: pass a handful of Vietnamese vs. English instructions
   with matching meaning through the frozen VLM text encoder; check whether
   paraphrase/semantic clustering roughly holds in the embedding space. Distinguishes
   "under-trained but recoverable via finetuning" from "effectively untrained for
   Vietnamese."

**Output**: a short diagnostic report (fertility numbers + embedding examples) that gates
whether Stage 1 proceeds as planned or Stage 4's continued-pretraining path should be
front-loaded instead.

## Stage 1 — Data: Vietnamese LIBERO variant

- Extract the task strings LIBERO uses (from `HuggingFaceVLA/libero`'s task metadata).
- Machine-translate all of them to Vietnamese (40 unique strings: 4 suites × 10 tasks each,
  confirmed against `data/libero/meta/tasks.parquet`'s 40 rows), then a full
  manual review pass — the set is small enough for complete human review, avoiding unnatural
  MT phrasing (wrong object names, awkward imperatives) from polluting the training signal.
- Build a new dataset variant (e.g. `<user>/libero-vi`) that reuses the same
  episodes/actions/video/images unchanged, only swapping the task-instruction text to
  Vietnamese — avoid duplicating heavy video data.
- Hold out a small, separately-verified eval slice (e.g. one task per suite) that is not used
  to seed or spot-check the training-set translations, so translation-quality leakage doesn't
  inflate eval numbers.

## Stage 1b — Eval-side instruction override (code change)

Because the LIBERO eval rollout reads its instruction from the `libero` package directly
(see "Current state"), Stage 3 eval is meaningless without a way to substitute Vietnamese
text for `task.language` at rollout time. Add a small override mechanism:

- A JSON file mapping `task_id → Vietnamese string` per suite (reusing the same translations
  produced in Stage 1, keyed consistently with the LIBERO suite's own task ordering — not
  `tasks.parquet`'s `task_index`, which is a different, dataset-local numbering).
- Thread a `task_language_overrides: dict[int, str] | None` (or a path to the JSON file,
  loaded once) through: the `LiberoEnv` `EnvConfig` dataclass (`envs/configs.py`) →
  `create_libero_envs`/`_make_env_fns` (`envs/libero.py:386-422`) → `LiberoEnv.__init__`
  (`envs/libero.py:110-130`), applied at `envs/libero.py:172`:
  `self.task_description = overrides.get(task_id, task.language)`.
- `src/lerobot/envs/libero.py` and `src/lerobot/envs/configs.py` are both under CLAUDE.md's
  strict-mypy module list — new parameters need full type annotations.
- When no override is supplied, behavior is unchanged (falls back to `task.language`), so this
  is additive and doesn't affect existing English-instruction eval runs.

## Stage 2 — Baseline LoRA finetune

- Start from `lerobot/smolvla_base` (pretrained), same starting point as the current
  `run.sh`.
- **Widen `--peft.target_modules`** beyond SmolVLA's default: also target the VLM
  text-model's attention (and optionally MLP) layers, not just the action expert's q/v
  projections. Language understanding lives in the VLM backbone, so adapting only the action
  expert would not move the needle on instruction comprehension.
- No new vocabulary or embedding-table surgery initially — the existing tokenizer already
  maps Vietnamese subwords to real (if underused) embeddings. Add `embed_tokens`/`lm_head` as
  LoRA targets only if Stage 3 results point specifically to embeddings being the bottleneck.
- Raise `tokenizer_max_length` above the default 48 if Stage 0's fertility check shows
  Vietnamese phrasing needs more headroom.
- Start small (`steps≈5,000–10,000`) to validate the pipeline before committing to a full-size
  run, consistent with the single-GPU budget.

## Stage 3 — Evaluation

- Reuse the existing `lerobot-eval` LIBERO protocol (4 suites × 10 episodes) from `run.sh`,
  run with the Stage 1b instruction overrides active so the policy actually receives the
  Stage 1 held-out Vietnamese eval strings during rollout.
- **Primary comparison**: Vietnamese-LoRA model on Vietnamese instructions vs. the current
  English baseline on English instructions (success rate).
- **Secondary reference**: the *English*-trained baseline run zero-shot on the Vietnamese eval
  set — quantifies the gap that LoRA finetuning is actually closing.
- Break results down per suite (`spatial`/`object`/`goal`/`10`), since `goal` and `10` involve
  longer, more compositional instructions than `spatial`/`object`.

## Stage 4 — Escalation gates (only if Stage 3 underperforms)

Escalation is explicitly gated, not a default next step — per the existing project principle
("đừng pretrain nếu finetune đã đủ", `vlai-docs/15-pretraining-va-roadmap.md`):

- **→ Continued pretraining**: trigger if Vietnamese success rate lags the English baseline by
  a large margin (e.g. >15–20 points) even after widened LoRA, *and* Stage 0 already flagged
  poor tokenizer fertility or embedding quality. Approach: lightweight adapter/projection-level
  continued pretraining on a Vietnamese image-caption corpus, then repeat Stage 2.
- **→ Backbone swap**: last resort, only if continued pretraining still doesn't close the gap.
  Requires a checkpoint that is both more multilingual *and* architecturally compatible —
  loads via `AutoModelForImageTextToText` and exposes the same `text_model.layers` /
  `text_config` shape that `smolvlm_with_expert.py`'s expert-construction code assumes.
  Verify compatibility with a smoke-load before committing further work.

## Deliverables

- Stage 0 diagnostic report (fertility numbers, embedding sanity examples).
- `<user>/libero-vi` dataset + translation review notes.
- `LiberoEnv` instruction-override mechanism (code change) + the Vietnamese eval-instruction
  JSON file it consumes.
- LoRA-finetuned checkpoint + training config.
- Eval table: success rate by suite, Vietnamese-finetuned vs. English baseline vs.
  zero-shot-Vietnamese, with example failure cases.
- Decision log noting whether/why Stage 4 escalation was triggered.

## Risks / open questions

- Translation quality of imperative, object-manipulation phrasing is a known risk even with
  manual review — LIBERO's vocabulary (object names, spatial relations) may not have a single
  natural Vietnamese rendering; document translation choices for reproducibility.
- Widened LoRA targets increase trainable parameter count and GPU memory vs. the SmolVLA
  default — validate this still fits the single-GPU budget before scaling `steps`.
- Stage 4's backbone-swap path is architecturally constrained (see "Current state" above) and
  should not be treated as a drop-in option without a compatibility smoke-test.
- Two separate translation artifacts now need to stay consistent: the dataset's
  `meta/tasks.parquet` (keyed by dataset-local `task_index`) and the Stage 1b eval-override
  JSON (keyed by the `libero` package's own per-suite `task_id`). They are different key
  spaces for the same underlying task set — translate from one shared source-of-truth mapping
  (English string → Vietnamese string) and derive both files from it, rather than translating
  twice.
