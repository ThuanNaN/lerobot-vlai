# Arm C Vocab-Only Backbone Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a third SmolVLA backbone arm ("Arm C") that has the Vietnamese
57,344-token vocabulary but zero Vietnamese language-pretraining signal, run it
through the existing frozen-8K A/B recipe, and compare it against Arm A (`smolvla`,
EN) and Arm B (`smolvla-vi`, VI-pretrained) to isolate whether the frozen-backbone
gap (−9.3 overall, −13.0 on `libero_spatial`) is a vocab-size artifact or a genuine
language-understanding effect.

**Architecture:** Task 1 writes and unit-tests a small standalone script that builds
Arm C's backbone (EN weights + VI tokenizer, `resize_token_embeddings(mean_resizing=True)`
on the new rows, no further training) — this is new, testable Python code. Tasks 2-5
are sequential *execution* steps (build the backbone for real, train it, eval it,
compare) that reuse existing infra (`run_vi.sh`, `run_eval_vi.sh`, `run_eval_en.sh`,
`compare_eval.py`) unchanged — these are long-running GPU/network operations, not
unit-testable, so each has explicit shell commands and concrete pass/fail checks
instead of pytest steps.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face `transformers` 5.5.x
(`AutoModelForImageTextToText`, `AutoProcessor`, `AutoTokenizer`), `uv`, pytest,
`lerobot-train` / `lerobot-eval` CLIs (this repo).

## Global Constraints

- All Python execution goes through `uv run` (per `CLAUDE.md`).
- New standalone research code lives under `vlai-experiments/vi-instructions/`
  (existing convention — see Task 1). Because that directory name has a hyphen it is
  not a dotted Python package; tests import sibling modules via
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` then a plain
  `import build_vocab_only_backbone`.
- `outputs/` is entirely gitignored (`.gitignore:167`) — nothing under
  `outputs/backbones/`, `outputs/train_vi_vocab_only/`, `outputs/eval_vi_vocab_only/`,
  `outputs/eval_en_vocab_only/`, or edits to `outputs/report_smolvla_en_vs_vi.md` gets
  committed. Only Task 1's script + test file are committed to git.
- No new dependencies — `transformers`, `torch` are already present via the `smolvla`
  extra (`uv sync --locked --extra smolvla --extra libero`, already run by `run_vi.sh`).
- Both backbones are already in the local HF cache (verified):
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and `thuanan/SmolVLM2-500M-vi-stage1` —
  no first-time multi-GB download expected for Task 2.
- Source spec: `docs/superpowers/specs/2026-07-28-vi-vocab-only-backbone-ablation-design.md`.
- Reuses, unmodified: `run_vi.sh`, `run_eval_vi.sh`, `run_eval_en.sh`,
  `vlai-experiments/vi-instructions/compare_eval.py` (already supports N-way
  `--run NAME=PATH`, no change needed for a 3-way comparison).

---

### Task 1: `build_vocab_only_backbone.py` — vocab-resize logic + CLI

**Files:**
- Create: `vlai-experiments/vi-instructions/build_vocab_only_backbone.py`
- Test: `vlai-experiments/vi-instructions/tests/test_build_vocab_only_backbone.py`

**Interfaces:**
- Consumes: nothing (first task, no dependencies on other tasks in this plan).
- Produces (used by Task 2's manual run, not by other Python code):
  - `resize_vocab_to_match(model, target_vocab_size: int, mean_resizing: bool = True) -> None`
    — mutates `model` in place: calls `model.resize_token_embeddings(new_num_tokens=target_vocab_size, mean_resizing=mean_resizing)`, then sets `model.config.vocab_size = target_vocab_size`.
  - `assert_ids_in_vocab(token_ids: list[int], vocab_size: int) -> None` — raises
    `ValueError` listing any out-of-range ids; returns `None` if all ids are in
    `[0, vocab_size)`.
  - `build_vocab_only_backbone(en_model_id: str, vi_tokenizer_id: str) -> tuple[PreTrainedModel, ProcessorMixin]`
    — real HF I/O (not unit tested): loads tokenizer from `vi_tokenizer_id`, model from
    `en_model_id`, calls `resize_vocab_to_match`, loads processor from
    `vi_tokenizer_id`, returns `(model, processor)`.
  - CLI (`__main__`): `--en-backbone` (default `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`),
    `--vi-backbone` (default `thuanan/SmolVLM2-500M-vi-stage1`), `--out-dir` (default
    `outputs/backbones/smolvlm2_vi_vocab_only`). Calls `build_vocab_only_backbone`,
    saves both `model.save_pretrained(out_dir)` and `processor.save_pretrained(out_dir)`,
    then sanity-checks by tokenizing a sample Vietnamese LIBERO instruction and calling
    `assert_ids_in_vocab`.

- [ ] **Step 1: Write the failing tests**

```python
# vlai-experiments/vi-instructions/tests/test_build_vocab_only_backbone.py
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from build_vocab_only_backbone import assert_ids_in_vocab, resize_vocab_to_match


class _StubModel:
    """Mimics the slice of transformers.PreTrainedModel this script depends on."""

    def __init__(self, vocab_size: int):
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.resize_calls: list[tuple[int, bool]] = []

    def resize_token_embeddings(self, new_num_tokens: int, mean_resizing: bool = True):
        self.resize_calls.append((new_num_tokens, mean_resizing))
        # Real transformers behavior: resizes the embedding table but does NOT touch
        # config.vocab_size on this composite-config model — verified interactively
        # against HuggingFaceTB/SmolVLM2-500M-Video-Instruct (transformers 5.5.4).


def test_resize_vocab_to_match_calls_resize_with_mean_resizing_true():
    model = _StubModel(vocab_size=49280)

    resize_vocab_to_match(model, target_vocab_size=57344)

    assert model.resize_calls == [(57344, True)]


def test_resize_vocab_to_match_updates_config_vocab_size():
    model = _StubModel(vocab_size=49280)

    resize_vocab_to_match(model, target_vocab_size=57344)

    assert model.config.vocab_size == 57344


def test_resize_vocab_to_match_respects_mean_resizing_flag():
    model = _StubModel(vocab_size=49280)

    resize_vocab_to_match(model, target_vocab_size=57344, mean_resizing=False)

    assert model.resize_calls == [(57344, False)]


def test_assert_ids_in_vocab_accepts_ids_in_range():
    assert_ids_in_vocab([0, 100, 57343], vocab_size=57344)  # must not raise


def test_assert_ids_in_vocab_rejects_id_at_or_above_vocab_size():
    with pytest.raises(ValueError, match="57344"):
        assert_ids_in_vocab([0, 57344], vocab_size=57344)


def test_assert_ids_in_vocab_rejects_negative_id():
    with pytest.raises(ValueError, match="-1"):
        assert_ids_in_vocab([-1, 5], vocab_size=57344)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_build_vocab_only_backbone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_vocab_only_backbone'`
(the module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/build_vocab_only_backbone.py
"""Build a SmolVLM2 backbone that has the Vietnamese 57,344-token vocabulary
(`thuanan/SmolVLM2-500M-vi-stage1`'s tokenizer) but zero Vietnamese
language-pretraining signal: the new 8,064 token rows are resized into the stock
English backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) via
`resize_token_embeddings(mean_resizing=True)` and nothing else is trained.

This isolates the vocab-size confound from the language-pretraining effect in the
EN-vs-VI SmolVLA A/B experiment — see
docs/superpowers/specs/2026-07-28-vi-vocab-only-backbone-ablation-design.md.

Output is a standalone HF backbone directory (model + processor/tokenizer files),
meant to be passed as VLM_MODEL to run_vi.sh — NOT a SmolVLA policy checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

EN_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VI_BACKBONE = "thuanan/SmolVLM2-500M-vi-stage1"
SANITY_INSTRUCTION = "nhấc cái bát đen trên hộp bánh quy lên rồi đặt nó lên đĩa"


def resize_vocab_to_match(model: Any, target_vocab_size: int, mean_resizing: bool = True) -> None:
    """Resize `model`'s input/output embeddings to `target_vocab_size` in place.

    `resize_token_embeddings` does not update `model.config.vocab_size` on
    composite-config VLMs (verified against SmolVLM2), so set it explicitly.
    """
    model.resize_token_embeddings(new_num_tokens=target_vocab_size, mean_resizing=mean_resizing)
    model.config.vocab_size = target_vocab_size


def assert_ids_in_vocab(token_ids: list[int], vocab_size: int) -> None:
    out_of_range = [i for i in token_ids if i < 0 or i >= vocab_size]
    if out_of_range:
        raise ValueError(f"token ids out of vocab range [0, {vocab_size}): {out_of_range}")


def build_vocab_only_backbone(en_model_id: str, vi_tokenizer_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(vi_tokenizer_id)
    model = AutoModelForImageTextToText.from_pretrained(
        en_model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    resize_vocab_to_match(model, target_vocab_size=len(tokenizer))
    processor = AutoProcessor.from_pretrained(vi_tokenizer_id)
    return model, processor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-backbone", default=EN_BACKBONE)
    parser.add_argument("--vi-backbone", default=VI_BACKBONE)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/backbones/smolvlm2_vi_vocab_only")
    )
    args = parser.parse_args()

    print(f"loading EN weights ({args.en_backbone}) + VI tokenizer ({args.vi_backbone}) ...")
    model, processor = build_vocab_only_backbone(args.en_backbone, args.vi_backbone)

    vocab_size = len(processor.tokenizer)
    print(f"resized embeddings to vocab_size={vocab_size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    sample_ids = processor.tokenizer(SANITY_INSTRUCTION)["input_ids"]
    assert_ids_in_vocab(sample_ids, vocab_size)

    print(f"wrote vocab-only backbone to {args.out_dir} (vocab={vocab_size}, sanity check passed)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_build_vocab_only_backbone.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add vlai-experiments/vi-instructions/build_vocab_only_backbone.py \
        vlai-experiments/vi-instructions/tests/test_build_vocab_only_backbone.py
git commit -m "feat(vi-instructions): add vocab-only backbone build script for Arm C ablation"
```

---

### Task 2: Build Arm C backbone for real

Not unit-testable (real multi-GB model I/O against Hugging Face Hub / local cache) —
this is a one-time execution step with concrete verification commands. Depends on
Task 1's script.

**Files:**
- Produces (gitignored, not committed): `outputs/backbones/smolvlm2_vi_vocab_only/`

- [ ] **Step 1: Run the build script**

```bash
uv run python vlai-experiments/vi-instructions/build_vocab_only_backbone.py
```

Expected: prints `resized embeddings to vocab_size=57344` then
`wrote vocab-only backbone to outputs/backbones/smolvlm2_vi_vocab_only (vocab=57344, sanity check passed)`.
No exception. Takes well under a minute — both source backbones are already cached
locally (`~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct`,
`models--thuanan--SmolVLM2-500M-vi-stage1`).

- [ ] **Step 2: Verify the output directory round-trips correctly**

```bash
uv run python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

out = 'outputs/backbones/smolvlm2_vi_vocab_only'
model = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.bfloat16, low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(out)

assert model.get_input_embeddings().weight.shape[0] == 57344, model.get_input_embeddings().weight.shape
assert model.get_output_embeddings().weight.shape[0] == 57344, model.get_output_embeddings().weight.shape
assert len(processor.tokenizer) == 57344

ids = processor.tokenizer('nhấc cái bát đen trên hộp bánh quy lên rồi đặt nó lên đĩa')['input_ids']
assert max(ids) < 57344 and min(ids) >= 0, ids
print('OK: embeddings, lm_head, and tokenizer are all 57344-vocab and consistent')
"
```

Expected: prints `OK: embeddings, lm_head, and tokenizer are all 57344-vocab and consistent`,
no `AssertionError`.

No commit for this task — `outputs/` is gitignored (see Global Constraints).

---

### Task 3: Train Arm C (frozen-backbone, 8K steps, action-expert only)

Long-running GPU training run (~same duration as Arm A/B's existing 8K frozen runs,
already in `outputs/train_vi_smolvla/` and `outputs/train_vi_vi/` for reference
timing). Depends on Task 2's backbone directory existing.

**Files:**
- Produces (gitignored): `outputs/train_vi_vocab_only/checkpoints/last/pretrained_model/`

- [ ] **Step 1: Launch training**

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> RUN_TAG=vocab_only \
  VLM_MODEL=outputs/backbones/smolvlm2_vi_vocab_only ./run_vi.sh
```

(Replace `<idle-gpu>` per the runbook's prerequisite: pin an idle GPU if the box is
shared. Uses every other default from `run_vi.sh`: `train_expert_only=true`,
`SEED=1000`, `STEPS=8000`, `BATCH_SIZE=16` — identical to Arm A/B except `VLM_MODEL`.)

- [ ] **Step 2: Wait for training to finish, then verify**

```bash
uv run python -c "
import json
cfg = json.load(open('outputs/train_vi_vocab_only/checkpoints/last/pretrained_model/config.json'))
assert cfg['vlm_model_name'] == 'outputs/backbones/smolvlm2_vi_vocab_only', cfg['vlm_model_name']
assert cfg['train_expert_only'] is True
print('OK: checkpoint config points at the vocab-only backbone, expert-only training confirmed')
"
```

Expected: `OK: checkpoint config points at the vocab-only backbone, expert-only
training confirmed`. Also check the training log for `num_learnable_params` ≈ 100M
(expert only, matching the runbook's verified figure for Arm A/B:
`99,880,992 / 450,046,176`).

No commit for this task — `outputs/` is gitignored.

---

### Task 4: Eval Arm C (VI primary, EN secondary sanity check)

Depends on Task 3's trained checkpoint. Two eval passes on the same checkpoint, no
further training.

**Files:**
- Produces (gitignored): `outputs/eval_vi_vocab_only/eval_info.json`,
  `outputs/eval_en_vocab_only/eval_info.json`

- [ ] **Step 1: Run the primary VI-instruction eval**

```bash
./run_eval_vi.sh outputs/train_vi_vocab_only/checkpoints/last/pretrained_model outputs/eval_vi_vocab_only
```

Expected: writes `outputs/eval_vi_vocab_only/eval_info.json` covering all 4 suites
(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), 10 episodes/task = 400
episodes total (matching Arm A/B's eval methodology).

- [ ] **Step 2: Run the secondary EN-instruction sanity eval**

```bash
./run_eval_en.sh outputs/train_vi_vocab_only/checkpoints/last/pretrained_model outputs/eval_en_vocab_only
```

Expected: writes `outputs/eval_en_vocab_only/eval_info.json`, same shape as Step 1.

- [ ] **Step 3: Verify both eval_info.json files are well-formed**

```bash
uv run python -c "
import json
for path in ['outputs/eval_vi_vocab_only/eval_info.json', 'outputs/eval_en_vocab_only/eval_info.json']:
    info = json.load(open(path))
    assert set(info['per_group'].keys()) == {'libero_spatial', 'libero_object', 'libero_goal', 'libero_10'}, info['per_group'].keys()
    assert 'pc_success' in info['overall']
    print(path, '-> overall pc_success =', info['overall']['pc_success'])
"
```

Expected: prints both `overall pc_success` values, no `AssertionError`/`KeyError`.

No commit for this task — `outputs/` is gitignored.

---

### Task 5: 3-way comparison + fold into the report

Depends on Task 4's two `eval_info.json` files. `compare_eval.py` already supports
repeatable `--run NAME=PATH` (verified in the design step — no code change needed).

**Files:**
- Modify: `outputs/report_smolvla_en_vs_vi.md` (gitignored — edit only, no commit)

- [ ] **Step 1: Generate the 3-way VI-instruction comparison table**

```bash
uv run python vlai-experiments/vi-instructions/compare_eval.py \
  --run smolvla=outputs/eval_vi_smolvla/eval_info.json \
  --run smolvla-vi=outputs/eval_vi_vi/eval_info.json \
  --run vocab-only=outputs/eval_vi_vocab_only/eval_info.json
```

Expected: a markdown table with columns `smolvla`, `smolvla-vi`, `vocab-only` and rows
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `overall`.

- [ ] **Step 2: Generate the EN-instruction sanity comparison (Arm A vs Arm C, both EN)**

```bash
uv run python vlai-experiments/vi-instructions/compare_eval.py \
  --run smolvla=outputs/eval_vi_smolvla/eval_info.json \
  --run vocab-only-en=outputs/eval_en_vocab_only/eval_info.json
```

Expected: a 2-column markdown table — read this to check whether mean-init vocab
expansion alone degraded the model's original English performance.

- [ ] **Step 3: Read the numbers and apply the design's decision criteria**

Per `docs/superpowers/specs/2026-07-28-vi-vocab-only-backbone-ablation-design.md`:
- `vocab-only` overall ≈ `smolvla-vi` overall (~20%, within noise) → gap is a
  vocab-size artifact.
- `vocab-only` overall ≪ `smolvla-vi` overall (collapses toward 0%) → confirms VI
  pretraining (not just vocab) drives Arm B's capability.
- Otherwise: report inconclusive at n=1 seed / 10 episodes-per-task — do not
  over-interpret a single-digit-point difference (same caveat that already applies
  to Arms A/B elsewhere in the report).

- [ ] **Step 4: Fold the result into `outputs/report_smolvla_en_vs_vi.md`**

Add a new `##` section after the existing "Hybrid backbone" section (before "Nguồn dữ
liệu thô"), following the file's existing style (per-suite table, then a short
Vietnamese-language interpretation paragraph, then a link to the raw
`eval_vi_vocab_only/eval_info.json` / `eval_en_vocab_only/eval_info.json` paths). Use
Step 1's and Step 2's actual table output and Step 3's actual conclusion — do not
invent numbers.

No commit — `outputs/report_smolvla_en_vs_vi.md` is gitignored, matching how the
existing report file (already git-untracked despite being on disk) has always been
handled in this project.
