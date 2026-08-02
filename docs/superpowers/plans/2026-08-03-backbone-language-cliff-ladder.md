# Backbone Language-Cliff Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled ladder of SmolVLM2 backbones — varying Vietnamese pretraining dose at fixed 500M, and model scale at fixed English-centric pretraining — train SmolVLA on English LIBERO at each rung, and evaluate zero-shot on Vietnamese, to locate the backbone language cliff.

**Architecture:** Three new Python tools plus one driver in the `lerobot` repo (`vlai-experiments/vi-instructions/`), and two small env-override edits in the separate `smollm-vi` repo that owns the stage-1 continued-pretraining stack. The pipeline is: generate a dose-scaled mixture YAML → run stage-1 in `smollm-vi` → merge the LoRA+token adapter into a standalone HF backbone → validate it (embedding shape, tokenizer round-trip, bits-per-character on held-out Vietnamese) → feed it to the existing `run_vi.sh` / `run_eval_en.sh` / `run_eval_vi.sh` scripts through a queue driver → aggregate into a ladder report.

**Tech Stack:** Python 3.12 · PyTorch · transformers · PEFT 0.19.1 · `uv run` · bash · pytest

## Global Constraints

- **Two repos.** `lerobot` = `/home/thuandn/Repository/lerobot` (all new Python + driver). `smollm-vi` = `/home/thuandn/Repository/smollm-vi` (stage-1 training stack; two env-override edits only). Commit in each repo separately; never add one as a submodule of the other.
- **Run Python via `uv run`** in `lerobot` (per `CLAUDE.md`). In `smollm-vi`, use its own environment exactly as `train_gpus.sh` does — do not port it into `lerobot`.
- **Vocab sizes are fixed constants:** stock English = `49280`, Vietnamese-extended = `57344`, new-token count = `8064`, `trainable_token_start = 49280`.
- **Stage-1 hyperparameters are frozen** across every dose: 2 epochs, per-device batch 4 × 3 GPU × 4 grad-accum = effective 48, LR 1e-4 cosine, weight decay 0.1, `model_max_length` 2048, `image_target_size` 1536, bf16 + tf32, gradient checkpointing, LoRA r=32 α=64 dropout=0.1 on `q_proj k_proj v_proj o_proj gate_proj up_proj out_proj`, `--trainable_token_start 49280`. **The only permitted variation is the mixture sampling strategy and the warmup schedule** (see Task 2).
- **SmolVLA training recipe is frozen** across every rung: `TRAIN_EXPERT_ONLY=false`, `STEPS=50000`, `BATCH_SIZE=16`, `SAVE_FREQ=10000`, `ENV_EVAL_FREQ=10000`, `LOG_FREQ=250`. Effective batch must stay 16 everywhere — if a backbone OOMs, lower `BATCH_SIZE` and raise grad-accum to compensate, never lower the effective batch.
- **English LIBERO dataset repo id is `HuggingFaceVLA/libero`.** Vietnamese is `VLAIResearchLab/lerobot_libero_vi`.
- **Never evaluate `checkpoints/last`** — always the explicit `checkpoints/050000/pretrained_model` path (a symlink race corrupted an earlier result).
- **Ladder point names** are exactly `stock`, `d0`, `d10`, `d25`, `d50`, `d100`, `s256m`, `s2200m`. **Language tags** are exactly `en`, `vi` (the language of the *training* data).
- **Do not modify or rename any existing `outputs/` directory.** The historical Arm A/B/C runs are referenced read-only through an alias table.
- **GPUs are busy** until roughly 2026-08-04 06:00 with the `outputs/ft_multiseed_pipeline` run. Tasks 1–8 are CPU-only or short and can proceed now; Tasks 9–13 need free GPUs.

---

## File Structure

**Created in `lerobot`:**
- `vlai-experiments/vi-instructions/dose_mixture.py` — pure functions that scale a stage-1 mixture YAML to a dose and emit a provenance manifest.
- `vlai-experiments/vi-instructions/merge_stage1_adapter.py` — sub-token-mean embedding resize + PEFT merge into a standalone HF backbone directory.
- `vlai-experiments/vi-instructions/build_bpc_corpus.py` — extract held-out Vietnamese text into a plain-text corpus for bits-per-character measurement.
- `vlai-experiments/vi-instructions/validate_backbone.py` — hard gate: embedding shape, tokenizer round-trip, bits-per-character.
- `vlai-experiments/vi-instructions/ladder_driver.sh` — GPU-count-agnostic job queue.
- `vlai-experiments/vi-instructions/ladder_report.py` — aggregate `eval_info.json` files into the ladder table, curves, and paired bootstrap.
- `vlai-experiments/vi-instructions/tests/test_dose_mixture.py`
- `vlai-experiments/vi-instructions/tests/test_merge_stage1_adapter.py`
- `vlai-experiments/vi-instructions/tests/test_validate_backbone.py`
- `vlai-experiments/vi-instructions/tests/test_ladder_report.py`

**Modified in `smollm-vi`:**
- `vision/experiments/pretraining/vietnamese/train_gpus.sh` — add `MIXTURE_TEMPLATE`, `WARMUP_RATIO`, `RUN_NAME` env overrides.

**Produced artifacts (not committed):**
- `outputs/backbones/vi_dose_{0,10,25,50}/` — new backbones (`d100` is the published `thuanan/SmolVLM2-500M-vi-stage1`; `stock`, `s256m`, `s2200m` are stock Hub ids).
- `outputs/train_ladder_<point>_<lang>[_seed<N>]/` and `outputs/eval_ladder_<point>_<lang>[_seed<N>]_{en,vi}/`
- `outputs/report_backbone_ladder.md`

---

### Task 1: Dose-scaled mixture generation

Scales every source's `sampling_strategy` in the stage-1 mixture YAML by a dose fraction, so `d10 ⊂ d25 ⊂ d50 ⊂ d100`.

Why nesting comes free: `smollm-vi`'s `vision/smolvlm2/smolvlm/datasets/dataset.py:475-499` implements `random:N%` as `random.seed(42); random.shuffle(list); list[:N]`. The permutation depends only on list length, which is constant across doses, so prefixes of the same permutation are nested. `first:N` is a raw prefix and is nested by construction. `all` is not shuffled, so at dose 1.0 it must stay `all` (any subset of it is trivially nested).

**Files:**
- Create: `vlai-experiments/vi-instructions/dose_mixture.py`
- Test: `vlai-experiments/vi-instructions/tests/test_dose_mixture.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `scale_sampling_strategy(strategy: str, dose: float) -> str`
  - `scale_mixture(mixture: dict, dose: float) -> dict` — takes the parsed YAML (a dict with exactly one top-level key whose value is a list of source dicts), returns a new dict, does not mutate the input.
  - `dose_manifest(dose: float, mixture: dict, scaled: dict, template_path: str, git_sha: str) -> dict`
  - Module constant `DOSES = (0.10, 0.25, 0.50, 1.00)`

- [ ] **Step 1: Write the failing test**

```python
# vlai-experiments/vi-instructions/tests/test_dose_mixture.py
from __future__ import annotations

import pytest

from vlai_experiments_path import ensure_on_path  # noqa: F401  (see Step 3 note)
from dose_mixture import DOSES, dose_manifest, scale_mixture, scale_sampling_strategy


class TestScaleSamplingStrategy:
    def test_all_stays_all_at_full_dose(self):
        assert scale_sampling_strategy("all", 1.0) == "all"

    def test_all_becomes_random_percent_below_full_dose(self):
        assert scale_sampling_strategy("all", 0.25) == "random:25%"

    def test_percent_strategy_is_multiplied(self):
        assert scale_sampling_strategy("random:15%", 0.25) == "random:3.75%"

    def test_percent_strategy_unchanged_at_full_dose(self):
        assert scale_sampling_strategy("random:15%", 1.0) == "random:15%"

    def test_first_count_is_multiplied_and_floored(self):
        assert scale_sampling_strategy("first:25000", 0.10) == "first:2500"

    def test_first_count_never_reaches_zero(self):
        assert scale_sampling_strategy("first:5", 0.10) == "first:1"

    def test_rejects_dose_outside_unit_interval(self):
        with pytest.raises(ValueError, match="dose"):
            scale_sampling_strategy("all", 1.5)

    def test_rejects_unknown_strategy_kind(self):
        with pytest.raises(ValueError, match="unsupported"):
            scale_sampling_strategy("end:100", 0.5)


class TestScaleMixture:
    @staticmethod
    def _template() -> dict:
        return {
            "vietnamese_stage1": [
                {"name": "viocrvqa", "sampling_strategy": "all"},
                {"name": "cauldron_vqav2", "sampling_strategy": "random:15%"},
                {"name": "viwiki_text", "sampling_strategy": "first:25000"},
            ]
        }

    def test_scales_every_source(self):
        out = scale_mixture(self._template(), 0.50)
        strategies = [s["sampling_strategy"] for s in out["vietnamese_stage1"]]
        assert strategies == ["random:50%", "random:7.5%", "first:12500"]

    def test_does_not_mutate_input(self):
        template = self._template()
        scale_mixture(template, 0.10)
        assert template["vietnamese_stage1"][0]["sampling_strategy"] == "all"

    def test_preserves_all_other_source_fields(self):
        template = self._template()
        template["vietnamese_stage1"][0]["json_path"] = "__DATA_FOLDER__/viocrvqa_train.json"
        out = scale_mixture(template, 0.25)
        assert out["vietnamese_stage1"][0]["json_path"] == "__DATA_FOLDER__/viocrvqa_train.json"
        assert out["vietnamese_stage1"][0]["name"] == "viocrvqa"

    def test_full_dose_is_identity(self):
        template = self._template()
        assert scale_mixture(template, 1.0) == template

    def test_rejects_mixture_with_multiple_top_level_keys(self):
        with pytest.raises(ValueError, match="exactly one"):
            scale_mixture({"a": [], "b": []}, 0.5)

    def test_rejects_source_missing_sampling_strategy(self):
        with pytest.raises(ValueError, match="sampling_strategy"):
            scale_mixture({"m": [{"name": "x"}]}, 0.5)


class TestDoseNesting:
    """The property the whole experiment rests on: a smaller dose must select a
    subset of a larger dose's samples. For `random:N%` that holds because
    smollm-vi shuffles with a fixed seed and takes a prefix, so it reduces to
    the prefix lengths being monotonic."""

    @staticmethod
    def _prefix_len(strategy: str, total: int) -> int:
        import math

        kind, amount = strategy.split(":") if ":" in strategy else ("all", None)
        if kind == "all":
            return total
        if amount.endswith("%"):
            return max(1, math.ceil(total * float(amount.rstrip("%")) / 100.0))
        return int(amount)

    def test_prefix_lengths_are_monotonic_across_doses(self):
        total = 19700
        lengths = [
            self._prefix_len(scale_sampling_strategy("all", d), total) for d in DOSES
        ]
        assert lengths == sorted(lengths)
        assert len(set(lengths)) == len(DOSES)

    def test_percent_prefix_lengths_are_monotonic(self):
        total = 82700
        lengths = [
            self._prefix_len(scale_sampling_strategy("random:15%", d), total) for d in DOSES
        ]
        assert lengths == sorted(lengths)


class TestDoseManifest:
    def test_records_dose_and_provenance(self):
        template = {"m": [{"name": "x", "sampling_strategy": "all"}]}
        scaled = scale_mixture(template, 0.25)
        manifest = dose_manifest(
            dose=0.25,
            mixture=template,
            scaled=scaled,
            template_path="/tmp/vietnamese_stage1.yaml",
            git_sha="abc123",
        )
        assert manifest["dose"] == 0.25
        assert manifest["git_sha"] == "abc123"
        assert manifest["template_path"] == "/tmp/vietnamese_stage1.yaml"
        assert manifest["sources"] == [
            {"name": "x", "original": "all", "scaled": "random:25%"}
        ]
        assert "created_at" in manifest
```

Create the tiny path shim the tests import (the existing tests in this directory rely on `conftest.py` for the same purpose — check whether one already exists and reuse it rather than adding a second mechanism):

```python
# vlai-experiments/vi-instructions/tests/conftest.py  (create ONLY if absent)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

If `conftest.py` already exists with this shim, delete the `from vlai_experiments_path import ensure_on_path` line from the test file instead of creating anything.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_dose_mixture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dose_mixture'`

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/dose_mixture.py
"""Scale a smollm-vi stage-1 mixture YAML to a fraction of its data ("dose").

The backbone language-cliff ladder needs backbones pretrained on 10%, 25%, 50%
and 100% of the Vietnamese stage-1 mixture, with the smaller doses being strict
subsets of the larger ones. smollm-vi's
`vision/smolvlm2/smolvlm/datasets/dataset.py:_apply_sampling_strategy` implements
`random:N%` as `random.seed(42); shuffle(items); items[:N]` -- a fixed permutation
followed by a prefix -- so scaling the percentages is enough to get nesting for
free. `first:N` is already a raw prefix.

See docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

DOSES = (0.10, 0.25, 0.50, 1.00)

# smollm-vi only implements these prefix-style kinds deterministically; `end:` is
# supported there but would break nesting (a suffix of one length is not a subset
# of a suffix of another), so it is rejected here rather than silently mis-scaled.
_SUPPORTED_KINDS = ("random", "first")


def _check_dose(dose: float) -> None:
    if not 0.0 < dose <= 1.0:
        raise ValueError(f"dose must be in (0, 1], got {dose}")


def _format_percent(value: float) -> str:
    """Render a percentage without trailing zeros: 25.0 -> '25', 3.75 -> '3.75'."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def scale_sampling_strategy(strategy: str, dose: float) -> str:
    """Return `strategy` scaled down to `dose` of the data it currently selects."""
    _check_dose(dose)

    if strategy == "all":
        return "all" if dose == 1.0 else f"random:{_format_percent(dose * 100)}%"

    if ":" not in strategy:
        raise ValueError(f"unsupported sampling_strategy: {strategy!r}")

    kind, amount = strategy.split(":", 1)
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"unsupported sampling_strategy kind: {kind!r} (in {strategy!r})")

    if amount.endswith("%"):
        scaled = float(amount.rstrip("%")) * dose
        return f"{kind}:{_format_percent(scaled)}%"

    # A bare count: keep at least one sample so a source never silently vanishes
    # from the mixture at a small dose, which would change the mixture's shape
    # rather than only its size.
    return f"{kind}:{max(1, int(int(amount) * dose))}"


def scale_mixture(mixture: dict, dose: float) -> dict:
    """Scale every source's sampling_strategy in a parsed stage-1 mixture YAML."""
    _check_dose(dose)

    if len(mixture) != 1:
        raise ValueError(f"mixture must have exactly one top-level key, got {sorted(mixture)}")

    scaled = copy.deepcopy(mixture)
    (sources,) = scaled.values()
    for source in sources:
        if "sampling_strategy" not in source:
            raise ValueError(f"source missing sampling_strategy: {source.get('name', source)!r}")
        source["sampling_strategy"] = scale_sampling_strategy(source["sampling_strategy"], dose)
    return scaled


def dose_manifest(
    dose: float, mixture: dict, scaled: dict, template_path: str, git_sha: str
) -> dict:
    """Provenance record written next to each dose-scaled mixture."""
    (original_sources,) = mixture.values()
    (scaled_sources,) = scaled.values()
    return {
        "dose": dose,
        "template_path": template_path,
        "git_sha": git_sha,
        "created_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                "name": original["name"],
                "original": original["sampling_strategy"],
                "scaled": new["sampling_strategy"],
            }
            for original, new in zip(original_sources, scaled_sources, strict=True)
        ],
    }


if __name__ == "__main__":
    import argparse
    import subprocess

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path.home()
        / "Repository/smollm-vi/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml",
        help="stage-1 mixture YAML to scale (keeps its __DATA_FOLDER__ placeholders)",
    )
    parser.add_argument("--dose", type=float, required=True, help="fraction in (0, 1]")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    template = yaml.safe_load(args.template.read_text())
    scaled = scale_mixture(template, args.dose)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mixture_path = args.out_dir / "mixture_dose.yaml"
    mixture_path.write_text(yaml.safe_dump(scaled, sort_keys=False, allow_unicode=True))
    (args.out_dir / "dose_manifest.json").write_text(
        json.dumps(
            dose_manifest(args.dose, template, scaled, str(args.template), git_sha),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"wrote {mixture_path} (dose={args.dose})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_dose_mixture.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Verify against the real template**

Run:
```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/dose_mixture.py \
  --dose 0.25 --out-dir /tmp/dose_check
cat /tmp/dose_check/mixture_dose.yaml
```
Expected: seven sources; `viocrvqa`/`openvivqa`/`uitviic` become `random:25%`; `cauldron_vqav2` becomes `random:3.75%`; `cauldron_ocrvqa` becomes `random:1%`; `cauldron_textvqa` becomes `random:6.25%`; `viwiki_text` becomes `first:6250`; every `__DATA_FOLDER__` placeholder preserved.

- [ ] **Step 6: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/dose_mixture.py \
        vlai-experiments/vi-instructions/tests/test_dose_mixture.py \
        vlai-experiments/vi-instructions/tests/conftest.py
git commit -m "feat(vi-instructions): dose-scaled stage-1 mixture generation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Env overrides in the smollm-vi trainer

`train_gpus.sh` hardcodes the mixture template and `--warmup_steps 100`. Both must become overridable: the ladder needs a per-dose mixture, and a fixed 100-step warmup would consume ~70% of a dose-10% run (the full run is 1436 steps, so 100 steps is 7%; at dose 0.10 the run is ~144 steps). Holding the *shape* of the LR schedule constant across doses means switching to `--warmup_ratio 0.07` with `--warmup_steps 0`.

**Files:**
- Modify: `/home/thuandn/Repository/smollm-vi/vision/experiments/pretraining/vietnamese/train_gpus.sh`

**Interfaces:**
- Consumes: `mixture_dose.yaml` produced by Task 1.
- Produces: `train_gpus.sh` honouring `MIXTURE_TEMPLATE`, `WARMUP_RATIO`, `RUN_NAME` env vars. Task 6 invokes it.

- [ ] **Step 1: Read the current script and confirm the three anchor lines**

Run:
```bash
cd /home/thuandn/Repository/smollm-vi
grep -n 'MIXTURE_TEMPLATE=\|warmup_steps\|run_name' vision/experiments/pretraining/vietnamese/train_gpus.sh
```
Expected: `MIXTURE_TEMPLATE="$REPO_ROOT/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml"`, `--warmup_steps 100 \`, `--run_name vietnamese_stage1_3gpu_v2`.

- [ ] **Step 2: Make the mixture template overridable**

Replace:
```bash
MIXTURE_TEMPLATE="$REPO_ROOT/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml"
```
with:
```bash
# Overridable so the language-cliff ladder can feed in a dose-scaled mixture
# (lerobot: vlai-experiments/vi-instructions/dose_mixture.py). Default is the
# full 100% stage-1 mixture, i.e. the original behaviour.
MIXTURE_TEMPLATE="${MIXTURE_TEMPLATE:-$REPO_ROOT/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml}"
```

- [ ] **Step 3: Make warmup dose-proportional and the run name overridable**

Add near the other env defaults (just below the `GRAD_ACCUM` block):
```bash
# Warmup as a fraction of total steps, not an absolute count: the ladder trains
# the same mixture at 10%-100% of its size, so a fixed 100-step warmup would be
# 7% of the full run but ~70% of a dose-10% run. 0.07 reproduces the original
# 100/1436 ratio of the published stage-1 run.
WARMUP_RATIO="${WARMUP_RATIO:-0.07}"
RUN_NAME="${RUN_NAME:-vietnamese_stage1_3gpu_v2}"
```

Replace the flag line:
```bash
    --warmup_steps 100 \
```
with:
```bash
    --warmup_steps 0 \
    --warmup_ratio "$WARMUP_RATIO" \
```

(HF `TrainingArguments` gives `warmup_steps` precedence over `warmup_ratio` when it is non-zero, so it must be explicitly zeroed.)

Replace:
```bash
    --run_name vietnamese_stage1_3gpu_v2
```
with:
```bash
    --run_name "$RUN_NAME"
```

- [ ] **Step 4: Verify the script still parses and the defaults are unchanged**

Run:
```bash
cd /home/thuandn/Repository/smollm-vi
bash -n vision/experiments/pretraining/vietnamese/train_gpus.sh && echo "syntax OK"
grep -n 'MIXTURE_TEMPLATE\|WARMUP_RATIO\|RUN_NAME\|warmup' vision/experiments/pretraining/vietnamese/train_gpus.sh
```
Expected: `syntax OK`; defaults resolve to the original template path, `0.07`, and `vietnamese_stage1_3gpu_v2`.

- [ ] **Step 5: Smoke test that the overrides reach torchrun (20 steps, ~3 min)**

Only run this once GPUs are free. If they are still busy, defer this step to the start of Task 6 and note it as deferred.

```bash
cd /home/thuandn/Repository/smollm-vi
MAX_STEPS=20 NUM_GPUS=1 PER_DEVICE_BATCH=1 GRAD_ACCUM=1 \
  RUN_NAME=ladder_smoke OUTPUT_DIR=/tmp/ladder_smoke \
  ./vision/experiments/pretraining/vietnamese/train_gpus.sh 2>&1 | tail -30
```
Expected: training starts, reaches step 20, exits 0. Confirm `/tmp/ladder_smoke/mixture_resolved.yaml` exists with `__DATA_FOLDER__` substituted.

- [ ] **Step 6: Commit (in smollm-vi)**

```bash
cd /home/thuandn/Repository/smollm-vi
git add vision/experiments/pretraining/vietnamese/train_gpus.sh
git commit -m "feat(vietnamese): env-overridable mixture, warmup ratio, run name

Needed by the lerobot backbone language-cliff ladder, which trains this same
stage-1 recipe at 10/25/50/100% of the mixture. Warmup moves from a fixed 100
steps to ratio 0.07 (= 100/1436, the published run's ratio) so the LR schedule
shape is identical at every dose.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Merge a stage-1 adapter into a standalone backbone

Stage-1 saves a PEFT adapter (`adapter_model.safetensors`), not a full model. To use a dose backbone as SmolVLA's `vlm_model_name`, it must be merged into a standalone HF directory.

Two details that make this exact rather than approximate:

1. The base must be resized to 57,344 **using the same initialisation stage-1 used** before the adapter is applied, because `trainable_tokens_delta` stores a *delta from that initialisation*. `smollm-vi`'s `train.py:238-249` initialises each new row as the **mean of the row's sub-token embeddings under the base tokenizer** — fully deterministic. This is *not* what `build_vocab_only_backbone.py` does (HF's `mean_resizing=True` draws from a fitted multivariate normal), which is why Task 5 exists.
2. The adapter's `trainable_token_indices` covers `embed_tokens` and `lm_head`; PEFT 0.19.1's `merge_and_unload()` folds both the LoRA deltas and the token deltas.

**Files:**
- Create: `vlai-experiments/vi-instructions/merge_stage1_adapter.py`
- Test: `vlai-experiments/vi-instructions/tests/test_merge_stage1_adapter.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `subtoken_mean_init(model, base_tokenizer, new_tokenizer, old_vocab_size: int) -> int` — resizes in place, returns the new vocab size.
  - `merge_stage1_adapter(base_model_id: str, adapter_dir: Path, tokenizer_dir: Path) -> tuple[PreTrainedModel, ProcessorMixin]`
  - Module constants `STOCK_VOCAB_SIZE = 49280`, `VI_VOCAB_SIZE = 57344`, `EN_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`
  - Task 5 imports `subtoken_mean_init`; Task 6 runs this module's CLI.

- [ ] **Step 1: Write the failing test**

The heavy model load is not unit-testable; the *initialisation arithmetic* is, so the test drives `subtoken_mean_init` against tiny stand-ins.

```python
# vlai-experiments/vi-instructions/tests/test_merge_stage1_adapter.py
from __future__ import annotations

import pytest
import torch
from torch import nn

from merge_stage1_adapter import STOCK_VOCAB_SIZE, VI_VOCAB_SIZE, subtoken_mean_init


class FakeTokenizer:
    """Minimal stand-in exposing only what subtoken_mean_init uses."""

    def __init__(self, decode_map: dict[int, str], encode_map: dict[str, list[int]]):
        self._decode_map = decode_map
        self._encode_map = encode_map

    def decode(self, ids: list[int]) -> str:
        return self._decode_map[ids[0]]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._encode_map[text]


class FakeModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self._in = nn.Embedding(vocab_size, hidden)
        self._out = nn.Linear(hidden, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out

    def resize_token_embeddings(self, new_num_tokens: int):
        hidden = self._in.weight.shape[1]
        old = self._in.weight.shape[0]
        new_in = nn.Embedding(new_num_tokens, hidden)
        with torch.no_grad():
            new_in.weight[:old] = self._in.weight
        self._in = new_in
        new_out = nn.Linear(hidden, new_num_tokens, bias=False)
        with torch.no_grad():
            new_out.weight[:old] = self._out.weight
        self._out = new_out


class TestSubtokenMeanInit:
    @staticmethod
    def _setup():
        model = FakeModel(vocab_size=3, hidden=4)
        with torch.no_grad():
            model.get_input_embeddings().weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            model.get_input_embeddings().weight[1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
            model.get_input_embeddings().weight[2] = torch.tensor([0.0, 0.0, 4.0, 0.0])
            model.get_output_embeddings().weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            model.get_output_embeddings().weight[1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
            model.get_output_embeddings().weight[2] = torch.tensor([0.0, 0.0, 4.0, 0.0])
        base_tok = FakeTokenizer({}, {"xy": [0, 1], "zz": [2], "??": []})
        new_tok = FakeTokenizer({3: "xy", 4: "zz", 5: "??"}, {})
        return model, base_tok, new_tok

    def test_returns_new_vocab_size(self):
        model, base_tok, new_tok = self._setup()
        assert subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6) == 6

    def test_new_row_is_mean_of_its_subtokens(self):
        model, base_tok, new_tok = self._setup()
        subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6)
        expected = (torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.tensor([0.0, 2.0, 0.0, 0.0])) / 2
        assert torch.allclose(model.get_input_embeddings().weight[3], expected)

    def test_single_subtoken_row_is_copied(self):
        model, base_tok, new_tok = self._setup()
        subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6)
        assert torch.allclose(
            model.get_input_embeddings().weight[4], torch.tensor([0.0, 0.0, 4.0, 0.0])
        )

    def test_row_with_no_valid_subtokens_falls_back_to_global_mean(self):
        model, base_tok, new_tok = self._setup()
        subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6)
        expected = torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 2.0, 0.0, 0.0]),
                torch.tensor([0.0, 0.0, 4.0, 0.0]),
            ]
        ).mean(dim=0)
        assert torch.allclose(model.get_input_embeddings().weight[5], expected)

    def test_original_rows_are_untouched(self):
        model, base_tok, new_tok = self._setup()
        subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6)
        assert torch.allclose(
            model.get_input_embeddings().weight[1], torch.tensor([0.0, 2.0, 0.0, 0.0])
        )

    def test_output_embeddings_are_initialised_too(self):
        model, base_tok, new_tok = self._setup()
        subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6)
        expected = (torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.tensor([0.0, 2.0, 0.0, 0.0])) / 2
        assert torch.allclose(model.get_output_embeddings().weight[3], expected)

    def test_is_deterministic(self):
        first, base_tok, new_tok = self._setup()
        subtoken_mean_init(first, base_tok, new_tok, old_vocab_size=3, new_vocab_size=6)
        second, base_tok2, new_tok2 = self._setup()
        subtoken_mean_init(second, base_tok2, new_tok2, old_vocab_size=3, new_vocab_size=6)
        assert torch.equal(
            first.get_input_embeddings().weight, second.get_input_embeddings().weight
        )

    def test_rejects_shrinking_vocab(self):
        model, base_tok, new_tok = self._setup()
        with pytest.raises(ValueError, match="new_vocab_size"):
            subtoken_mean_init(model, base_tok, new_tok, old_vocab_size=3, new_vocab_size=2)


class TestConstants:
    def test_vocab_constants_match_the_experiment(self):
        assert STOCK_VOCAB_SIZE == 49280
        assert VI_VOCAB_SIZE == 57344
        assert VI_VOCAB_SIZE - STOCK_VOCAB_SIZE == 8064
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_merge_stage1_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'merge_stage1_adapter'`

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/merge_stage1_adapter.py
"""Merge a smollm-vi stage-1 LoRA + trainable-token adapter into a standalone
SmolVLM2 backbone directory usable as SmolVLA's `vlm_model_name`.

The adapter stores `trainable_tokens_delta` for `embed_tokens` and `lm_head` --
deltas relative to how the 8,064 new rows were initialised during training. That
initialisation is smollm-vi's sub-token mean (each new token's embedding is the
mean of the embeddings its text decomposes into under the *base* tokenizer; see
smollm-vi vision/smolvlm2/smolvlm/train/train.py:238-249), NOT HuggingFace's
`resize_token_embeddings(mean_resizing=True)` multivariate draw. Reproducing the
same deterministic init here is what makes the merge exact.

See docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

STOCK_VOCAB_SIZE = 49280
VI_VOCAB_SIZE = 57344
EN_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def subtoken_mean_init(
    model, base_tokenizer, new_tokenizer, old_vocab_size: int, new_vocab_size: int
) -> int:
    """Resize `model` to `new_vocab_size`, initialising each new row as the mean of
    the base-vocabulary embeddings its token text decomposes into.

    Rows whose text has no in-range sub-token fall back to the mean of all original
    rows. Mirrors smollm-vi's stage-1 initialisation exactly, and is deterministic.
    """
    if new_vocab_size <= old_vocab_size:
        raise ValueError(
            f"new_vocab_size {new_vocab_size} must exceed old_vocab_size {old_vocab_size}"
        )

    model.resize_token_embeddings(new_vocab_size)

    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings().weight
        in_fallback = in_emb[:old_vocab_size].mean(dim=0)
        out_fallback = out_emb[:old_vocab_size].mean(dim=0)

        for idx in range(old_vocab_size, new_vocab_size):
            text = new_tokenizer.decode([idx])
            sub_ids = [
                s
                for s in base_tokenizer.encode(text, add_special_tokens=False)
                if s < old_vocab_size
            ]
            if sub_ids:
                in_emb[idx] = in_emb[sub_ids].mean(dim=0)
                out_emb[idx] = out_emb[sub_ids].mean(dim=0)
            else:
                in_emb[idx] = in_fallback
                out_emb[idx] = out_fallback

    model.config.vocab_size = new_vocab_size
    return new_vocab_size


def merge_stage1_adapter(base_model_id: str, adapter_dir: Path, tokenizer_dir: Path):
    """Load base -> resize with sub-token mean -> apply adapter -> merge -> return."""
    from peft import PeftModel

    new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    old_vocab = model.get_input_embeddings().weight.shape[0]
    subtoken_mean_init(model, base_tokenizer, new_tokenizer, old_vocab, len(new_tokenizer))

    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.config.vocab_size = len(new_tokenizer)

    processor = AutoProcessor.from_pretrained(tokenizer_dir)
    return merged, processor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=EN_BACKBONE)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path.home()
        / "Repository/smollm-vi/vision/experiments/pretraining/vietnamese/data/tokenizer_vi",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} exists and is non-empty; pass --overwrite")

    print(f"merging {args.adapter_dir} into {args.base_model} ...")
    model, processor = merge_stage1_adapter(
        args.base_model, args.adapter_dir, args.tokenizer_dir
    )

    vocab_size = len(processor.tokenizer)
    n_in = model.get_input_embeddings().weight.shape[0]
    n_out = model.get_output_embeddings().weight.shape[0]
    if n_in != vocab_size or n_out != vocab_size:
        raise ValueError(f"embedding rows {n_in}/{n_out} != tokenizer vocab {vocab_size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    (args.out_dir / "build_metadata.json").write_text(
        json.dumps(
            {
                "kind": "stage1_merged",
                "base_model": args.base_model,
                "adapter_dir": str(args.adapter_dir),
                "tokenizer_dir": str(args.tokenizer_dir),
                "vocab_size": vocab_size,
                "init_method": "subtoken_mean",
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote merged backbone to {args.out_dir} (vocab={vocab_size})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_merge_stage1_adapter.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Verify the merge round-trips against the published checkpoint**

The published `thuanan/SmolVLM2-500M-vi-stage1` was produced from the same adapter. Merging locally must reproduce it closely (bf16 rounding aside).

```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/merge_stage1_adapter.py \
  --adapter-dir ~/Repository/smollm-vi/checkpoints/vietnamese_stage1_3gpu_v2 \
  --out-dir /tmp/d100_local_check
uv run python - <<'PY'
import torch
from transformers import AutoModelForImageTextToText
a = AutoModelForImageTextToText.from_pretrained("/tmp/d100_local_check", dtype=torch.float32)
b = AutoModelForImageTextToText.from_pretrained("thuanan/SmolVLM2-500M-vi-stage1", dtype=torch.float32)
wa = a.get_input_embeddings().weight
wb = b.get_input_embeddings().weight
print("shapes:", wa.shape, wb.shape)
print("max abs diff (new rows):", (wa[49280:] - wb[49280:]).abs().max().item())
print("max abs diff (old rows):", (wa[:49280] - wb[:49280]).abs().max().item())
PY
```
Expected: both shapes `[57344, 960]`; max abs diff below `1e-2` on both slices.

**If the diff is large:** the published checkpoint was built from a different adapter (candidates: `checkpoints/vietnamese_stage1_3gpu`, or a specific `checkpoint-NNNN` subdirectory). Retry against each until one matches, and record the winner in the commit message — every dose backbone must be merged the same way as `d100` or the ladder is not controlled. Do not proceed to Task 6 until one matches.

- [ ] **Step 6: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/merge_stage1_adapter.py \
        vlai-experiments/vi-instructions/tests/test_merge_stage1_adapter.py
git commit -m "feat(vi-instructions): merge stage-1 adapter into standalone backbone

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Held-out Vietnamese corpus for bits-per-character

The ladder's second x-axis is a language-capability measure independent of the robot metric. It must be comparable across a 49,280-token and a 57,344-token vocabulary, so it is reported in **bits per character**, not per token.

The corpus must be text no dose ever trained on. Stage-1 trains on `*_train.json` files; the dev/test splits of the same datasets are genuinely disjoint (`smollm-vi/vision/evaluation/vietnamese/tasks.py:40-60` documents zero image-id overlap for ViOCRVQA's dev split).

**Files:**
- Create: `vlai-experiments/vi-instructions/build_bpc_corpus.py`
- Create (generated, committed): `vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `vi_bpc_corpus.txt` — one Vietnamese sentence per line, UTF-8, no blank lines. Task 5's `validate_backbone.py` reads it.

- [ ] **Step 1: Write the extractor**

```python
# vlai-experiments/vi-instructions/build_bpc_corpus.py
"""Extract a held-out Vietnamese text corpus for bits-per-character measurement.

Sources are the *dev/test* splits of the same datasets stage-1 trains on (stage-1
uses only the `_train.json` files), so no ladder rung has seen this text. Output is
plain UTF-8, one sentence per line, so `validate_backbone.py` can score any
backbone regardless of its tokenizer.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

MIN_CHARS = 20
MAX_LINES_PER_SOURCE = 2000


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def openvivqa_dev_texts() -> list[str]:
    path = hf_hub_download(
        "uitnlp/OpenViVQA-dataset", "vlsp2023_dev_data.json", repo_type="dataset"
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    texts: list[str] = []
    for item in data["annotations"].values():
        for field in ("question", "answer"):
            value = _clean(str(item.get(field, "")))
            if len(value) >= MIN_CHARS:
                texts.append(value)
    return texts


def viocrvqa_dev_texts() -> list[str]:
    zip_path = hf_hub_download(
        "huyhuy123/ViOCRVQA", "data_ViOCRVQA.zip", repo_type="dataset"
    )
    with zipfile.ZipFile(zip_path) as zf:
        data = json.load(zf.open("data/dev.json"))
    texts: list[str] = []
    for item in data["annotations"].values():
        for field in ("question", "answers"):
            value = item.get(field, "")
            values = value if isinstance(value, list) else [value]
            for v in values:
                cleaned = _clean(str(v))
                if len(cleaned) >= MIN_CHARS:
                    texts.append(cleaned)
    return texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt"),
    )
    args = parser.parse_args()

    lines: list[str] = []
    for name, fn in (("openvivqa_dev", openvivqa_dev_texts), ("viocrvqa_dev", viocrvqa_dev_texts)):
        got = fn()[:MAX_LINES_PER_SOURCE]
        print(f"{name}: {len(got)} lines")
        lines.extend(got)

    deduped = list(dict.fromkeys(lines))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(deduped) + "\n", encoding="utf-8")
    print(f"wrote {len(deduped)} lines ({sum(len(x) for x in deduped)} chars) to {args.out}")
```

- [ ] **Step 2: Generate the corpus**

Run:
```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/build_bpc_corpus.py
wc -l vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt
head -3 vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt
```
Expected: 1,000–4,000 lines of Vietnamese text.

**If a source's JSON layout differs from the code above** (e.g. `annotations` is a list, not a dict), inspect the actual structure and adjust the extractor. `smollm-vi/vision/evaluation/vietnamese/tasks.py` contains working loaders for exactly these files — copy its field access.

- [ ] **Step 3: Verify disjointness from the training data**

```bash
cd /home/thuandn/Repository/lerobot
uv run python - <<'PY'
import json
from pathlib import Path

train_dir = Path.home() / "Repository/smollm-vi/vision/experiments/pretraining/vietnamese/data"
train_text = set()
for name in ("openvivqa_train.json", "viocrvqa_train.json", "uitviic_train.json"):
    blob = json.loads((train_dir / name).read_text(encoding="utf-8"))
    train_text.update(
        " ".join(turn.get("value", "").split())
        for item in blob
        for turn in item.get("conversations", [])
    )

corpus = [
    line.strip()
    for line in Path("vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
overlap = [line for line in corpus if line in train_text]
print(f"corpus lines: {len(corpus)}, overlapping with train: {len(overlap)}")
print(f"overlap rate: {len(overlap) / len(corpus):.4%}")
PY
```
Expected: overlap rate below 1%. **If it exceeds 1%**, filter the overlapping lines out in `build_bpc_corpus.py` and regenerate before continuing — a contaminated corpus would flatter the high-dose rungs.

- [ ] **Step 4: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/build_bpc_corpus.py \
        vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt
git commit -m "feat(vi-instructions): held-out Vietnamese corpus for BPC measurement

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Backbone validation gate

A hard gate every rung must pass before a ~10-hour training job is launched. It exists because this project has twice shipped a silently-wrong backbone: a `policy_preprocessor.json` whose `tokenizer_name` still pointed at the 49,280-token English tokenizer while the model's embedding table was 57,344 rows. That does not crash — it produces a confident 0.0% that looks like a scientific finding.

**Files:**
- Create: `vlai-experiments/vi-instructions/validate_backbone.py`
- Test: `vlai-experiments/vi-instructions/tests/test_validate_backbone.py`

**Interfaces:**
- Consumes: `merge_stage1_adapter.STOCK_VOCAB_SIZE`, `VI_VOCAB_SIZE`; the corpus from Task 4.
- Produces:
  - `assert_embeddings_match_vocab(model, vocab_size: int) -> None`
  - `bits_per_character(model, tokenizer, texts: list[str], device: str) -> float`
  - `assert_policy_tokenizer_matches(checkpoint_dir: Path) -> None` — used post-training by the driver.
  - CLI exit code 0 = pass, 1 = fail. Task 7's driver depends on that contract.

- [ ] **Step 1: Write the failing test**

```python
# vlai-experiments/vi-instructions/tests/test_validate_backbone.py
from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn

from validate_backbone import (
    assert_embeddings_match_vocab,
    assert_policy_tokenizer_matches,
    bits_per_character,
)


class TinyLM(nn.Module):
    """Deterministic stand-in: always predicts a uniform distribution, so the
    negative log-likelihood per token is exactly log(vocab_size)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self._in = nn.Embedding(vocab_size, 2)
        self._out = nn.Linear(2, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out

    def forward(self, input_ids, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size)
        return type("Out", (), {"logits": logits})()


class CharTokenizer:
    """One token per character, so tokens-per-character is exactly 1 and the
    expected bits-per-character is analytically known."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [(ord(c) % self.vocab_size) for c in text]
        return {"input_ids": torch.tensor([ids])}


class TestAssertEmbeddingsMatchVocab:
    def test_passes_when_both_match(self):
        assert_embeddings_match_vocab(TinyLM(64), 64)

    def test_raises_when_vocab_differs(self):
        with pytest.raises(ValueError, match="embedding rows"):
            assert_embeddings_match_vocab(TinyLM(64), 57344)


class TestBitsPerCharacter:
    def test_uniform_model_gives_log2_vocab_bits_per_token(self):
        vocab = 64
        model = TinyLM(vocab)
        tokenizer = CharTokenizer(vocab)
        bpc = bits_per_character(model, tokenizer, ["abcdefgh"], device="cpu")
        # 1 token per char, uniform over `vocab` -> log2(vocab) bits per char.
        # The first token has no preceding context and is not scored, so the
        # denominator is the full char count while the numerator covers n-1 tokens.
        assert bpc == pytest.approx(math.log2(vocab) * 7 / 8, rel=1e-3)

    def test_is_deterministic(self):
        model, tokenizer = TinyLM(64), CharTokenizer(64)
        texts = ["xin chào các bạn", "hôm nay trời đẹp quá"]
        assert bits_per_character(model, tokenizer, texts, device="cpu") == bits_per_character(
            model, tokenizer, texts, device="cpu"
        )

    def test_rejects_empty_corpus(self):
        with pytest.raises(ValueError, match="empty"):
            bits_per_character(TinyLM(64), CharTokenizer(64), [], device="cpu")

    def test_skips_lines_too_short_to_score(self):
        model, tokenizer = TinyLM(64), CharTokenizer(64)
        # A 1-char line yields a single token with no context; it must not divide by zero.
        assert bits_per_character(model, tokenizer, ["a", "abcdefgh"], device="cpu") > 0


class TestAssertPolicyTokenizerMatches:
    def test_passes_when_names_agree(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"vlm_model_name": "thuanan/SmolVLM2-500M-vi-stage1"})
        )
        (tmp_path / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": "tokenizer_processor",
                            "config": {"tokenizer_name": "thuanan/SmolVLM2-500M-vi-stage1"},
                        }
                    ]
                }
            )
        )
        assert_policy_tokenizer_matches(tmp_path)

    def test_raises_on_the_stale_tokenizer_bug(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"vlm_model_name": "thuanan/SmolVLM2-500M-vi-stage1"})
        )
        (tmp_path / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": "tokenizer_processor",
                            "config": {
                                "tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                            },
                        }
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="tokenizer_name"):
            assert_policy_tokenizer_matches(tmp_path)

    def test_raises_when_preprocessor_file_is_missing(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"vlm_model_name": "x"}))
        with pytest.raises(FileNotFoundError):
            assert_policy_tokenizer_matches(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_validate_backbone.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'validate_backbone'`

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/validate_backbone.py
"""Hard gate run before any ladder training job, and after any ladder checkpoint.

Checks, in order:
  1. input and output embedding rows both equal the tokenizer's vocab size;
  2. a Vietnamese LIBERO instruction round-trips into in-range token ids;
  3. bits-per-character on the held-out Vietnamese corpus (Task 4) -- the
     ladder's language-capability x-axis. Per *character*, not per token, so a
     49,280-token and a 57,344-token vocabulary are directly comparable.

Exit code 0 = pass, 1 = fail. ladder_driver.sh refuses to launch on non-zero.

Motivation: twice in this project a checkpoint shipped with a
`policy_preprocessor.json` whose `tokenizer_name` still pointed at the English
tokenizer while the embedding table was the Vietnamese one. Token ids get looked
up against the wrong rows -- silently wrong, no crash, a confident 0.0%.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

SANITY_INSTRUCTION = "nhấc cái bát đen trên hộp bánh quy lên rồi đặt nó lên đĩa"
DEFAULT_CORPUS = Path(__file__).parent / "data" / "vi_bpc_corpus.txt"


def assert_embeddings_match_vocab(model, vocab_size: int) -> None:
    n_in = model.get_input_embeddings().weight.shape[0]
    n_out = model.get_output_embeddings().weight.shape[0]
    if n_in != vocab_size or n_out != vocab_size:
        raise ValueError(f"embedding rows {n_in}/{n_out} != tokenizer vocab {vocab_size}")


def assert_ids_in_vocab(token_ids: list[int], vocab_size: int) -> None:
    out_of_range = [i for i in token_ids if i < 0 or i >= vocab_size]
    if out_of_range:
        raise ValueError(f"token ids out of vocab range [0, {vocab_size}): {out_of_range}")


@torch.no_grad()
def bits_per_character(model, tokenizer, texts: list[str], device: str = "cpu") -> float:
    """Mean bits per character over `texts` under `model`.

    The first token of each line has no preceding context and is not scored, but
    every character still counts toward the denominator -- consistent across
    backbones, so cross-vocabulary comparison stays fair.
    """
    if not texts:
        raise ValueError("corpus is empty")

    total_nats = 0.0
    total_chars = 0
    for text in texts:
        if not text.strip():
            continue
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        total_chars += len(text)
        if ids.shape[1] < 2:
            continue
        logits = model(input_ids=ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = ids[:, 1:]
        total_nats += -log_probs.gather(2, targets.unsqueeze(-1)).sum().item()

    if total_chars == 0:
        raise ValueError("corpus is empty after filtering blank lines")
    return total_nats / total_chars / math.log(2)


def assert_policy_tokenizer_matches(checkpoint_dir: Path) -> None:
    """Post-training gate: the trained policy's preprocessor must reference the same
    tokenizer as the backbone it was built on.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    preprocessor_path = checkpoint_dir / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"missing {preprocessor_path}")

    vlm_model_name = json.loads(config_path.read_text())["vlm_model_name"]
    preprocessor = json.loads(preprocessor_path.read_text())

    tokenizer_names = [
        step.get("config", {}).get("tokenizer_name")
        for step in preprocessor.get("steps", [])
        if "tokenizer_name" in step.get("config", {})
    ]
    if not tokenizer_names:
        raise ValueError(f"no tokenizer_name found in {preprocessor_path}")
    for name in tokenizer_names:
        if name != vlm_model_name:
            raise ValueError(
                f"stale tokenizer_name {name!r} != vlm_model_name {vlm_model_name!r} "
                f"in {preprocessor_path}"
            )


if __name__ == "__main__":
    import argparse
    import sys

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", help="HF id or local directory")
    parser.add_argument("--expected-vocab", type=int, required=True, help="49280 or 57344")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument(
        "--report", type=Path, default=None, help="optional JSON path to write results to"
    )
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        if len(tokenizer) != args.expected_vocab:
            raise ValueError(f"tokenizer vocab {len(tokenizer)} != expected {args.expected_vocab}")

        model = AutoModelForImageTextToText.from_pretrained(
            args.backbone, dtype=torch.float32, low_cpu_mem_usage=True
        ).to(args.device)
        model.eval()

        assert_embeddings_match_vocab(model, args.expected_vocab)
        assert_ids_in_vocab(tokenizer(SANITY_INSTRUCTION)["input_ids"], args.expected_vocab)

        texts = [
            line for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()
        ][: args.max_lines]
        # The VLM wrapper exposes the text tower's LM head; score through it.
        bpc = bits_per_character(model, tokenizer, texts, device=args.device)

        result = {
            "backbone": args.backbone,
            "vocab_size": args.expected_vocab,
            "bits_per_character": round(bpc, 4),
            "corpus_lines": len(texts),
            "status": "pass",
        }
        print(json.dumps(result, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 -- the CLI's contract is its exit code
        print(json.dumps({"backbone": args.backbone, "status": "fail", "error": str(exc)}, indent=2))
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_validate_backbone.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the gate against the two backbones that already exist**

```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/validate_backbone.py \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct --expected-vocab 49280 --device cpu --max-lines 50
uv run python vlai-experiments/vi-instructions/validate_backbone.py \
  thuanan/SmolVLM2-500M-vi-stage1 --expected-vocab 57344 --device cpu --max-lines 50
```
Expected: both exit 0. `thuanan/SmolVLM2-500M-vi-stage1` must show a **lower** bits-per-character than the stock English backbone — if it does not, the BPC computation is reaching the wrong head and must be fixed before it is used as an axis.

**If `model(input_ids=...)` fails** because the VLM wrapper requires image inputs, call the text tower directly: `model.model.text_model` for hidden states plus `model.lm_head`, or use `AutoModelForCausalLM` on the same directory. Adjust `bits_per_character`'s forward call and re-run the unit tests.

- [ ] **Step 6: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/validate_backbone.py \
        vlai-experiments/vi-instructions/tests/test_validate_backbone.py
git commit -m "feat(vi-instructions): backbone validation gate with bits-per-character

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Build the `d0` dose-zero anchor

The existing Arm C backbone (`outputs/backbones/smolvlm2_vi_vocab_only/`) initialises the 8,064 new rows with HuggingFace's `resize_token_embeddings(mean_resizing=True)` — a draw from a multivariate normal fitted to the English rows. Stage-1 instead uses the deterministic sub-token mean. Two different initialisations means the existing Arm C backbone is **not** the dose-0 point of this ladder's curve, and the first segment of the curve would confound dose with initialisation method.

This task builds the true anchor. The existing Arm C backbone is left untouched — it stays the reference for the published Arm C result.

**Files:**
- Create: `outputs/backbones/vi_dose_0/` (artifact, not committed)

**Interfaces:**
- Consumes: `merge_stage1_adapter.subtoken_mean_init`, `EN_BACKBONE`, `STOCK_VOCAB_SIZE`, `VI_VOCAB_SIZE`.
- Produces: `outputs/backbones/vi_dose_0/` — the `d0` rung.

- [ ] **Step 1: Build it**

```bash
cd /home/thuandn/Repository/lerobot
uv run python - <<'PY'
import json, sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, "vlai-experiments/vi-instructions")
from merge_stage1_adapter import EN_BACKBONE, STOCK_VOCAB_SIZE, VI_VOCAB_SIZE, subtoken_mean_init

TOKENIZER_DIR = Path.home() / "Repository/smollm-vi/vision/experiments/pretraining/vietnamese/data/tokenizer_vi"
out = Path("outputs/backbones/vi_dose_0")

new_tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
base_tok = AutoTokenizer.from_pretrained(EN_BACKBONE)
assert len(new_tok) == VI_VOCAB_SIZE, len(new_tok)

model = AutoModelForImageTextToText.from_pretrained(EN_BACKBONE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
subtoken_mean_init(model, base_tok, new_tok, STOCK_VOCAB_SIZE, VI_VOCAB_SIZE)

out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out)
AutoProcessor.from_pretrained(TOKENIZER_DIR).save_pretrained(out)
(out / "build_metadata.json").write_text(json.dumps({
    "kind": "dose_zero_anchor",
    "base_model": EN_BACKBONE,
    "tokenizer_dir": str(TOKENIZER_DIR),
    "vocab_size": VI_VOCAB_SIZE,
    "init_method": "subtoken_mean",
    "dose": 0.0,
    "note": "Dose-0 rung of the language-cliff ladder. Distinct from "
            "outputs/backbones/smolvlm2_vi_vocab_only (HF mean_resizing draw, Arm C).",
    "created_at": datetime.now(UTC).isoformat(),
}, indent=2) + "\n")
print("wrote", out)
PY
```

- [ ] **Step 2: Validate it**

Run:
```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/validate_backbone.py \
  outputs/backbones/vi_dose_0 --expected-vocab 57344 --device cpu --max-lines 50 \
  --report outputs/backbones/vi_dose_0/validation.json
```
Expected: exit 0. Its bits-per-character should sit **between** stock English and `thuanan/SmolVLM2-500M-vi-stage1` — better than stock (the Vietnamese tokenizer fragments less) but worse than the trained backbone.

- [ ] **Step 3: Confirm it differs from the Arm C backbone**

```bash
cd /home/thuandn/Repository/lerobot
uv run python - <<'PY'
import torch
from transformers import AutoModelForImageTextToText
a = AutoModelForImageTextToText.from_pretrained("outputs/backbones/vi_dose_0", dtype=torch.float32)
b = AutoModelForImageTextToText.from_pretrained("outputs/backbones/smolvlm2_vi_vocab_only", dtype=torch.float32)
wa, wb = a.get_input_embeddings().weight, b.get_input_embeddings().weight
print("old rows identical:", torch.allclose(wa[:49280], wb[:49280], atol=1e-3))
print("new rows max diff:", (wa[49280:] - wb[49280:]).abs().max().item())
PY
```
Expected: old rows identical `True`; new rows max diff clearly non-zero — confirming the two initialisations differ, which is exactly why this anchor was needed.

- [ ] **Step 4: Commit the note (artifacts are gitignored)**

```bash
cd /home/thuandn/Repository/lerobot
git status --short outputs/ | head
```
If `outputs/` is gitignored, nothing to commit — record the build in the Task 7 commit message instead.

---

### Task 7: Build the `d10`, `d25`, `d50` dose backbones

Each dose is one stage-1 run in `smollm-vi` followed by one merge in `lerobot`. Needs free GPUs.

**Files:**
- Create: `outputs/backbones/vi_dose_{10,25,50}/` (artifacts)
- Create: `outputs/backbones/mixtures/dose_{10,25,50}/mixture_dose.yaml` + `dose_manifest.json`

**Interfaces:**
- Consumes: `dose_mixture.py` CLI (Task 1), `train_gpus.sh` overrides (Task 2), `merge_stage1_adapter.py` CLI (Task 3), `validate_backbone.py` CLI (Task 5).
- Produces: three validated backbone directories, consumed by Task 9.

- [ ] **Step 1: Verify GPUs are free**

Run: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
Expected: no processes. If the `ft_multiseed_pipeline` run is still going, **stop here** and wait — do not preempt it, its seeds are needed for the published Arm A/B/C statistics.

- [ ] **Step 2: Generate the three dose mixtures**

```bash
cd /home/thuandn/Repository/lerobot
for pct in 10 25 50; do
  uv run python vlai-experiments/vi-instructions/dose_mixture.py \
    --dose "0.$(printf '%02d' "$pct")" \
    --out-dir "outputs/backbones/mixtures/dose_${pct}"
done
grep -h sampling_strategy outputs/backbones/mixtures/dose_*/mixture_dose.yaml
```
Expected: 21 lines (7 sources × 3 doses), each strictly smaller as the dose drops.

Note: `0.50` is written by the loop as `0.50`; confirm `dose_manifest.json` records `0.5`, `0.25`, `0.1` respectively.

- [ ] **Step 3: Run stage-1 at each dose (sequentially, ~1–3h each)**

```bash
cd /home/thuandn/Repository/smollm-vi
for pct in 10 25 50; do
  MIXTURE_TEMPLATE="/home/thuandn/Repository/lerobot/outputs/backbones/mixtures/dose_${pct}/mixture_dose.yaml" \
  OUTPUT_DIR="/home/thuandn/Repository/smollm-vi/checkpoints/vietnamese_stage1_dose_${pct}" \
  RUN_NAME="vietnamese_stage1_dose_${pct}" \
    ./vision/experiments/pretraining/vietnamese/train_gpus.sh \
    2>&1 | tee "/tmp/stage1_dose_${pct}.log"
done
```
Expected: each run completes with `Done. Checkpoints in: ...`. Total steps should scale roughly with dose (full run = 1436 steps, so expect ≈144 / ≈359 / ≈718).

- [ ] **Step 4: Confirm the step counts scale with dose**

```bash
for pct in 10 25 50; do
  echo -n "dose_${pct}: "
  uv run python -c "
import json,sys
print(json.load(open('/home/thuandn/Repository/smollm-vi/checkpoints/vietnamese_stage1_dose_${pct}/trainer_state.json'))['global_step'])
"
done
```
Expected: monotonically increasing, roughly 10%/25%/50% of 1436. **If they are equal or unordered**, the `MIXTURE_TEMPLATE` override did not take effect — check `checkpoints/vietnamese_stage1_dose_*/mixture_resolved.yaml` and fix before continuing.

- [ ] **Step 5: Merge each adapter into a standalone backbone**

```bash
cd /home/thuandn/Repository/lerobot
for pct in 10 25 50; do
  uv run python vlai-experiments/vi-instructions/merge_stage1_adapter.py \
    --adapter-dir "/home/thuandn/Repository/smollm-vi/checkpoints/vietnamese_stage1_dose_${pct}" \
    --out-dir "outputs/backbones/vi_dose_${pct}"
done
```
Expected: three directories, each reporting `vocab=57344`.

- [ ] **Step 6: Validate all six Phase 1 rungs**

```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/validate_backbone.py \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct --expected-vocab 49280 \
  --report outputs/backbones/validation_stock.json
for name in vi_dose_0 vi_dose_10 vi_dose_25 vi_dose_50; do
  uv run python vlai-experiments/vi-instructions/validate_backbone.py \
    "outputs/backbones/${name}" --expected-vocab 57344 \
    --report "outputs/backbones/${name}/validation.json"
done
uv run python vlai-experiments/vi-instructions/validate_backbone.py \
  thuanan/SmolVLM2-500M-vi-stage1 --expected-vocab 57344 \
  --report outputs/backbones/validation_d100.json
```
Expected: all six exit 0, and **bits-per-character decreases monotonically** across `d0 → d10 → d25 → d50 → d100`.

**If BPC is not monotonic**, do not proceed — it means either a dose run failed to train properly or the merge picked up the wrong adapter. Investigate before spending ~84 GPU-hours on Task 9.

- [ ] **Step 7: Commit the dose manifests**

```bash
cd /home/thuandn/Repository/lerobot
git add -f outputs/backbones/mixtures/dose_*/dose_manifest.json \
           outputs/backbones/mixtures/dose_*/mixture_dose.yaml
git commit -m "chore(vi-instructions): record dose-scaled stage-1 mixtures and manifests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Ladder driver

A queue that fills whatever GPUs are free, survives SSH loss, and gates every job on backbone validation.

**Files:**
- Create: `vlai-experiments/vi-instructions/ladder_driver.sh`

**Interfaces:**
- Consumes: `validate_backbone.py` CLI exit code; `run_vi.sh`, `run_eval_en.sh`, `run_eval_vi.sh` at the repo root.
- Produces: `outputs/train_ladder_<point>_<lang>[_seed<N>]/` and `outputs/eval_ladder_<point>_<lang>[_seed<N>]_{en,vi}/`, plus `outputs/ladder_pipeline/driver.log`. Task 10 reads these.

- [ ] **Step 1: Write the driver**

```bash
#!/usr/bin/env bash
# Backbone language-cliff ladder driver: a GPU-count-agnostic job queue.
#
# Modelled on outputs/ft_multiseed_pipeline/driver.sh, with three changes:
#   * every job is gated on validate_backbone.py before a GPU is committed;
#   * both an EN and a VI eval run per checkpoint;
#   * the job table is generated from a phase argument rather than hardcoded.
#
# Usage:
#   ./vlai-experiments/vi-instructions/ladder_driver.sh <phase> [seed]
#     phase: 1a | 1b | 2 | vi
#     seed:  default 1000
#
# Launch detached so it survives SSH loss:
#   setsid nohup ./vlai-experiments/vi-instructions/ladder_driver.sh 1a \
#     > outputs/ladder_pipeline/driver.stdout 2>&1 &
set -uo pipefail

REPO="/home/thuandn/Repository/lerobot"
PHASE="${1:?Usage: ladder_driver.sh <1a|1b|2|vi> [seed]}"
SEED="${2:-1000}"
OUT="${REPO}/outputs/ladder_pipeline"
LOG="${OUT}/driver_phase${PHASE}_seed${SEED}.log"
STEPS=50000
BATCH_SIZE=16
NEED_MIB=9000
EN_DATASET="HuggingFaceVLA/libero"
VI_DATASET="VLAIResearchLab/lerobot_libero_vi"

mkdir -p "${OUT}"
cd "${REPO}" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
ckpt_ok() { [[ -f "$1/config.json" && -f "$1/model.safetensors" ]]; }
errored() { grep -qiE "Traceback|CUDA out of memory|Error:|Exception" "$1" 2>/dev/null; }

# point|backbone|expected_vocab
BACKBONES=(
  "stock|HuggingFaceTB/SmolVLM2-500M-Video-Instruct|49280"
  "d0|${REPO}/outputs/backbones/vi_dose_0|57344"
  "d10|${REPO}/outputs/backbones/vi_dose_10|57344"
  "d25|${REPO}/outputs/backbones/vi_dose_25|57344"
  "d50|${REPO}/outputs/backbones/vi_dose_50|57344"
  "d100|thuanan/SmolVLM2-500M-vi-stage1|57344"
  "s256m|HuggingFaceTB/SmolVLM2-256M-Video-Instruct|49280"
  "s2200m|HuggingFaceTB/SmolVLM2-2.2B-Instruct|49280"
)

backbone_for() {
  local want="$1"
  for entry in "${BACKBONES[@]}"; do
    IFS='|' read -r point path vocab <<< "${entry}"
    if [[ "${point}" == "${want}" ]]; then echo "${path}|${vocab}"; return 0; fi
  done
  return 1
}

case "${PHASE}" in
  1a|1b) POINTS=(stock d0 d10 d25 d50 d100); LANG=en; DATASET="${EN_DATASET}" ;;
  2)     POINTS=(s256m s2200m);              LANG=en; DATASET="${EN_DATASET}" ;;
  vi)    POINTS=(d10 d25 d50);               LANG=vi; DATASET="${VI_DATASET}" ;;
  *)     echo "unknown phase: ${PHASE}" >&2; exit 1 ;;
esac

SUFFIX=""
[[ "${SEED}" != "1000" ]] && SUFFIX="_seed${SEED}"

log "=== ladder driver: phase=${PHASE} seed=${SEED} points=${POINTS[*]} lang=${LANG} ==="

# --- Gate: validate every backbone before committing any GPU time ------------
for point in "${POINTS[@]}"; do
  IFS='|' read -r path vocab <<< "$(backbone_for "${point}")"
  log "validating ${point} (${path}, vocab=${vocab})"
  if ! uv run python vlai-experiments/vi-instructions/validate_backbone.py \
        "${path}" --expected-vocab "${vocab}" --max-lines 200 \
        --report "${OUT}/validate_${point}.json" >> "${LOG}" 2>&1; then
    log "FATAL: ${point} failed validation -- aborting before any training"
    exit 1
  fi
done
log "all ${#POINTS[@]} backbones validated"

# --- Queue ------------------------------------------------------------------
free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=', ' read -r idx used total; do
    if (( total - used > NEED_MIB )); then echo "${idx}"; return 0; fi
  done
}

declare -A GPU_OF_JOB
next=0
running=0

while (( next < ${#POINTS[@]} )) || (( running > 0 )); do
  # Launch as many queued jobs as there are free GPUs.
  while (( next < ${#POINTS[@]} )); do
    gpu="$(free_gpu | head -1)"
    [[ -z "${gpu}" ]] && break

    point="${POINTS[$next]}"
    IFS='|' read -r path vocab <<< "$(backbone_for "${point}")"
    tag="ladder_${point}_${LANG}${SUFFIX}"
    train_dir="${REPO}/outputs/train_${tag}"
    train_log="${OUT}/train_${tag}.log"

    log "launch ${tag} on GPU ${gpu} (backbone=${path})"
    env CUDA_VISIBLE_DEVICES="${gpu}" SKIP_SYNC=1 \
        VLM_MODEL="${path}" DATASET_REPO="${DATASET}" \
        RUN_TAG="${tag}" OUTPUT_DIR="${train_dir}/" SEED="${SEED}" \
        STEPS="${STEPS}" BATCH_SIZE="${BATCH_SIZE}" TRAIN_EXPERT_ONLY=false \
        SAVE_FREQ=10000 ENV_EVAL_FREQ=10000 LOG_FREQ=250 \
        ./run_vi.sh > "${train_log}" 2>&1 &
    GPU_OF_JOB["${tag}"]="${gpu}:$!"
    next=$((next + 1))
    running=$((running + 1))
    sleep 120   # let the job claim its VRAM before free_gpu is polled again
  done

  sleep 300

  # Reap finished jobs and run their evals.
  for tag in "${!GPU_OF_JOB[@]}"; do
    IFS=':' read -r gpu pid <<< "${GPU_OF_JOB[$tag]}"
    kill -0 "${pid}" 2>/dev/null && continue

    unset "GPU_OF_JOB[$tag]"
    running=$((running - 1))
    train_log="${OUT}/train_${tag}.log"
    # Explicit step directory, never checkpoints/last -- a symlink race once made
    # an earlier experiment read a 10k checkpoint while training ran to 50k.
    ckpt="${REPO}/outputs/train_${tag}/checkpoints/$(printf '%06d' "${STEPS}")/pretrained_model"

    if ! ckpt_ok "${ckpt}"; then
      log "FAILED ${tag}: no final checkpoint at ${ckpt}"
      errored "${train_log}" && log "  (error found in ${train_log})"
      continue
    fi

    if ! uv run python vlai-experiments/vi-instructions/validate_backbone.py \
          --help > /dev/null 2>&1; then :; fi
    if ! uv run python -c "
import sys; sys.path.insert(0, 'vlai-experiments/vi-instructions')
from validate_backbone import assert_policy_tokenizer_matches
assert_policy_tokenizer_matches('${ckpt}')
" >> "${LOG}" 2>&1; then
      log "FAILED ${tag}: stale tokenizer_name in policy_preprocessor.json -- skipping eval"
      continue
    fi

    for eval_lang in en vi; do
      eval_dir="${REPO}/outputs/eval_${tag}_${eval_lang}"
      log "eval ${tag} (${eval_lang}) on GPU ${gpu}"
      env CUDA_VISIBLE_DEVICES="${gpu}" \
        "./run_eval_${eval_lang}.sh" "${ckpt}" "${eval_dir}" \
        > "${OUT}/eval_${tag}_${eval_lang}.log" 2>&1 ||
        log "  eval ${eval_lang} FAILED for ${tag}"
    done
    log "DONE ${tag}"
  done
done

log "=== phase ${PHASE} seed ${SEED} complete ==="
```

- [ ] **Step 2: Verify syntax and the validation gate**

Run:
```bash
cd /home/thuandn/Repository/lerobot
chmod +x vlai-experiments/vi-instructions/ladder_driver.sh
bash -n vlai-experiments/vi-instructions/ladder_driver.sh && echo "syntax OK"
```
Expected: `syntax OK`

- [ ] **Step 3: Verify the gate aborts on an invalid backbone**

```bash
cd /home/thuandn/Repository/lerobot
mkdir -p /tmp/bogus_backbone
sed -i 's#d0|${REPO}/outputs/backbones/vi_dose_0#d0|/tmp/bogus_backbone#' \
  /tmp/ladder_driver_test.sh 2>/dev/null || \
  sed 's#\${REPO}/outputs/backbones/vi_dose_0#/tmp/bogus_backbone#' \
    vlai-experiments/vi-instructions/ladder_driver.sh > /tmp/ladder_driver_test.sh
bash /tmp/ladder_driver_test.sh 1a; echo "exit=$?"
```
Expected: exits non-zero with `FATAL: d0 failed validation -- aborting before any training`, and **no** `outputs/train_ladder_*` directory is created.

- [ ] **Step 4: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/ladder_driver.sh
git commit -m "feat(vi-instructions): GPU-agnostic ladder driver with validation gate

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Ladder report

**Files:**
- Create: `vlai-experiments/vi-instructions/ladder_report.py`
- Test: `vlai-experiments/vi-instructions/tests/test_ladder_report.py`

**Interfaces:**
- Consumes: `eval_info.json` files produced by Task 8; `validate_*.json` files from the driver's gate.
- Produces:
  - `LADDER_ALIASES: dict[str, dict[str, str]]` — maps historical Arm A/B/C runs into ladder cells.
  - `collect_runs(outputs_dir: Path) -> dict[str, dict]`
  - `build_ladder_rows(runs: dict) -> list[dict]`
  - `paired_bootstrap(a: list[float], b: list[float], n_resamples: int, seed: int) -> tuple[float, float, float]` — returns `(mean_diff, ci_low, ci_high)`.
  - `render_report(rows, bpc, missing) -> str`

- [ ] **Step 1: Write the failing test**

```python
# vlai-experiments/vi-instructions/tests/test_ladder_report.py
from __future__ import annotations

import json

import pytest

from ladder_report import (
    LADDER_ALIASES,
    build_ladder_rows,
    collect_runs,
    paired_bootstrap,
    render_report,
)


def _write_eval(path, per_group, overall):
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_info.json").write_text(
        json.dumps({"per_group": per_group, "overall": {"pc_success": overall}})
    )


class TestCollectRuns:
    def test_finds_ladder_eval_dirs(self, tmp_path):
        _write_eval(tmp_path / "eval_ladder_d10_en_vi", {"libero_10": {"pc_success": 3.0}}, 3.0)
        _write_eval(tmp_path / "eval_ladder_d10_en_en", {"libero_10": {"pc_success": 60.0}}, 60.0)
        runs = collect_runs(tmp_path)
        assert set(runs) == {"d10_en_vi", "d10_en_en"}
        assert runs["d10_en_en"]["overall"]["pc_success"] == 60.0

    def test_ignores_unrelated_directories(self, tmp_path):
        _write_eval(tmp_path / "eval_vi_smolvla", {"libero_10": {"pc_success": 1.0}}, 1.0)
        assert collect_runs(tmp_path) == {}

    def test_skips_dirs_without_eval_info(self, tmp_path):
        (tmp_path / "eval_ladder_d25_en_vi").mkdir(parents=True)
        assert collect_runs(tmp_path) == {}


class TestBuildLadderRows:
    def test_one_row_per_point_with_en_vi_and_gap(self):
        runs = {
            "d10_en_en": {"overall": {"pc_success": 60.0}, "per_group": {}},
            "d10_en_vi": {"overall": {"pc_success": 4.0}, "per_group": {}},
        }
        rows = build_ladder_rows(runs)
        assert len(rows) == 1
        assert rows[0]["point"] == "d10"
        assert rows[0]["en"] == 60.0
        assert rows[0]["vi"] == 4.0
        assert rows[0]["gap"] == 56.0

    def test_orders_points_by_ladder_position_not_alphabetically(self):
        runs = {
            f"{p}_en_{lang}": {"overall": {"pc_success": 1.0}, "per_group": {}}
            for p in ("d100", "stock", "d10")
            for lang in ("en", "vi")
        }
        assert [r["point"] for r in build_ladder_rows(runs)] == ["stock", "d10", "d100"]

    def test_missing_leg_yields_none_not_a_crash(self):
        rows = build_ladder_rows({"d50_en_en": {"overall": {"pc_success": 55.0}, "per_group": {}}})
        assert rows[0]["vi"] is None
        assert rows[0]["gap"] is None


class TestPairedBootstrap:
    def test_identical_inputs_give_zero_mean_difference(self):
        values = [10.0, 20.0, 30.0, 40.0]
        mean, low, high = paired_bootstrap(values, values, n_resamples=200, seed=0)
        assert mean == 0.0
        assert low == 0.0 and high == 0.0

    def test_constant_offset_is_recovered(self):
        a = [10.0, 20.0, 30.0, 40.0]
        b = [15.0, 25.0, 35.0, 45.0]
        mean, low, high = paired_bootstrap(a, b, n_resamples=500, seed=0)
        assert mean == pytest.approx(-5.0)
        assert low <= -5.0 <= high

    def test_is_deterministic_for_a_fixed_seed(self):
        a, b = [1.0, 5.0, 9.0, 2.0], [3.0, 4.0, 8.0, 7.0]
        assert paired_bootstrap(a, b, 300, seed=7) == paired_bootstrap(a, b, 300, seed=7)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap([1.0, 2.0], [1.0], 100, seed=0)


class TestRenderReport:
    def test_lists_missing_cells_explicitly(self):
        rows = [{"point": "d10", "en": 60.0, "vi": None, "gap": None}]
        text = render_report(rows, bpc={}, missing=["d10_en_vi"])
        assert "d10_en_vi" in text
        assert "Missing" in text or "thiếu" in text.lower()

    def test_includes_bpc_column_when_available(self):
        rows = [{"point": "d10", "en": 60.0, "vi": 4.0, "gap": 56.0}]
        text = render_report(rows, bpc={"d10": 2.31}, missing=[])
        assert "2.31" in text


class TestAliases:
    def test_historical_arms_map_to_vi_train_cells(self):
        assert LADDER_ALIASES["stock_vi_vi"]["dir"] == "eval_vi_ft_smolvla"
        assert LADDER_ALIASES["d100_vi_vi"]["dir"] == "eval_vi_ft_vi"
        assert LADDER_ALIASES["d0_vi_vi"]["dir"] == "eval_vi_vocab_only_ft"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_ladder_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ladder_report'`

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/ladder_report.py
"""Aggregate backbone language-cliff ladder evals into one report.

Reads outputs/eval_ladder_<point>_<train_lang>_<eval_lang>/eval_info.json plus the
driver's per-backbone validate_*.json (for bits-per-character), and emits the
ladder table, the EN-VI gap per rung, and paired bootstrap confidence intervals
over the 40 LIBERO tasks.

Historical Arm A/B/C runs predate this naming scheme; LADDER_ALIASES maps them
into the VI-train cells read-only. No existing outputs/ directory is renamed.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

# Ladder order is semantic (increasing Vietnamese dose, then increasing scale),
# never alphabetical -- the curve is meaningless in any other order.
LADDER_ORDER = ("stock", "d0", "d10", "d25", "d50", "d100", "s256m", "s2200m")

LADDER_ALIASES: dict[str, dict[str, str]] = {
    "stock_vi_vi": {"dir": "eval_vi_ft_smolvla", "note": "Arm A, full-FT 50k, seed 1000"},
    "d100_vi_vi": {"dir": "eval_vi_ft_vi", "note": "Arm B, full-FT 50k, seed 1000"},
    "d0_vi_vi": {
        "dir": "eval_vi_vocab_only_ft",
        "note": "Arm C, full-FT 50k, seed 1000 -- NOTE: HF mean_resizing init, "
        "not the sub-token-mean init used by the d0 ladder anchor",
    },
}


def collect_runs(outputs_dir: Path) -> dict[str, dict]:
    """Map '<point>_<train_lang>_<eval_lang>' -> parsed eval_info.json."""
    runs: dict[str, dict] = {}
    for path in sorted(Path(outputs_dir).glob("eval_ladder_*")):
        info = path / "eval_info.json"
        if not info.is_file():
            continue
        runs[path.name.removeprefix("eval_ladder_")] = json.loads(info.read_text())
    return runs


def _point_of(key: str) -> str:
    return key.split("_")[0]


def build_ladder_rows(runs: dict[str, dict]) -> list[dict]:
    """One row per ladder point: EN score, VI score, and the EN-VI gap."""
    points = {_point_of(k) for k in runs}
    ordered = [p for p in LADDER_ORDER if p in points]
    ordered += sorted(points - set(LADDER_ORDER))

    rows: list[dict] = []
    for point in ordered:
        en = runs.get(f"{point}_en_en", {}).get("overall", {}).get("pc_success")
        vi = runs.get(f"{point}_en_vi", {}).get("overall", {}).get("pc_success")
        rows.append(
            {
                "point": point,
                "en": en,
                "vi": vi,
                "gap": None if en is None or vi is None else en - vi,
            }
        )
    return rows


def paired_bootstrap(
    a: list[float], b: list[float], n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap over per-task scores. Returns (mean_diff, ci_low, ci_high)
    for a - b at the 95% level."""
    if len(a) != len(b):
        raise ValueError(f"a and b must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("cannot bootstrap an empty sample")

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(n_resamples):
        means.append(statistics.fmean(rng.choices(diffs, k=n)))
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(int(0.975 * n_resamples), n_resamples - 1)]
    return statistics.fmean(diffs), lo, hi


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def render_report(rows: list[dict], bpc: dict[str, float], missing: list[str]) -> str:
    lines = [
        "# Backbone language-cliff ladder",
        "",
        "EN-train / eval EN and VI (zero-shot cross-lingual). "
        "`gap` = EN − VI, comparable to arXiv:2606.11906's Vietnamese numbers "
        "(OpenVLA-OFT 59.6, π₀.₅ 57.3).",
        "",
        "| point | bits/char (VI) | EN | VI | gap |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        bpc_cell = f"{bpc[row['point']]:.2f}" if row["point"] in bpc else "—"
        lines.append(
            f"| `{row['point']}` | {bpc_cell} | {_fmt(row['en'])} | "
            f"{_fmt(row['vi'])} | {_fmt(row['gap'])} |"
        )
    if missing:
        lines += ["", "## Missing cells", ""]
        lines += [f"- `{name}`" for name in missing]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/report_backbone_ladder.md"))
    args = parser.parse_args()

    runs = collect_runs(args.outputs_dir)
    rows = build_ladder_rows(runs)

    bpc: dict[str, float] = {}
    for report in sorted((args.outputs_dir / "ladder_pipeline").glob("validate_*.json")):
        data = json.loads(report.read_text())
        if data.get("status") == "pass":
            bpc[report.stem.removeprefix("validate_")] = data["bits_per_character"]

    expected = [f"{r['point']}_en_{lang}" for r in rows for lang in ("en", "vi")]
    missing = [key for key in expected if key not in runs]

    args.out.write_text(render_report(rows, bpc, missing))
    print(args.out.read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/test_ladder_report.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the whole new test suite**

Run: `cd /home/thuandn/Repository/lerobot && uv run pytest vlai-experiments/vi-instructions/tests/ -v`
Expected: PASS — the four new test files plus the seven pre-existing ones.

- [ ] **Step 6: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/ladder_report.py \
        vlai-experiments/vi-instructions/tests/test_ladder_report.py
git commit -m "feat(vi-instructions): ladder report with paired bootstrap

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Phase 1a — run the dose ladder (~84 GPU-h)

**Files:** produces `outputs/train_ladder_{stock,d0,d10,d25,d50,d100}_en/` and matching `eval_*_{en,vi}/`.

**Interfaces:**
- Consumes: Task 7's backbones, Task 8's driver.
- Produces: the six EN-train rungs consumed by Task 9's report.

- [ ] **Step 1: Confirm GPUs are free and launch detached**

```bash
cd /home/thuandn/Repository/lerobot
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
mkdir -p outputs/ladder_pipeline
uv sync --locked --extra smolvla --extra libero
setsid nohup ./vlai-experiments/vi-instructions/ladder_driver.sh 1a \
  > outputs/ladder_pipeline/driver.stdout 2>&1 &
echo "driver pid $!"
```

- [ ] **Step 2: Confirm the validation gate passed and jobs launched**

```bash
sleep 600
tail -30 /home/thuandn/Repository/lerobot/outputs/ladder_pipeline/driver_phase1a_seed1000.log
```
Expected: `all 6 backbones validated`, then `launch ladder_<point>_en on GPU N` lines.

- [ ] **Step 3: Monitor to completion (~1–2 days wall-clock)**

```bash
cd /home/thuandn/Repository/lerobot
for f in outputs/ladder_pipeline/train_ladder_*_en.log; do
  echo -n "$(basename "$f"): "
  tr '\r' '\n' < "$f" | grep -o '[0-9]*/50000 \[[^]]*\]' | tail -1
done
```
Expected: all six advancing at roughly 1.4 step/s (~10h each).

- [ ] **Step 4: Build the report**

```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/ladder_report.py
```
Expected: a six-row table with no missing cells.

- [ ] **Step 5: Read the curve and decide whether Phase 1b is worth running**

Per the spec's early-stop rule: **if all six rungs are ≈0% on the VI leg**, there is no cliff in this dose range. Stop Phase 1b, record the negative result, and go to Task 11 (scale ladder). Otherwise continue.

Record the decision and the reasoning in `outputs/report_backbone_ladder.md` before proceeding.

- [ ] **Step 6: Commit the report**

```bash
cd /home/thuandn/Repository/lerobot
git add -f outputs/report_backbone_ladder.md
git commit -m "results(ladder): Phase 1a dose ladder, EN-train / EN+VI eval

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Phase 2 — scale ladder (~30 GPU-h)

**Interfaces:**
- Consumes: Task 8's driver (phase `2`).
- Produces: `s256m` and `s2200m` rungs.

- [ ] **Step 1: Check whether SmolVLM2-2.2B fits at batch 16**

```bash
cd /home/thuandn/Repository/lerobot
env CUDA_VISIBLE_DEVICES=0 SKIP_SYNC=1 \
  VLM_MODEL=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  DATASET_REPO=HuggingFaceVLA/libero \
  RUN_TAG=oom_probe OUTPUT_DIR=/tmp/oom_probe/ SEED=1000 \
  STEPS=20 BATCH_SIZE=16 TRAIN_EXPERT_ONLY=false \
  SAVE_FREQ=1000000 ENV_EVAL_FREQ=1000000 LOG_FREQ=5 \
  ./run_vi.sh 2>&1 | tail -20
```
Expected: reaches step 20 without `CUDA out of memory`.

**If it OOMs:** the global constraint is that effective batch stays 16. `run_vi.sh` exposes no grad-accum knob, so add one — pass `--optim.grad_accum_steps` (confirm the exact flag with `uv run lerobot-train --help | grep -i accum`) and run 2.2B at `BATCH_SIZE=4` with 4 accumulation steps. Re-probe before launching the real run.

- [ ] **Step 2: Launch phase 2**

```bash
cd /home/thuandn/Repository/lerobot
setsid nohup ./vlai-experiments/vi-instructions/ladder_driver.sh 2 \
  > outputs/ladder_pipeline/driver_phase2.stdout 2>&1 &
```

- [ ] **Step 3: Rebuild the report once both finish**

```bash
cd /home/thuandn/Repository/lerobot
uv run python vlai-experiments/vi-instructions/ladder_report.py
```
Expected: eight rows. The scientific read: if `s256m`, `stock` and `s2200m` all sit near 0 on the VI leg while the dose rungs climb, the cliff is about language exposure, not scale.

- [ ] **Step 4: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add -f outputs/report_backbone_ladder.md
git commit -m "results(ladder): Phase 2 scale ladder (256M / 500M / 2.2B)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: VI-train branch (~42 GPU-h)

Completes the 6 × 2 matrix by filling the three dose rungs on Vietnamese training data. The other three cells already exist as Arm A/B/C and are pulled in through `LADDER_ALIASES`.

- [ ] **Step 1: Launch**

```bash
cd /home/thuandn/Repository/lerobot
setsid nohup ./vlai-experiments/vi-instructions/ladder_driver.sh vi \
  > outputs/ladder_pipeline/driver_phasevi.stdout 2>&1 &
```

- [ ] **Step 2: Verify the three runs used the Vietnamese dataset**

```bash
cd /home/thuandn/Repository/lerobot
grep -h "dataset.repo_id" outputs/ladder_pipeline/train_ladder_d*_vi.log | head -3
```
Expected: `VLAIResearchLab/lerobot_libero_vi` in all three.

- [ ] **Step 3: Extend the report with the VI-train column**

Add a second table to `render_report` covering the `*_vi_vi` cells, sourcing the three historical arms through `LADDER_ALIASES` and the three new runs from `collect_runs`. Write a test first, mirroring `TestBuildLadderRows`, that asserts the aliased Arm A cell reports 45.0 when `outputs/eval_vi_ft_smolvla/eval_info.json` says so.

- [ ] **Step 4: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add vlai-experiments/vi-instructions/ladder_report.py \
        vlai-experiments/vi-instructions/tests/test_ladder_report.py
git add -f outputs/report_backbone_ladder.md
git commit -m "results(ladder): VI-train branch completes the 6x2 matrix

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Phase 1b — seeds 2000 and 3000 (~168 GPU-h)

Every quantitative claim in the paper depends on this task. Phase 1a establishes only the *shape* of the curve.

- [ ] **Step 1: Launch both seeds sequentially**

```bash
cd /home/thuandn/Repository/lerobot
setsid nohup bash -c '
  ./vlai-experiments/vi-instructions/ladder_driver.sh 1b 2000
  ./vlai-experiments/vi-instructions/ladder_driver.sh 1b 3000
' > outputs/ladder_pipeline/driver_phase1b.stdout 2>&1 &
```

- [ ] **Step 2: Add per-task paired bootstrap across seeds to the report**

Extend `ladder_report.py` to read `per_group` per-task scores for each rung at all three seeds, average across seeds per task, and call `paired_bootstrap` between adjacent rungs (`stock` vs `d0`, `d0` vs `d10`, and so on). Write the test first: assert that comparing a rung against itself yields a mean difference of 0.0 and a CI containing 0.

- [ ] **Step 3: Rebuild and commit**

```bash
cd /home/thuandn/Repository/lerobot
uv run pytest vlai-experiments/vi-instructions/tests/ -v
uv run python vlai-experiments/vi-instructions/ladder_report.py
git add vlai-experiments/vi-instructions/ladder_report.py \
        vlai-experiments/vi-instructions/tests/test_ladder_report.py
git add -f outputs/report_backbone_ladder.md
git commit -m "results(ladder): 3-seed Phase 1b with paired bootstrap CIs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: Amend the spec with what the build revealed

Two findings during implementation contradict the spec as written. The spec is the paper's methods section in embryo; leaving it wrong is a correctness problem, not bookkeeping.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md`

- [ ] **Step 1: Record the two initialisation methods**

In §4.2, note that `d0` is a **new** backbone built with stage-1's sub-token-mean initialisation, distinct from `outputs/backbones/smolvlm2_vi_vocab_only/` (Arm C, HF `mean_resizing=True` multivariate draw). State that the VI-train `d0` cell in §4.4 therefore uses the Arm C init, and is flagged as such in the report.

- [ ] **Step 2: Record the warmup change**

In §5.1, replace "giữ nguyên mọi siêu tham số stage-1" with the exception: `--warmup_steps 100` becomes `--warmup_ratio 0.07` (= 100/1436) so the LR schedule shape is identical at every dose.

- [ ] **Step 3: Resolve the deferred BPC corpus decision**

In §5.2, replace the "chốt cụ thể ở bước plan" placeholder with what Task 4 actually built: the held-out dev splits of OpenViVQA and ViOCRVQA, extracted by `build_bpc_corpus.py`, verified <1% overlap with the stage-1 training JSONs.

- [ ] **Step 4: Commit**

```bash
cd /home/thuandn/Repository/lerobot
git add docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md
git commit -m "docs(specs): amend ladder spec with build findings

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** §4.1 protocol → Tasks 8, 10. §4.2 dose ladder → Tasks 1, 2, 6, 7, 10. §4.3 scale ladder → Task 11. §4.4 VI-train branch → Task 12. §4.5 budget → Tasks 10–13. §5.1 dose pretraining → Tasks 1, 2, 7. §5.2 validation + BPC → Tasks 4, 5. §5.3 driver → Task 8. §5.4 report → Task 9. §5.5 naming → Global Constraints and Task 8. §6 error handling: E1 stale tokenizer → Task 5 (`assert_policy_tokenizer_matches`) + Task 8 gate; E2 symlink race → Task 8 explicit step path; E3 processor files → Task 3 saves the processor, Task 8's `ckpt_ok`; E4 2.2B OOM → Task 11 Step 1; E5 job death → Task 8's reap loop and Task 9's `missing` list. §7 statistics → Tasks 9, 13. §7 unit tests → Tasks 1, 3, 5, 9. §8 early stop → Task 10 Step 5.

**Gaps found and closed.** The spec assumed one embedding initialisation; the code has two — added Task 6 and Task 14. The spec froze all stage-1 hyperparameters; a fixed warmup breaks at low dose — added the `WARMUP_RATIO` override in Task 2 and the amendment in Task 14. The spec deferred the BPC corpus choice — Task 4 resolves it concretely.

**Type consistency.** `subtoken_mean_init` takes `(model, base_tokenizer, new_tokenizer, old_vocab_size, new_vocab_size)` in Task 3's implementation, its tests, and Task 6's usage. `bits_per_character(model, tokenizer, texts, device)` matches between Task 5's implementation, its tests, and the CLI. `paired_bootstrap(a, b, n_resamples, seed) -> (mean, lo, hi)` matches between Task 9's implementation, its tests, and Task 13's usage. `LADDER_ALIASES` keys are `<point>_<train_lang>_<eval_lang>`, consistent with `collect_runs`'s key format.
