"""Merge a smollm-vi stage-1 LoRA + trainable-token adapter into a standalone
SmolVLM2 backbone directory usable as SmolVLA's `vlm_model_name`.

The adapter stores `trainable_tokens_delta` for `embed_tokens` and `lm_head` --
deltas relative to how the 8,064 new rows were initialised during training. That
initialisation is smollm-vi's sub-token mean (each new token's embedding is the mean
of the embeddings its text decomposes into under the *base* tokenizer; see smollm-vi
vision/smolvlm2/smolvlm/train/train.py:238-249), NOT HuggingFace's
`resize_token_embeddings(mean_resizing=True)` multivariate draw. Reproducing the same
deterministic init here is what makes the merge exact.

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
DEFAULT_TOKENIZER_DIR = (
    Path.home() / "Repository/smollm-vi/vision/experiments/pretraining/vietnamese/data/tokenizer_vi"
)


def load_backbone_bf16(model_id: str):
    """Load a SmolVLM2 backbone in bfloat16, across the transformers 4/5 kwarg change.

    This module has to run under smollm-vi's environment (transformers 4.50 + peft
    0.19.1 -- the exact peft that wrote the stage-1 adapters), while the rest of the
    lerobot repo is on transformers 5.5.4, where `torch_dtype=` was renamed `dtype=`.
    """
    kwargs = {"low_cpu_mem_usage": True}
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, **kwargs
        )


def subtoken_mean_init(
    model, base_tokenizer, new_tokenizer, old_vocab_size: int, new_vocab_size: int
) -> int:
    """Resize `model` to `new_vocab_size`, initialising each new row as the mean of the
    base-vocabulary embeddings its token text decomposes into.

    Rows whose text has no in-range sub-token fall back to the mean of the *original*
    rows only (computed before the resize, so freshly allocated rows cannot leak in).
    Mirrors smollm-vi's stage-1 initialisation exactly, and is deterministic.
    """
    if new_vocab_size <= old_vocab_size:
        raise ValueError(
            f"new_vocab_size {new_vocab_size} must exceed old_vocab_size {old_vocab_size}"
        )

    with torch.no_grad():
        in_fallback = model.get_input_embeddings().weight[:old_vocab_size].mean(dim=0).clone()
        out_fallback = model.get_output_embeddings().weight[:old_vocab_size].mean(dim=0).clone()

    model.resize_token_embeddings(new_vocab_size)

    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings().weight
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

    # resize_token_embeddings does not update vocab_size on composite-config VLMs
    # (verified against SmolVLM2), so set it explicitly.
    model.config.vocab_size = new_vocab_size
    return new_vocab_size


def merge_stage1_adapter(base_model_id: str, adapter_dir: Path | None, tokenizer_dir: Path):
    """Load base -> resize with sub-token mean -> apply adapter -> merge -> return.

    `adapter_dir=None` builds the dose-zero rung of the language-cliff ladder: the same
    pipeline, stopped before the adapter is applied. Going through this one function
    rather than a separate script is what makes "dose 0" mean literally "identical
    construction, zero Vietnamese training" -- any divergence in resize behaviour,
    dtype, or save format would otherwise land exactly on the first segment of the
    curve, where the cliff is expected.
    """
    new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = load_backbone_bf16(base_model_id)

    old_vocab = model.get_input_embeddings().weight.shape[0]
    subtoken_mean_init(model, base_tokenizer, new_tokenizer, old_vocab, len(new_tokenizer))

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.config.vocab_size = len(new_tokenizer)

    processor = AutoProcessor.from_pretrained(tokenizer_dir)
    return model, processor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=EN_BACKBONE)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="stage-1 checkpoint to merge; omit with --no-adapter for the dose-0 rung",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        default=False,
        help="build the dose-0 anchor: resize with sub-token mean, apply nothing",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()

    if args.no_adapter and args.adapter_dir is not None:
        parser.error("--no-adapter and --adapter-dir are mutually exclusive")
    if not args.no_adapter and args.adapter_dir is None:
        parser.error("pass --adapter-dir, or --no-adapter to build the dose-0 rung")

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} exists and is non-empty; pass --overwrite")

    if args.no_adapter:
        print(f"building dose-0 anchor from {args.base_model} (no adapter) ...")
    else:
        print(f"merging {args.adapter_dir} into {args.base_model} ...")
    model, processor = merge_stage1_adapter(args.base_model, args.adapter_dir, args.tokenizer_dir)

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
                "kind": "dose_zero_anchor" if args.no_adapter else "stage1_merged",
                "dose": 0.0 if args.no_adapter else None,
                "base_model": args.base_model,
                "adapter_dir": None if args.adapter_dir is None else str(args.adapter_dir),
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
