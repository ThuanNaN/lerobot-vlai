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

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, PreTrainedModel
from transformers.processing_utils import ProcessorMixin

EN_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VI_BACKBONE = "thuanan/SmolVLM2-500M-vi-stage1"
SANITY_INSTRUCTION = "nhấc cái bát đen trên hộp bánh quy lên rồi đặt nó lên đĩa"

# The only mean_resizing value this script currently exercises (resize_vocab_to_match's
# default) — kept as one constant so the build_metadata.json record can't drift from
# what actually ran.
MEAN_RESIZING_DEFAULT = True


def resize_vocab_to_match(
    model: PreTrainedModel, target_vocab_size: int, mean_resizing: bool = MEAN_RESIZING_DEFAULT
) -> None:
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


def assert_embeddings_match_vocab(model: PreTrainedModel, vocab_size: int) -> None:
    """Verify the resize in `resize_vocab_to_match` actually took effect on the model's
    embedding tables — the CLI's tokenizer-only sanity check can't catch a resize that
    silently no-op'd, since token ids drawn from the tokenizer are trivially in-range
    regardless of what happened to the model.
    """
    n_in = model.get_input_embeddings().weight.shape[0]
    n_out = model.get_output_embeddings().weight.shape[0]
    if n_in != vocab_size or n_out != vocab_size:
        raise ValueError(f"embedding rows {n_in}/{n_out} != tokenizer vocab {vocab_size}")


def check_out_dir_overwrite(out_dir: Path, overwrite: bool) -> None:
    """Refuse to silently clobber an existing non-empty --out-dir unless --overwrite
    was passed explicitly.
    """
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists and is non-empty; pass --overwrite to replace it")


def build_metadata(en_backbone: str, vi_backbone: str, target_vocab_size: int, mean_resizing: bool) -> dict:
    """Provenance record for the produced backbone: which sources and settings built it."""
    return {
        "en_backbone": en_backbone,
        "vi_backbone": vi_backbone,
        "target_vocab_size": target_vocab_size,
        "mean_resizing": mean_resizing,
        "created_at": datetime.now(UTC).isoformat(),
    }


def build_vocab_only_backbone(
    en_model_id: str, vi_tokenizer_id: str
) -> tuple[PreTrainedModel, ProcessorMixin]:
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/backbones/smolvlm2_vi_vocab_only"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="allow overwriting a non-empty --out-dir",
    )
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()

    check_out_dir_overwrite(args.out_dir, args.overwrite)

    print(f"loading EN weights ({args.en_backbone}) + VI tokenizer ({args.vi_backbone}) ...")
    model, processor = build_vocab_only_backbone(args.en_backbone, args.vi_backbone)

    vocab_size = len(processor.tokenizer)
    print(f"resized embeddings to vocab_size={vocab_size}")

    assert_embeddings_match_vocab(model, vocab_size)
    sample_ids = processor.tokenizer(SANITY_INSTRUCTION)["input_ids"]
    assert_ids_in_vocab(sample_ids, vocab_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    metadata = build_metadata(
        en_backbone=args.en_backbone,
        vi_backbone=args.vi_backbone,
        target_vocab_size=vocab_size,
        mean_resizing=MEAN_RESIZING_DEFAULT,
    )
    (args.out_dir / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote vocab-only backbone to {args.out_dir} (vocab={vocab_size}, sanity check passed)")
