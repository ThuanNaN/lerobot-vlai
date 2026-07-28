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
