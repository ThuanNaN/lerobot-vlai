"""Hard gate run before any ladder training job, and after any ladder checkpoint.

Checks, in order:
  1. input and output embedding rows both equal the tokenizer's vocab size;
  2. a Vietnamese LIBERO instruction round-trips into in-range token ids;
  3. bits-per-character on the held-out Vietnamese corpus (build_bpc_corpus.py) --
     the ladder's language-capability x-axis. Per *character*, not per token, so a
     49,280-token and a 57,344-token vocabulary are directly comparable.

Exit code 0 = pass, 1 = fail. ladder_driver.sh refuses to launch on non-zero.

Motivation: twice in this project a checkpoint shipped with a
`policy_preprocessor.json` whose `tokenizer_name` still pointed at the English
tokenizer while the embedding table was the Vietnamese one. Token ids get looked up
against the wrong rows -- silently wrong, no crash, a confident 0.0%.

See docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md.
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

    The first token of each line has no preceding context and is not scored, but every
    character still counts toward the denominator -- applied identically to every
    backbone, so cross-vocabulary comparison stays fair.
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
    """Post-training gate: a trained policy's preprocessor must reference the same
    tokenizer as the backbone it was built on.
    """
    checkpoint_dir = Path(checkpoint_dir)
    preprocessor_path = checkpoint_dir / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"missing {preprocessor_path}")

    vlm_model_name = json.loads((checkpoint_dir / "config.json").read_text())["vlm_model_name"]
    preprocessor = json.loads(preprocessor_path.read_text())

    tokenizer_names = [
        step["config"]["tokenizer_name"]
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


def load_text_model(backbone: str, device: str):
    """Load a SmolVLM2 backbone's *text* tower as a causal LM.

    The image-text-to-text wrapper's forward expects pixel inputs, and this metric is
    text-only, so score through the text tower's own LM head instead.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        model = AutoModelForCausalLM.from_pretrained(backbone, low_cpu_mem_usage=True)
    except (ValueError, KeyError, OSError):
        model = AutoModelForImageTextToText.from_pretrained(backbone, low_cpu_mem_usage=True)
    return model.to(device).eval()


if __name__ == "__main__":
    import argparse
    import sys

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", help="HF id or local directory")
    parser.add_argument("--expected-vocab", type=int, required=True, help="49280 or 57344")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument("--report", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        if len(tokenizer) != args.expected_vocab:
            raise ValueError(f"tokenizer vocab {len(tokenizer)} != expected {args.expected_vocab}")

        model = load_text_model(args.backbone, args.device)
        assert_embeddings_match_vocab(model, args.expected_vocab)
        assert_ids_in_vocab(tokenizer(SANITY_INSTRUCTION)["input_ids"], args.expected_vocab)

        texts = [
            line for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()
        ][: args.max_lines]
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
