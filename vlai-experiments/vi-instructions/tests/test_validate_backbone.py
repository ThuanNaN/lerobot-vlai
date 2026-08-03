from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn
from validate_backbone import (
    assert_embeddings_match_vocab,
    assert_ids_in_vocab,
    assert_policy_tokenizer_matches,
    bits_per_character,
)


class TinyLM(nn.Module):
    """Deterministic stand-in that always predicts a uniform distribution, so the
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
    """One token per character, so tokens-per-character is exactly 1 and the expected
    bits-per-character is analytically known."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [(ord(c) % self.vocab_size) for c in text]
        return {"input_ids": torch.tensor([ids])}


def _write_policy(tmp_path: Path, vlm_model_name: str, tokenizer_name: str | None) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"vlm_model_name": vlm_model_name}))
    if tokenizer_name is not None:
        (tmp_path / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": "tokenizer_processor",
                            "config": {"tokenizer_name": tokenizer_name},
                        }
                    ]
                }
            )
        )
    return tmp_path


class TestAssertEmbeddingsMatchVocab:
    def test_passes_when_both_match(self):
        assert_embeddings_match_vocab(TinyLM(64), 64)

    def test_raises_when_vocab_differs(self):
        with pytest.raises(ValueError, match="embedding rows"):
            assert_embeddings_match_vocab(TinyLM(64), 57344)


class TestAssertIdsInVocab:
    def test_passes_for_in_range_ids(self):
        assert_ids_in_vocab([0, 5, 63], 64)

    def test_raises_for_ids_past_the_end(self):
        with pytest.raises(ValueError, match="out of vocab range"):
            assert_ids_in_vocab([0, 64], 64)


class TestBitsPerCharacter:
    def test_uniform_model_gives_expected_bits_per_char(self):
        vocab = 64
        bpc = bits_per_character(TinyLM(vocab), CharTokenizer(vocab), ["abcdefgh"], device="cpu")
        # 1 token per char, uniform over `vocab` -> log2(vocab) bits per scored token.
        # The first token has no preceding context and is not scored, but all 8 chars
        # count toward the denominator, so 7/8 of log2(64).
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
        # A 1-char line yields a single token with no context; it must not divide by zero.
        assert bits_per_character(TinyLM(64), CharTokenizer(64), ["a", "abcdefgh"], "cpu") > 0

    def test_rejects_corpus_of_only_blank_lines(self):
        with pytest.raises(ValueError, match="empty"):
            bits_per_character(TinyLM(64), CharTokenizer(64), ["", "   "], device="cpu")


class TestAssertPolicyTokenizerMatches:
    def test_passes_when_names_agree(self, tmp_path):
        path = _write_policy(
            tmp_path, "thuanan/SmolVLM2-500M-vi-stage1", "thuanan/SmolVLM2-500M-vi-stage1"
        )
        assert_policy_tokenizer_matches(path)

    def test_raises_on_the_stale_tokenizer_bug(self, tmp_path):
        path = _write_policy(
            tmp_path,
            "thuanan/SmolVLM2-500M-vi-stage1",
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        )
        with pytest.raises(ValueError, match="tokenizer_name"):
            assert_policy_tokenizer_matches(path)

    def test_raises_when_preprocessor_file_is_missing(self, tmp_path):
        path = _write_policy(tmp_path, "x", None)
        with pytest.raises(FileNotFoundError):
            assert_policy_tokenizer_matches(path)

    def test_raises_when_no_tokenizer_name_present(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"vlm_model_name": "x"}))
        (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": []}))
        with pytest.raises(ValueError, match="no tokenizer_name"):
            assert_policy_tokenizer_matches(tmp_path)

    def test_accepts_a_local_backbone_path(self, tmp_path):
        local = "/home/thuandn/Repository/lerobot/outputs/backbones/vi_dose_10"
        assert_policy_tokenizer_matches(_write_policy(tmp_path, local, local))
