from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from merge_stage1_adapter import STOCK_VOCAB_SIZE, VI_VOCAB_SIZE, subtoken_mean_init
from torch import nn


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
        self.config = SimpleNamespace(vocab_size=vocab_size)

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out

    def resize_token_embeddings(self, new_num_tokens: int):
        hidden = self._in.weight.shape[1]
        old = self._in.weight.shape[0]
        new_in = nn.Embedding(new_num_tokens, hidden)
        new_out = nn.Linear(hidden, new_num_tokens, bias=False)
        with torch.no_grad():
            new_in.weight[:old] = self._in.weight
            new_out.weight[:old] = self._out.weight
        self._in = new_in
        self._out = new_out


def _setup():
    """Vocab of 3 known rows; new ids 3/4/5 decode to text whose base-vocab
    decomposition is respectively two tokens, one token, and none."""
    model = FakeModel(vocab_size=3, hidden=4)
    with torch.no_grad():
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            emb.weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            emb.weight[1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
            emb.weight[2] = torch.tensor([0.0, 0.0, 4.0, 0.0])
    base_tok = FakeTokenizer({}, {"xy": [0, 1], "zz": [2], "??": []})
    new_tok = FakeTokenizer({3: "xy", 4: "zz", 5: "??"}, {})
    return model, base_tok, new_tok


class TestSubtokenMeanInit:
    def test_returns_new_vocab_size(self):
        model, base_tok, new_tok = _setup()
        assert subtoken_mean_init(model, base_tok, new_tok, 3, 6) == 6

    def test_updates_config_vocab_size(self):
        model, base_tok, new_tok = _setup()
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        assert model.config.vocab_size == 6

    def test_new_row_is_mean_of_its_subtokens(self):
        model, base_tok, new_tok = _setup()
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        expected = (torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.tensor([0.0, 2.0, 0.0, 0.0])) / 2
        assert torch.allclose(model.get_input_embeddings().weight[3], expected)

    def test_single_subtoken_row_is_copied(self):
        model, base_tok, new_tok = _setup()
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        assert torch.allclose(
            model.get_input_embeddings().weight[4], torch.tensor([0.0, 0.0, 4.0, 0.0])
        )

    def test_row_with_no_valid_subtokens_falls_back_to_global_mean(self):
        model, base_tok, new_tok = _setup()
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        expected = torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 2.0, 0.0, 0.0]),
                torch.tensor([0.0, 0.0, 4.0, 0.0]),
            ]
        ).mean(dim=0)
        assert torch.allclose(model.get_input_embeddings().weight[5], expected)

    def test_original_rows_are_untouched(self):
        model, base_tok, new_tok = _setup()
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        assert torch.allclose(
            model.get_input_embeddings().weight[1], torch.tensor([0.0, 2.0, 0.0, 0.0])
        )

    def test_output_embeddings_are_initialised_too(self):
        model, base_tok, new_tok = _setup()
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        expected = (torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.tensor([0.0, 2.0, 0.0, 0.0])) / 2
        assert torch.allclose(model.get_output_embeddings().weight[3], expected)

    def test_is_deterministic(self):
        first, base_tok, new_tok = _setup()
        subtoken_mean_init(first, base_tok, new_tok, 3, 6)
        second, base_tok2, new_tok2 = _setup()
        subtoken_mean_init(second, base_tok2, new_tok2, 3, 6)
        assert torch.equal(
            first.get_input_embeddings().weight, second.get_input_embeddings().weight
        )

    def test_rejects_shrinking_vocab(self):
        model, base_tok, new_tok = _setup()
        with pytest.raises(ValueError, match="new_vocab_size"):
            subtoken_mean_init(model, base_tok, new_tok, 3, 2)

    def test_fallback_uses_only_original_rows_not_freshly_resized_ones(self):
        """The global-mean fallback must be computed from the pre-resize rows. If it
        were computed after the resize over all rows, HF's uninitialised new rows
        would leak into it and the init would stop being deterministic."""
        model, base_tok, new_tok = _setup()
        with torch.no_grad():
            model.get_input_embeddings().weight[:] = torch.zeros(3, 4)
            model.get_input_embeddings().weight[0] = torch.tensor([3.0, 3.0, 3.0, 3.0])
        subtoken_mean_init(model, base_tok, new_tok, 3, 6)
        assert torch.allclose(
            model.get_input_embeddings().weight[5], torch.tensor([1.0, 1.0, 1.0, 1.0])
        )


class TestConstants:
    def test_vocab_constants_match_the_experiment(self):
        assert STOCK_VOCAB_SIZE == 49280
        assert VI_VOCAB_SIZE == 57344
        assert VI_VOCAB_SIZE - STOCK_VOCAB_SIZE == 8064
