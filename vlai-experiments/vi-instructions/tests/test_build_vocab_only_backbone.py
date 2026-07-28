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
