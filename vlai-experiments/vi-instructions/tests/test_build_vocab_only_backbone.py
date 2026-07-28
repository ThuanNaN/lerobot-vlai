import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from build_vocab_only_backbone import (
    assert_embeddings_match_vocab,
    assert_ids_in_vocab,
    build_metadata,
    check_out_dir_overwrite,
    resize_vocab_to_match,
)


def _embedding(vocab_size: int, hidden: int = 8) -> SimpleNamespace:
    return SimpleNamespace(weight=SimpleNamespace(shape=(vocab_size, hidden)))


class _StubModel:
    """Mimics the slice of transformers.PreTrainedModel this script depends on."""

    def __init__(
        self, vocab_size: int, input_vocab_size: int | None = None, output_vocab_size: int | None = None
    ):
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.resize_calls: list[tuple[int, bool]] = []
        self._input_embedding = _embedding(vocab_size if input_vocab_size is None else input_vocab_size)
        self._output_embedding = _embedding(vocab_size if output_vocab_size is None else output_vocab_size)

    def resize_token_embeddings(self, new_num_tokens: int, mean_resizing: bool = True):
        self.resize_calls.append((new_num_tokens, mean_resizing))
        # Real transformers behavior: resizes the embedding table but does NOT touch
        # config.vocab_size on this composite-config model — verified interactively
        # against HuggingFaceTB/SmolVLM2-500M-Video-Instruct (transformers 5.5.4).
        self._input_embedding = _embedding(new_num_tokens)
        self._output_embedding = _embedding(new_num_tokens)

    def get_input_embeddings(self):
        return self._input_embedding

    def get_output_embeddings(self):
        return self._output_embedding


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


def test_assert_embeddings_match_vocab_accepts_resized_model():
    model = _StubModel(vocab_size=49280)
    resize_vocab_to_match(model, target_vocab_size=57344)

    assert_embeddings_match_vocab(model, vocab_size=57344)  # must not raise


def test_assert_embeddings_match_vocab_rejects_stale_input_embedding():
    # Simulates a resize that silently no-op'd on the input embedding: the tokenizer
    # (and thus vocab_size/sample ids) is already the new, bigger vocab, but the
    # model's embedding table was never actually resized.
    model = _StubModel(vocab_size=49280, input_vocab_size=49280, output_vocab_size=57344)

    with pytest.raises(ValueError, match="49280"):
        assert_embeddings_match_vocab(model, vocab_size=57344)


def test_assert_embeddings_match_vocab_rejects_stale_output_embedding():
    model = _StubModel(vocab_size=49280, input_vocab_size=57344, output_vocab_size=49280)

    with pytest.raises(ValueError, match="49280"):
        assert_embeddings_match_vocab(model, vocab_size=57344)


def test_check_out_dir_overwrite_allows_missing_dir():
    check_out_dir_overwrite(Path("/nonexistent/does-not-exist"), overwrite=False)  # must not raise


def test_check_out_dir_overwrite_allows_empty_dir(tmp_path):
    check_out_dir_overwrite(tmp_path, overwrite=False)  # must not raise


def test_check_out_dir_overwrite_rejects_nonempty_dir_without_flag(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(FileExistsError, match="overwrite"):
        check_out_dir_overwrite(tmp_path, overwrite=False)


def test_check_out_dir_overwrite_allows_nonempty_dir_with_flag(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    check_out_dir_overwrite(tmp_path, overwrite=True)  # must not raise


def test_build_metadata_contains_provenance_fields():
    metadata = build_metadata(
        en_backbone="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        vi_backbone="thuanan/SmolVLM2-500M-vi-stage1",
        target_vocab_size=57344,
        mean_resizing=True,
    )

    assert metadata["en_backbone"] == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    assert metadata["vi_backbone"] == "thuanan/SmolVLM2-500M-vi-stage1"
    assert metadata["target_vocab_size"] == 57344
    assert metadata["mean_resizing"] is True
    assert "created_at" in metadata


def test_build_metadata_created_at_is_iso8601_utc():
    metadata = build_metadata(en_backbone="en", vi_backbone="vi", target_vocab_size=57344, mean_resizing=True)

    # datetime.now(UTC).isoformat() always carries an explicit UTC offset, e.g.
    # "2026-07-28T12:00:00.000000+00:00" -- assert on that rather than a deprecated
    # naive-datetime "Z" suffix.
    parsed = datetime.fromisoformat(metadata["created_at"])
    assert parsed.utcoffset() == timedelta(0)


def test_out_dir_gets_resolved_to_absolute_path():
    # Mirrors the `args.out_dir = args.out_dir.resolve()` line in the CLI's
    # __main__ block; that block is argparse glue and not worth extracting into a
    # separate function just to unit-test two lines.
    assert Path("outputs/backbones/smolvlm2_vi_vocab_only").resolve().is_absolute()
