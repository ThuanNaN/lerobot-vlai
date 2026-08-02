from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
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
    subset of a larger dose's samples. smollm-vi implements `random:N%` as
    `random.seed(42); shuffle(items); items[:N]` -- a fixed permutation followed by
    a prefix -- so nesting reduces to the prefix lengths being monotonic.
    """

    @staticmethod
    def _prefix_len(strategy: str, total: int) -> int:
        kind, amount = strategy.split(":") if ":" in strategy else ("all", None)
        if kind == "all":
            return total
        if amount.endswith("%"):
            return max(1, math.ceil(total * float(amount.rstrip("%")) / 100.0))
        return int(amount)

    def test_prefix_lengths_are_monotonic_across_doses(self):
        total = 19700
        lengths = [self._prefix_len(scale_sampling_strategy("all", d), total) for d in DOSES]
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
        assert manifest["sources"] == [{"name": "x", "original": "all", "scaled": "random:25%"}]
        assert "created_at" in manifest
