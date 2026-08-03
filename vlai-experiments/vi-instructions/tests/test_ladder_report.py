from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from ladder_report import (
    LADDER_ALIASES,
    build_ladder_rows,
    collect_runs,
    paired_bootstrap,
    per_task_scores,
    render_report,
)


def _eval_info(per_group: dict, overall: float, per_task: list | None = None) -> dict:
    return {
        "per_group": per_group,
        "overall": {"pc_success": overall},
        "per_task": per_task if per_task is not None else [],
    }


def _write_eval(path: Path, per_group: dict, overall: float, per_task: list | None = None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_info.json").write_text(json.dumps(_eval_info(per_group, overall, per_task)))


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

    def test_ignores_stale_directories_moved_aside_by_the_driver(self, tmp_path):
        _write_eval(
            tmp_path / "eval_ladder_d10_en_vi.stale-20260803081500",
            {"libero_10": {"pc_success": 99.0}},
            99.0,
        )
        assert collect_runs(tmp_path) == {}


class TestBuildLadderRows:
    def test_one_row_per_point_with_en_vi_and_gap(self):
        runs = {
            "d10_en_en": _eval_info({}, 60.0),
            "d10_en_vi": _eval_info({}, 4.0),
        }
        rows = build_ladder_rows(runs)
        assert len(rows) == 1
        assert rows[0]["point"] == "d10"
        assert rows[0]["en"] == 60.0
        assert rows[0]["vi"] == 4.0
        assert rows[0]["gap"] == 56.0

    def test_orders_points_by_ladder_position_not_alphabetically(self):
        runs = {
            f"{p}_en_{lang}": _eval_info({}, 1.0)
            for p in ("d100", "stock", "d10")
            for lang in ("en", "vi")
        }
        assert [r["point"] for r in build_ladder_rows(runs)] == ["stock", "d10", "d100"]

    def test_missing_leg_yields_none_not_a_crash(self):
        rows = build_ladder_rows({"d50_en_en": _eval_info({}, 55.0)})
        assert rows[0]["vi"] is None
        assert rows[0]["gap"] is None

    def test_unknown_point_is_kept_and_sorted_last(self):
        runs = {"zz_en_en": _eval_info({}, 1.0), "stock_en_en": _eval_info({}, 2.0)}
        assert [r["point"] for r in build_ladder_rows(runs)] == ["stock", "zz"]


class TestPerTaskScores:
    def test_returns_percent_success_per_task(self):
        info = _eval_info(
            {},
            50.0,
            [
                {
                    "task_group": "libero_10",
                    "task_id": 1,
                    "metrics": {"successes": [True, False, True, False]},
                },
                {
                    "task_group": "libero_10",
                    "task_id": 0,
                    "metrics": {"successes": [True, True, True, True]},
                },
            ],
        )
        # Sorted by (task_group, task_id), so task 0 comes first.
        assert per_task_scores(info) == [100.0, 50.0]

    def test_ordering_is_stable_across_runs_so_tasks_pair_correctly(self):
        def make(order):
            return _eval_info(
                {},
                0.0,
                [
                    {"task_group": g, "task_id": i, "metrics": {"successes": [True]}}
                    for g, i in order
                ],
            )

        a = make([("libero_spatial", 0), ("libero_10", 3)])
        b = make([("libero_10", 3), ("libero_spatial", 0)])
        assert len(per_task_scores(a)) == len(per_task_scores(b)) == 2

    def test_empty_per_task_gives_empty_list(self):
        assert per_task_scores(_eval_info({}, 0.0)) == []


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

    def test_rejects_empty_sample(self):
        with pytest.raises(ValueError, match="empty"):
            paired_bootstrap([], [], 100, seed=0)


class TestRenderReport:
    def test_lists_missing_cells_explicitly(self):
        rows = [{"point": "d10", "en": 60.0, "vi": None, "gap": None}]
        text = render_report(rows, bpc={}, missing=["d10_en_vi"])
        assert "d10_en_vi" in text
        assert "Missing" in text

    def test_includes_bpc_column_when_available(self):
        rows = [{"point": "d10", "en": 60.0, "vi": 4.0, "gap": 56.0}]
        assert "2.31" in render_report(rows, bpc={"d10": 2.31}, missing=[])

    def test_renders_em_dash_for_absent_values(self):
        rows = [{"point": "d10", "en": None, "vi": None, "gap": None}]
        assert "—" in render_report(rows, bpc={}, missing=[])


class TestAliases:
    def test_historical_arms_map_to_the_directories_that_actually_exist(self):
        assert LADDER_ALIASES["stock_vi_vi"]["dir"] == "eval_vi_ft_smolvla_50k"
        assert LADDER_ALIASES["d100_vi_vi"]["dir"] == "eval_vi_ft_vi_50k"
        assert LADDER_ALIASES["d0_vi_vi"]["dir"] == "eval_vi_vocab_only_ft"

    def test_arm_c_alias_records_the_differing_init(self):
        assert "init" in LADDER_ALIASES["d0_vi_vi"]["note"].lower()
