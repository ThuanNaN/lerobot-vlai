import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_eval import build_comparison_table, render_markdown_table


def test_build_comparison_table_aligns_suites_across_runs():
    runs = {
        "vi_lora_on_vi": {"overall": {"pc_success": 40.0}, "libero_spatial": {"pc_success": 50.0}},
        "en_baseline_on_en": {"overall": {"pc_success": 70.0}, "libero_spatial": {"pc_success": 80.0}},
    }

    rows = build_comparison_table(runs)

    assert rows == [
        {"suite": "libero_spatial", "vi_lora_on_vi": 50.0, "en_baseline_on_en": 80.0},
        {"suite": "overall", "vi_lora_on_vi": 40.0, "en_baseline_on_en": 70.0},
    ]


def test_render_markdown_table_produces_expected_header_and_rows():
    rows = [{"suite": "overall", "run_a": 40.0}]
    table = render_markdown_table(rows, ["run_a"])
    lines = table.splitlines()
    assert lines[0] == "| suite | run_a |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| overall | 40.0 |"
