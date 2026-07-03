from __future__ import annotations

import json
from pathlib import Path


def load_eval_info(path: Path) -> dict:
    return json.loads(path.read_text())


def build_comparison_table(runs: dict[str, dict]) -> list[dict[str, str | float]]:
    suites = sorted({suite for info in runs.values() for suite in info if suite != "overall"})
    rows: list[dict[str, str | float]] = []
    for suite in [*suites, "overall"]:
        row: dict[str, str | float] = {"suite": suite}
        for run_name, info in runs.items():
            row[run_name] = info.get(suite, {}).get("pc_success", float("nan"))
        rows.append(row)
    return rows


def render_markdown_table(rows: list[dict[str, str | float]], run_names: list[str]) -> str:
    header = "| suite | " + " | ".join(run_names) + " |"
    separator = "|---" * (len(run_names) + 1) + "|"
    lines = [header, separator]
    for row in rows:
        cells = [str(row["suite"])] + [
            f"{row[name]:.1f}" if isinstance(row[name], float) else str(row[name]) for name in run_names
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vi-lora-on-vi", type=Path, required=True)
    parser.add_argument("--en-baseline-on-en", type=Path, required=True)
    parser.add_argument("--en-baseline-zero-shot-vi", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "vi_lora_on_vi": load_eval_info(args.vi_lora_on_vi),
        "en_baseline_on_en": load_eval_info(args.en_baseline_on_en),
        "en_baseline_zero_shot_vi": load_eval_info(args.en_baseline_zero_shot_vi),
    }
    rows = build_comparison_table(runs)
    print(render_markdown_table(rows, list(runs.keys())))
