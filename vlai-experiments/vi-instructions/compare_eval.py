from __future__ import annotations

import json
from pathlib import Path


def load_eval_info(path: Path) -> dict:
    return json.loads(path.read_text())


def build_comparison_table(runs: dict[str, dict]) -> list[dict[str, str | float]]:
    suites = sorted({suite for info in runs.values() for suite in info.get("per_group", {})})
    rows: list[dict[str, str | float]] = []
    for suite in suites:
        row: dict[str, str | float] = {"suite": suite}
        for run_name, info in runs.items():
            row[run_name] = info.get("per_group", {}).get(suite, {}).get("pc_success", float("nan"))
        rows.append(row)

    overall_row: dict[str, str | float] = {"suite": "overall"}
    for run_name, info in runs.items():
        overall_row[run_name] = info.get("overall", {}).get("pc_success", float("nan"))
    rows.append(overall_row)

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

    parser = argparse.ArgumentParser(
        description="Compare LIBERO eval_info.json runs. Pass --run NAME=PATH for each run "
        "(order is preserved as table columns), e.g. "
        "--run smolvla=outputs/eval_vi_smolvla/eval_info.json "
        "--run smolvla-vi=outputs/eval_vi_vi/eval_info.json"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A named eval_info.json to include (repeatable, at least two).",
    )
    args = parser.parse_args()

    runs: dict[str, dict] = {}
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run must be NAME=PATH, got: {spec!r}")
        name, path = spec.split("=", 1)
        if name in runs:
            parser.error(f"duplicate run name: {name!r}")
        runs[name] = load_eval_info(Path(path))
    if len(runs) < 2:
        parser.error("need at least two --run entries to compare")

    rows = build_comparison_table(runs)
    print(render_markdown_table(rows, list(runs.keys())))
