from __future__ import annotations

import csv
import json
from pathlib import Path


def load_tasks_vi_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_eval_overrides(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for row in rows:
        suite = row["suite"]
        task_id = row["task_id"]
        vietnamese = row["vietnamese"]
        if not vietnamese.strip():
            raise ValueError(f"Empty Vietnamese translation for suite={suite} task_id={task_id}")
        overrides.setdefault(suite, {})[task_id] = vietnamese
    return overrides


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-vi-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_vi.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("vlai-experiments/vi-instructions/data/eval_overrides.json")
    )
    args = parser.parse_args()

    rows = load_tasks_vi_csv(args.tasks_vi_csv)
    overrides = build_eval_overrides(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overrides, indent=2, ensure_ascii=False))
    print(
        f"Wrote overrides for {sum(len(v) for v in overrides.values())} tasks across {len(overrides)} suites"
    )
