"""Aggregate backbone language-cliff ladder evals into one report.

Reads outputs/eval_ladder_<point>_<train_lang>_<eval_lang>/eval_info.json plus the
driver's per-backbone validate_*.json (for bits-per-character), and emits the ladder
table, the EN-VI gap per rung, and paired bootstrap confidence intervals over the 40
LIBERO tasks.

Historical Arm A/B/C runs predate this naming scheme; LADDER_ALIASES maps them into
the VI-train cells read-only. No existing outputs/ directory is renamed.

See docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

# Ladder order is semantic (increasing Vietnamese dose, then increasing scale), never
# alphabetical -- the curve is meaningless in any other order.
LADDER_ORDER = ("stock", "d0", "d10", "d25", "d50", "d100", "s256m", "s2200m")

LADDER_ALIASES: dict[str, dict[str, str]] = {
    "stock_vi_vi": {"dir": "eval_vi_ft_smolvla_50k", "note": "Arm A, full-FT 50k, seed 1000"},
    "d100_vi_vi": {"dir": "eval_vi_ft_vi_50k", "note": "Arm B, full-FT 50k, seed 1000"},
    "d0_vi_vi": {
        "dir": "eval_vi_vocab_only_ft",
        "note": "Arm C, full-FT 50k, seed 1000 -- NOTE: built with HF mean_resizing "
        "init, not the sub-token-mean init used by the d0 ladder anchor",
    },
}


def collect_runs(outputs_dir: Path) -> dict[str, dict]:
    """Map '<point>_<train_lang>_<eval_lang>' -> parsed eval_info.json."""
    runs: dict[str, dict] = {}
    for path in sorted(Path(outputs_dir).glob("eval_ladder_*")):
        # The drivers move superseded results to `<dir>.stale-<timestamp>`; those are
        # kept for forensics but must never re-enter a report.
        if ".stale-" in path.name:
            continue
        info = path / "eval_info.json"
        if not info.is_file():
            continue
        runs[path.name.removeprefix("eval_ladder_")] = json.loads(info.read_text())
    return runs


def _point_of(key: str) -> str:
    return key.split("_")[0]


def build_ladder_rows(runs: dict[str, dict]) -> list[dict]:
    """One row per ladder point: EN score, VI score, and the EN-VI gap."""
    points = {_point_of(k) for k in runs}
    ordered = [p for p in LADDER_ORDER if p in points]
    ordered += sorted(points - set(LADDER_ORDER))

    rows: list[dict] = []
    for point in ordered:
        en = runs.get(f"{point}_en_en", {}).get("overall", {}).get("pc_success")
        vi = runs.get(f"{point}_en_vi", {}).get("overall", {}).get("pc_success")
        rows.append(
            {
                "point": point,
                "en": en,
                "vi": vi,
                "gap": None if en is None or vi is None else en - vi,
            }
        )
    return rows


def per_task_scores(info: dict) -> list[float]:
    """Percent success for each of the 40 LIBERO tasks, ordered by (group, task_id).

    A stable order is what lets two runs be paired task-by-task for the bootstrap.
    """
    tasks = sorted(info.get("per_task", []), key=lambda t: (t["task_group"], t["task_id"]))
    return [100.0 * statistics.fmean(t["metrics"]["successes"]) for t in tasks]


def paired_bootstrap(
    a: list[float], b: list[float], n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap over per-task scores.

    Returns (mean_diff, ci_low, ci_high) for a - b at the 95% level.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("cannot bootstrap an empty sample")

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(n_resamples))
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(int(0.975 * n_resamples), n_resamples - 1)]
    return statistics.fmean(diffs), lo, hi


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def render_report(rows: list[dict], bpc: dict[str, float], missing: list[str]) -> str:
    lines = [
        "# Backbone language-cliff ladder",
        "",
        "EN-train / eval EN and VI (zero-shot cross-lingual). `gap` = EN − VI, "
        "comparable to arXiv:2606.11906's Vietnamese numbers (OpenVLA-OFT 59.6, "
        "π₀.₅ 57.3).",
        "",
        "| point | bits/char (VI) | EN | VI | gap |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        bpc_cell = f"{bpc[row['point']]:.2f}" if row["point"] in bpc else "—"
        lines.append(
            f"| `{row['point']}` | {bpc_cell} | {_fmt(row['en'])} | "
            f"{_fmt(row['vi'])} | {_fmt(row['gap'])} |"
        )
    if missing:
        lines += ["", "## Missing cells", ""]
        lines += [f"- `{name}`" for name in missing]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/report_backbone_ladder.md"))
    args = parser.parse_args()

    runs = collect_runs(args.outputs_dir)
    rows = build_ladder_rows(runs)

    bpc: dict[str, float] = {}
    for report in sorted((args.outputs_dir / "ladder_pipeline").glob("validate_*.json")):
        data = json.loads(report.read_text())
        if data.get("status") == "pass":
            bpc[report.stem.removeprefix("validate_")] = data["bits_per_character"]

    expected = [f"{r['point']}_en_{lang}" for r in rows for lang in ("en", "vi")]
    missing = [key for key in expected if key not in runs]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(rows, bpc, missing))
    print(args.out.read_text())
