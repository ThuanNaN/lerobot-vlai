"""Scale a smollm-vi stage-1 mixture YAML to a fraction of its data ("dose").

The backbone language-cliff ladder needs backbones pretrained on 10%, 25%, 50% and
100% of the Vietnamese stage-1 mixture, with the smaller doses being strict subsets
of the larger ones. smollm-vi's
`vision/smolvlm2/smolvlm/datasets/dataset.py:_apply_sampling_strategy` implements
`random:N%` as `random.seed(42); shuffle(items); items[:N]` -- a fixed permutation
followed by a prefix -- so scaling the percentages is enough to get nesting for
free. `first:N` is already a raw prefix.

See docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

DOSES = (0.10, 0.25, 0.50, 1.00)

# smollm-vi also implements `end:N`, but a suffix of one length is not a subset of a
# suffix of another, so it would break dose nesting. Reject it rather than scale it.
_SUPPORTED_KINDS = ("random", "first")


def _check_dose(dose: float) -> None:
    if not 0.0 < dose <= 1.0:
        raise ValueError(f"dose must be in (0, 1], got {dose}")


def _format_percent(value: float) -> str:
    """Render a percentage without trailing zeros: 25.0 -> '25', 3.75 -> '3.75'."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def scale_sampling_strategy(strategy: str, dose: float) -> str:
    """Return `strategy` scaled down to `dose` of the data it currently selects."""
    _check_dose(dose)

    if strategy == "all":
        return "all" if dose == 1.0 else f"random:{_format_percent(dose * 100)}%"

    if ":" not in strategy:
        raise ValueError(f"unsupported sampling_strategy: {strategy!r}")

    kind, amount = strategy.split(":", 1)
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"unsupported sampling_strategy kind: {kind!r} (in {strategy!r})")

    if amount.endswith("%"):
        return f"{kind}:{_format_percent(float(amount.rstrip('%')) * dose)}%"

    # A bare count: keep at least one sample so a source never silently vanishes from
    # the mixture at a small dose, which would change the mixture's shape rather than
    # only its size.
    return f"{kind}:{max(1, int(int(amount) * dose))}"


def scale_mixture(mixture: dict, dose: float) -> dict:
    """Scale every source's sampling_strategy in a parsed stage-1 mixture YAML."""
    _check_dose(dose)

    if len(mixture) != 1:
        raise ValueError(f"mixture must have exactly one top-level key, got {sorted(mixture)}")

    scaled = copy.deepcopy(mixture)
    (sources,) = scaled.values()
    for source in sources:
        if "sampling_strategy" not in source:
            raise ValueError(f"source missing sampling_strategy: {source.get('name', source)!r}")
        source["sampling_strategy"] = scale_sampling_strategy(source["sampling_strategy"], dose)
    return scaled


def dose_manifest(
    dose: float, mixture: dict, scaled: dict, template_path: str, git_sha: str
) -> dict:
    """Provenance record written next to each dose-scaled mixture."""
    (original_sources,) = mixture.values()
    (scaled_sources,) = scaled.values()
    return {
        "dose": dose,
        "template_path": template_path,
        "git_sha": git_sha,
        "created_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                "name": original["name"],
                "original": original["sampling_strategy"],
                "scaled": new["sampling_strategy"],
            }
            for original, new in zip(original_sources, scaled_sources, strict=True)
        ],
    }


if __name__ == "__main__":
    import argparse
    import subprocess

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path.home()
        / "Repository/smollm-vi/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml",
        help="stage-1 mixture YAML to scale (keeps its __DATA_FOLDER__ placeholders)",
    )
    parser.add_argument("--dose", type=float, required=True, help="fraction in (0, 1]")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    template = yaml.safe_load(args.template.read_text())
    scaled = scale_mixture(template, args.dose)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mixture_path = args.out_dir / "mixture_dose.yaml"
    mixture_path.write_text(yaml.safe_dump(scaled, sort_keys=False, allow_unicode=True))
    (args.out_dir / "dose_manifest.json").write_text(
        json.dumps(
            dose_manifest(args.dose, template, scaled, str(args.template), git_sha),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"wrote {mixture_path} (dose={args.dose})")
