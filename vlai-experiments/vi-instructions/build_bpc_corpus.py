"""Extract a held-out Vietnamese text corpus for bits-per-character measurement.

Two kinds of source, deliberately:

* **In-domain, held out** -- the *dev* splits of OpenViVQA and ViOCRVQA. stage-1 trains
  only on the `_train.json` files, so these images are unseen. But the datasets'
  disjointness guarantee is about image ids, not text: VQA questions are formulaic
  ("cửa hàng này tên gì ?") and 21% of dev question strings appear verbatim in train.
  Every such line is filtered out here -- measured 2026-08-03.
* **Out-of-domain** -- the test split of `sepidmnorozy/Vietnamese_sentiment`, natural
  Vietnamese prose unrelated to the stage-1 mixture (VQA / captions / Wikipedia /
  cauldron). Without it the metric would largely reward memorising VQA phrasing rather
  than modelling Vietnamese.

Output is plain UTF-8, one sentence per line, so `validate_backbone.py` can score any
backbone regardless of its tokenizer.

The two VQA sources disagree on layout, which is why field access is defensive:
OpenViVQA's `annotations` is a dict keyed by id with a singular `answer`; ViOCRVQA's is
a list with a plural `answers` list.

See docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

MIN_CHARS = 20
MAX_LINES_PER_SOURCE = 2000
DEFAULT_TRAIN_DIR = (
    Path.home() / "Repository/smollm-vi/vision/experiments/pretraining/vietnamese/data"
)
TRAIN_JSONS = ("openvivqa_train.json", "viocrvqa_train.json", "uitviic_train.json")


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _iter_annotations(annotations):
    """Yield annotation dicts from either a dict-keyed-by-id or a plain list."""
    return annotations.values() if isinstance(annotations, dict) else annotations


def _texts_from(annotations, fields: tuple[str, ...]) -> list[str]:
    texts: list[str] = []
    for item in _iter_annotations(annotations):
        for field in fields:
            value = item.get(field)
            if value is None:
                continue
            for candidate in value if isinstance(value, list) else [value]:
                cleaned = _clean(candidate)
                if len(cleaned) >= MIN_CHARS:
                    texts.append(cleaned)
    return texts


def openvivqa_dev_texts() -> list[str]:
    path = hf_hub_download(
        "uitnlp/OpenViVQA-dataset", "vlsp2023_dev_data.json", repo_type="dataset"
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _texts_from(data["annotations"], ("question", "answer"))


def viocrvqa_dev_texts() -> list[str]:
    zip_path = hf_hub_download("huyhuy123/ViOCRVQA", "data_ViOCRVQA.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        data = json.load(zf.open("data/dev.json"))
    return _texts_from(data["annotations"], ("question", "answers"))


def sentiment_test_texts() -> list[str]:
    """Out-of-domain Vietnamese prose: none of the stage-1 sources overlap this."""
    from datasets import load_dataset

    ds = load_dataset("sepidmnorozy/Vietnamese_sentiment", split="test")
    return [t for t in (_clean(row["text"]) for row in ds) if len(t) >= MIN_CHARS]


SOURCES = (
    ("openvivqa_dev", openvivqa_dev_texts),
    ("viocrvqa_dev", viocrvqa_dev_texts),
    ("sentiment_test", sentiment_test_texts),
)


def load_train_utterances(train_dir: Path) -> set[str]:
    """Every Vietnamese utterance stage-1 was trained on, normalised for comparison."""
    utterances: set[str] = set()
    for name in TRAIN_JSONS:
        blob = json.loads((train_dir / name).read_text(encoding="utf-8"))
        for item in blob:
            for turn in item.get("conversations", []):
                # Drop the <image> sentinel so the comparison is on Vietnamese text alone.
                utterances.add(_clean(turn.get("value", "").replace("<image>", " ")))
    return utterances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt"),
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    args = parser.parse_args()

    train_text = load_train_utterances(args.train_dir)
    print(f"stage-1 training utterances: {len(train_text)}")

    lines: list[str] = []
    for name, fn in SOURCES:
        raw = fn()
        kept = [line for line in raw if line not in train_text][:MAX_LINES_PER_SOURCE]
        dropped = len(raw) - len([line for line in raw if line not in train_text])
        print(f"{name}: {len(kept)} kept, {dropped} dropped as seen in training")
        lines.extend(kept)

    deduped = list(dict.fromkeys(lines))
    contaminated = [line for line in deduped if line in train_text]
    if contaminated:
        raise AssertionError(f"{len(contaminated)} contaminated lines survived filtering")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(deduped) + "\n", encoding="utf-8")
    print(f"wrote {len(deduped)} lines ({sum(len(x) for x in deduped)} chars) to {args.out}")
