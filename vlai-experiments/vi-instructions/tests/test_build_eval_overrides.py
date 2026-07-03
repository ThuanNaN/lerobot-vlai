import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from build_eval_overrides import build_eval_overrides


def test_build_eval_overrides_groups_by_suite():
    rows = [
        {
            "suite": "libero_spatial",
            "task_id": "0",
            "english": "pick up the bowl",
            "vietnamese": "nhấc cái bát lên",
        },
        {
            "suite": "libero_spatial",
            "task_id": "1",
            "english": "open the drawer",
            "vietnamese": "mở ngăn kéo",
        },
        {
            "suite": "libero_object",
            "task_id": "0",
            "english": "turn on the stove",
            "vietnamese": "bật bếp lên",
        },
    ]

    overrides = build_eval_overrides(rows)

    assert overrides == {
        "libero_spatial": {"0": "nhấc cái bát lên", "1": "mở ngăn kéo"},
        "libero_object": {"0": "bật bếp lên"},
    }


def test_build_eval_overrides_rejects_empty_translation():
    rows = [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
    with pytest.raises(ValueError):
        build_eval_overrides(rows)
