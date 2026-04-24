"""Regression tests against frozen per-paper outputs.

Skipped per-paper if the corresponding workspace (``runs/v1/<paper_id>/``)
doesn't exist yet -- build it via ``scripts/build_dataset.py`` first.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ._golden import (
    GOLDEN_DIR,
    compare_fingerprints,
    fingerprint,
    load_annotations,
    load_golden,
)


def golden_paper_ids() -> list[str]:
    if not GOLDEN_DIR.exists():
        return []
    return sorted(p.stem for p in GOLDEN_DIR.glob("*.json"))


@pytest.mark.parametrize("paper_id", golden_paper_ids())
def test_golden_match(paper_id: str) -> None:
    ann = load_annotations(paper_id)
    if ann is None:
        pytest.skip(
            f"workspace for {paper_id} not cached at "
            f"runs/v1/{paper_id}/dataset/annotations.json -- run "
            f"scripts/build_dataset.py"
        )
    golden = load_golden(paper_id)
    assert golden is not None, f"golden file missing for {paper_id}"

    fp = fingerprint(ann)
    messages = compare_fingerprints(fp, golden, bbox_tol=int(os.environ.get("ALX_BBOX_TOL", "2")))
    assert not messages, "\n".join(messages)
