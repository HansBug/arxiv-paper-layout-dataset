"""Shared helpers for golden-output regression tests.

Each test run:

1. Looks up the cached pipeline workspace under ``runs/v1/<paper_id>/``
   (or the override passed via ``ALX_RUNS_ROOT``).
2. Turns that paper's ``dataset/annotations.json`` into a stable fingerprint
   (labels sorted, bbox rounded to integer pixels).
3. Compares the fingerprint to the committed snapshot in
   ``tests/golden/<paper_id>.json``.

Running the pipeline from scratch takes minutes per paper, so tests do NOT
recompile; they assume a recent ``scripts/build_dataset.py`` has populated
``runs/v1``. A freshly-cloned checkout that hasn't built the pipeline yet
will ``skip`` the per-paper test rather than fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"


def runs_root() -> Path:
    """Primary location of cached per-paper pipeline outputs.

    Tests additionally fall through to ``runs/v2_extra/`` when looking up
    a paper, since the extra arXiv algorithm / listing papers live there.
    Override with ``ALX_RUNS_ROOT`` (single root) or with
    ``ALX_RUNS_ROOTS`` (``:``-separated list).
    """
    return Path(os.environ.get("ALX_RUNS_ROOT", REPO / "runs" / "v2_validated"))


def runs_roots() -> list[Path]:
    override = os.environ.get("ALX_RUNS_ROOTS")
    if override:
        return [Path(p) for p in override.split(":") if p]
    roots = [runs_root()]
    fallback = REPO / "runs" / "v2_extra"
    if fallback != runs_root() and fallback.exists():
        roots.append(fallback)
    return roots


def annotations_path(paper_id: str) -> Path:
    for root in runs_roots():
        p = root / paper_id / "dataset" / "annotations.json"
        if p.exists():
            return p
    # return the primary path so callers can surface the expected location
    return runs_root() / paper_id / "dataset" / "annotations.json"


def fingerprint(annotations: dict) -> dict:
    """Stable, diff-friendly summary of an annotations.json.

    - Bbox rounded to integer pixels (1px slack on either side is below the
      typically observed per-run variance at 200 DPI).
    - Labels keyed on a stable id (kind + label_id).
    - Category list kept sorted.
    """
    kinds = {c["id"]: c["name"] for c in annotations["categories"]}
    labels = []
    for a in annotations["annotations"]:
        labels.append(
            {
                "label_id": a["label_id"],
                "kind": kinds[a["category_id"]],
                "page": a["image_id"],
                "bbox": [int(round(v)) for v in a["bbox"]],
            }
        )
    labels.sort(key=lambda x: (x["page"], x["kind"], x["label_id"]))
    return {
        "categories": sorted({kinds[c["id"]] for c in annotations["categories"]}),
        "num_pages": len(annotations["images"]),
        "labels": labels,
    }


def load_annotations(paper_id: str) -> dict | None:
    path = annotations_path(paper_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_golden(paper_id: str) -> dict | None:
    path = GOLDEN_DIR / f"{paper_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_golden(paper_id: str, fp: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / f"{paper_id}.json").write_text(
        json.dumps(fp, indent=2) + "\n", encoding="utf-8"
    )


def bbox_close(a: list[int], b: list[int], tol: int = 2) -> bool:
    """A pair of bboxes agrees if every coordinate is within ``tol`` pixels."""
    return len(a) == len(b) == 4 and all(abs(x - y) <= tol for x, y in zip(a, b))


def compare_fingerprints(actual: dict, golden: dict, bbox_tol: int = 2) -> list[str]:
    """Return a list of human-readable mismatch messages (empty -> match)."""
    messages: list[str] = []
    if actual["categories"] != golden["categories"]:
        messages.append(
            f"categories changed: actual={actual['categories']} golden={golden['categories']}"
        )
    if actual["num_pages"] != golden["num_pages"]:
        messages.append(
            f"num_pages changed: actual={actual['num_pages']} golden={golden['num_pages']}"
        )
    golden_by_id = {(l["page"], l["label_id"]): l for l in golden["labels"]}
    actual_by_id = {(l["page"], l["label_id"]): l for l in actual["labels"]}
    for key in sorted(set(golden_by_id) | set(actual_by_id)):
        if key not in golden_by_id:
            messages.append(f"extra label: {key} {actual_by_id[key]}")
            continue
        if key not in actual_by_id:
            messages.append(f"missing label: {key}")
            continue
        g = golden_by_id[key]
        a = actual_by_id[key]
        if a["kind"] != g["kind"]:
            messages.append(f"kind drift {key}: actual={a['kind']} golden={g['kind']}")
        if not bbox_close(a["bbox"], g["bbox"], tol=bbox_tol):
            messages.append(
                f"bbox drift {key}: actual={a['bbox']} golden={g['bbox']}"
            )
    return messages
