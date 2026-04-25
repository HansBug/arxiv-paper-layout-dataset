#!/usr/bin/env python3
"""One-shot balance snapshot for the continuous corpus driver.

Designed to be wrapped in a slow polling loop (e.g. every 10 minutes)
so an agent can decide whether to edit ``<root>/control.json`` and
steer the crawler toward underrepresented kinds / archives.

Prints a single ``SNAPSHOT ...`` line with:

- totals (``papers_total``, ``papers_ok``, ``papers_failed``,
  ``pages_total``, ``pages_with_labels``, ``total_labels``)
- ``kinds``                    — full per-class histogram
- ``archives``                 — driver query-bucket histogram
- ``archive_coverage``         — e.g. ``20/20``
- ``untouched_archives``       — archives with zero OK paper
- ``top_cats``                 — top 10 ``primary_category``
- ``fail_reasons``             — histogram of failure reasons
- ``fails_by_cat``             — top 8 primary categories among failures
- ``control``                  — the currently-loaded control.json
                                 (minus ``note``) so the monitor sees
                                 which interventions are active.

Then, on a separate block, prints a 3-column **SUBSETS** table across
the three class-sets we care about at training time, applying the
spatial-pair (N:1 with containment) filter:

- ``8``  = all classes (figure / figure_cap / table / table_cap /
           algorithm / algorithm_cap / listing / listing_cap)
- ``6``  = drop listing pair (figure / figure_cap / table / table_cap /
           algorithm / algorithm_cap)  — the user's "6 label" meaning
- ``4``  = drop listing + algorithm (figure / figure_cap / table / table_cap)

For each subset: ``papers_pass`` (passing spatial-pair), ``pages_total``
(all pages of passing papers), ``pages_without_labels`` (pages of
passing papers with zero active-class instance — future YOLO negative
samples), and per-kind instance counts from passing papers.

Usage::

    python3 scripts/corpus_snapshot.py                 # default root
    python3 scripts/corpus_snapshot.py --root path/to/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arxiv_layout.corpus import ARXIV_ARCHIVES  # noqa: E402
from arxiv_layout.spatial_pair import (  # noqa: E402
    CLASSES,
    CLASS_SUBSETS,
    count_kinds,
    pages_label_stats,
    paper_passes_spatial_pairing,
)


def _fmt_cell(v: object) -> str:
    """Integers right-padded so the 3-column table stays readable."""
    if isinstance(v, int):
        return f"{v:>10}"
    return f"{str(v):>10}"


def _compute_subsets(
    workspaces_root: Path,
    papers_state: dict,
) -> dict[str, dict]:
    """Walk every OK paper's annotations.json, apply spatial-pair for
    each subset, and accumulate the stats the 3-column table needs.

    ``papers_state`` is ``state.papers`` so we can skip FAIL entries
    without touching the filesystem; ``spatial_pair_ok`` on the record
    is also honoured as a fast-path, falling back to a fresh re-check
    if it's missing (i.e. records ingested before the field existed).
    """
    per_subset: dict[str, dict] = {
        name: {
            "papers_pass": 0,
            "pages_total": 0,
            "pages_without_labels": 0,
            "kinds": {c: 0 for c in CLASSES},
        }
        for name in CLASS_SUBSETS
    }
    if not workspaces_root.is_dir():
        return per_subset

    for arxiv_id, rec in papers_state.items():
        if rec.get("status") != "ok":
            continue
        workspace = rec.get("workspace") or ""
        if not workspace:
            continue
        ann_path = Path(workspace) / "dataset" / "annotations.json"
        if not ann_path.is_file():
            continue
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue

        # cached qualification per subset ("8"/"6"/"4" -> bool)
        cached = rec.get("spatial_pair_ok") or {}
        kind_hist = count_kinds(ann)

        for name, classes in CLASS_SUBSETS.items():
            if name in cached:
                passes = bool(cached[name])
            else:
                passes = paper_passes_spatial_pairing(ann, classes)
            if not passes:
                continue
            bucket = per_subset[name]
            bucket["papers_pass"] += 1
            pages_total, pages_without = pages_label_stats(ann, classes)
            bucket["pages_total"] += pages_total
            bucket["pages_without_labels"] += pages_without
            for cls, n in kind_hist.items():
                if cls in bucket["kinds"]:
                    bucket["kinds"][cls] += n
    return per_subset


def _print_subset_table(subsets: dict[str, dict]) -> None:
    """Print a compact three-column text table.

    Row order: a human-facing explanation of each subset's label scope,
    then the size metrics, then the 8 per-class counts. **Classes that
    aren't in a subset's active list render as ``--``** — e.g. the
    6-label column shows ``--`` for ``algorithm`` / ``algorithm_cap``,
    and the 4-label column shows ``--`` for those plus ``listing`` /
    ``listing_cap``. A concrete number there would be misleading
    because those instances get dropped at export time under that
    subset.
    """
    names = list(CLASS_SUBSETS.keys())
    label_w = 18
    col_w = 12

    def _row(cells: list[str]) -> str:
        out = [cells[0].ljust(label_w)]
        for cell in cells[1:]:
            out.append(cell.rjust(col_w))
        return "  ".join(out)

    print("SUBSETS (spatial-pair N:1 with containment, IoU thresh 0.9)")
    for name in names:
        print(f"  {name}-label = {', '.join(CLASS_SUBSETS[name])}")
    print(_row(["", *[f"{name}-label" for name in names]]))
    print(_row(["-" * label_w, *["-" * col_w] * len(names)]))

    metrics = [
        ("papers_pass", "papers_pass"),
        ("pages_total", "pages_total"),
        ("pages_no_label", "pages_without_labels"),
    ]
    for label, key in metrics:
        print(_row([label, *[str(subsets[n][key]) for n in names]]))

    print(_row(["-- kinds --", *["" for _ in names]]))
    for cls in CLASSES:
        cells = [cls]
        for name in names:
            if cls in CLASS_SUBSETS[name]:
                cells.append(str(subsets[name]["kinds"][cls]))
            else:
                cells.append("--")
        print(_row(cells))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/corpus"))
    parser.add_argument(
        "--no-subsets",
        action="store_true",
        help="skip the 3-column SUBSETS table (useful for lightweight "
        "polling when you only want the single SNAPSHOT line).",
    )
    args = parser.parse_args()

    state_path = args.root / "state.json"
    control_path = args.root / "control.json"

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"SNAPSHOT-ERR state={state_path} err={exc}")
        return 0

    stats = state.get("stats", {}) or {}
    fails: dict[str, int] = {}
    fails_by_cat: dict[str, int] = {}
    for paper in (state.get("papers", {}) or {}).values():
        if paper.get("status") != "failed":
            continue
        reason = paper.get("reason") or "?"
        fails[reason] = fails.get(reason, 0) + 1
        cat = paper.get("primary_category") or "?"
        fails_by_cat[cat] = fails_by_cat.get(cat, 0) + 1

    top_cats = dict(
        sorted(
            (stats.get("primary_category_histogram") or {}).items(),
            key=lambda kv: -kv[1],
        )[:10]
    )
    top_fails_by_cat = dict(
        sorted(fails_by_cat.items(), key=lambda kv: -kv[1])[:8]
    )

    try:
        ctl = json.loads(control_path.read_text(encoding="utf-8"))
    except Exception:
        ctl = {}
    ctl_summary = {k: v for k, v in ctl.items() if k != "note"}

    archives_hist = stats.get("archive_histogram", {}) or {}
    untouched = sorted(a for a in ARXIV_ARCHIVES if a not in archives_hist)
    covered = sorted(a for a in ARXIV_ARCHIVES if a in archives_hist)

    papers_ok = stats.get("papers_ok", 0)
    papers_failed = stats.get("papers_failed", 0)
    papers_total = papers_ok + papers_failed
    kinds = stats.get("labels_by_kind", {}) or {}
    total_labels = sum(kinds.values())
    pages_total = stats.get("pages_total", 0)
    pages_with_labels = stats.get("pages_with_labels", 0)

    parts = [
        "SNAPSHOT",
        f"papers_total={papers_total}",
        f"papers_ok={papers_ok}",
        f"papers_failed={papers_failed}",
        f"pages_total={pages_total}",
        f"pages_with_labels={pages_with_labels}",
        f"total_labels={total_labels}",
        "kinds=" + json.dumps(kinds, ensure_ascii=False),
        "archives=" + json.dumps(archives_hist, ensure_ascii=False),
        f"archive_coverage={len(covered)}/{len(ARXIV_ARCHIVES)}",
        "untouched_archives=" + json.dumps(untouched, ensure_ascii=False),
        "top_cats=" + json.dumps(top_cats, ensure_ascii=False),
        "fail_reasons=" + json.dumps(fails, ensure_ascii=False),
        "fails_by_cat=" + json.dumps(top_fails_by_cat, ensure_ascii=False),
        "control=" + json.dumps(ctl_summary, ensure_ascii=False),
    ]
    print(" ".join(parts))

    if not args.no_subsets:
        subsets = _compute_subsets(
            args.root / "workspaces",
            state.get("papers", {}) or {},
        )
        _print_subset_table(subsets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
