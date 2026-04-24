#!/usr/bin/env python3
"""One-shot balance snapshot for the continuous corpus driver.

Designed to be wrapped in a slow polling loop (e.g. every 10 minutes)
so an agent can decide whether to edit ``<root>/control.json`` and
steer the crawler toward underrepresented kinds / archives.

Prints a single ``SNAPSHOT ...`` line with:

- totals (``papers_ok``, ``papers_failed``, ``pages_with_labels``)
- ``kinds``                    — full per-class histogram
- ``archives``                 — driver query-bucket histogram
- ``top_cats``                 — top 10 ``primary_category``
- ``fail_reasons``             — histogram of failure reasons
- ``fails_by_cat``             — top 8 primary categories among failures
- ``control``                  — the currently-loaded control.json
                                 (minus ``note``) so the monitor sees
                                 which interventions are active.

Usage::

    python3 scripts/corpus_snapshot.py                 # default root
    python3 scripts/corpus_snapshot.py --root path/to/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/corpus"))
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

    parts = [
        "SNAPSHOT",
        f"papers_ok={stats.get('papers_ok', 0)}",
        f"papers_failed={stats.get('papers_failed', 0)}",
        f"pages_with_labels={stats.get('pages_with_labels', 0)}",
        "kinds=" + json.dumps(stats.get("labels_by_kind", {}), ensure_ascii=False),
        "archives="
        + json.dumps(stats.get("archive_histogram", {}), ensure_ascii=False),
        "top_cats=" + json.dumps(top_cats, ensure_ascii=False),
        "fail_reasons=" + json.dumps(fails, ensure_ascii=False),
        "fails_by_cat=" + json.dumps(top_fails_by_cat, ensure_ascii=False),
        "control=" + json.dumps(ctl_summary, ensure_ascii=False),
    ]
    print(" ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
