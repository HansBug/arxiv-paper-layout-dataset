#!/usr/bin/env python3
"""Regenerate tests/golden/*.json snapshots from the current ``runs/v1``.

Usage:

    python scripts/regen_golden.py                  # every paper present
    python scripts/regen_golden.py --papers 2604.21245v1_... 2604.21800v1_...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._golden import (  # noqa: E402
    fingerprint,
    load_annotations,
    runs_root,
    save_golden,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="*", help="restrict to these paper ids")
    args = parser.parse_args()

    root = runs_root()
    if not root.exists():
        print(f"no cached workspace at {root}", file=sys.stderr)
        return 1

    targets = args.papers or sorted(p.name for p in root.iterdir() if p.is_dir())
    any_written = False
    for paper_id in targets:
        ann = load_annotations(paper_id)
        if ann is None:
            print(f"[skip] {paper_id} (no annotations)", file=sys.stderr)
            continue
        fp = fingerprint(ann)
        save_golden(paper_id, fp)
        any_written = True
        print(f"[wrote] {paper_id}: {len(fp['labels'])} labels, {fp['num_pages']} pages")

    return 0 if any_written else 1


if __name__ == "__main__":
    sys.exit(main())
