#!/usr/bin/env python3
"""Pretty-print the live corpus stats from ``state.json``.

Useful while ``run_corpus_pipeline.py`` is running in the background:
run this from a second terminal (or via a cron / tmux tile) to watch
the distributions evolve without interrupting the driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


def fmt(counter: dict[str, int], top: int) -> list[str]:
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [f"  {k:30s} {v:6d}" for k, v in items]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True, help="corpus state.json")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if not args.state.is_file():
        print(f"no state file at {args.state}", file=sys.stderr)
        return 1

    data = json.loads(args.state.read_text(encoding="utf-8"))
    stats = data["stats"]
    papers = data["papers"]

    print(f"corpus: {args.state}")
    print(f"last-updated:    {stats.get('last_updated', '?')}")
    print(f"papers seen:     {len(papers)}")
    print(f"papers ok:       {stats['papers_ok']}")
    print(f"papers failed:   {stats['papers_failed']}")
    print(f"pages total:     {stats['pages_total']}")
    print(f"pages w/ labels: {stats['pages_with_labels']}")
    print(f"labels total:    {stats['labels_total']}")

    for header, key in [
        ("archives", "archive_histogram"),
        ("primary categories", "primary_category_histogram"),
        ("years", "year_histogram"),
        ("kinds", "labels_by_kind"),
        ("boxes / page", "box_counts_histogram"),
    ]:
        print()
        print(f"top {args.top} {header}:")
        for line in fmt(stats.get(key, {}), args.top):
            print(line)

    # breakdown of failure reasons
    reasons = Counter(
        p.get("reason", "") for p in papers.values() if p["status"] == "failed"
    )
    if reasons:
        print()
        print(f"top {args.top} failure reasons:")
        for line in fmt(reasons, args.top):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
