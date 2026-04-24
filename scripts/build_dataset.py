#!/usr/bin/env python3
"""Run the bbox pipeline over every validated paper in a source root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arxiv_layout.pipeline import process_paper  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/zhangshaoang/texlive-arxiv-validation/runs/full-20260424-132920"),
        help="Dir containing one subdirectory per paper, each with a src/ folder",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "v2",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--papers", nargs="*", default=None, help="restrict to these paper ids (by folder name)")
    parser.add_argument("--engine-hint", type=str, default=None)
    args = parser.parse_args()

    args.work_root.mkdir(parents=True, exist_ok=True)

    papers = []
    for entry in sorted(args.source_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "src").exists():
            continue
        if args.papers and entry.name not in args.papers:
            continue
        papers.append(entry)
        if args.limit and len(papers) >= args.limit:
            break

    results = []
    for paper in papers:
        print(f"[start] {paper.name}", flush=True)
        outcome = process_paper(paper, args.work_root, dpi=args.dpi, original_engine=args.engine_hint)
        print(
            f"[done] {paper.name} ok={outcome.ok} labels={outcome.labels} "
            f"pages={outcome.pages} reason={outcome.reason or '-'}",
            flush=True,
        )
        results.append(
            {
                "paper_id": outcome.paper_id,
                "ok": outcome.ok,
                "reason": outcome.reason,
                "pages": outcome.pages,
                "labels": outcome.labels,
                "labels_by_kind": outcome.labels_by_kind,
                "workspace": outcome.workspace,
            }
        )

    summary = {
        "source_root": str(args.source_root),
        "dpi": args.dpi,
        "papers": results,
        "total_ok": sum(1 for r in results if r["ok"]),
        "total_labels": sum(r["labels"] for r in results if r["ok"]),
    }
    (args.work_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
