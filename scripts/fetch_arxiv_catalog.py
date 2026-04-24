#!/usr/bin/env python3
"""Bulk-fetch arXiv paper metadata, evenly across archives.

Queries the arXiv export API (``export.arxiv.org/api/query``) for every
top-level archive (cs, math, physics, astro-ph, cond-mat, gr-qc, hep-ex,
hep-lat, hep-ph, hep-th, math-ph, nlin, nucl-ex, nucl-th, q-bio, q-fin,
quant-ph, stat, eess, econ) and saves the flattened metadata to parquet.

For each paper we record

- ``arxiv_id``                    -- canonical id, e.g. ``2604.21931v1``
- ``abs_url``                     -- HTML abstract page
- ``pdf_url``                     -- PDF on arxiv.org
- ``source_url``                  -- e-print (source tarball); may be a
                                     ``.pdf`` passthrough if the author
                                     didn't upload latex, but most papers
                                     do carry latex source
- ``title`` / ``summary``         -- free text
- ``authors``                     -- list[str]
- ``year`` / ``updated``          -- publication timeline
- ``primary_category``            -- e.g. ``cs.CV``
- ``categories``                  -- all cross-listed categories
- ``comment`` / ``journal_ref`` / ``doi``

The fetcher respects the arxiv API's 3-second rate limit per request. A
paper that appears in multiple archives (cross-listed) is emitted once:
we dedupe by ``arxiv_id``.

Usage::

    python3 scripts/fetch_arxiv_catalog.py \\
        --out runs/arxiv_catalog.parquet \\
        --per-archive 100
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import feedparser
import pandas as pd


ARXIV_ARCHIVES = (
    "astro-ph",
    "cond-mat",
    "cs",
    "econ",
    "eess",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "physics",
    "q-bio",
    "q-fin",
    "quant-ph",
    "stat",
)


API = "http://export.arxiv.org/api/query"
USER_AGENT = "arxiv-paper-layout-dataset/1.0 (+https://github.com/HansBug/arxiv-paper-layout-dataset)"
RATE_LIMIT_SEC = 3.1  # arxiv asks for 3 seconds between queries


def fetch_archive_batch(archive: str, start: int, page_size: int) -> feedparser.FeedParserDict:
    # Some archives (quant-ph, gr-qc, hep-*, math-ph, nucl-*) have no
    # dot-subcategories so ``cat:<archive>.*`` returns 0 papers -- we need
    # ``cat:<archive>`` for those. Others (cs, math, physics, stat, ...) use
    # subcats like ``cs.CV``; ``cat:<archive>`` misses them. Query the
    # disjunction to cover both shapes in one API call.
    query = f"cat:{archive} OR cat:{archive}.*"
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": str(start),
        "max_results": str(page_size),
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to fetch {url}: {last_exc}")
    return feedparser.parse(data)


def parse_entry(entry: feedparser.FeedParserDict) -> dict:
    arxiv_id = entry.id.rsplit("/abs/", 1)[-1]
    primary_cat = None
    all_cats: list[str] = []
    if hasattr(entry, "tags"):
        for tag in entry.tags:
            term = tag.get("term")
            if term:
                all_cats.append(term)
        if all_cats:
            primary_cat = all_cats[0]
    # arxiv api carries a dedicated primary category element
    primary = getattr(entry, "arxiv_primary_category", None)
    if primary is not None:
        primary_cat = primary.get("term", primary_cat)

    authors = [a.get("name", "") for a in getattr(entry, "authors", [])]

    return {
        "arxiv_id": arxiv_id,
        "abs_url": entry.id,
        "pdf_url": next(
            (l.href for l in entry.links if getattr(l, "type", None) == "application/pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}",
        ),
        "source_url": f"https://arxiv.org/e-print/{arxiv_id}",
        "title": (entry.title or "").strip(),
        "summary": (entry.summary or "").strip(),
        "authors": authors,
        "primary_category": primary_cat,
        "categories": all_cats,
        "published": getattr(entry, "published", ""),
        "updated": getattr(entry, "updated", ""),
        "year": (getattr(entry, "published", "") or "")[:4],
        "comment": getattr(entry, "arxiv_comment", ""),
        "journal_ref": getattr(entry, "arxiv_journal_ref", ""),
        "doi": getattr(entry, "arxiv_doi", ""),
        "status": "ok",
    }


def fetch_one_archive(archive: str, target: int, page_size: int = 100) -> list[dict]:
    collected: list[dict] = []
    start = 0
    seen: set[str] = set()
    while len(collected) < target:
        want = min(page_size, target - len(collected))
        parsed = fetch_archive_batch(archive, start=start, page_size=want)
        entries = list(parsed.entries)
        if not entries:
            break
        for entry in entries:
            row = parse_entry(entry)
            if row["arxiv_id"] in seen:
                continue
            seen.add(row["arxiv_id"])
            collected.append(row)
            if len(collected) >= target:
                break
        start += page_size
        time.sleep(RATE_LIMIT_SEC)
        # arxiv sometimes returns fewer than requested without advancing; bail
        if len(entries) < want:
            break
    return collected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="output parquet path")
    parser.add_argument(
        "--per-archive",
        type=int,
        default=100,
        help="max number of papers to fetch per archive (default 100)",
    )
    parser.add_argument(
        "--archives",
        nargs="*",
        default=list(ARXIV_ARCHIVES),
        help="which arxiv archives to query (default: all 20)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="arxiv API max_results per request (<= 2000; default 50)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for archive in args.archives:
        print(f"[fetch] {archive} target={args.per_archive}", flush=True)
        try:
            rows = fetch_one_archive(archive, args.per_archive, page_size=args.page_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {archive}: {exc}", flush=True)
            continue
        print(f"  got {len(rows)} papers", flush=True)
        for row in rows:
            row["queried_archive"] = archive
        all_rows.extend(rows)

    if not all_rows:
        print("no rows collected", file=sys.stderr)
        return 1

    # dedupe by arxiv_id across archives; keep the first occurrence but remember
    # every archive that surfaced the paper
    by_id: dict[str, dict] = {}
    for row in all_rows:
        existing = by_id.get(row["arxiv_id"])
        if existing is None:
            row["also_in_archives"] = [row["queried_archive"]]
            by_id[row["arxiv_id"]] = row
        else:
            existing["also_in_archives"].append(row["queried_archive"])
    unique = list(by_id.values())

    df = pd.DataFrame(unique)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(
        f"[done] {len(df)} unique papers written to {args.out} "
        f"(total {len(all_rows)} hits incl. cross-listed duplicates)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
