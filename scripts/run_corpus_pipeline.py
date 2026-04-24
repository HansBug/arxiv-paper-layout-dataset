#!/usr/bin/env python3
"""Continuous balance-aware corpus-building driver.

Loop forever (or until ``--max-images`` is reached):

1. Look at the current per-archive paper counts in ``state.json``.
2. Pick the least-covered ``(archive, year_bucket)`` pair.
3. Query the arXiv export API for a few candidate papers in that
   bucket we haven't seen yet.
4. For each candidate:
   - download the e-print,
   - stand up a per-paper workspace,
   - run the ``process_paper`` pipeline (inject / compile / extract),
   - update the state + stats,
   - atomically persist state to disk,
   - print a live summary.
5. Sleep a beat for arXiv rate limits; repeat.

Design pillars:

- **Resumable** -- state.json is the single source of truth. Any paper
  already logged (success OR failure) is skipped on restart, so you
  can Ctrl-C / lose the node / hit OOM and just re-launch.
- **Real-time visibility** -- after every paper, the driver prints an
  ASCII table of archive / year / label-kind distributions. The same
  blob is persisted to state.json, so external dashboards can read it
  without a round-trip through the driver.
- **Self-balancing** -- the scheduler picks archives inversely to
  their current share. This keeps the dataset broad-coverage as it
  grows without manual babysitting.
- **Clean-up** -- by default a successful paper keeps only
  ``dataset/annotations.json`` + ``pages/*.png`` (small). Compile
  intermediates (``.aux`` / ``.log`` / .tex after inject / PDF) are
  deleted so the corpus doesn't explode on disk. Use
  ``--keep-workspace`` to preserve the full per-paper tree.

Usage (typical cluster invocation, see README)::

    python3 scripts/run_corpus_pipeline.py \\
        --root runs/corpus \\
        --max-images 1_000_000 \\
        --archive-quota 50000 \\
        --candidates-per-query 25
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import shutil
import signal
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import feedparser  # fetcher dependency

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arxiv_layout.corpus import (  # noqa: E402
    ARXIV_ARCHIVES,
    BalancedQueryStrategy,
    CorpusState,
    CorpusStats,
    PaperRecord,
    STATUS_FAILED,
    STATUS_FETCHED,
    STATUS_OK,
    slug_for_paper,
)
from arxiv_layout.pipeline import process_paper  # noqa: E402


ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "arxiv-paper-layout-dataset/1.0 (continuous-pipeline)"
RATE_LIMIT_SEC = 3.1

CONTROL_FILENAME = "control.json"


def _load_control(root: Path) -> dict:
    """Hot-reload an intervention file each step without restarting the
    driver. Missing / malformed files degrade gracefully to no-op. Keys:

    - ``skip_primary_cats`` (list[str]): candidate papers whose
      ``primary_category`` is in this list are dropped before the
      pipeline runs. Use this to blacklist a sub-archive that keeps
      failing (e.g. ``nlin.SI`` when it's mostly un-instrumentable
      pure-math papers).
    - ``skip_archive_query`` (list[str]): archives excluded from
      :meth:`BalancedQueryStrategy.pick`'s rotation.
    - ``force_next_archive`` (str | null): override the scheduler for
      the next query (useful to prioritise an underrepresented class
      like ``cs`` when ``algorithm`` / ``listing`` are still 0).
    - ``note`` (str): free-form comment (ignored by code, shown in log).
    """
    path = root / CONTROL_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[control] WARN: failed to read {path}: {exc}", flush=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _arxiv_api_call(url: str, attempts: int = 5, timeout: int = 60) -> bytes:
    """Call the arXiv export API with 429-aware exponential backoff.

    arXiv's listing API is aggressive about throttling bursty pollers
    (especially right after a lot of metadata queries). Generic
    ``time.sleep(2*(attempt+1))`` is too short — a 429 typically needs
    30-120s of silence. Detect 429 explicitly and sleep 30s, 60s,
    120s, 240s, 480s on successive retries. Other errors keep the
    original short linear backoff.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                delay = 30 * (2 ** attempt)  # 30, 60, 120, 240, 480
                print(
                    f"  [rate-limit] 429 from arxiv; sleeping {delay}s "
                    f"(attempt {attempt + 1}/{attempts})",
                    flush=True,
                )
                time.sleep(delay)
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"arxiv api call failed after {attempts} attempts: {last_exc}") from last_exc


def fetch_candidates(
    archive: str,
    year_lo: int,
    year_hi: int,
    *,
    max_results: int,
    seen_ids: set[str],
    offset: int = 0,
) -> list[dict]:
    """Query arxiv for papers in ``cat:<archive> OR cat:<archive>.*`` within
    the given submission-year window. Returns lightweight metadata dicts.
    """
    query = f"cat:{archive} OR cat:{archive}.*"
    # arxiv's query syntax doesn't officially support year-in-submittedDate
    # filtering cleanly, so we over-fetch and filter by published year client-side.
    # We still sort by recency so the first pages are recent papers.
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": str(offset),
        "max_results": str(max_results),
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    data = _arxiv_api_call(url)
    parsed = feedparser.parse(data)

    out: list[dict] = []
    for entry in parsed.entries:
        arxiv_id = entry.id.rsplit("/abs/", 1)[-1]
        if arxiv_id in seen_ids:
            continue
        published = getattr(entry, "published", "") or ""
        try:
            year = int(published[:4])
        except ValueError:
            year = 0
        if year and not (year_lo <= year <= year_hi):
            continue
        primary = getattr(entry, "arxiv_primary_category", None)
        primary_cat = primary.get("term") if primary else None
        categories = [t.get("term", "") for t in getattr(entry, "tags", [])]
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": (entry.title or "").strip(),
                "abs_url": entry.id,
                "source_url": f"https://arxiv.org/e-print/{arxiv_id}",
                "primary_category": primary_cat,
                "categories": categories,
                "year": str(year) if year else "",
                "published": published,
            }
        )
    return out


def _download_bytes(url: str, dest: Path, timeout: int = 180) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    dest.write_bytes(payload)


def _extract_source_blob(blob: Path, dest: Path, fallback_name: str) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    raw = blob.read_bytes()

    def safe_extract(tar: tarfile.TarFile) -> None:
        base = dest.resolve()
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing symlink in tar: {member.name}")
            p = (dest / member.name).resolve()
            if not str(p).startswith(str(base)):
                raise RuntimeError(f"escape: {member.name}")
        tar.extractall(dest)

    for payload in (raw, None):
        if payload is None:
            try:
                payload = gzip.decompress(raw)
            except OSError:
                continue
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tar:
                safe_extract(tar)
            return
        except tarfile.TarError:
            continue

    text = raw.decode("utf-8", errors="ignore")
    if "\\documentclass" in text:
        (dest / f"{fallback_name}.tex").write_text(text, encoding="utf-8")
        return

    raise RuntimeError("unable to extract arxiv source: not tar, not gz, not tex")


def prepare_paper_workspace(candidate: dict, archive: str, sources_root: Path) -> Path:
    """Download + unpack arXiv e-print into ``sources_root/<slug>/src``.

    Returns the paper root (``sources_root/<slug>``); it is the input to
    ``pipeline.process_paper``, which in turn writes its output under a
    different root (``workspaces_root``).
    """
    slug = slug_for_paper(candidate["arxiv_id"], candidate["title"], archive)
    paper_src_root = sources_root / slug
    src = paper_src_root / "src"
    downloads = paper_src_root / "downloads"
    shutil.rmtree(paper_src_root, ignore_errors=True)
    src.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    blob = downloads / "source.blob"
    _download_bytes(candidate["source_url"], blob)
    _extract_source_blob(blob, src, fallback_name=slug)
    return paper_src_root


def box_count_per_page(annotations: dict) -> Counter[int]:
    page_counts: Counter[int] = Counter()
    for ann in annotations["annotations"]:
        page_counts[ann["image_id"]] += 1
    hist: Counter[int] = Counter()
    # include pages with 0 annotations (they're in annotations["images"] but
    # absent from annotations["annotations"])
    seen_pages = {a["image_id"] for a in annotations["annotations"]}
    for img in annotations["images"]:
        if img["id"] not in seen_pages:
            hist[0] += 1
    for _, cnt in page_counts.items():
        hist[cnt] += 1
    return hist


def _finalise_workspace(workspace: Path, keep_full: bool) -> None:
    """Trim a successful workspace to the minimum needed for dataset export."""
    if keep_full:
        return
    # delete: src/ (compile products), downloads/ -- huge and redundant.
    shutil.rmtree(workspace / "src", ignore_errors=True)
    shutil.rmtree(workspace / "downloads", ignore_errors=True)


def process_candidate(
    candidate: dict,
    archive: str,
    sources_root: Path,
    workspaces_root: Path,
    *,
    keep_workspace: bool,
    dpi: int,
) -> PaperRecord:
    """Download + compile + extract one candidate paper. Returns a populated
    PaperRecord (status OK or failed)."""

    arxiv_id = candidate["arxiv_id"]
    rec = PaperRecord(
        arxiv_id=arxiv_id,
        archive=archive,
        primary_category=candidate.get("primary_category"),
        categories=candidate.get("categories", []),
        year=candidate.get("year", ""),
        title=candidate.get("title", ""),
        abs_url=candidate.get("abs_url", ""),
        source_url=candidate.get("source_url", ""),
        status=STATUS_FETCHED,
        started_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    try:
        paper_src_root = prepare_paper_workspace(candidate, archive, sources_root)
    except Exception as exc:  # noqa: BLE001
        rec.status = STATUS_FAILED
        rec.reason = f"download/extract: {exc}"
        rec.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return rec

    try:
        outcome = process_paper(paper_src_root, workspaces_root, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        rec.status = STATUS_FAILED
        rec.reason = f"pipeline: {exc}"
        rec.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return rec

    rec.workspace = outcome.workspace or ""
    if not outcome.ok:
        rec.status = STATUS_FAILED
        rec.reason = outcome.reason or "unknown pipeline failure"
        rec.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return rec

    rec.status = STATUS_OK
    rec.pages_total = outcome.pages
    rec.labels_total = outcome.labels
    rec.labels_by_kind = dict(outcome.labels_by_kind)

    # compute page-level stats from the just-written annotations.json
    ann_path = Path(outcome.workspace) / "dataset" / "annotations.json"
    if ann_path.is_file():
        import json as _json
        ann = _json.loads(ann_path.read_text(encoding="utf-8"))
        pages_with_labels = {a["image_id"] for a in ann["annotations"]}
        rec.pages_with_labels = len(pages_with_labels)
        hist = box_count_per_page(ann)
        rec.box_counts_histogram = {str(k): v for k, v in hist.items()}

    _finalise_workspace(Path(outcome.workspace), keep_full=keep_workspace)

    # remove the downloaded+extracted e-print now that the workspace has the
    # compiled results (the raw source is never needed again).
    if not keep_workspace:
        shutil.rmtree(paper_src_root, ignore_errors=True)

    rec.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return rec


# ---------------------------------------------------------------------------
# Stats pretty-printing


def _short_hist(counter: dict[str, int], top: int = 10) -> str:
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return ", ".join(f"{k}={v}" for k, v in items) if items else "(empty)"


def print_dashboard(state: CorpusState, started_at: float) -> None:
    stats = state.stats
    elapsed = time.time() - started_at
    lines = [
        "=" * 80,
        f"corpus @ {stats.last_updated}  (driver uptime: {elapsed:.0f}s)",
        f"papers: total={stats.papers_total}  ok={stats.papers_ok}  "
        f"failed={stats.papers_failed}",
        f"pages:  total={stats.pages_total}  "
        f"with-labels={stats.pages_with_labels}  labels={stats.labels_total}",
        "kinds:  " + _short_hist(stats.labels_by_kind, top=12),
        "archives (top 10): " + _short_hist(stats.archive_histogram, top=10),
        "years (top 10):    " + _short_hist(stats.year_histogram, top=10),
        "cats (top 10):     " + _short_hist(stats.primary_category_histogram, top=10),
        "box/page (top 10): " + _short_hist(stats.box_counts_histogram, top=10),
        "=" * 80,
    ]
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Driver


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.root.resolve()
        self.sources_root = self.root / "sources"
        self.workspaces_root = self.root / "workspaces"
        self.state_path = self.root / "state.json"
        self.state = CorpusState(self.state_path)
        self.strategy = BalancedQueryStrategy(
            archives=args.archives or ARXIV_ARCHIVES,
            archive_quota=args.archive_quota,
            rng_seed=args.seed,
        )
        self.started_at = time.time()
        self.stop_requested = False

        self.root.mkdir(parents=True, exist_ok=True)
        self.sources_root.mkdir(parents=True, exist_ok=True)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)

    def handle_signal(self, signum, _frame) -> None:  # noqa: ANN001
        print(f"[signal {signum}] requesting graceful stop...", flush=True)
        self.stop_requested = True

    # ------------------------------------------------------------------

    def step(self) -> bool:
        """Run a single query + process cycle. Returns True if work done."""
        control = _load_control(self.root)
        skip_primary_cats = set(control.get("skip_primary_cats") or [])
        skip_archive_query = control.get("skip_archive_query") or []
        force_next_archive = control.get("force_next_archive")
        if control:
            parts = []
            if skip_primary_cats:
                parts.append(f"skip_cats={sorted(skip_primary_cats)}")
            if skip_archive_query:
                parts.append(f"skip_archives={list(skip_archive_query)}")
            if force_next_archive:
                parts.append(f"force={force_next_archive}")
            if control.get("note"):
                parts.append(f"note={control['note']!r}")
            if parts:
                print("[control] " + " | ".join(parts), flush=True)

        pick = self.strategy.pick(
            self.state,
            avoid_archives=skip_archive_query,
            force_archive=force_next_archive,
        )
        if pick is None:
            print("[stop] no archives below quota.", flush=True)
            return False
        archive, (year_lo, year_hi) = pick
        print(
            f"[query] archive={archive} years={year_lo}-{year_hi} "
            f"seen={len(self.state.papers)}",
            flush=True,
        )

        seen_ids = set(self.state.papers.keys())
        try:
            candidates = fetch_candidates(
                archive,
                year_lo,
                year_hi,
                max_results=self.args.candidates_per_query,
                seen_ids=seen_ids,
                offset=self._offset_for(archive),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] query failed: {exc}", flush=True)
            time.sleep(RATE_LIMIT_SEC)
            return True

        time.sleep(RATE_LIMIT_SEC)

        if not candidates:
            print("  no fresh candidates; bumping offset", flush=True)
            self._bump_offset(archive)
            return True

        for cand in candidates:
            if self.stop_requested:
                return True
            if self.state.seen(cand["arxiv_id"]):
                continue
            primary = cand.get("primary_category") or ""
            if primary in skip_primary_cats:
                print(
                    f"  [skip-cat] {cand['arxiv_id']} [{primary}]  "
                    f"(primary in skip_primary_cats)",
                    flush=True,
                )
                continue
            print(
                f"  -> {cand['arxiv_id']} [{primary or '?'}]  "
                f"{cand['title'][:60]}",
                flush=True,
            )
            rec = process_candidate(
                cand,
                archive,
                sources_root=self.sources_root,
                workspaces_root=self.workspaces_root,
                keep_workspace=self.args.keep_workspace,
                dpi=self.args.dpi,
            )
            self.state.upsert(rec)
            self.state.save()
            tag = "OK" if rec.status == STATUS_OK else f"FAIL({rec.reason})"
            kind_str = ", ".join(
                f"{k}={v}" for k, v in sorted(rec.labels_by_kind.items())
            ) or "-"
            print(
                f"     {tag}  [{rec.primary_category or '?'}]  "
                f"pages={rec.pages_total}  "
                f"labels={rec.labels_total} ({kind_str})",
                flush=True,
            )

            if self.stop_requested:
                return True
            # Periodic dashboard
            if self.state.stats.papers_total % max(1, self.args.dashboard_every) == 0:
                print_dashboard(self.state, self.started_at)
            if self._target_reached():
                return False
        return True

    _offset_overrides: dict[str, int] = {}

    def _offset_for(self, archive: str) -> int:
        return self._offset_overrides.get(archive, 0)

    def _bump_offset(self, archive: str) -> None:
        self._offset_overrides[archive] = (
            self._offset_overrides.get(archive, 0) + self.args.candidates_per_query
        )

    def _target_reached(self) -> bool:
        if self.args.max_images is not None and self.state.total_images_with_labels() >= self.args.max_images:
            print(
                f"[stop] reached max-images={self.args.max_images}",
                flush=True,
            )
            return True
        if self.args.max_papers is not None and self.state.stats.papers_ok >= self.args.max_papers:
            print(
                f"[stop] reached max-papers={self.args.max_papers}",
                flush=True,
            )
            return True
        return False

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        print_dashboard(self.state, self.started_at)

        while not self.stop_requested:
            if self._target_reached():
                break
            continued = self.step()
            if self.stop_requested:
                break
            if not continued:
                break
            time.sleep(RATE_LIMIT_SEC)

        self.state.save()
        print_dashboard(self.state, self.started_at)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "runs" / "corpus",
        help="corpus root (state.json + workspaces/ live here)",
    )
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="stop when pages_with_labels >= this",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--archive-quota",
        type=int,
        default=None,
        help="upper bound on successful papers per archive (None = unlimited)",
    )
    parser.add_argument("--archives", nargs="*")
    parser.add_argument(
        "--candidates-per-query",
        type=int,
        default=15,
        help="how many papers to fetch from arxiv per API call",
    )
    parser.add_argument(
        "--dashboard-every",
        type=int,
        default=1,
        help="print the dashboard every N successful papers",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="keep src/ + downloads/ for every processed paper "
        "(debugging only; blows up disk)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    driver = Driver(args)
    return driver.run()


if __name__ == "__main__":
    sys.exit(main())
