"""Corpus-building utilities: resumable state store + balance-aware query
scheduler + running statistics.

The goal is to grow a detection dataset to million-image scale without
drifting off-balance in the domain / year distribution. Each arXiv
paper the driver processes produces one state entry. The aggregate
state + a small stats blob are recomputed after every successful
paper so a human (or a dashboard) can see the shape of the corpus
live.

State lives in a single ``state.json`` under the corpus root. All
mutating operations write to a tmp file and atomically rename, so a
killed / OOM'd / preempted driver can resume without corrupting data.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


# arXiv top-level archives -- we want each represented in roughly equal
# share. 20 buckets, pick the least-covered one when querying.
ARXIV_ARCHIVES: tuple[str, ...] = (
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


STATUS_QUEUED = "queued"
STATUS_FETCHED = "fetched"
STATUS_OK = "ok"
STATUS_FAILED = "failed"


@dataclass
class PaperRecord:
    """One paper's entry in the state file."""

    arxiv_id: str
    archive: str                     # which top-level archive it's counted under
    primary_category: str | None = None
    categories: list[str] = field(default_factory=list)
    year: str | None = None
    title: str = ""
    abs_url: str = ""
    source_url: str = ""
    status: str = STATUS_QUEUED
    reason: str = ""                 # failure reason (if status=failed)
    started_at: str = ""             # ISO
    finished_at: str = ""            # ISO
    pages_total: int = 0
    pages_with_labels: int = 0
    labels_total: int = 0
    labels_by_kind: dict[str, int] = field(default_factory=dict)
    box_counts_histogram: dict[str, int] = field(default_factory=dict)  # "{n}" -> count
    workspace: str = ""              # absolute path of per-paper workspace

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PaperRecord":
        data = dict(data)
        data.setdefault("categories", [])
        data.setdefault("labels_by_kind", {})
        data.setdefault("box_counts_histogram", {})
        return cls(**data)


@dataclass
class CorpusStats:
    """Rollup of the per-paper records. Kept in the state file so a live
    dashboard can read it without recomputing."""

    papers_total: int = 0
    papers_ok: int = 0
    papers_failed: int = 0
    pages_total: int = 0
    pages_with_labels: int = 0
    labels_total: int = 0
    labels_by_kind: dict[str, int] = field(default_factory=dict)
    archive_histogram: dict[str, int] = field(default_factory=dict)
    year_histogram: dict[str, int] = field(default_factory=dict)
    primary_category_histogram: dict[str, int] = field(default_factory=dict)
    box_counts_histogram: dict[str, int] = field(default_factory=dict)
    last_updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_papers(cls, records: Iterable[PaperRecord]) -> "CorpusStats":
        stats = cls()
        archive_h: Counter[str] = Counter()
        year_h: Counter[str] = Counter()
        cat_h: Counter[str] = Counter()
        box_h: Counter[str] = Counter()
        kind_h: Counter[str] = Counter()
        for rec in records:
            stats.papers_total += 1
            if rec.status == STATUS_OK:
                stats.papers_ok += 1
                stats.pages_total += rec.pages_total
                stats.pages_with_labels += rec.pages_with_labels
                stats.labels_total += rec.labels_total
                archive_h[rec.archive] += 1
                if rec.year:
                    year_h[rec.year] += 1
                if rec.primary_category:
                    cat_h[rec.primary_category] += 1
                for k, v in rec.labels_by_kind.items():
                    kind_h[k] += v
                for k, v in rec.box_counts_histogram.items():
                    box_h[k] += v
            elif rec.status == STATUS_FAILED:
                stats.papers_failed += 1
        stats.archive_histogram = dict(archive_h)
        stats.year_histogram = dict(year_h)
        stats.primary_category_histogram = dict(cat_h)
        stats.box_counts_histogram = dict(box_h)
        stats.labels_by_kind = dict(kind_h)
        stats.last_updated = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return stats


class CorpusState:
    """Thread-unsafe but process-crash-safe state store."""

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self.papers: dict[str, PaperRecord] = {}
        self.stats = CorpusStats()
        self.created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._dirty = False
        if path.is_file():
            self._load()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != self.VERSION:
            raise RuntimeError(f"unsupported state version {data.get('version')}")
        self.created_at = data.get("created_at", self.created_at)
        for rid, raw in data.get("papers", {}).items():
            self.papers[rid] = PaperRecord.from_dict(raw)
        if "stats" in data:
            stats_raw = dict(data["stats"])
            stats_raw.setdefault("labels_by_kind", {})
            stats_raw.setdefault("archive_histogram", {})
            stats_raw.setdefault("year_histogram", {})
            stats_raw.setdefault("primary_category_histogram", {})
            stats_raw.setdefault("box_counts_histogram", {})
            self.stats = CorpusStats(**stats_raw)

    def save(self) -> None:
        """Atomically persist state."""
        # refresh aggregate stats from the per-paper records, so stats is
        # always consistent with papers.
        self.stats = CorpusStats.from_papers(self.papers.values())
        data = {
            "version": self.VERSION,
            "created_at": self.created_at,
            "last_updated": self.stats.last_updated,
            "papers": {rid: rec.to_dict() for rid, rec in self.papers.items()},
            "stats": self.stats.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json.tmp",
            dir=str(self.path.parent),
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, self.path)
        self._dirty = False

    def upsert(self, rec: PaperRecord) -> None:
        self.papers[rec.arxiv_id] = rec
        self._dirty = True

    def get(self, arxiv_id: str) -> PaperRecord | None:
        return self.papers.get(arxiv_id)

    def seen(self, arxiv_id: str) -> bool:
        return arxiv_id in self.papers

    # --- stats helpers ----------------------------------------------------

    def archive_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for r in self.papers.values():
            if r.status == STATUS_OK:
                counts[r.archive] += 1
        return dict(counts)

    def total_images_with_labels(self) -> int:
        return sum(r.pages_with_labels for r in self.papers.values() if r.status == STATUS_OK)

    def total_labels(self) -> int:
        return sum(r.labels_total for r in self.papers.values() if r.status == STATUS_OK)


class BalancedQueryStrategy:
    """Picks the next ``(archive, year_bucket)`` to query.

    The rule is: pick the archive with the fewest successful papers so
    far; then inside that archive rotate through year buckets so we
    don't keep sampling the same corner of arxiv's timeline. Archives
    that have reached ``archive_quota`` are skipped.
    """

    def __init__(
        self,
        archives: Iterable[str] = ARXIV_ARCHIVES,
        archive_quota: int | None = None,
        year_buckets: tuple[tuple[int, int], ...] = (
            (2023, 2026),
            (2019, 2022),
            (2014, 2018),
            (2007, 2013),
        ),
        rng_seed: int = 0,
    ) -> None:
        self.archives = list(archives)
        self.archive_quota = archive_quota
        self.year_buckets = year_buckets
        self.rng = random.Random(rng_seed)

    def _next_year_bucket(self, state: CorpusState, archive: str) -> tuple[int, int]:
        # rotate buckets by least-covered year inside this archive
        year_counts: Counter[str] = Counter()
        for r in state.papers.values():
            if r.status == STATUS_OK and r.archive == archive and r.year:
                year_counts[r.year] += 1
        # score each bucket by total papers
        def score(bucket: tuple[int, int]) -> int:
            lo, hi = bucket
            total = 0
            for yr, cnt in year_counts.items():
                try:
                    y = int(yr[:4])
                except ValueError:
                    continue
                if lo <= y <= hi:
                    total += cnt
            return total
        buckets = sorted(self.year_buckets, key=score)
        # small randomness to avoid pathological determinism
        self.rng.shuffle(buckets[: max(1, len(buckets) // 2)])
        return buckets[0]

    def pick(
        self,
        state: CorpusState,
        *,
        avoid_archives: Iterable[str] | None = None,
        force_archive: str | None = None,
    ) -> tuple[str, tuple[int, int]] | None:
        """Choose the next ``(archive, year_bucket)`` to query.

        ``avoid_archives`` excludes archives from the rotation (e.g. a
        Monitor-driven intervention blacklisting one that's mostly
        failing). ``force_archive`` pins the archive if the caller wants
        to steer toward a specific underrepresented kind (still picks
        the least-covered year bucket inside).
        """
        avoid = set(avoid_archives or ())
        if force_archive and force_archive in self.archives and force_archive not in avoid:
            year_bucket = self._next_year_bucket(state, force_archive)
            return force_archive, year_bucket
        counts = state.archive_counts()
        candidates = [
            a
            for a in self.archives
            if a not in avoid
            and (self.archive_quota is None or counts.get(a, 0) < self.archive_quota)
        ]
        if not candidates:
            return None
        # pick archive with the smallest count, tie-break randomly among smallest
        min_count = min(counts.get(a, 0) for a in candidates)
        contenders = [a for a in candidates if counts.get(a, 0) == min_count]
        archive = self.rng.choice(contenders)
        year_bucket = self._next_year_bucket(state, archive)
        return archive, year_bucket


def arxiv_id_to_arxiv_stub(arxiv_id: str) -> str:
    """``2604.21931v1`` -> ``2604.21931`` (version-less id for slugging)."""
    return re.sub(r"v\d+$", "", arxiv_id)


def slug_for_paper(arxiv_id: str, title: str, archive: str) -> str:
    """Produce a stable, human-legible workspace slug for a paper.

    Pattern: ``<arxiv_id>_<archive>_<title-words>`` truncated to ~80 chars.
    The arxiv_id component already disambiguates, so we only need enough
    title to be recognizable in an ``ls``.
    """
    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    words = safe_title.split("_")[:6]
    suffix = "_".join(w for w in words if w)[:40]
    base = arxiv_id_to_arxiv_stub(arxiv_id).replace(".", "_")
    slug = f"{arxiv_id}_{archive.replace('-', '')}_{suffix}" if suffix else f"{arxiv_id}_{archive}"
    if len(slug) > 90:
        slug = slug[:90]
    return slug
