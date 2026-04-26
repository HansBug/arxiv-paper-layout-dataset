#!/usr/bin/env python3
"""Export a pipeline output tree (``runs/v2_validated`` + ``runs/v2_extra``)
into Ultralytics YOLO format, deterministically split 8:1:1.

The YOLO layout this produces::

    <out>/
      data.yaml                  -- class names, split paths
      dataset_meta.json          -- structured metadata (provenance, stats)
      train_recommended.yaml     -- ready-to-run ultralytics training config
      manifest.sha256            -- file integrity manifest
      README.md                  -- summary, embedded plots, provenance
      analysis/                  -- auto-generated diagnostic plots
        class_counts.png
        bbox_centers.png
        bbox_aspect.png
        bbox_size.png
        page_aspect.png
        labels_per_image.png
        archive_class.png
        preview.png              -- 4×4 sample mosaic with bboxes
      images/
        train/<paper_id>__page_<NNN>.jpg   (default: JPG, short-side <= 720)
        val/...
        test/...
      labels/
        train/<paper_id>__page_<NNN>.txt
        val/...
        test/...

Label files are plain YOLO: one row per instance, each row is::

    <class_index> <cx_norm> <cy_norm> <w_norm> <h_norm>

Pages with no annotations of any active class are exported as pure
negative samples (an image paired with an empty ``.txt`` label file).
YOLO treats these as background examples, which reduces false positives.
Use ``--skip-negatives`` to drop them instead.

Paper-level filter (spatial-pair, N:1 with containment) is ON by
default: a paper is kept iff every active body/cap pair is spatially
valid — every body bbox is mostly contained in some cap bbox, and
every cap bbox contains at least one body. Orphan body or empty cap
rejects the paper. ``--strict-1to1`` adds a per-page-count constraint
on top; ``--no-filter`` exports everything.

Subsets:
- ``--subset 4`` is shorthand for the 4-class figure/table subset.
- ``--subset 6`` adds algorithm.
- ``--subset 8`` is the full 8-class set (all caption pairs).
- Or pass ``--classes a,b,c`` for an explicit list.

Image processing:
- Default output format: JPG (``--format jpg``; ``--format png`` keeps PNG).
- Default resize policy: if ``min(width, height) > 720``, scale so the
  shorter side equals 720. Override with ``--max-short-side``; use 0 to
  disable resizing. Boxes are re-normalised automatically because YOLO
  labels are already relative to the (possibly-resized) image size.

Sampling (small smoke-test exports):
- ``--sample N`` caps the dataset to ``N`` images total.
- ``--sample-strategy`` controls which pages get picked:
  * ``balanced`` (default) — round-robin by archive, and within each
    archive prefer pages that contribute rarer classes.
  * ``class-balanced`` — assign a per-class quota and round-robin
    over classes; guarantees rarest classes appear in tiny samples.
  * ``by-archive`` — pure round-robin across archives.
  * ``random`` — deterministic shuffle by ``--sample-seed``.
- ``--neg-ratio FLOAT`` — if set, reserve this fraction of the sample
  for negative pages (e.g. ``--neg-ratio 0.3`` -> 30% negatives).

Quality / dedup filters (apply during candidate enumeration):
- ``--max-pages-per-paper N`` — protect against single-paper domination.
- ``--max-labels-per-paper N`` — drop long-tail mega-papers (~PhD theses).
- ``--min-bbox-area FRAC`` — drop bboxes smaller than this fraction of
  the page area; useful for stripping artifact-tiny boxes.

Split determinism:
- The file name contains both the arxiv paper id and the page number, so
  it uniquely identifies the sample.
- Split membership is picked by ``sha256(filename) % 10`` mapped to
  train (0-7) / val (8) / test (9). Given the same filename, the same
  split is always assigned regardless of re-runs or input order.

Dataset card / training:
- ``analysis/`` plots + ``README.md`` describe the export.
- ``dataset_meta.json`` is the structured equivalent for tooling.
- ``train_recommended.yaml`` is a sane starting config for
  ``ultralytics`` (imgsz / lr / aug). Feed it to ``yolo train ...``.
- ``manifest.sha256`` lists every emitted file's hash for downstream
  integrity checks.

Usage::

    # 100-image smoke test, 4 classes, with full dataset card
    python3 scripts/export_yolo.py \\
        --input runs/corpus/workspaces \\
        --out runs/yolo_smoke \\
        --subset 4 --sample 100

    # Full 6-class export, drop bbox below 0.1% page area
    python3 scripts/export_yolo.py \\
        --input runs/corpus/workspaces \\
        --out runs/yolo_full_6 \\
        --subset 6 --min-bbox-area 0.001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Shared spatial-pair judgement logic lives in the library so the
# driver's per-paper broadcast and Monitor B's subset table can apply
# the exact same predicates.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arxiv_layout.spatial_pair import (  # noqa: E402
    CLASSES,
    CLASS_SUBSETS,
    count_kinds,
    paper_passes_spatial_pairing,
    paper_passes_strict_1to1,
)

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}

# Map internal split key -> directory name on disk. We keep ``val`` as
# the internal key (Counters, stats) but write the canonical Roboflow
# layout (``valid/`` on disk) so the resulting dataset slots into any
# Ultralytics tooling that expects ``../valid/images``-style paths.
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pick_split(stem: str, weights: tuple[int, int, int]) -> str:
    """Return one of ``train/val/test`` deterministically from ``stem``.

    The same ``stem`` always maps to the same split. The probability
    distribution matches ``weights`` (normalised to 10 buckets).
    """
    total = sum(weights)
    assert total > 0
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % total
    cutoff = weights[0]
    if bucket < cutoff:
        return "train"
    cutoff += weights[1]
    if bucket < cutoff:
        return "val"
    return "test"


def build_stem(paper_id: str, page_id: int) -> str:
    """Single filename stem that captures ``(arxiv_id, page_id)``.

    The split picker hashes this, so anyone regenerating the export
    without touching the hash input gets the same split.
    """
    return f"{paper_id}__page_{page_id:03d}"


def yolo_label_lines(
    annotations: dict,
    image_id_to_size: dict[int, tuple[int, int]],
    class_to_index: dict[str, int] | None = None,
    min_bbox_area: float = 0.0,
) -> dict[int, list[str]]:
    """For each page image id, the list of YOLO rows to write.

    ``class_to_index`` maps active class names to their output YOLO index.
    Annotations whose kind is missing from this map are silently dropped
    (so passing ``{"figure": 0, "figure_cap": 1}`` produces a 2-class dataset).
    Defaults to the full 8-class mapping.

    ``min_bbox_area`` drops any normalized bbox whose ``w * h`` is below
    the threshold (page area = 1.0). 0.0 disables. A typical value of
    ``0.001`` strips boxes smaller than 0.1% of the page — usually the
    artifact-tiny ones from injection edge cases.
    """
    if class_to_index is None:
        class_to_index = CLASS_TO_INDEX
    kinds = {c["id"]: c["name"] for c in annotations["categories"]}
    per_image: dict[int, list[str]] = {}
    for item in annotations["annotations"]:
        kind = kinds.get(item["category_id"])
        if kind is None or kind not in class_to_index:
            continue
        x, y, w, h = item["bbox"]
        img_w, img_h = image_id_to_size.get(item["image_id"], (0, 0))
        if img_w <= 0 or img_h <= 0:
            continue
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        # clip into [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        if nw <= 0.0 or nh <= 0.0:
            continue
        if min_bbox_area > 0 and nw * nh < min_bbox_area:
            continue
        per_image.setdefault(item["image_id"], []).append(
            f"{class_to_index[kind]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
    return per_image


def iter_papers(input_root: Path):
    if not input_root.is_dir():
        return
    for entry in sorted(input_root.iterdir()):
        if not entry.is_dir():
            continue
        ann = entry / "dataset" / "annotations.json"
        pages_dir = entry / "pages"
        if not ann.is_file() or not pages_dir.is_dir():
            continue
        yield entry.name, ann, pages_dir


# ---------------------------------------------------------------------------
# Archive lookup
# ---------------------------------------------------------------------------


def _load_archive_lookup(
    corpus_state: Path | None, inputs: list[Path]
) -> dict[str, str]:
    """Map workspace dirname (slug) -> arxiv archive (e.g. "astro-ph").

    Source of truth is ``corpus/state.json`` if available — that's the
    only place that records which archive each paper came from. We try
    the explicit ``--corpus-state`` path first, then fall back to the
    typical ``runs/corpus/state.json`` next to ``runs/corpus/workspaces``.
    """
    paths_to_try: list[Path] = []
    if corpus_state is not None:
        paths_to_try.append(Path(corpus_state))
    for inp in inputs:
        candidate = inp.resolve().parent / "state.json"
        paths_to_try.append(candidate)

    state_path = next((p for p in paths_to_try if p.is_file()), None)
    if state_path is None:
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: dict[str, str] = {}
    papers = state.get("papers", {}) or {}
    for record in papers.values():
        archive = record.get("archive")
        if not archive:
            continue
        ws = record.get("workspace")
        if ws:
            slug = Path(ws).name
            out[slug] = archive
    return out


def _git_commit_short() -> str | None:
    """Best-effort: return the current repo's HEAD commit (short). Used
    only to record provenance — silent on failure."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Candidate enumeration + sampling
# ---------------------------------------------------------------------------


def _process_one_paper(
    paper_id: str,
    ann_path: Path,
    pages_dir: Path,
    weights: tuple[int, int, int],
    active_classes: tuple[str, ...],
    class_to_index: dict[str, int],
    filter_mode: str,
    spatial_iou_thresh: float,
    include_negatives: bool,
    archive_lookup: dict[str, str],
    min_bbox_area: float,
    max_labels_per_paper: int,
    max_pages_per_paper: int,
    rng_seed: int,
) -> dict:
    """Per-paper worker for ``_collect_candidates``: returns a dict with
    the paper's accepted page candidates *and* whatever counter deltas
    the caller should fold in. All inputs are immutable / per-call so
    this is safe to call from multiple threads in parallel.
    """
    out = {
        "candidates": [],
        "delta": {
            "skipped_no_labels": 0,
            "skipped_papers_not_1to1": 0,
            "skipped_papers_no_spatial_pair": 0,
            "skipped_papers_too_many_labels": 0,
            "skipped_pages_per_paper_cap": 0,
        },
    }
    try:
        annotations = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if filter_mode == "strict":
        if not paper_passes_strict_1to1(
            annotations, active_classes, spatial_iou_thresh
        ):
            out["delta"]["skipped_papers_not_1to1"] += 1
            return out
    elif filter_mode == "spatial":
        if not paper_passes_spatial_pairing(
            annotations, active_classes, spatial_iou_thresh
        ):
            out["delta"]["skipped_papers_no_spatial_pair"] += 1
            return out
    size_by_id = {
        img["id"]: (img["width"], img["height"]) for img in annotations["images"]
    }
    file_by_id = {img["id"]: img["file_name"] for img in annotations["images"]}
    labels = yolo_label_lines(
        annotations, size_by_id, class_to_index, min_bbox_area
    )
    archive = archive_lookup.get(paper_id, "unknown")

    if max_labels_per_paper > 0:
        total_lab = sum(len(v) for v in labels.values())
        if total_lab > max_labels_per_paper:
            out["delta"]["skipped_papers_too_many_labels"] += 1
            return out

    paper_pages: list[dict] = []
    for image_id, file_name in file_by_id.items():
        src_img = pages_dir / file_name
        if not src_img.is_file():
            continue
        rows = labels.get(image_id, [])
        if not rows and not include_negatives:
            out["delta"]["skipped_no_labels"] += 1
            continue
        stem = build_stem(paper_id, image_id)
        split = pick_split(stem, weights)
        paper_pages.append(
            {
                "paper_id": paper_id,
                "image_id": image_id,
                "src_img": src_img,
                "stem": stem,
                "split": split,
                "rows": rows,
                "size": size_by_id.get(image_id, (0, 0)),
                "archive": archive,
            }
        )

    if max_pages_per_paper > 0 and len(paper_pages) > max_pages_per_paper:
        rng = random.Random(rng_seed)
        pos = [p for p in paper_pages if p["rows"]]
        neg = [p for p in paper_pages if not p["rows"]]
        rng.shuffle(neg)
        merged = pos[:max_pages_per_paper]
        if len(merged) < max_pages_per_paper:
            merged += neg[: max_pages_per_paper - len(merged)]
        else:
            merged = merged[:max_pages_per_paper]
        out["delta"]["skipped_pages_per_paper_cap"] += (
            len(paper_pages) - len(merged)
        )
        paper_pages = merged

    out["candidates"] = paper_pages
    return out


def _collect_candidates(
    inputs: list[Path],
    weights: tuple[int, int, int],
    active_classes: tuple[str, ...],
    filter_mode: str,
    spatial_iou_thresh: float,
    include_negatives: bool,
    archive_lookup: dict[str, str],
    counts: dict[str, int],
    min_bbox_area: float = 0.0,
    max_labels_per_paper: int = 0,
    max_pages_per_paper: int = 0,
    workers: int = 8,
) -> list[dict]:
    """Walk every paper, filter, expand to one entry per page.

    Per-paper work (json parse + spatial-pair check + label generation)
    is independent, so we run it through a thread pool. ``counts`` is
    folded together at the end so callers still see why the corpus
    shrank between input and output.
    """
    class_to_index = {name: idx for idx, name in enumerate(active_classes)}

    # Stable rng seed per paper (so re-runs with the same input get the
    # same sub-sampled pages).
    paper_jobs: list[tuple[str, Path, Path, int]] = []
    rng_master = random.Random(0)
    for input_root in inputs:
        for i, (paper_id, ann_path, pages_dir) in enumerate(iter_papers(input_root)):
            paper_jobs.append((paper_id, ann_path, pages_dir, rng_master.randint(0, 2**31 - 1)))

    candidates: list[dict] = []
    if workers <= 1:
        for paper_id, ann_path, pages_dir, seed in paper_jobs:
            r = _process_one_paper(
                paper_id, ann_path, pages_dir, weights, active_classes,
                class_to_index, filter_mode, spatial_iou_thresh,
                include_negatives, archive_lookup, min_bbox_area,
                max_labels_per_paper, max_pages_per_paper, seed,
            )
            candidates.extend(r["candidates"])
            for k, v in r["delta"].items():
                counts[k] = counts.get(k, 0) + v
        return candidates

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _process_one_paper,
                paper_id, ann_path, pages_dir, weights, active_classes,
                class_to_index, filter_mode, spatial_iou_thresh,
                include_negatives, archive_lookup, min_bbox_area,
                max_labels_per_paper, max_pages_per_paper, seed,
            )
            for paper_id, ann_path, pages_dir, seed in paper_jobs
        ]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                continue
            candidates.extend(r["candidates"])
            for k, v in r["delta"].items():
                counts[k] = counts.get(k, 0) + v
    return candidates


def _split_pos_neg(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    pos = [c for c in candidates if c["rows"]]
    neg = [c for c in candidates if not c["rows"]]
    return pos, neg


def _round_robin_take(
    groups: dict[str, list[dict]],
    n: int,
    archives_order: list[str],
) -> list[dict]:
    """Round-robin pull at most ``n`` items, in the given archive order."""
    cursor: dict[str, int] = {a: 0 for a in archives_order}
    result: list[dict] = []
    while len(result) < n:
        progressed = False
        for a in archives_order:
            if cursor[a] < len(groups[a]):
                result.append(groups[a][cursor[a]])
                cursor[a] += 1
                progressed = True
                if len(result) >= n:
                    break
        if not progressed:
            break
    return result


def _cap_negatives(
    candidates: list[dict], neg_ratio: float, seed: int
) -> list[dict]:
    """Downsample negative pages so they end up at ``neg_ratio`` of the
    final dataset. Positives are always kept; only negatives are
    randomly thinned. No-op if there are already fewer negatives than
    the cap, or if ``neg_ratio`` is None / not in (0, 1)."""
    if neg_ratio is None or not (0.0 < neg_ratio < 1.0):
        return list(candidates)
    pos, neg = _split_pos_neg(candidates)
    if not pos or not neg:
        return list(candidates)
    target_total = len(pos) / max(1e-6, 1.0 - neg_ratio)
    target_neg = int(round(target_total * neg_ratio))
    if target_neg >= len(neg):
        return list(candidates)
    rng = random.Random(seed)
    neg = list(neg)
    rng.shuffle(neg)
    return pos + neg[:target_neg]


def _sample_candidates(
    candidates: list[dict],
    n: int,
    strategy: str,
    seed: int,
    num_classes: int,
    neg_ratio: float | None = None,
) -> list[dict]:
    """Reduce ``candidates`` to at most ``n`` entries using ``strategy``.

    Strategies:
    - ``random`` — shuffle by ``seed`` and slice.
    - ``by-archive`` — round-robin over archives.
    - ``balanced`` (default) — round-robin by archive, but within each
      archive prefer pages whose classes are rare in the candidate pool.
    - ``class-balanced`` — assign a per-class hard quota and pull pages
      that bring the rarest unfilled class first; guarantees every
      class gets representation in tiny samples.

    ``neg_ratio`` reserves a fraction of the sample for negative pages.
    The positive and negative budgets are sampled independently: the
    positive budget uses ``strategy``; the negative budget is always
    archive round-robin (negatives don't have classes to balance).
    """
    total = len(candidates)
    if n <= 0 or n >= total:
        return list(candidates)
    rng = random.Random(seed)

    pos, neg = _split_pos_neg(candidates)

    if neg_ratio is not None:
        want_neg = min(len(neg), int(round(n * neg_ratio)))
        want_pos = n - want_neg
    else:
        want_pos = n
        want_neg = 0

    # ----- positives -----
    if strategy == "random":
        shuffled = list(pos)
        rng.shuffle(shuffled)
        picked_pos = shuffled[:want_pos]
    elif strategy == "by-archive":
        groups: dict[str, list[dict]] = defaultdict(list)
        for c in pos:
            groups[c["archive"]].append(c)
        for lst in groups.values():
            rng.shuffle(lst)
        archives = list(groups.keys())
        rng.shuffle(archives)
        picked_pos = _round_robin_take(groups, want_pos, archives)
    elif strategy == "balanced":
        cls_freq: Counter = Counter()
        for c in pos:
            for row in c["rows"]:
                cls_idx = int(row.split()[0])
                cls_freq[cls_idx] += 1

        def rarity(c: dict) -> float:
            score = 0.0
            for row in c["rows"]:
                cls_idx = int(row.split()[0])
                score += 1.0 / max(1, cls_freq.get(cls_idx, 1))
            return score

        groups = defaultdict(list)
        for c in pos:
            groups[c["archive"]].append(c)
        for lst in groups.values():
            rng.shuffle(lst)
            lst.sort(key=lambda c: -rarity(c))
        archives = list(groups.keys())
        rng.shuffle(archives)
        picked_pos = _round_robin_take(groups, want_pos, archives)
    elif strategy == "class-balanced":
        # group pages by which classes they contain. A page with
        # multiple classes counts in each. We round-robin over classes,
        # picking the next page (de-duping by stem) until want_pos.
        by_class: dict[int, list[dict]] = defaultdict(list)
        for c in pos:
            seen_cls: set[int] = set()
            for row in c["rows"]:
                cls_idx = int(row.split()[0])
                if cls_idx not in seen_cls:
                    by_class[cls_idx].append(c)
                    seen_cls.add(cls_idx)
        for ci, lst in by_class.items():
            rng.shuffle(lst)
        classes_present = list(by_class.keys())
        if not classes_present:
            picked_pos = []
        else:
            rng.shuffle(classes_present)
            cursor = {ci: 0 for ci in classes_present}
            seen_stems: set[str] = set()
            picked_pos = []
            while len(picked_pos) < want_pos:
                progressed = False
                for ci in classes_present:
                    while cursor[ci] < len(by_class[ci]):
                        c = by_class[ci][cursor[ci]]
                        cursor[ci] += 1
                        if c["stem"] in seen_stems:
                            continue
                        seen_stems.add(c["stem"])
                        picked_pos.append(c)
                        progressed = True
                        if len(picked_pos) >= want_pos:
                            break
                        break  # advance to next class
                    if len(picked_pos) >= want_pos:
                        break
                if not progressed:
                    break
    else:
        raise ValueError(f"unknown sample strategy {strategy!r}")

    # ----- negatives -----
    picked_neg: list[dict] = []
    if want_neg > 0 and neg:
        groups = defaultdict(list)
        for c in neg:
            groups[c["archive"]].append(c)
        for lst in groups.values():
            rng.shuffle(lst)
        archives = list(groups.keys())
        rng.shuffle(archives)
        picked_neg = _round_robin_take(groups, want_neg, archives)

    return picked_pos + picked_neg


# ---------------------------------------------------------------------------
# Image emission
# ---------------------------------------------------------------------------


def _write_image(
    src: Path,
    dst: Path,
    fmt: str,
    max_short_side: int,
    jpg_quality: int,
) -> tuple[int, int]:
    """Save ``src`` into ``dst`` in ``fmt`` format, optionally resized so the
    shorter side is at most ``max_short_side`` (use 0 to disable). Returns
    the (width, height) of the saved image so callers can record it for
    the dataset card without having to re-open the file.
    """
    with Image.open(src) as im:
        w, h = im.size
        short = min(w, h)
        if max_short_side > 0 and short > max_short_side:
            scale = max_short_side / float(short)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
            w, h = new_size
        if fmt == "jpg":
            im = im.convert("RGB")
            im.save(dst, format="JPEG", quality=jpg_quality, optimize=True)
        elif fmt == "png":
            im.save(dst, format="PNG", optimize=True)
        else:
            raise ValueError(f"unsupported format {fmt!r}")
    return w, h


def _emit_one(
    cand: dict,
    out: Path,
    copy_images: bool,
    image_format: str,
    max_short_side: int,
    jpg_quality: int,
    ext: str,
) -> dict:
    """Single-page worker: write the image + its YOLO label. Returns a
    partial-result dict for the main thread to fold into ``stats``.

    This is the unit of work that ``_emit_candidates`` parallelises.
    PIL releases the GIL during decode/encode/resize so a thread pool
    sees real CPU concurrency, and the I/O wait on NFS amortises across
    threads as well.
    """
    src_img = cand["src_img"]
    rows = cand["rows"]
    split = cand["split"]
    stem = cand["stem"]
    archive = cand["archive"]

    split_dir = SPLIT_DIRS[split]
    dst_img = out / split_dir / "images" / f"{stem}{ext}"
    dst_lbl = out / split_dir / "labels" / f"{stem}.txt"

    can_symlink = (
        (not copy_images)
        and image_format == src_img.suffix.lstrip(".").lower()
        and max_short_side <= 0
    )
    if can_symlink:
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(src_img.resolve())
        with Image.open(src_img) as im:
            saved_w, saved_h = im.size
    else:
        saved_w, saved_h = _write_image(
            src_img, dst_img, image_format, max_short_side, jpg_quality
        )

    dst_lbl.write_text(
        ("\n".join(rows) + "\n") if rows else "", encoding="utf-8"
    )
    return {
        "split": split,
        "is_negative": not rows,
        "saved_size": (saved_w, saved_h),
        "rows": rows,
        "archive": archive,
        "paper_id": cand["paper_id"],
        "stem": stem,
        "image": dst_img,
        "label": dst_lbl,
    }


def _emit_candidates(
    candidates: list[dict],
    out: Path,
    copy_images: bool,
    image_format: str,
    max_short_side: int,
    jpg_quality: int,
    counts: dict[str, int],
    active_classes: tuple[str, ...],
    workers: int = 8,
    progress_every: int = 500,
) -> dict:
    """Write images + labels for every candidate in parallel.

    The expensive step (PIL decode → resize → JPEG encode → disk
    write) runs in a ``ThreadPoolExecutor`` of ``workers`` threads.
    Stats accumulation happens serially in the producer side so we
    don't need any locking — each worker returns a small dict and the
    main loop folds those into the running ``stats``.

    Pass ``workers=1`` to recover the old sequential behaviour (useful
    for debugging or when the underlying I/O does not benefit from
    parallelism).
    """
    ext = f".{image_format}"
    stats: dict = {
        "splits": Counter(),
        "split_positives": Counter(),
        "split_negatives": Counter(),
        "classes_per_split": defaultdict(Counter),
        "bbox_centers": [],
        "bbox_sizes": [],
        "page_sizes": [],
        "labels_per_image": [],
        "archives": Counter(),
        "archive_x_class": defaultdict(Counter),
        "papers": set(),
        "saved_files": [],
    }

    def absorb(r: dict) -> None:
        split = r["split"]
        is_negative = r["is_negative"]
        archive = r["archive"]
        rows = r["rows"]
        counts[split] += 1
        counts["negatives" if is_negative else "positives"] += 1
        stats["splits"][split] += 1
        if is_negative:
            stats["split_negatives"][split] += 1
        else:
            stats["split_positives"][split] += 1
        saved_w, saved_h = r["saved_size"]
        stats["page_sizes"].append((saved_w, saved_h))
        stats["labels_per_image"].append(len(rows))
        stats["archives"][archive] += 1
        stats["papers"].add(r["paper_id"])
        for row in rows:
            parts = row.split()
            cls_idx = int(parts[0])
            cx, cy, nw, nh = (float(p) for p in parts[1:5])
            stats["bbox_centers"].append((cls_idx, cx, cy))
            stats["bbox_sizes"].append((cls_idx, nw, nh))
            stats["archive_x_class"][archive][active_classes[cls_idx]] += 1
        stats["saved_files"].append(
            {
                "split": split,
                "image": r["image"],
                "label": r["label"],
                "stem": r["stem"],
                "archive": archive,
                "is_negative": is_negative,
                "saved_size": (saved_w, saved_h),
                "rows": rows,
            }
        )

    total = len(candidates)
    if total == 0:
        return stats

    t0 = time.monotonic()

    if workers <= 1:
        for i, cand in enumerate(candidates, 1):
            r = _emit_one(
                cand, out, copy_images, image_format,
                max_short_side, jpg_quality, ext,
            )
            absorb(r)
            if progress_every and i % progress_every == 0:
                elapsed = time.monotonic() - t0
                rate = i / max(0.001, elapsed)
                eta = (total - i) / max(0.001, rate)
                sys.stderr.write(
                    f"[emit] {i}/{total}  {rate:.1f} img/s  "
                    f"eta {eta/60:.1f} min\n"
                )
        return stats

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _emit_one, cand, out, copy_images, image_format,
                max_short_side, jpg_quality, ext,
            )
            for cand in candidates
        ]
        done = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                sys.stderr.write(f"[emit] worker error: {exc}\n")
                continue
            absorb(r)
            done += 1
            if progress_every and done % progress_every == 0:
                elapsed = time.monotonic() - t0
                rate = done / max(0.001, elapsed)
                eta = (total - done) / max(0.001, rate)
                sys.stderr.write(
                    f"[emit] {done}/{total}  {rate:.1f} img/s  "
                    f"eta {eta/60:.1f} min\n"
                )

    return stats


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save_class_counts(
    path: Path, stats: dict, active_classes: tuple[str, ...]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    splits = ["train", "val", "test"]
    matrix = np.zeros((len(splits), len(active_classes)), dtype=int)
    for c_idx, _ in enumerate(active_classes):
        for s_idx, split in enumerate(splits):
            matrix[s_idx, c_idx] = stats["classes_per_split"][split].get(
                active_classes[c_idx], 0
            )

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(active_classes)), 4.5))
    bottom = np.zeros(len(active_classes), dtype=int)
    colors = ["#4C72B0", "#DD8452", "#55A467"]
    for s_idx, split in enumerate(splits):
        ax.bar(
            active_classes,
            matrix[s_idx],
            bottom=bottom,
            label=split,
            color=colors[s_idx],
        )
        bottom = bottom + matrix[s_idx]
    for x, total in enumerate(bottom):
        ax.text(x, total, str(int(total)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("instance count")
    ax.set_title("Per-class label count (stacked by split)")
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_bbox_centers(
    path: Path, stats: dict, active_classes: tuple[str, ...]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(active_classes)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False
    )
    bins = 30
    by_class: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for cls_idx, cx, cy in stats["bbox_centers"]:
        by_class[cls_idx].append((cx, cy))
    for c_idx, name in enumerate(active_classes):
        ax = axes[c_idx // cols][c_idx % cols]
        pts = by_class.get(c_idx, [])
        if pts:
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            h, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, 1], [0, 1]])
            ax.imshow(
                h.T, origin="upper", extent=(0, 1, 1, 0),
                aspect="auto", cmap="viridis",
            )
            ax.set_title(f"{name} (n={len(pts)})")
        else:
            ax.set_title(f"{name} (n=0)")
            ax.text(0.5, 0.5, "no boxes", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xlabel("cx (page-normalized)")
        ax.set_ylabel("cy (page-normalized)")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("BBox center distribution per class (page-normalized; y grows downward)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_bbox_aspect(
    path: Path, stats: dict, active_classes: tuple[str, ...]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_class: dict[int, list[float]] = defaultdict(list)
    for cls_idx, nw, nh in stats["bbox_sizes"]:
        if nh > 0:
            by_class[cls_idx].append(nw / nh)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(np.log10(0.2), np.log10(20), 40)
    for c_idx, name in enumerate(active_classes):
        vals = by_class.get(c_idx, [])
        if not vals:
            continue
        ax.hist(
            vals, bins=bins, histtype="step", linewidth=1.6,
            label=f"{name} (n={len(vals)})",
        )
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", linewidth=0.6, linestyle="--")
    ax.set_xlabel("bbox aspect ratio  w / h  (log scale)")
    ax.set_ylabel("count")
    ax.set_title("BBox aspect ratio (per class)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_bbox_size(
    path: Path, stats: dict, active_classes: tuple[str, ...]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_class: dict[int, list[float]] = defaultdict(list)
    for cls_idx, nw, nh in stats["bbox_sizes"]:
        by_class[cls_idx].append(nw * nh)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    for c_idx, name in enumerate(active_classes):
        vals = by_class.get(c_idx, [])
        if not vals:
            continue
        ax.hist(
            vals, bins=bins, histtype="step", linewidth=1.6,
            label=f"{name} (n={len(vals)})",
        )
    ax.set_xlabel("bbox area / page area")
    ax.set_ylabel("count")
    ax.set_title("BBox area as fraction of page (per class)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_page_aspect(path: Path, stats: dict) -> None:
    import matplotlib.pyplot as plt

    aspects = [w / h for w, h in stats["page_sizes"] if h > 0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if aspects:
        ax.hist(aspects, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(1.0, color="k", linewidth=0.6, linestyle="--")
    ax.set_xlabel("page width / height")
    ax.set_ylabel("page count")
    ax.set_title(f"Page aspect ratio (n={len(aspects)})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_labels_per_image(path: Path, stats: dict) -> None:
    import matplotlib.pyplot as plt

    counts = list(stats["labels_per_image"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if counts:
        max_c = min(50, max(counts))
        ax.hist(
            counts, bins=range(0, max_c + 2),
            color="#55A467", edgecolor="white",
        )
        ax.set_xlabel(
            f"labels per image  (capped display @ {max_c}; "
            f"max actual = {max(counts)})"
        )
    else:
        ax.set_xlabel("labels per image")
    ax.set_ylabel("page count")
    pos = sum(1 for c in counts if c > 0)
    neg = sum(1 for c in counts if c == 0)
    ax.set_title(f"Labels per page (positive {pos} / negative {neg})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_archive_class(
    path: Path, stats: dict, active_classes: tuple[str, ...]
) -> None:
    """archive × class label-count heatmap. Skipped if there's no archive
    information at all (i.e. `archive` came back as 'unknown' for every
    page)."""
    import matplotlib.pyplot as plt
    import numpy as np

    archives = sorted(stats["archive_x_class"].keys())
    if not archives:
        return
    matrix = np.zeros((len(archives), len(active_classes)), dtype=int)
    for r, a in enumerate(archives):
        for c, name in enumerate(active_classes):
            matrix[r, c] = stats["archive_x_class"][a].get(name, 0)
    fig, ax = plt.subplots(
        figsize=(max(7, 0.9 * len(active_classes)), max(4, 0.4 * len(archives)))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(active_classes)))
    ax.set_xticklabels(active_classes, rotation=20)
    ax.set_yticks(range(len(archives)))
    ax.set_yticklabels(archives)
    ax.set_title("Archive × class label-count")
    for r in range(len(archives)):
        for c in range(len(active_classes)):
            v = int(matrix[r, c])
            if v == 0:
                continue
            ax.text(
                c, r, str(v),
                ha="center", va="center",
                color="white" if v > matrix.max() / 2 else "lightgray",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


_PREVIEW_PALETTE = [
    "#FF595E", "#FFCA3A", "#8AC926", "#1982C4",
    "#6A4C93", "#F45B69", "#06D6A0", "#118AB2",
]


def _save_preview_mosaic(
    path: Path,
    stats: dict,
    active_classes: tuple[str, ...],
    n_thumbs: int = 16,
    seed: int = 0,
) -> None:
    """Render up to ``n_thumbs`` sample pages with their bboxes overlaid,
    tiled into a square mosaic. Uses positive pages where possible."""

    from PIL import ImageDraw, ImageFont

    rng = random.Random(seed)
    files = list(stats["saved_files"])
    if not files:
        return
    positives = [f for f in files if not f["is_negative"]]
    if len(positives) >= n_thumbs:
        chosen = rng.sample(positives, n_thumbs)
    else:
        # mix in negatives to fill the mosaic
        rest = [f for f in files if f["is_negative"]]
        rng.shuffle(rest)
        chosen = list(positives) + rest[: n_thumbs - len(positives)]
        chosen = chosen[:n_thumbs]
    if not chosen:
        return

    # decide grid
    cols = int(round(len(chosen) ** 0.5))
    cols = max(2, min(cols, 6))
    rows_n = (len(chosen) + cols - 1) // cols

    # thumbnail size — keep file <= ~2 MB
    thumb_w = 320
    thumb_h = 420

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
        )
    except Exception:
        font = ImageFont.load_default()

    canvas = Image.new(
        "RGB",
        (cols * thumb_w, rows_n * thumb_h),
        color=(15, 15, 18),
    )
    draw_canvas = ImageDraw.Draw(canvas)

    for i, entry in enumerate(chosen):
        with Image.open(entry["image"]) as im:
            im = im.convert("RGB")
            ow, oh = im.size
            scale = min(thumb_w / ow, thumb_h / oh)
            nw, nh = int(ow * scale), int(oh * scale)
            im = im.resize((nw, nh), Image.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_h), (30, 30, 35))
            offset = ((thumb_w - nw) // 2, (thumb_h - nh) // 2)
            tile.paste(im, offset)
            draw = ImageDraw.Draw(tile)
            for row in entry["rows"]:
                parts = row.split()
                cls_idx = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                # bbox in thumb space
                x0 = (cx - bw / 2) * nw + offset[0]
                y0 = (cy - bh / 2) * nh + offset[1]
                x1 = (cx + bw / 2) * nw + offset[0]
                y1 = (cy + bh / 2) * nh + offset[1]
                color = _PREVIEW_PALETTE[cls_idx % len(_PREVIEW_PALETTE)]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                lbl = active_classes[cls_idx]
                draw.text((x0 + 2, max(0, y0 - 12)), lbl, fill=color, font=font)
            # caption strip with archive + split + filename stem
            caption = (
                f"[{entry['split']}] {entry['archive']}  "
                f"{entry['stem'][:32]}"
            )
            draw.rectangle([0, 0, thumb_w, 14], fill=(0, 0, 0))
            draw.text((4, 0), caption, fill=(220, 220, 220), font=font)
            cx = (i % cols) * thumb_w
            cy = (i // cols) * thumb_h
            canvas.paste(tile, (cx, cy))

    # legend at the bottom
    legend_h = 28
    full = Image.new(
        "RGB",
        (canvas.width, canvas.height + legend_h),
        color=(15, 15, 18),
    )
    full.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(full)
    x = 6
    y = canvas.height + 6
    draw.text(
        (x, y - 2), "classes:", fill=(255, 255, 255), font=font
    )
    x += 60
    for ci, name in enumerate(active_classes):
        color = _PREVIEW_PALETTE[ci % len(_PREVIEW_PALETTE)]
        draw.rectangle([x, y, x + 14, y + 14], outline=color, width=2)
        draw.text((x + 18, y - 1), name, fill=color, font=font)
        x += 18 + max(60, 9 * len(name) + 12)
    full.save(path, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# README + structured artifacts
# ---------------------------------------------------------------------------


def _split_table_row(stats: dict, split: str) -> str:
    total = stats["splits"].get(split, 0)
    pos = stats["split_positives"].get(split, 0)
    neg = stats["split_negatives"].get(split, 0)
    return f"| {split} | {total} | {pos} | {neg} |"


def _write_readme(
    path: Path,
    out: Path,
    stats: dict,
    counts: dict[str, int],
    args_repr: str,
    active_classes: tuple[str, ...],
    sample_n: int,
    sample_strategy: str,
    neg_ratio: float | None,
    filter_mode: str,
    image_format: str,
    max_short_side: int,
    spatial_iou_thresh: float,
    min_bbox_area: float,
    max_pages_per_paper: int,
    max_labels_per_paper: int,
    has_preview: bool,
    has_archive_class: bool,
) -> None:
    total_images = sum(stats["splits"].values())
    total_labels = len(stats["bbox_sizes"])
    pages_with_labels = sum(1 for n in stats["labels_per_image"] if n > 0)
    pages_neg = sum(1 for n in stats["labels_per_image"] if n == 0)

    lines: list[str] = []
    lines.append("# arxiv-paper-layout YOLO export")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_"
    )
    lines.append("")
    lines.append(f"**Generator command**:  `{args_repr}`")
    lines.append("")

    # 1. At a glance
    lines.append("## 1. At a glance")
    lines.append("")
    lines.append(f"- **classes (output index order)**: {list(active_classes)}")
    lines.append(
        f"- **paper-level filter**: `{filter_mode}` "
        f"(IoU thresh {spatial_iou_thresh})"
    )
    if sample_n > 0:
        neg_str = f", neg_ratio={neg_ratio}" if neg_ratio is not None else ""
        lines.append(
            f"- **sampling**: `--sample {sample_n}` strategy=`{sample_strategy}`"
            f"{neg_str} (output is a smoke-test slice, not the full corpus)"
        )
    else:
        lines.append("- **sampling**: full export (no `--sample`)")
    qc_bits: list[str] = []
    if min_bbox_area > 0:
        qc_bits.append(f"min_bbox_area={min_bbox_area}")
    if max_pages_per_paper > 0:
        qc_bits.append(f"max_pages_per_paper={max_pages_per_paper}")
    if max_labels_per_paper > 0:
        qc_bits.append(f"max_labels_per_paper={max_labels_per_paper}")
    if qc_bits:
        lines.append("- **QC filters**: " + ", ".join(qc_bits))
    lines.append(
        f"- **image format**: `{image_format}` "
        f"(short side <= {max_short_side or 'unbounded'})"
    )
    lines.append(
        f"- **papers covered**: {len(stats['papers'])}  | "
        f"**archives covered**: {len(stats['archives'])}"
    )
    lines.append("")
    lines.append("| split | pages | positive | negative |")
    lines.append("|---|---:|---:|---:|")
    for split in ("train", "val", "test"):
        lines.append(_split_table_row(stats, split))
    lines.append(
        f"| **all** | **{total_images}** | "
        f"**{pages_with_labels}** | **{pages_neg}** |"
    )
    lines.append("")

    if has_preview:
        lines.append("### Sample preview")
        lines.append("")
        lines.append(
            "Random pages from the export with bboxes drawn — quick "
            "eyeball of what the labels look like."
        )
        lines.append("")
        lines.append("![preview](analysis/preview.png)")
        lines.append("")

    # 2. Per-class counts
    lines.append("## 2. Per-class label counts")
    lines.append("")
    lines.append("| class | train | val | test | total |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in active_classes:
        tr = stats["classes_per_split"]["train"].get(name, 0)
        va = stats["classes_per_split"]["val"].get(name, 0)
        te = stats["classes_per_split"]["test"].get(name, 0)
        lines.append(f"| `{name}` | {tr} | {va} | {te} | **{tr + va + te}** |")
    lines.append(f"| **all classes** |  |  |  | **{total_labels}** |")
    lines.append("")
    lines.append("![class counts](analysis/class_counts.png)")
    lines.append("")

    # 3. Archive coverage
    lines.append("## 3. Archive coverage")
    lines.append("")
    lines.append("| archive | pages |")
    lines.append("|---|---:|")
    for archive, n in sorted(stats["archives"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {archive} | {n} |")
    lines.append("")

    if has_archive_class:
        lines.append("### Archive × class")
        lines.append("")
        lines.append(
            "Where each class lives. Cell value is the number of label "
            "instances of that class contributed by papers from that archive."
        )
        lines.append("")
        lines.append("![archive×class](analysis/archive_class.png)")
        lines.append("")

    # 4. Spatial / size analytics
    lines.append("## 4. BBox spatial distribution")
    lines.append("")
    lines.append(
        "Heatmap of bbox centers on a page-normalised canvas. Origin "
        "is top-left and y grows downward to match image coordinates, "
        "so pages have visually the same orientation as the rendered PDF."
    )
    lines.append("")
    lines.append("![bbox centers](analysis/bbox_centers.png)")
    lines.append("")
    lines.append("## 5. BBox size distribution")
    lines.append("")
    lines.append(
        "Per-class histograms of `area / page_area`. Most floats sit "
        "below 40% of the page; the long tail above 70% is dominated "
        "by full-page tables / figures and is rare."
    )
    lines.append("")
    lines.append("![bbox size](analysis/bbox_size.png)")
    lines.append("")
    lines.append("## 6. BBox aspect ratio (w/h)")
    lines.append("")
    lines.append(
        "Anchor-box / detector hint: tables typically peak in the "
        "*very-wide* band (>2.5), figures cluster around 1.0–2.0, "
        "and `tall` boxes (<0.4) are rare across the corpus."
    )
    lines.append("")
    lines.append("![bbox aspect](analysis/bbox_aspect.png)")
    lines.append("")
    lines.append("## 7. Page aspect ratio")
    lines.append("")
    lines.append(
        "Most arxiv pages are A4/letter portrait (w/h ≈ 0.7–0.8). "
        "Landscape pages exist for wide tables / posters."
    )
    lines.append("")
    lines.append("![page aspect](analysis/page_aspect.png)")
    lines.append("")
    lines.append("## 8. Labels per page")
    lines.append("")
    lines.append(
        "Distribution of `labels_per_image` across the export. A wide "
        "left peak at zero is expected: most pages of a paper are "
        "text-only, kept here as YOLO background examples."
    )
    lines.append("")
    lines.append("![labels per image](analysis/labels_per_image.png)")
    lines.append("")

    # 9. Layout
    lines.append("## 9. Filesystem layout")
    lines.append("")
    lines.append("```")
    lines.append(f"{out.name}/")
    lines.append("  data.yaml                # ultralytics YOLO data spec")
    lines.append("  train_recommended.yaml   # ready-to-use ultralytics train cfg")
    lines.append("  dataset_meta.json        # structured metadata")
    lines.append("  manifest.sha256          # file integrity manifest")
    lines.append("  README.md                # this file")
    lines.append("  analysis/                # auto-generated diagnostic plots")
    lines.append("    class_counts.png")
    lines.append("    archive_class.png")
    lines.append("    bbox_centers.png")
    lines.append("    bbox_size.png")
    lines.append("    bbox_aspect.png")
    lines.append("    page_aspect.png")
    lines.append("    labels_per_image.png")
    lines.append("    preview.png")
    lines.append(
        f"  train/images/<paper_id>__page_NNN.{image_format}"
    )
    lines.append("  train/labels/<paper_id>__page_NNN.txt")
    lines.append(
        f"  valid/images/<paper_id>__page_NNN.{image_format}"
    )
    lines.append("  valid/labels/<paper_id>__page_NNN.txt")
    lines.append(
        f"  test/images/<paper_id>__page_NNN.{image_format}"
    )
    lines.append("  test/labels/<paper_id>__page_NNN.txt")
    lines.append("```")
    lines.append("")

    # 10. Train command
    lines.append("## 10. Train command")
    lines.append("")
    lines.append(
        "Pre-baked Ultralytics config sits at `train_recommended.yaml`. "
        "To kick off training:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("yolo train cfg=train_recommended.yaml")
    lines.append("```")
    lines.append("")
    lines.append(
        "The config presets `imgsz=1280`, mosaic on, vertical-flip off, "
        "and class weights tuned to the per-class frequency in this "
        "export. Override anything via `key=value` on the CLI."
    )
    lines.append("")

    # 11. Provenance
    lines.append("## 11. Provenance & caveats")
    lines.append("")
    lines.append(
        "- Source pipeline: `arxiv-paper-layout-dataset` (`scripts/export_yolo.py`)."
    )
    if sample_n > 0:
        lines.append(
            f"- This export is a sampled slice (`{sample_strategy}`, "
            f"capped at {sample_n} images). **Do not generalise "
            "distributional claims** from it; use the full export "
            "for that."
        )
    lines.append(
        f"- `paper-level filter = {filter_mode}` rejects papers whose "
        "bbox structure violates the body/cap spatial pairing rule."
    )
    if min_bbox_area > 0:
        lines.append(
            f"- `min_bbox_area = {min_bbox_area}` strips bboxes below "
            "this fraction of the page area (kills artifact-tiny boxes)."
        )
    lines.append(
        "- Caption boxes (`*_cap`) include the caption *region*, not "
        "just the caption *text*. They cover the full caption as it "
        "appears in the PDF — mostly multi-line."
    )
    lines.append(
        "- Boxes are clipped into [0, 1] on emission; `cx / cy / nw / "
        "nh` are normalised to the *saved* image (post-resize) per the "
        "Ultralytics convention."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dataset_meta(
    path: Path,
    stats: dict,
    counts: dict[str, int],
    args_repr: str,
    active_classes: tuple[str, ...],
    sample_n: int,
    sample_strategy: str,
    sample_seed: int,
    neg_ratio: float | None,
    filter_mode: str,
    image_format: str,
    max_short_side: int,
    spatial_iou_thresh: float,
    min_bbox_area: float,
    max_pages_per_paper: int,
    max_labels_per_paper: int,
) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit_short(),
        "generator_command": args_repr,
        "classes": list(active_classes),
        "filter": {
            "mode": filter_mode,
            "spatial_iou_thresh": spatial_iou_thresh,
        },
        "qc": {
            "min_bbox_area": min_bbox_area,
            "max_pages_per_paper": max_pages_per_paper,
            "max_labels_per_paper": max_labels_per_paper,
        },
        "image": {
            "format": image_format,
            "max_short_side": max_short_side,
        },
        "sampling": {
            "n": sample_n,
            "strategy": sample_strategy,
            "seed": sample_seed,
            "neg_ratio": neg_ratio,
        },
        "splits": dict(stats["splits"]),
        "split_positives": dict(stats["split_positives"]),
        "split_negatives": dict(stats["split_negatives"]),
        "classes_per_split": {
            split: dict(c) for split, c in stats["classes_per_split"].items()
        },
        "archive_x_class": {
            archive: dict(c) for archive, c in stats["archive_x_class"].items()
        },
        "archives": dict(stats["archives"]),
        "papers": len(stats["papers"]),
        "totals": {
            "images": sum(stats["splits"].values()),
            "labels": len(stats["bbox_sizes"]),
            "positive_pages": sum(1 for n in stats["labels_per_image"] if n > 0),
            "negative_pages": sum(1 for n in stats["labels_per_image"] if n == 0),
        },
        "counts": {k: v for k, v in counts.items()},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_train_yaml(
    path: Path,
    out: Path,
    stats: dict,
    active_classes: tuple[str, ...],
    image_format: str,
    max_short_side: int,
) -> None:
    """Recommended Ultralytics training config tuned to this export.

    Class weights are inverse-proportional to per-class instance counts
    (clipped to [1.0, 30.0]) so rare classes (algorithm/listing) get
    upweighted without exploding the loss.
    """
    counts_by_class: list[tuple[str, int]] = []
    for c in active_classes:
        n = sum(stats["classes_per_split"][s].get(c, 0) for s in ("train", "val", "test"))
        counts_by_class.append((c, n))
    max_n = max((n for _, n in counts_by_class), default=1) or 1
    weights: list[float] = []
    for _, n in counts_by_class:
        if n <= 0:
            weights.append(30.0)
        else:
            w = max(1.0, min(30.0, max_n / n))
            weights.append(round(w, 2))

    imgsz = 1280 if (max_short_side or 0) >= 1024 else 960
    if max_short_side > 0:
        imgsz = max(640, min(1280, max_short_side + 256))

    body = [
        "# Auto-generated by scripts/export_yolo.py.",
        "# Run from the dataset root:",
        "#   cd <dataset_dir>",
        "#   yolo train cfg=train_recommended.yaml",
        "# Paths below are relative to this file so the whole dataset",
        "# directory stays portable.",
        "",
        f"task: detect",
        f"mode: train",
        f"model: yolov8m.pt",
        f"data: data.yaml",
        f"imgsz: {imgsz}",
        f"epochs: 100",
        f"patience: 20",
        f"batch: 16",
        f"workers: 8",
        f"device: 0",
        f"optimizer: AdamW",
        f"lr0: 0.001",
        f"cos_lr: true",
        f"# augmentation — fliplr safe, flipud unsafe (caption-below relation)",
        f"mosaic: 1.0",
        f"mixup: 0.0",
        f"fliplr: 0.5",
        f"flipud: 0.0",
        f"degrees: 0.0",
        f"hsv_v: 0.2",
        f"hsv_h: 0.0",
        f"hsv_s: 0.2",
        f"copy_paste: 0.0",
        f"# class weights (inverse class frequency, clipped 1.0–30.0)",
        "cls_weights: ["
        + ", ".join(str(w) for w in weights)
        + "]",
        "",
        f"# class counts at export time:",
    ]
    for name, n in counts_by_class:
        body.append(f"#   {name:<14} {n}")
    body.append("")
    path.write_text("\n".join(body), encoding="utf-8")


def _sha256_file(p: Path) -> tuple[str, str]:
    """sha256 a single file, return ``(rel_or_abs_path_str, hex)``.

    The path is returned as-is so the caller can decide how to
    relativise it (the worker doesn't know about the ``out`` root).
    """
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return str(p), h.hexdigest()


def _write_manifest(
    path: Path, stats: dict, out: Path, workers: int = 8
) -> None:
    """Sha256 manifest of every emitted artifact (images + labels +
    plots). Order is stable so re-runs that emit the same files
    produce a byte-identical manifest. Hashing runs in a thread pool
    because it is per-file independent and disk-bound."""
    seen: set[Path] = set()
    files: list[Path] = []
    for entry in stats["saved_files"]:
        for p in (entry["image"], entry["label"]):
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    for sub in ("data.yaml", "train_recommended.yaml", "README.md", "dataset_meta.json"):
        p = out / sub
        if p.is_file() and p not in seen:
            seen.add(p)
            files.append(p)
    analysis = out / "analysis"
    if analysis.is_dir():
        for p in sorted(analysis.iterdir()):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)

    items: list[tuple[str, str]] = []
    if workers <= 1:
        for p in files:
            if not p.is_file():
                continue
            _, hexd = _sha256_file(p)
            items.append((str(p.relative_to(out)), hexd))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_sha256_file, p) for p in files if p.is_file()]
            for fut in as_completed(futures):
                try:
                    pstr, hexd = fut.result()
                except Exception:
                    continue
                items.append(
                    (str(Path(pstr).relative_to(out)), hexd)
                )
    items.sort()
    path.write_text(
        "\n".join(f"{h}  {rel}" for rel, h in items) + "\n",
        encoding="utf-8",
    )


def verify_dataset(out: Path, num_classes: int) -> tuple[int, int, list[str]]:
    """Walk the emitted dataset and return ``(images, issues, sample_msgs)``.

    Checks performed:
    - every label file has a matching image file
    - every label row parses as ``<int> <float> <float> <float> <float>``
    - class index in [0, num_classes)
    - all coords in [0, 1]
    - every image opens via PIL
    """
    issues: list[str] = []
    img_count = 0
    for split in ("train", "val", "test"):
        split_dir = SPLIT_DIRS[split]
        img_dir = out / split_dir / "images"
        lbl_dir = out / split_dir / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            img_count += 1
            stem = img.stem
            lbl = lbl_dir / f"{stem}.txt"
            if not lbl.is_file():
                issues.append(f"missing label: {img.relative_to(out)}")
                continue
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as exc:
                issues.append(f"image broken: {img.relative_to(out)} ({exc})")
                continue
            try:
                txt = lbl.read_text(encoding="utf-8").strip()
            except Exception as exc:
                issues.append(f"label unreadable: {lbl.relative_to(out)} ({exc})")
                continue
            if not txt:
                continue  # empty label = negative sample, valid
            for ln, raw in enumerate(txt.splitlines(), 1):
                parts = raw.split()
                if len(parts) != 5:
                    issues.append(
                        f"{lbl.relative_to(out)}:{ln} bad row: {raw!r}"
                    )
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(p) for p in parts[1:5])
                except Exception:
                    issues.append(
                        f"{lbl.relative_to(out)}:{ln} parse error: {raw!r}"
                    )
                    continue
                if not (0 <= cls < num_classes):
                    issues.append(
                        f"{lbl.relative_to(out)}:{ln} class {cls} OOB "
                        f"[0,{num_classes})"
                    )
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    issues.append(
                        f"{lbl.relative_to(out)}:{ln} coord out of [0,1]: {raw!r}"
                    )
    return img_count, len(issues), issues[:20]


def write_dataset_card(
    out: Path,
    stats: dict,
    counts: dict[str, int],
    args_repr: str,
    active_classes: tuple[str, ...],
    sample_n: int,
    sample_strategy: str,
    sample_seed: int,
    neg_ratio: float | None,
    filter_mode: str,
    image_format: str,
    max_short_side: int,
    spatial_iou_thresh: float,
    min_bbox_area: float,
    max_pages_per_paper: int,
    max_labels_per_paper: int,
) -> None:
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    _save_class_counts(analysis / "class_counts.png", stats, active_classes)
    _save_bbox_centers(analysis / "bbox_centers.png", stats, active_classes)
    _save_bbox_aspect(analysis / "bbox_aspect.png", stats, active_classes)
    _save_bbox_size(analysis / "bbox_size.png", stats, active_classes)
    _save_page_aspect(analysis / "page_aspect.png", stats)
    _save_labels_per_image(analysis / "labels_per_image.png", stats)

    has_archive_class = (
        len(stats["archive_x_class"]) > 0
        and any(stats["archive_x_class"].values())
    )
    if has_archive_class:
        _save_archive_class(
            analysis / "archive_class.png", stats, active_classes
        )

    has_preview = bool(stats["saved_files"])
    if has_preview:
        try:
            _save_preview_mosaic(
                analysis / "preview.png", stats, active_classes, n_thumbs=16
            )
        except Exception as exc:
            sys.stderr.write(f"[warn] preview mosaic failed: {exc}\n")
            has_preview = False

    _write_readme(
        out / "README.md",
        out,
        stats,
        counts,
        args_repr,
        active_classes,
        sample_n,
        sample_strategy,
        neg_ratio,
        filter_mode,
        image_format,
        max_short_side,
        spatial_iou_thresh,
        min_bbox_area,
        max_pages_per_paper,
        max_labels_per_paper,
        has_preview=has_preview,
        has_archive_class=has_archive_class,
    )


# ---------------------------------------------------------------------------
# Top-level export()
# ---------------------------------------------------------------------------


def export(
    inputs: list[Path],
    out: Path,
    weights: tuple[int, int, int],
    copy_images: bool = True,
    image_format: str = "jpg",
    max_short_side: int = 720,
    jpg_quality: int = 90,
    active_classes: tuple[str, ...] = CLASSES,
    strict_1to1: bool = False,
    spatial_pair: bool = False,
    spatial_iou_thresh: float = 0.9,
    include_negatives: bool = True,
    sample: int = 0,
    sample_strategy: str = "balanced",
    sample_seed: int = 0,
    neg_ratio: float | None = None,
    write_readme: bool = True,
    write_train_yaml: bool = True,
    write_meta: bool = True,
    write_manifest: bool = True,
    do_verify: bool = True,
    corpus_state: Path | None = None,
    args_repr: str | None = None,
    min_bbox_area: float = 0.0,
    max_pages_per_paper: int = 0,
    max_labels_per_paper: int = 0,
    workers: int = 8,
) -> dict[str, int]:
    """Orchestrate the full export."""
    out.mkdir(parents=True, exist_ok=True)
    # Clean any leftover images/labels from a previous (possibly aborted)
    # run into the same output dir. Without this, an interrupted run
    # leaves orphan files that survive into the next export and confuse
    # the manifest / verify counts. Layout is Roboflow-style:
    # ``<out>/<split_dir>/{images,labels}/`` where split_dir is one of
    # train/valid/test.
    # Wipe legacy ``<out>/images/{train,val,test}`` if present so old
    # exports don't pollute new runs.
    for legacy in ("images", "labels"):
        legacy_root = out / legacy
        if legacy_root.is_dir():
            for split_legacy in ("train", "val", "valid", "test"):
                d = legacy_root / split_legacy
                if d.is_dir():
                    shutil.rmtree(d)
    for split in ("train", "val", "test"):
        split_dir = SPLIT_DIRS[split]
        for kind in ("images", "labels"):
            d = out / split_dir / kind
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "train": 0,
        "val": 0,
        "test": 0,
        "positives": 0,
        "negatives": 0,
        "skipped_no_labels": 0,
        "skipped_papers_not_1to1": 0,
        "skipped_papers_no_spatial_pair": 0,
        "skipped_papers_too_many_labels": 0,
        "skipped_pages_per_paper_cap": 0,
        "candidates": 0,
        "sampled": 0,
        "verify_issues": 0,
    }

    if strict_1to1:
        filter_mode = "strict"
    elif spatial_pair:
        filter_mode = "spatial"
    else:
        filter_mode = "none"

    archive_lookup = _load_archive_lookup(corpus_state, inputs)

    candidates = _collect_candidates(
        inputs,
        weights,
        active_classes,
        filter_mode,
        spatial_iou_thresh,
        include_negatives,
        archive_lookup,
        counts,
        min_bbox_area=min_bbox_area,
        max_labels_per_paper=max_labels_per_paper,
        max_pages_per_paper=max_pages_per_paper,
        workers=workers,
    )
    counts["candidates"] = len(candidates)

    if sample > 0 and sample < len(candidates):
        candidates = _sample_candidates(
            candidates,
            sample,
            sample_strategy,
            sample_seed,
            num_classes=len(active_classes),
            neg_ratio=neg_ratio,
        )
    elif neg_ratio is not None:
        # Full export with a negative-ratio cap: keep every positive
        # page, randomly thin the negatives down to the requested
        # share of the final dataset.
        candidates = _cap_negatives(candidates, neg_ratio, sample_seed)
    counts["sampled"] = len(candidates)

    stats = _emit_candidates(
        candidates,
        out,
        copy_images,
        image_format,
        max_short_side,
        jpg_quality,
        counts,
        active_classes,
        workers=workers,
    )

    # Per-split per-class counts (emit only fills aggregates).
    for cand in candidates:
        for row in cand["rows"]:
            cls_idx = int(row.split()[0])
            stats["classes_per_split"][cand["split"]][active_classes[cls_idx]] += 1

    data_yaml = out / "data.yaml"
    # Roboflow-style data.yaml: train/val/test paths use ``../`` so they
    # resolve against the *parent* of the data.yaml's resolved path.
    # Layout on disk:
    #     <root>/
    #       data.yaml
    #       train/images/...
    #       valid/images/...
    #       test/images/...
    # ``path:`` is intentionally omitted — having it confuses several
    # downstream consumers that prefer the Roboflow flat layout.
    names_inline = "[" + ", ".join(f"'{n}'" for n in active_classes) + "]"
    data_yaml.write_text(
        "\n".join(
            [
                "train: ../train/images",
                "val: ../valid/images",
                "test: ../test/images",
                "",
                f"nc: {len(active_classes)}",
                f"names: {names_inline}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if write_train_yaml:
        _write_train_yaml(
            out / "train_recommended.yaml",
            out,
            stats,
            active_classes,
            image_format,
            max_short_side,
        )

    if write_readme:
        write_dataset_card(
            out,
            stats,
            counts,
            args_repr or "",
            active_classes,
            sample,
            sample_strategy,
            sample_seed,
            neg_ratio,
            filter_mode,
            image_format,
            max_short_side,
            spatial_iou_thresh,
            min_bbox_area,
            max_pages_per_paper,
            max_labels_per_paper,
        )

    if write_meta:
        _write_dataset_meta(
            out / "dataset_meta.json",
            stats,
            counts,
            args_repr or "",
            active_classes,
            sample,
            sample_strategy,
            sample_seed,
            neg_ratio,
            filter_mode,
            image_format,
            max_short_side,
            spatial_iou_thresh,
            min_bbox_area,
            max_pages_per_paper,
            max_labels_per_paper,
        )

    if do_verify:
        img_n, issues_n, sample_msgs = verify_dataset(
            out, num_classes=len(active_classes)
        )
        counts["verify_issues"] = issues_n
        if issues_n:
            sys.stderr.write(
                f"[verify] {issues_n} issue(s) across {img_n} image(s); "
                f"first {len(sample_msgs)}:\n"
            )
            for msg in sample_msgs:
                sys.stderr.write(f"  - {msg}\n")
        else:
            print(f"[verify] OK ({img_n} images checked)")

    if write_manifest:
        _write_manifest(out / "manifest.sha256", stats, out, workers=workers)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_classes(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--classes must list at least one class")
    bad = [p for p in parts if p not in CLASS_TO_INDEX]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--classes unknown names: {bad!r}; valid: {list(CLASSES)}"
        )
    if len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError(f"--classes has duplicates: {parts!r}")
    return tuple(parts)


def parse_weights(raw: str) -> tuple[int, int, int]:
    parts = [p for p in raw.replace(",", ":").split(":") if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--split expected three ints separated by `:` or `,`; got {raw!r}"
        )
    ints = tuple(int(p) for p in parts)
    if any(v < 0 for v in ints) or sum(ints) == 0:
        raise argparse.ArgumentTypeError("--split values must be >= 0 and sum > 0")
    return ints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="one or more run roots (each with <paper_id>/dataset/annotations.json)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=parse_weights,
        default=(8, 1, 1),
        help="train:val:test ratio (default 8:1:1)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="symlink images instead of copying (saves disk; only usable if "
        "the source tree persists AND format matches AND no resize).",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        help="output image format; default jpg for a compact shippable dataset",
    )
    parser.add_argument(
        "--max-short-side",
        type=int,
        default=1024,
        help="downscale images so min(width, height) == this value when the "
        "source image is larger; 0 disables resizing (default 1024 — chosen "
        "so figure/table boxes that are 5-10%% of the page are still ~50px "
        "after resize, which is healthy for YOLO small-object recall).",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=90,
        help="JPEG quality when --format jpg (default 90)",
    )

    cls_group = parser.add_mutually_exclusive_group()
    cls_group.add_argument(
        "--classes",
        type=parse_classes,
        default=None,
        help="comma-separated subset of classes to export. Output index "
        "matches CLI order. Mutually exclusive with --subset.",
    )
    cls_group.add_argument(
        "--subset",
        choices=tuple(CLASS_SUBSETS.keys()),
        default=None,
        help="shorthand for the canonical 4/6/8-label class set "
        "(see arxiv_layout.spatial_pair.CLASS_SUBSETS).",
    )

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--strict-1to1",
        dest="filter_mode",
        action="store_const",
        const="strict",
        help="Only keep papers where every active pair is 1:1 AND "
        "spatially valid.",
    )
    filter_group.add_argument(
        "--spatial-pair",
        dest="filter_mode",
        action="store_const",
        const="spatial",
        help="[default] Only keep papers where every active pair is "
        "spatially valid, relaxed to N:1.",
    )
    filter_group.add_argument(
        "--no-filter",
        dest="filter_mode",
        action="store_const",
        const="none",
        help="Disable paper-level filtering.",
    )
    parser.set_defaults(filter_mode="spatial")
    parser.add_argument(
        "--spatial-iou-thresh",
        type=float,
        default=0.9,
        help="threshold for 'body mostly inside cap' (default 0.9).",
    )

    neg_group = parser.add_mutually_exclusive_group()
    neg_group.add_argument(
        "--include-negatives",
        dest="include_negatives",
        action="store_true",
        default=True,
        help="Export pages with no active-class annotations as pure "
        "negative samples (empty .txt). Default: on.",
    )
    neg_group.add_argument(
        "--skip-negatives",
        dest="include_negatives",
        action="store_false",
        help="Drop pages with no active-class annotations.",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, cap the export to this many images total.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("balanced", "class-balanced", "by-archive", "random"),
        default="balanced",
        help="Sampling strategy when --sample is set. `balanced` "
        "(default) round-robins by archive AND prefers rarer-class "
        "pages first. `class-balanced` enforces per-class quotas. "
        "`by-archive` is plain round-robin. `random` is shuffled slice.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for sampling shuffles (default 0, reproducible).",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=None,
        help="Target fraction of negative (no-label) pages in the "
        "final dataset, e.g. 0.3 -> 30%% negatives. Works in two "
        "modes: (a) with --sample, reserves this fraction of the "
        "sample budget for negatives; (b) without --sample (full "
        "export), keeps every positive page and randomly downsamples "
        "negatives to hit the ratio. Default: no cap (raw 67%% "
        "negatives are all kept).",
    )
    parser.add_argument(
        "--min-bbox-area",
        type=float,
        default=0.0005,
        help="Drop bboxes below this fraction of the page area. Default "
        "0.0005 (0.05%% of page) strips artifact-tiny boxes that would "
        "otherwise pollute training. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-pages-per-paper",
        type=int,
        default=30,
        help="Cap pages contributed by any one paper. Default 30 "
        "protects against PhD-thesis domination (the long tail at "
        "100+ pages). Use 0 to disable.",
    )
    parser.add_argument(
        "--max-labels-per-paper",
        type=int,
        default=200,
        help="Reject papers whose total active-label count exceeds "
        "this. Default 200 — well above the 99th percentile (~108) so "
        "real long papers survive but extreme outliers don't dominate. "
        "Use 0 to disable.",
    )

    parser.add_argument(
        "--no-readme",
        dest="write_readme",
        action="store_false",
        default=True,
        help="Skip the README.md + analysis/*.png generation.",
    )
    parser.add_argument(
        "--no-train-yaml",
        dest="write_train_yaml",
        action="store_false",
        default=True,
        help="Skip the train_recommended.yaml generation.",
    )
    parser.add_argument(
        "--no-meta",
        dest="write_meta",
        action="store_false",
        default=True,
        help="Skip the dataset_meta.json generation.",
    )
    parser.add_argument(
        "--no-manifest",
        dest="write_manifest",
        action="store_false",
        default=True,
        help="Skip the manifest.sha256 generation.",
    )
    parser.add_argument(
        "--no-verify",
        dest="do_verify",
        action="store_false",
        default=True,
        help="Skip the post-emit dataset integrity check.",
    )
    parser.add_argument(
        "--corpus-state",
        type=Path,
        default=None,
        help="Path to corpus state.json (default: auto-detected at "
        "`<input>/../state.json`).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4)),
        help="Worker thread count for image emit + manifest hashing. "
        "PIL releases the GIL during heavy ops so threads scale near "
        "linearly until I/O saturates. Default: min(16, cpu_count). "
        "Use 1 for sequential; bump higher on big NFS-backed runs.",
    )
    args = parser.parse_args()

    # Resolve --classes / --subset (subset wins if both somehow set, but
    # they're mutually exclusive at the parser level).
    if args.subset is not None:
        active_classes = CLASS_SUBSETS[args.subset]
    elif args.classes is not None:
        active_classes = args.classes
    else:
        active_classes = CLASSES

    args_repr = "python3 scripts/export_yolo.py " + " ".join(sys.argv[1:])

    counts = export(
        inputs=[p.resolve() for p in args.input],
        out=args.out.resolve(),
        weights=args.split,
        copy_images=not args.symlink,
        image_format=args.format,
        max_short_side=args.max_short_side,
        jpg_quality=args.jpg_quality,
        active_classes=active_classes,
        strict_1to1=(args.filter_mode == "strict"),
        spatial_pair=(args.filter_mode == "spatial"),
        spatial_iou_thresh=args.spatial_iou_thresh,
        include_negatives=args.include_negatives,
        sample=args.sample,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        neg_ratio=args.neg_ratio,
        write_readme=args.write_readme,
        write_train_yaml=args.write_train_yaml,
        write_meta=args.write_meta,
        write_manifest=args.write_manifest,
        do_verify=args.do_verify,
        corpus_state=args.corpus_state,
        args_repr=args_repr,
        min_bbox_area=args.min_bbox_area,
        max_pages_per_paper=args.max_pages_per_paper,
        max_labels_per_paper=args.max_labels_per_paper,
        workers=args.workers,
    )
    mode = {
        "strict": "strict-1to1",
        "spatial": "spatial-pair (default)",
        "none": "no-filter",
    }[args.filter_mode]
    neg_mode = "include-negatives" if args.include_negatives else "skip-negatives"
    print(
        f"[classes] {list(active_classes)}  filter={mode}  "
        f"iou_thresh={args.spatial_iou_thresh}  negatives={neg_mode}"
    )
    if args.sample > 0:
        print(
            f"[sample] {args.sample} max  strategy={args.sample_strategy}  "
            f"seed={args.sample_seed}  neg_ratio={args.neg_ratio}  "
            f"candidates={counts['candidates']}  sampled={counts['sampled']}"
        )
    total_written = counts["train"] + counts["val"] + counts["test"]
    print(f"[done] {total_written} images -> {args.out}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
