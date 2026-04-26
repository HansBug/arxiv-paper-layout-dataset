#!/usr/bin/env python3
"""Export a pipeline output tree (``runs/v2_validated`` + ``runs/v2_extra``)
into Ultralytics YOLO format, deterministically split 8:1:1.

The YOLO layout this produces::

    <out>/
      data.yaml                  -- class names, split paths
      README.md                  -- summary, embedded plots, provenance
      analysis/                  -- auto-generated diagnostic plots
        class_counts.png
        bbox_centers.png
        bbox_aspect.png
        bbox_size.png
        page_aspect.png
        labels_per_image.png
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
rejects the paper. This is the user's preferred "clean enough for
training" gate — it tolerates the common sub-figure pattern that
strict 1:1 would wrongly throw away. Use ``--strict-1to1`` for the
smaller strictly-1:1 subset or ``--no-filter`` to export everything.

Image processing:
- Default output format: JPG (``--format jpg``; ``--format png`` keeps PNG).
- Default resize policy: if ``min(width, height) > 720``, scale so the
  shorter side equals 720. Override with ``--max-short-side``; use 0 to
  disable resizing. Boxes are re-normalised automatically because YOLO
  labels are already relative to the (possibly-resized) image size.

Sampling (small smoke-test exports):
- ``--sample N`` caps the dataset to ``N`` images total. Useful for
  shipping a small "what does this dataset look like" preview.
- ``--sample-strategy`` controls which pages get picked:
  * ``balanced`` (default) — round-robin by archive, and within each
    archive prefer pages that contribute rarer classes (so even a tiny
    sample exposes algorithm/listing pages, not just figure pages).
  * ``by-archive`` — pure round-robin across archives; even archive
    coverage but no class-rarity bias.
  * ``random`` — deterministic shuffle by ``--sample-seed``; simplest
    and most reproducible for QC tooling.
- Splitting is unaffected by sampling: the same page always lands in
  the same split (sha256 hash of the filename).

Split determinism:
- The file name contains both the arxiv paper id and the page number, so
  it uniquely identifies the sample.
- Split membership is picked by ``sha256(filename) % 10`` mapped to
  train (0-7) / val (8) / test (9). Given the same filename, the same
  split is always assigned regardless of re-runs or input order.

Dataset card:
- After exporting, a ``README.md`` plus a small ``analysis/`` directory
  of PNG plots is written into the output. The README has split sizes,
  per-class counts, archive / paper coverage, and embeds the plots so
  reviewers can read the dataset's properties at a glance. Disable with
  ``--no-readme``.

Usage::

    python3 scripts/export_yolo.py \\
        --input runs/corpus/workspaces \\
        --out runs/yolo_smoke \\
        --classes figure,figure_cap,table,table_cap \\
        --sample 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
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
    count_kinds,
    paper_passes_spatial_pairing,
    paper_passes_strict_1to1,
)

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}


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
) -> dict[int, list[str]]:
    """For each page image id, the list of YOLO rows to write.

    ``class_to_index`` maps active class names to their output YOLO index.
    Annotations whose kind is missing from this map are silently dropped
    (so passing ``{"figure": 0, "figure_cap": 1}`` produces a 2-class dataset).
    Defaults to the full 8-class mapping.
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
        # workspaces dir lives at runs/corpus/workspaces, state at runs/corpus/state.json
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
        # The workspace dirname is what ``iter_papers`` yields as
        # ``paper_id`` — derive it from the full ``workspace`` path
        # rather than relying on a separate slug field.
        ws = record.get("workspace")
        if ws:
            slug = Path(ws).name
            out[slug] = archive
    return out


# ---------------------------------------------------------------------------
# Candidate enumeration + sampling
# ---------------------------------------------------------------------------


def _collect_candidates(
    inputs: list[Path],
    weights: tuple[int, int, int],
    active_classes: tuple[str, ...],
    filter_mode: str,
    spatial_iou_thresh: float,
    include_negatives: bool,
    archive_lookup: dict[str, str],
    counts: dict[str, int],
) -> list[dict]:
    """Walk every paper, filter, expand to one entry per page that we'd
    like to write. Sampling later picks a subset; emission writes them.

    ``counts`` is updated with paper-level reject counters so the user
    can see *why* the corpus shrank between input and output.
    """
    class_to_index = {name: idx for idx, name in enumerate(active_classes)}
    candidates: list[dict] = []

    for input_root in inputs:
        for paper_id, ann_path, pages_dir in iter_papers(input_root):
            try:
                annotations = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if filter_mode == "strict":
                if not paper_passes_strict_1to1(
                    annotations, active_classes, spatial_iou_thresh
                ):
                    counts["skipped_papers_not_1to1"] += 1
                    continue
            elif filter_mode == "spatial":
                if not paper_passes_spatial_pairing(
                    annotations, active_classes, spatial_iou_thresh
                ):
                    counts["skipped_papers_no_spatial_pair"] += 1
                    continue
            size_by_id = {
                img["id"]: (img["width"], img["height"]) for img in annotations["images"]
            }
            file_by_id = {img["id"]: img["file_name"] for img in annotations["images"]}
            labels = yolo_label_lines(annotations, size_by_id, class_to_index)
            archive = archive_lookup.get(paper_id, "unknown")

            for image_id, file_name in file_by_id.items():
                src_img = pages_dir / file_name
                if not src_img.is_file():
                    continue
                rows = labels.get(image_id, [])
                if not rows and not include_negatives:
                    counts["skipped_no_labels"] += 1
                    continue
                stem = build_stem(paper_id, image_id)
                split = pick_split(stem, weights)
                candidates.append(
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
    return candidates


def _sample_candidates(
    candidates: list[dict],
    n: int,
    strategy: str,
    seed: int,
) -> list[dict]:
    """Reduce ``candidates`` to at most ``n`` entries using ``strategy``.

    No-op when ``n <= 0`` or the candidate list is already smaller. The
    output preserves enough diversity for a smoke test:

    - ``random``: shuffle with ``seed`` and slice. Predictable, no
      attempt to balance anything.
    - ``by-archive``: round-robin over archives; equal slot per archive.
    - ``balanced`` (default): same archive round-robin, but within an
      archive the first picks are pages whose classes are rare in the
      full candidate pool. Tiny samples therefore still surface
      algorithm/listing pages, not just figure-only pages.
    """
    total = len(candidates)
    if n <= 0 or n >= total:
        return list(candidates)
    rng = random.Random(seed)

    if strategy == "random":
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled[:n]

    # group by archive (used by both by-archive and balanced)
    groups: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        groups[cand["archive"]].append(cand)

    if strategy == "balanced":
        # rarity score per page = sum 1 / class_freq across the candidate pool
        cls_freq: Counter = Counter()
        for cand in candidates:
            for row in cand["rows"]:
                cls_idx = int(row.split()[0])
                cls_freq[cls_idx] += 1

        def rarity(cand: dict) -> float:
            score = 0.0
            for row in cand["rows"]:
                cls_idx = int(row.split()[0])
                score += 1.0 / max(1, cls_freq.get(cls_idx, 1))
            # negative pages have rarity 0 — they sort to the back, so
            # tiny samples still prefer label-bearing pages first.
            return score

        for archive, lst in groups.items():
            # randomise tie-break so identical-rarity pages don't always
            # come from the same paper.
            rng.shuffle(lst)
            lst.sort(key=lambda c: -rarity(c))
    elif strategy == "by-archive":
        for lst in groups.values():
            rng.shuffle(lst)
    else:
        raise ValueError(f"unknown sample strategy {strategy!r}")

    archives = list(groups.keys())
    rng.shuffle(archives)  # round-robin order shouldn't be alphabetic
    cursor = {a: 0 for a in archives}
    result: list[dict] = []
    while len(result) < n:
        progressed = False
        for a in archives:
            if cursor[a] < len(groups[a]):
                result.append(groups[a][cursor[a]])
                cursor[a] += 1
                progressed = True
                if len(result) >= n:
                    break
        if not progressed:
            break
    return result


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


def _emit_candidates(
    candidates: list[dict],
    out: Path,
    copy_images: bool,
    image_format: str,
    max_short_side: int,
    jpg_quality: int,
    counts: dict[str, int],
) -> dict:
    """Write images + labels for each candidate. Accumulates statistics
    needed by the dataset card and returns them as a plain dict.
    """
    ext = f".{image_format}"
    stats: dict = {
        "splits": Counter(),
        "split_positives": Counter(),
        "split_negatives": Counter(),
        "classes_per_split": defaultdict(Counter),
        "bbox_centers": [],  # list[(cls_idx, cx, cy)]
        "bbox_sizes": [],    # list[(cls_idx, nw, nh)]
        "page_sizes": [],    # list[(w, h)] of saved image
        "labels_per_image": [],
        "archives": Counter(),
        "archive_x_class": defaultdict(Counter),
        "papers": set(),
    }

    for cand in candidates:
        src_img = cand["src_img"]
        rows = cand["rows"]
        split = cand["split"]
        stem = cand["stem"]
        archive = cand["archive"]
        is_negative = not rows

        dst_img = out / "images" / split / f"{stem}{ext}"
        dst_lbl = out / "labels" / split / f"{stem}.txt"

        # Symlink only makes sense when src and dst share the exact same
        # byte content: we can't symlink across format + resize.
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

        # Empty `.txt` (0 rows) is a valid YOLO negative sample.
        dst_lbl.write_text(
            ("\n".join(rows) + "\n") if rows else "", encoding="utf-8"
        )

        counts[split] += 1
        counts["negatives" if is_negative else "positives"] += 1

        stats["splits"][split] += 1
        if is_negative:
            stats["split_negatives"][split] += 1
        else:
            stats["split_positives"][split] += 1
        stats["page_sizes"].append((saved_w, saved_h))
        stats["labels_per_image"].append(len(rows))
        stats["archives"][archive] += 1
        stats["papers"].add(cand["paper_id"])
        for row in rows:
            parts = row.split()
            cls_idx = int(parts[0])
            cx, cy, nw, nh = (float(p) for p in parts[1:5])
            stats["bbox_centers"].append((cls_idx, cx, cy))
            stats["bbox_sizes"].append((cls_idx, nw, nh))

    return stats


# ---------------------------------------------------------------------------
# Dataset card (README + plots)
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
            h, _, _ = np.histogram2d(
                xs, ys, bins=bins, range=[[0, 1], [0, 1]]
            )
            ax.imshow(
                h.T,
                origin="upper",
                extent=(0, 1, 1, 0),
                aspect="auto",
                cmap="viridis",
            )
            ax.set_title(f"{name} (n={len(pts)})")
        else:
            ax.set_title(f"{name} (n=0)")
            ax.text(0.5, 0.5, "no boxes", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # page y grows downward
        ax.set_xlabel("cx (page-normalized)")
        ax.set_ylabel("cy (page-normalized)")
    # blank unused axes
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
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.6,
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
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.6,
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

    aspects: list[float] = []
    for w, h in stats["page_sizes"]:
        if h > 0:
            aspects.append(w / h)
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
    import numpy as np

    counts = list(stats["labels_per_image"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if counts:
        max_c = min(50, max(counts))
        ax.hist(
            counts,
            bins=range(0, max_c + 2),
            color="#55A467",
            edgecolor="white",
        )
        ax.set_xlabel(f"labels per image  (capped display @ {max_c}; "
                      f"max actual = {max(counts)})")
    else:
        ax.set_xlabel("labels per image")
    ax.set_ylabel("page count")
    pos = sum(1 for c in counts if c > 0)
    neg = sum(1 for c in counts if c == 0)
    ax.set_title(f"Labels per page (positive {pos} / negative {neg})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_readme(
    path: Path,
    out: Path,
    stats: dict,
    counts: dict[str, int],
    args_repr: str,
    active_classes: tuple[str, ...],
    sample_n: int,
    sample_strategy: str,
    filter_mode: str,
    image_format: str,
    max_short_side: int,
    spatial_iou_thresh: float,
) -> None:
    """Write README.md describing what the dataset is, how it was made,
    and what its label/bbox distribution looks like."""

    total_images = sum(stats["splits"].values())
    total_labels = len(stats["bbox_sizes"])
    pages_with_labels = sum(1 for n in stats["labels_per_image"] if n > 0)
    pages_neg = sum(1 for n in stats["labels_per_image"] if n == 0)

    # Per-class totals
    class_totals: Counter = Counter()
    for split, c in stats["classes_per_split"].items():
        for k, v in c.items():
            class_totals[k] += v

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
    lines.append(f"- **paper-level filter**: `{filter_mode}` "
                 f"(IoU thresh {spatial_iou_thresh})")
    if sample_n > 0:
        lines.append(
            f"- **sampling**: `--sample {sample_n}` with strategy "
            f"`{sample_strategy}` (output is a smoke-test slice, not the "
            "full corpus)"
        )
    else:
        lines.append("- **sampling**: full export (no `--sample`)")
    lines.append(f"- **image format**: `{image_format}` "
                 f"(short side <= {max_short_side or 'unbounded'})")
    lines.append(
        f"- **papers covered**: {len(stats['papers'])}  | "
        f"**archives covered**: {len(stats['archives'])}"
    )
    lines.append("")
    lines.append("| split | pages | positive | negative |")
    lines.append("|---|---:|---:|---:|")
    for split in ("train", "val", "test"):
        total = stats["splits"].get(split, 0)
        pos = stats["split_positives"].get(split, 0)
        neg = stats["split_negatives"].get(split, 0)
        lines.append(f"| {split} | {total} | {pos} | {neg} |")
    lines.append(
        f"| **all** | **{total_images}** | "
        f"**{pages_with_labels}** | **{pages_neg}** |"
    )
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
        lines.append(
            f"| `{name}` | {tr} | {va} | {te} | **{tr + va + te}** |"
        )
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

    # 4. Spatial / size analytics
    lines.append("## 4. BBox spatial distribution")
    lines.append("")
    lines.append(
        "Heatmap of bbox centers on a page-normalised canvas. Origin "
        "is top-left and y grows downward to match image coordinates, "
        "so pages have *visually* the same orientation as the rendered PDF."
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
        "and ``tall`` boxes (<0.4) are rare across the corpus."
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
    lines.append("  data.yaml              # ultralytics YOLO data spec")
    lines.append("  README.md              # this file")
    lines.append("  analysis/              # auto-generated diagnostic plots")
    lines.append("    class_counts.png")
    lines.append("    bbox_centers.png")
    lines.append("    bbox_size.png")
    lines.append("    bbox_aspect.png")
    lines.append("    page_aspect.png")
    lines.append("    labels_per_image.png")
    lines.append("  images/{train,val,test}/<paper_id>__page_NNN."
                 + image_format)
    lines.append("  labels/{train,val,test}/<paper_id>__page_NNN.txt")
    lines.append("```")
    lines.append("")

    # 10. Provenance
    lines.append("## 10. Provenance & caveats")
    lines.append("")
    lines.append(
        "- Source pipeline: `arxiv-paper-layout-dataset` "
        "(`scripts/export_yolo.py`)."
    )
    if sample_n > 0:
        lines.append(
            f"- This export is a sampled slice (`{sample_strategy}`, "
            f"capped at {sample_n} images). "
            "**Do not generalise distributional claims** from it; use "
            "the full export for that."
        )
    lines.append(
        f"- `paper-level filter = {filter_mode}` rejects papers whose "
        "bbox structure violates the body/cap spatial pairing rule."
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
    lines.append(
        "- Detector hint: anchor-free heads (YOLOv8/9 default) work "
        "well; if you bake fixed anchors, prioritise wide aspect "
        "ratios and skip the tall (<0.4) band — the corpus has "
        "essentially zero tall floats."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _split_per_class_counts(stats: dict, active_classes: tuple[str, ...]) -> None:
    """Walk per-row stats again to populate stats['classes_per_split']
    properly (we only added totals during emit)."""
    # already done during emit by caller — kept here for future use


def write_dataset_card(
    out: Path,
    stats: dict,
    counts: dict[str, int],
    args_repr: str,
    active_classes: tuple[str, ...],
    sample_n: int,
    sample_strategy: str,
    filter_mode: str,
    image_format: str,
    max_short_side: int,
    spatial_iou_thresh: float,
) -> None:
    """Emit ``analysis/`` plots and ``README.md`` describing the export."""

    # Recompute per-split per-class counts since emit stored class counts
    # only at the aggregate level. We still have bbox rows in
    # stats["bbox_sizes"] paired with class index; combine with
    # stats["splits"] is not enough — we need per-page split. But we
    # know the candidate's split because we wrote one image per split;
    # easiest: use stats["classes_per_split"] which the emit step did
    # populate.

    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    _save_class_counts(analysis / "class_counts.png", stats, active_classes)
    _save_bbox_centers(analysis / "bbox_centers.png", stats, active_classes)
    _save_bbox_aspect(analysis / "bbox_aspect.png", stats, active_classes)
    _save_bbox_size(analysis / "bbox_size.png", stats, active_classes)
    _save_page_aspect(analysis / "page_aspect.png", stats)
    _save_labels_per_image(analysis / "labels_per_image.png", stats)
    _write_readme(
        out / "README.md",
        out,
        stats,
        counts,
        args_repr,
        active_classes,
        sample_n,
        sample_strategy,
        filter_mode,
        image_format,
        max_short_side,
        spatial_iou_thresh,
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
    write_readme: bool = True,
    corpus_state: Path | None = None,
    args_repr: str | None = None,
) -> dict[str, int]:
    """Orchestrate the full export.

    Phases:
      1. Walk ``inputs``, apply the paper-level filter, and build a list
         of page candidates. Each candidate already knows its split.
      2. Optionally subsample candidates with ``--sample N``.
      3. Emit images + labels.
      4. Write data.yaml and (unless ``write_readme=False``) the
         dataset card under ``out/analysis/`` plus ``out/README.md``.
    """
    out.mkdir(parents=True, exist_ok=True)
    for subset in ("train", "val", "test"):
        (out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out / "labels" / subset).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "train": 0,
        "val": 0,
        "test": 0,
        "positives": 0,
        "negatives": 0,
        "skipped_no_labels": 0,
        "skipped_papers_not_1to1": 0,
        "skipped_papers_no_spatial_pair": 0,
        "candidates": 0,
        "sampled": 0,
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
    )
    counts["candidates"] = len(candidates)

    if sample > 0 and sample < len(candidates):
        candidates = _sample_candidates(
            candidates, sample, sample_strategy, sample_seed
        )
    counts["sampled"] = len(candidates)

    stats = _emit_candidates(
        candidates,
        out,
        copy_images,
        image_format,
        max_short_side,
        jpg_quality,
        counts,
    )

    # Populate per-split per-class counts from the candidate list since
    # _emit_candidates only fills aggregates. We walk one more pass over
    # candidates because rows already exist in memory.
    for cand in candidates:
        for row in cand["rows"]:
            cls_idx = int(row.split()[0])
            stats["classes_per_split"][cand["split"]][active_classes[cls_idx]] += 1

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(active_classes)],
                "",
            ]
        ),
        encoding="utf-8",
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
            filter_mode,
            image_format,
            max_short_side,
            spatial_iou_thresh,
        )

    return counts


def parse_classes(raw: str) -> tuple[str, ...]:
    """``--classes figure,figure_cap,table,table_cap`` -> ``('figure', ...)``.

    Preserves user-specified order so indices 0..N-1 in the output
    dataset match the CLI order exactly. Unknown names raise.
    """
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
        help="symlink images instead of copying (saves disk; only usable if the "
        "source tree persists AND format matches AND no resize).",
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
        default=720,
        help="downscale images so min(width, height) == this value when the "
        "source image is larger; 0 disables resizing (default 720)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=90,
        help="JPEG quality when --format jpg (default 90)",
    )
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=CLASSES,
        help="comma-separated subset of classes to export; index in the "
        "output dataset follows this order. Default: all 8 classes. "
        f"Valid: {','.join(CLASSES)}",
    )
    # Paper-level filter. Default is spatial-pair (N:1 with bbox
    # containment) — the user's preferred "clean-enough-for-training"
    # gate. Callers pick a stricter (--strict-1to1) or looser
    # (--no-filter) alternative.
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--strict-1to1",
        dest="filter_mode",
        action="store_const",
        const="strict",
        help="Only keep papers where every active pair is 1:1 AND "
        "spatially valid (same per-page count, every body bbox mostly "
        "inside some cap bbox, every cap bbox holds at least one body). "
        "Smaller but cleanest possible subset.",
    )
    filter_group.add_argument(
        "--spatial-pair",
        dest="filter_mode",
        action="store_const",
        const="spatial",
        help="[default] Only keep papers where every active pair is "
        "spatially valid, relaxed to N:1 — one cap can hold multiple "
        "bodies, which covers the common multi-subfigure pattern. "
        "Orphan body (not inside any cap) or empty cap (no body "
        "inside) still rejects the paper. Passed explicitly this flag "
        "is a no-op (already the default); kept for self-documentation.",
    )
    filter_group.add_argument(
        "--no-filter",
        dest="filter_mode",
        action="store_const",
        const="none",
        help="Disable paper-level filtering — export every paper even "
        "if its bbox spatial structure is messy. Use this when you "
        "want to pretrain on the largest possible corpus and deal "
        "with noise downstream.",
    )
    parser.set_defaults(filter_mode="spatial")
    parser.add_argument(
        "--spatial-iou-thresh",
        type=float,
        default=0.9,
        help="threshold for 'body mostly inside cap' (intersection_area "
        "/ body_area). Default 0.9 — a body bbox is accepted if ≥90%% "
        "of its area falls inside the cap bbox. Lower values tolerate "
        "more body-bbox overflow.",
    )
    neg_group = parser.add_mutually_exclusive_group()
    neg_group.add_argument(
        "--include-negatives",
        dest="include_negatives",
        action="store_true",
        default=True,
        help="Export pages with no active-class annotations as pure "
        "negative samples (empty .txt label file). YOLO uses them as "
        "background examples, which reduces false positives. Default: on.",
    )
    neg_group.add_argument(
        "--skip-negatives",
        dest="include_negatives",
        action="store_false",
        help="Drop pages with no active-class annotations instead of "
        "exporting them as negatives. Use this if you only want "
        "label-bearing pages.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, cap the export to this many images total. Useful "
        "for shipping a small smoke-test slice. 0 (default) exports "
        "every page that survived the filters.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("balanced", "by-archive", "random"),
        default="balanced",
        help="How to choose pages when --sample is set. Default "
        "`balanced` round-robins by archive AND prefers rarer-class "
        "pages first, so even tiny samples surface algorithm/listing "
        "pages. `by-archive` is plain round-robin without class "
        "rebalancing. `random` is a shuffled slice (predictable; no "
        "diversity guarantee).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for sampling shuffles. Default 0 — pinned for "
        "reproducible smoke-test datasets.",
    )
    parser.add_argument(
        "--no-readme",
        dest="write_readme",
        action="store_false",
        default=True,
        help="Skip the README.md + analysis/*.png generation. Default: "
        "write them (fast, ~1s, useful for everyone).",
    )
    parser.add_argument(
        "--corpus-state",
        type=Path,
        default=None,
        help="Path to corpus state.json so the dataset card can show "
        "archive coverage. Default: auto-detected at "
        "`<input>/../state.json`.",
    )
    args = parser.parse_args()

    # Reconstructed command-line that produced this export — useful for
    # the README's provenance section.
    args_repr = "python3 scripts/export_yolo.py " + " ".join(sys.argv[1:])

    counts = export(
        inputs=[p.resolve() for p in args.input],
        out=args.out.resolve(),
        weights=args.split,
        copy_images=not args.symlink,
        image_format=args.format,
        max_short_side=args.max_short_side,
        jpg_quality=args.jpg_quality,
        active_classes=args.classes,
        strict_1to1=(args.filter_mode == "strict"),
        spatial_pair=(args.filter_mode == "spatial"),
        spatial_iou_thresh=args.spatial_iou_thresh,
        include_negatives=args.include_negatives,
        sample=args.sample,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        write_readme=args.write_readme,
        corpus_state=args.corpus_state,
        args_repr=args_repr,
    )
    mode = {
        "strict": "strict-1to1",
        "spatial": "spatial-pair (default)",
        "none": "no-filter",
    }[args.filter_mode]
    neg_mode = "include-negatives" if args.include_negatives else "skip-negatives"
    print(
        f"[classes] {list(args.classes)}  filter={mode}  "
        f"iou_thresh={args.spatial_iou_thresh}  negatives={neg_mode}"
    )
    if args.sample > 0:
        print(
            f"[sample] {args.sample} max  strategy={args.sample_strategy}  "
            f"seed={args.sample_seed}  candidates={counts['candidates']}  "
            f"sampled={counts['sampled']}"
        )
    total_written = counts["train"] + counts["val"] + counts["test"]
    print(f"[done] {total_written} images -> {args.out}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
