#!/usr/bin/env python3
"""Export a pipeline output tree (``runs/v2_validated`` + ``runs/v2_extra``)
into Ultralytics YOLO format, deterministically split 8:1:1.

The YOLO layout this produces::

    <out>/
      data.yaml                  -- class names, split paths
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

Image processing:
- Default output format: JPG (``--format jpg``; ``--format png`` keeps PNG).
- Default resize policy: if ``min(width, height) > 720``, scale so the
  shorter side equals 720. Override with ``--max-short-side``; use 0 to
  disable resizing. Boxes are re-normalised automatically because YOLO
  labels are already relative to the (possibly-resized) image size.

Split determinism:
- The file name contains both the arxiv paper id and the page number, so
  it uniquely identifies the sample.
- Split membership is picked by ``sha256(filename) % 10`` mapped to
  train (0-7) / val (8) / test (9). Given the same filename, the same
  split is always assigned regardless of re-runs or input order.

Usage::

    python3 scripts/export_yolo.py \\
        --input runs/v2_validated runs/v2_extra \\
        --out runs/yolo_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image


CLASSES: tuple[str, ...] = (
    "fig",
    "fig_cap",
    "table",
    "table_cap",
    "algorithm",
    "algorithm_cap",
    "listing",
    "listing_cap",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}

# Body/caption pairs used for the strict-1:1 filter. If the caller limits
# the export to a subset of classes via ``--classes``, only pairs whose
# BOTH members are selected are enforced.
CAPTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("fig", "fig_cap"),
    ("table", "table_cap"),
    ("algorithm", "algorithm_cap"),
    ("listing", "listing_cap"),
)


def count_kinds(annotations: dict) -> dict[str, int]:
    """Per-class instance count for a paper's COCO-style ``annotations.json``."""
    kind_by_cat = {c["id"]: c["name"] for c in annotations.get("categories", [])}
    out: dict[str, int] = {}
    for item in annotations.get("annotations", []):
        name = kind_by_cat.get(item["category_id"])
        if name is not None:
            out[name] = out.get(name, 0) + 1
    return out


def _pair_per_page_bboxes(
    annotations: dict,
    active_classes: tuple[str, ...],
):
    """Yield ``(body_kind, cap_kind, bodies_xywh, caps_xywh)`` per page, per
    active pair. Pairs whose both members aren't in ``active_classes`` are
    skipped. Pages with neither body nor cap are skipped.
    """
    active = set(active_classes)
    active_pairs = [(b, c) for b, c in CAPTION_PAIRS
                    if b in active and c in active]
    if not active_pairs:
        return
    kind_by_cat = {c["id"]: c["name"] for c in annotations.get("categories", [])}
    per_page: dict[int, dict[str, list]] = {}
    for item in annotations.get("annotations", []):
        kind = kind_by_cat.get(item["category_id"])
        if kind is None:
            continue
        per_page.setdefault(item["image_id"], {}).setdefault(kind, []).append(item["bbox"])
    for page_kinds in per_page.values():
        for body_kind, cap_kind in active_pairs:
            bodies = page_kinds.get(body_kind, [])
            caps = page_kinds.get(cap_kind, [])
            if not bodies and not caps:
                continue
            yield body_kind, cap_kind, bodies, caps


def paper_passes_strict_1to1(
    annotations: dict,
    active_classes: tuple[str, ...],
    iou_thresh: float = 0.9,
) -> bool:
    """Strict 1:1 with spatial validity:

    - On every page, ``count(body) == count(cap)`` for each active pair.
    - Every body is mostly contained in some cap, every cap holds >=1
      body (orphan body / empty cap -> reject).
    - At least one pair somewhere is non-empty.
    """
    any_pair_nonempty = False
    for body_kind, cap_kind, bodies, caps in _pair_per_page_bboxes(
        annotations, active_classes
    ):
        if len(bodies) != len(caps):
            return False
        for bb in bodies:
            if not any(_body_mostly_inside_cap(bb, cb, iou_thresh) for cb in caps):
                return False
        for cb in caps:
            if not any(_body_mostly_inside_cap(bb, cb, iou_thresh) for bb in bodies):
                return False
        any_pair_nonempty = True
    return any_pair_nonempty


def _body_mostly_inside_cap(body_xywh, cap_xywh, thresh: float) -> bool:
    """True iff ``body`` is mostly contained in ``cap``.

    ``intersection_area / body_area >= thresh``, so a fig bbox that sits
    cleanly inside its fig_cap bbox (or overshoots it by a small
    amount) counts as contained.
    """
    xa, ya, wa, ha = body_xywh
    xc, yc, wc, hc = cap_xywh
    ix0 = max(xa, xc); iy0 = max(ya, yc)
    ix1 = min(xa + wa, xc + wc); iy1 = min(ya + ha, yc + hc)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    body_area = max(1e-9, wa * ha)
    return (inter / body_area) >= thresh


def paper_passes_spatial_pairing(
    annotations: dict,
    active_classes: tuple[str, ...],
    iou_thresh: float = 0.9,
) -> bool:
    """Relaxed alternative to strict 1:1 based on bbox containment.

    For every pair ``(body, cap)`` where BOTH names are in
    ``active_classes``, on every page:

    - Every ``body`` bbox must be *mostly contained* in some ``cap``
      bbox on that page (``intersection / body_area >= iou_thresh``).
      An orphan body (no parent cap) rejects the paper.
    - Every ``cap`` bbox must have at least one body mostly contained
      in it. An empty cap (no child body) rejects the paper.
    - At least one (body, cap) pair must be non-empty *somewhere* in
      the paper, otherwise there's nothing to learn.

    Unlike :func:`paper_passes_strict_1to1`, per-page counts do not
    have to match: one ``fig_cap`` enclosing multiple ``fig`` bboxes
    (the common subfigure pattern) is accepted.
    """
    any_pair_nonempty = False
    for body_kind, cap_kind, bodies, caps in _pair_per_page_bboxes(
        annotations, active_classes
    ):
        for bb in bodies:
            if not any(_body_mostly_inside_cap(bb, cb, iou_thresh) for cb in caps):
                return False
        for cb in caps:
            if not any(_body_mostly_inside_cap(bb, cb, iou_thresh) for bb in bodies):
                return False
        if bodies and caps:
            any_pair_nonempty = True
    return any_pair_nonempty


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
    (so passing ``{"fig": 0, "fig_cap": 1}`` produces a 2-class dataset).
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
            f"{CLASS_TO_INDEX[kind]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
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


def _write_image(
    src: Path,
    dst: Path,
    fmt: str,
    max_short_side: int,
    jpg_quality: int,
) -> None:
    """Save ``src`` into ``dst`` in ``fmt`` format, optionally resized so the
    shorter side is at most ``max_short_side`` (use 0 to disable).
    """
    with Image.open(src) as im:
        w, h = im.size
        short = min(w, h)
        if max_short_side > 0 and short > max_short_side:
            scale = max_short_side / float(short)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
        if fmt == "jpg":
            im = im.convert("RGB")
            im.save(dst, format="JPEG", quality=jpg_quality, optimize=True)
        elif fmt == "png":
            im.save(dst, format="PNG", optimize=True)
        else:
            raise ValueError(f"unsupported format {fmt!r}")


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
) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    for subset in ("train", "val", "test"):
        (out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out / "labels" / subset).mkdir(parents=True, exist_ok=True)

    class_to_index = {name: idx for idx, name in enumerate(active_classes)}
    ext = f".{image_format}"
    counts: dict[str, int] = {
        "train": 0,
        "val": 0,
        "test": 0,
        "positives": 0,
        "negatives": 0,
        "skipped_no_labels": 0,
        "skipped_papers_not_1to1": 0,
        "skipped_papers_no_spatial_pair": 0,
    }

    for input_root in inputs:
        for paper_id, ann_path, pages_dir in iter_papers(input_root):
            annotations = json.loads(ann_path.read_text(encoding="utf-8"))
            if strict_1to1:
                if not paper_passes_strict_1to1(
                    annotations, active_classes, spatial_iou_thresh
                ):
                    counts["skipped_papers_not_1to1"] += 1
                    continue
            elif spatial_pair:
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

            for image_id, file_name in file_by_id.items():
                src_img = pages_dir / file_name
                if not src_img.is_file():
                    continue
                stem = build_stem(paper_id, image_id)
                split = pick_split(stem, weights)

                rows = labels.get(image_id, [])
                if not rows:
                    if not include_negatives:
                        counts["skipped_no_labels"] += 1
                        continue
                    is_negative = True
                else:
                    is_negative = False

                dst_img = out / "images" / split / f"{stem}{ext}"
                dst_lbl = out / "labels" / split / f"{stem}.txt"
                # Symlink only makes sense when src and dst share the exact
                # same byte content: we can't symlink across format + resize.
                can_symlink = (
                    (not copy_images)
                    and image_format == src_img.suffix.lstrip(".").lower()
                    and max_short_side <= 0
                )
                if can_symlink:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    dst_img.symlink_to(src_img.resolve())
                else:
                    _write_image(
                        src_img, dst_img, image_format, max_short_side, jpg_quality
                    )
                # Empty `.txt` (0 rows) is a valid YOLO negative sample.
                dst_lbl.write_text(
                    ("\n".join(rows) + "\n") if rows else "", encoding="utf-8"
                )
                counts[split] += 1
                counts["negatives" if is_negative else "positives"] += 1

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
    return counts


def parse_classes(raw: str) -> tuple[str, ...]:
    """``--classes fig,fig_cap,table,table_cap`` -> ``('fig', ...)``.

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
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--strict-1to1",
        action="store_true",
        help="Only keep papers where every active pair is 1:1 AND "
        "spatially valid (same per-page count, every body bbox mostly "
        "inside some cap bbox, every cap bbox holds at least one body). "
        "Use this for the cleanest possible training subset.",
    )
    filter_group.add_argument(
        "--spatial-pair",
        action="store_true",
        help="Only keep papers where every active pair is spatially "
        "valid, relaxed to N:1 — one cap can hold multiple bodies, "
        "which covers the common multi-subfigure pattern. Orphan body "
        "(not inside any cap) or empty cap (no body inside) still "
        "rejects the paper. Larger subset than --strict-1to1.",
    )
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
    args = parser.parse_args()

    counts = export(
        inputs=[p.resolve() for p in args.input],
        out=args.out.resolve(),
        weights=args.split,
        copy_images=not args.symlink,
        image_format=args.format,
        max_short_side=args.max_short_side,
        jpg_quality=args.jpg_quality,
        active_classes=args.classes,
        strict_1to1=args.strict_1to1,
        spatial_pair=args.spatial_pair,
        spatial_iou_thresh=args.spatial_iou_thresh,
        include_negatives=args.include_negatives,
    )
    mode = (
        "strict-1to1" if args.strict_1to1
        else "spatial-pair" if args.spatial_pair
        else "all (no filter)"
    )
    neg_mode = "include-negatives" if args.include_negatives else "skip-negatives"
    print(
        f"[classes] {list(args.classes)}  filter={mode}  "
        f"iou_thresh={args.spatial_iou_thresh}  negatives={neg_mode}"
    )
    total_written = counts["train"] + counts["val"] + counts["test"]
    print(f"[done] {total_written} images -> {args.out}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
