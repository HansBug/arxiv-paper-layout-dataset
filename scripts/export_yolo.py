#!/usr/bin/env python3
"""Export a pipeline output tree (``runs/v2_validated`` + ``runs/v2_extra``)
into Ultralytics YOLO format, deterministically split 8:1:1.

The YOLO layout this produces::

    <out>/
      data.yaml                  -- class names, split paths
      images/
        train/<paper_id>__page_<NNN>.png
        val/<paper_id>__page_<NNN>.png
        test/<paper_id>__page_<NNN>.png
      labels/
        train/<paper_id>__page_<NNN>.txt
        val/<paper_id>__page_<NNN>.txt
        test/<paper_id>__page_<NNN>.txt

Label files are plain YOLO: one row per instance, each row is::

    <class_index> <cx_norm> <cy_norm> <w_norm> <h_norm>

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
) -> dict[int, list[str]]:
    """For each page image id, the list of YOLO rows to write."""
    kinds = {c["id"]: c["name"] for c in annotations["categories"]}
    per_image: dict[int, list[str]] = {}
    for item in annotations["annotations"]:
        kind = kinds.get(item["category_id"])
        if kind is None or kind not in CLASS_TO_INDEX:
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


def export(
    inputs: list[Path],
    out: Path,
    weights: tuple[int, int, int],
    copy_images: bool = True,
) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    for subset in ("train", "val", "test"):
        (out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out / "labels" / subset).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "skipped_no_labels": 0}

    for input_root in inputs:
        for paper_id, ann_path, pages_dir in iter_papers(input_root):
            annotations = json.loads(ann_path.read_text(encoding="utf-8"))
            size_by_id = {
                img["id"]: (img["width"], img["height"]) for img in annotations["images"]
            }
            file_by_id = {img["id"]: img["file_name"] for img in annotations["images"]}
            labels = yolo_label_lines(annotations, size_by_id)

            for image_id, file_name in file_by_id.items():
                src_img = pages_dir / file_name
                if not src_img.is_file():
                    continue
                stem = build_stem(paper_id, image_id)
                split = pick_split(stem, weights)

                rows = labels.get(image_id, [])
                if not rows:
                    counts["skipped_no_labels"] += 1
                    continue

                dst_img = out / "images" / split / f"{stem}.png"
                dst_lbl = out / "labels" / split / f"{stem}.txt"
                if copy_images:
                    shutil.copyfile(src_img, dst_img)
                else:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    dst_img.symlink_to(src_img.resolve())
                dst_lbl.write_text("\n".join(rows) + "\n", encoding="utf-8")
                counts[split] += 1

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
                *[f"  {idx}: {name}" for idx, name in enumerate(CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return counts


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
        "source tree persists)",
    )
    args = parser.parse_args()

    counts = export(
        inputs=[p.resolve() for p in args.input],
        out=args.out.resolve(),
        weights=args.split,
        copy_images=not args.symlink,
    )
    print(
        f"[done] {sum(v for k, v in counts.items() if k != 'skipped_no_labels')} images "
        f"-> {args.out}"
    )
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
