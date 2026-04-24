"""Draw QC bboxes onto rendered pages."""

from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf

from .render import BBox, ResolvedLabel, bbox_pt_to_px


COLORS = {
    "fig": (230, 25, 75),         # red
    "fig_cap": (255, 150, 150),
    "table": (60, 180, 75),       # green
    "table_cap": (180, 230, 180),
    "algorithm": (0, 130, 200),   # blue
    "algorithm_cap": (150, 190, 230),
    "listing": (245, 130, 48),    # orange
    "listing_cap": (250, 200, 150),
}


def _draw_rect(pix: fitz.Pixmap, bbox: BBox, color: tuple[int, int, int], thickness: int = 4) -> None:
    """Draw a rectangle on a fitz pixmap in-place using PIL fallback."""
    from PIL import Image, ImageDraw

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    draw.rectangle([bbox.x0, bbox.y0, bbox.x1, bbox.y1], outline=color, width=thickness)
    # write back into pix - actually simpler: return the PIL image and let the
    # caller save it.
    return img


def draw_labels_on_image(
    page_image_path: Path,
    labels: list[ResolvedLabel],
    dpi: int,
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(page_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for lab in labels:
        color = COLORS.get(lab.kind, (255, 0, 255))
        b_px = bbox_pt_to_px(lab.bbox_pt, dpi)
        # cap boxes are drawn thinner + semi-offset so they don't overwhelm
        thickness = 3 if lab.kind.endswith("_cap") else 5
        draw.rectangle(
            [b_px.x0, b_px.y0, b_px.x1, b_px.y1],
            outline=color,
            width=thickness,
        )
        # label text at top-left, stacked if multiple labels share corners
        text = f"{lab.kind}:{lab.label_id}"
        tx, ty = b_px.x0 + 4, max(0, b_px.y0 + 4)
        # white background pad
        try:
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=color)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
