"""Project anchors into page images and emit labels.

zref-savepos coordinates are in scaled points (sp), with origin at the
bottom-left of the current page's paper (not the text block). PyMuPDF pages
use the PDF's default coordinate system: origin at the top-left in points.

So the transform is:

    x_pt_pdf = posx_sp / 65536
    y_pt_pdf = page_height_pt - posy_sp / 65536

At the PDF resolution we pick (DPI), pixels are ``pt * dpi / 72``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf

from .extractor import SP_PER_PT, Anchor, BoxDims, MarkDims, PageInfo
from .injector import InjectionManifest, LabeledAnchor


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def clipped(self, width: float, height: float) -> "BBox":
        return BBox(
            x0=max(0.0, min(self.x0, width)),
            y0=max(0.0, min(self.y0, height)),
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
        )


@dataclass
class ResolvedLabel:
    label_id: str
    kind: str
    page: int  # 1-indexed abs page from LaTeX
    bbox_pt: BBox  # in PDF points, origin top-left, PDF coords


def _pt_from_sp(sp: int) -> float:
    return sp / SP_PER_PT


def resolve_labels(
    manifest: InjectionManifest,
    anchors: dict[str, Anchor],
    boxes: dict[str, BoxDims],
    marks: dict[str, MarkDims],
    page_info: PageInfo,
    pdf_page_heights_pt: dict[int, float],
) -> list[ResolvedLabel]:
    """Convert anchors + manifest into per-label PDF-coordinate bboxes."""

    resolved: list[ResolvedLabel] = []
    for label in manifest.labels:
        if label.method == "wrap":
            if len(label.anchor_names) != 2:
                continue
            pre = anchors.get(label.anchor_names[0])
            post = anchors.get(label.anchor_names[1])
            dims = boxes.get(label.label_id)
            if pre is None or post is None or dims is None:
                continue
            if pre.abspage != post.abspage:
                continue
            page = pre.abspage
            page_h = pdf_page_heights_pt.get(page)
            if page_h is None:
                continue
            # pre anchor sits at the content's left edge on the baseline; post
            # anchor at the right edge on the baseline. baseline = pre.posy.
            # Content extends upward by height, downward by depth.
            baseline_pt = page_h - _pt_from_sp(pre.posy_sp)
            x0 = _pt_from_sp(pre.posx_sp)
            x1 = _pt_from_sp(post.posx_sp)
            if x1 < x0:
                x0, x1 = x1, x0
            # validate against natural width to catch cases where post landed
            # on a new line due to an unexpected line break inside the box.
            width_pt = dims.width_sp / SP_PER_PT
            if width_pt > 0 and abs((x1 - x0) - width_pt) / max(width_pt, 1.0) > 0.05:
                # trust width over post anchor x (post was on a different line)
                x1 = x0 + width_pt
            y_top = baseline_pt - dims.height_sp / SP_PER_PT
            y_bot = baseline_pt + dims.depth_sp / SP_PER_PT
            resolved.append(
                ResolvedLabel(
                    label_id=label.label_id,
                    kind=label.kind,
                    page=page,
                    bbox_pt=BBox(x0=x0, y0=y_top, x1=x1, y1=y_bot),
                )
            )
        elif label.method == "span":
            # ``anchor_names`` is a flat list of (top, bot) pairs tried in
            # order. Pick the first pair where both ends resolved -- they
            # don't have to be on the same page (longtable spans multiple
            # pages). When the top and bot pair disagrees on page, we emit
            # one bbox per intermediate page, clipped to the text block's
            # vertical range.
            if len(label.anchor_names) % 2 != 0:
                continue
            chosen_top: Anchor | None = None
            chosen_bot: Anchor | None = None
            chosen_top_name: str | None = None
            for i in range(0, len(label.anchor_names), 2):
                top_name = label.anchor_names[i]
                bot_name = label.anchor_names[i + 1]
                t = anchors.get(top_name)
                b = anchors.get(bot_name)
                if t is None or b is None:
                    continue
                chosen_top = t
                chosen_bot = b
                chosen_top_name = top_name
                break
            if chosen_top is None or chosen_bot is None:
                continue

            # X-extent: use \the\hsize reported at the mark's location when
            # available; otherwise fall back to column geometry.
            mark_info = marks.get(chosen_top_name) if chosen_top_name else None
            if mark_info is not None and mark_info.hsize_sp > 0:
                hsize_pt = mark_info.hsize_sp / SP_PER_PT
            else:
                hsize_pt = page_info.columnwidth_pt or page_info.textwidth_pt
            x0 = _pt_from_sp(chosen_top.posx_sp)
            x1 = x0 + hsize_pt

            top_page = chosen_top.abspage
            bot_page = chosen_bot.abspage
            if top_page > bot_page:
                top_page, bot_page = bot_page, top_page

            # For multi-page spans we clip to the textblock vertical bounds
            # so the emitted bbox doesn't bleed into headers / page numbers
            # / footnote areas. Single-page spans keep their tight top-mark /
            # bot-mark y's.
            text_top = page_info.text_top_pt
            text_bot = page_info.text_bot_pt

            for p in range(top_page, bot_page + 1):
                page_h = pdf_page_heights_pt.get(p)
                if page_h is None:
                    continue
                if p == chosen_top.abspage and p == chosen_bot.abspage:
                    y_top = page_h - _pt_from_sp(chosen_top.posy_sp)
                    y_bot = page_h - _pt_from_sp(chosen_bot.posy_sp)
                    y0, y1 = min(y_top, y_bot), max(y_top, y_bot)
                elif p == chosen_top.abspage:
                    y0 = page_h - _pt_from_sp(chosen_top.posy_sp)
                    y1 = min(text_bot, page_h)
                elif p == chosen_bot.abspage:
                    y0 = max(0.0, text_top)
                    y1 = page_h - _pt_from_sp(chosen_bot.posy_sp)
                else:
                    y0 = max(0.0, text_top)
                    y1 = min(text_bot, page_h)
                resolved.append(
                    ResolvedLabel(
                        label_id=label.label_id,
                        kind=label.kind,
                        page=p,
                        bbox_pt=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    )
                )
    return resolved


def union_span_with_bodies(
    labels: list[ResolvedLabel],
    manifest: InjectionManifest,
    anchors: dict[str, Anchor] | None = None,
    pdf_page_heights_pt: dict[int, float] | None = None,
) -> list[ResolvedLabel]:
    """Expand a ``*_cap`` box so it covers

    - every body box in the same float on the same page, and
    - the actual caption region reported by our ``\\@makecaption`` hook
      (``alx@cap@top@<cap_id>`` / ``alx@cap@bot@<cap_id>``).

    The ``\\@makecaption`` anchors are the only reliable way to learn the
    caption's true bottom: in-source ``\\alxmark{...-inner-bot}`` sits after
    whatever the author stuck between ``\\caption{…}`` and ``\\end{figure}``
    (``\\label``, ``\\vspace*{-…}``, etc), and a negative vspace slides the
    mark up inside the caption body.
    """

    body_by_float: dict[str, list[ResolvedLabel]] = {}
    cap_by_float: dict[str, ResolvedLabel] = {}
    float_of = {entry.label_id: entry.float_id for entry in manifest.labels}

    for lab in labels:
        fid = float_of.get(lab.label_id)
        if fid is None:
            continue
        if lab.kind.endswith("_cap"):
            cap_by_float[fid] = lab
        else:
            body_by_float.setdefault(fid, []).append(lab)

    for fid, cap in cap_by_float.items():
        c = cap.bbox_pt
        x0, y0, x1, y1 = c.x0, c.y0, c.x1, c.y1

        bodies_on_page = [b for b in body_by_float.get(fid, []) if b.page == cap.page]
        if bodies_on_page:
            x0 = min(x0, *(b.bbox_pt.x0 for b in bodies_on_page))
            x1 = max(x1, *(b.bbox_pt.x1 for b in bodies_on_page))
            y0 = min(y0, *(b.bbox_pt.y0 for b in bodies_on_page))
            y1 = max(y1, *(b.bbox_pt.y1 for b in bodies_on_page))

        # Extend using @makecaption anchors if they're on the same page.
        if anchors is not None and pdf_page_heights_pt is not None:
            top_a = anchors.get(f"alx@cap@top@{cap.label_id}")
            bot_a = anchors.get(f"alx@cap@bot@{cap.label_id}")
            for a in (top_a, bot_a):
                if a is None or a.abspage != cap.page:
                    continue
                page_h = pdf_page_heights_pt.get(cap.page)
                if page_h is None:
                    continue
                y_here = page_h - a.posy_sp / SP_PER_PT
                y0 = min(y0, y_here)
                y1 = max(y1, y_here)

        cap.bbox_pt = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    return labels


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_meta: list[dict] = []
    with fitz.open(str(pdf_path)) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img_path = out_dir / f"page_{idx:03d}.png"
            pix.save(str(img_path))
            pages_meta.append(
                {
                    "abspage": idx,
                    "image_path": str(img_path),
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "width_pt": page.rect.width,
                    "height_pt": page.rect.height,
                    "dpi": dpi,
                }
            )
    return pages_meta


def bbox_pt_to_px(bbox_pt: BBox, dpi: int) -> BBox:
    factor = dpi / 72.0
    return BBox(
        x0=bbox_pt.x0 * factor,
        y0=bbox_pt.y0 * factor,
        x1=bbox_pt.x1 * factor,
        y1=bbox_pt.y1 * factor,
    )


def page_heights_pt(pages_meta: list[dict]) -> dict[int, float]:
    return {p["abspage"]: p["height_pt"] for p in pages_meta}


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
