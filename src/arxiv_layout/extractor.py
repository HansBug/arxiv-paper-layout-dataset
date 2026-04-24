"""Read zref anchor coordinates + page geometry.

Page geometry comes from a single ``ARXIVLAYOUT-PAGEINFO`` line we typeout
into the .log. Anchor coordinates come straight from the compiled ``.aux``
file -- parsing the raw ``\\zref@newlabel`` lines is more reliable than
typeouting ``\\zposx`` because the typeout happens at ``\\AtEndDocument``,
which is *before* the aux file for the *current* run has been closed and
re-read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SP_PER_PT = 65536


@dataclass
class Anchor:
    name: str
    posx_sp: int
    posy_sp: int
    abspage: int


@dataclass
class BoxDims:
    box_id: str
    width_sp: int
    height_sp: int
    depth_sp: int
    hsize_sp: int = 0


@dataclass
class MarkDims:
    mark_id: str
    hsize_sp: int
    linewidth_sp: int
    columnwidth_sp: int


@dataclass
class PageInfo:
    paperwidth_sp: int
    paperheight_sp: int
    textwidth_sp: int
    textheight_sp: int
    oddsidemargin_sp: int
    topmargin_sp: int
    headheight_sp: int
    headsep_sp: int
    columnwidth_sp: int = 0
    columnsep_sp: int = 0

    @property
    def paperwidth_pt(self) -> float:
        return self.paperwidth_sp / SP_PER_PT

    @property
    def paperheight_pt(self) -> float:
        return self.paperheight_sp / SP_PER_PT

    @property
    def text_left_pt(self) -> float:
        # LaTeX: x of text block = 1in + \hoffset + \oddsidemargin.
        # \hoffset default 0. 1in = 72.27pt.
        return 72.27 + self.oddsidemargin_sp / SP_PER_PT

    @property
    def text_right_pt(self) -> float:
        return self.text_left_pt + self.textwidth_sp / SP_PER_PT

    @property
    def textwidth_pt(self) -> float:
        return self.textwidth_sp / SP_PER_PT

    @property
    def columnwidth_pt(self) -> float:
        if self.columnwidth_sp > 0:
            return self.columnwidth_sp / SP_PER_PT
        return self.textwidth_pt

    @property
    def columnsep_pt(self) -> float:
        return self.columnsep_sp / SP_PER_PT

    @property
    def num_columns_est(self) -> float:
        """Approximate column count derived from textwidth / columnwidth.

        A floating point so callers can tell "almost one column" (1.0) from
        "one-and-a-bit" (e.g., multicol with mixed widths). Falls back to 1.0
        when geometry is missing.
        """
        if self.columnwidth_sp <= 0 or self.textwidth_sp <= 0:
            return 1.0
        return self.textwidth_sp / self.columnwidth_sp


_AUX_ZREF_RE = re.compile(
    r"\\zref@newlabel\{(?P<name>[^{}]+)\}\{(?P<body>[^{}]*"
    r"(?:\{[^{}]*\}[^{}]*)*)\}"
)
_POSX_RE = re.compile(r"\\posx\{(-?\d+)\}")
_POSY_RE = re.compile(r"\\posy\{(-?\d+)\}")
_ABSPAGE_RE = re.compile(r"\\abspage\{(-?\d+)\}")

_PAGEINFO_RE = re.compile(r"ARXIVLAYOUT-PAGEINFO\s+(?P<rest>.+)")
_BOX_RE = re.compile(
    r"ARXIVLAYOUT-BOX\s+id=(?P<id>\S+)\s+width=(?P<w>-?[\d.]+pt)\s+height=(?P<h>-?[\d.]+pt)\s+depth=(?P<d>-?[\d.]+pt)(?:\s+hsize=(?P<hs>-?[\d.]+pt))?"
)
_MARK_RE = re.compile(
    r"ARXIVLAYOUT-MARK\s+id=(?P<id>\S+)\s+hsize=(?P<hs>-?[\d.]+pt)\s+linewidth=(?P<lw>-?[\d.]+pt)\s+columnwidth=(?P<cw>-?[\d.]+pt)"
)
_DIM_RE = re.compile(r"(?P<name>\w+)=(?P<value>-?[\d.]+)(?P<unit>pt|sp)?")


def _parse_dim(raw: str) -> int:
    """Parse ``\\the\\paperwidth``-style output like ``614.295pt``. Returns sp."""
    m = re.match(r"\s*(-?[\d.]+)\s*pt\s*$", raw)
    if m:
        return int(round(float(m.group(1)) * SP_PER_PT))
    m = re.match(r"\s*(-?[\d.]+)\s*sp\s*$", raw)
    if m:
        return int(round(float(m.group(1))))
    return 0


def parse_aux(aux_path: Path) -> dict[str, Anchor]:
    text = aux_path.read_text(encoding="utf-8", errors="ignore")
    anchors: dict[str, Anchor] = {}
    for match in _AUX_ZREF_RE.finditer(text):
        name = match.group("name")
        body = match.group("body")
        posx_match = _POSX_RE.search(body)
        posy_match = _POSY_RE.search(body)
        abspage_match = _ABSPAGE_RE.search(body)
        if not posx_match or not posy_match:
            continue
        anchors[name] = Anchor(
            name=name,
            posx_sp=int(posx_match.group(1)),
            posy_sp=int(posy_match.group(1)),
            abspage=int(abspage_match.group(1)) if abspage_match else 0,
        )
    return anchors


def parse_log_for_pageinfo(log_path: Path) -> PageInfo | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    page_match = _PAGEINFO_RE.search(text)
    if not page_match:
        return None
    fields: dict[str, int] = {}
    rest = page_match.group("rest")
    for token in rest.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = _parse_dim(value)
    return PageInfo(
        paperwidth_sp=fields.get("paperwidth", 0),
        paperheight_sp=fields.get("paperheight", 0),
        textwidth_sp=fields.get("textwidth", 0),
        textheight_sp=fields.get("textheight", 0),
        oddsidemargin_sp=fields.get("oddsidemargin", 0),
        topmargin_sp=fields.get("topmargin", 0),
        headheight_sp=fields.get("headheight", 0),
        headsep_sp=fields.get("headsep", 0),
        columnwidth_sp=fields.get("columnwidth", 0),
        columnsep_sp=fields.get("columnsep", 0),
    )


def parse_log_for_boxes(log_path: Path) -> dict[str, BoxDims]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, BoxDims] = {}
    for match in _BOX_RE.finditer(text):
        out[match.group("id")] = BoxDims(
            box_id=match.group("id"),
            width_sp=_parse_dim(match.group("w")),
            height_sp=_parse_dim(match.group("h")),
            depth_sp=_parse_dim(match.group("d")),
            hsize_sp=_parse_dim(match.group("hs")) if match.group("hs") else 0,
        )
    return out


def parse_log_for_marks(log_path: Path) -> dict[str, MarkDims]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, MarkDims] = {}
    for match in _MARK_RE.finditer(text):
        out[match.group("id")] = MarkDims(
            mark_id=match.group("id"),
            hsize_sp=_parse_dim(match.group("hs")),
            linewidth_sp=_parse_dim(match.group("lw")),
            columnwidth_sp=_parse_dim(match.group("cw")),
        )
    return out


def parse_compile_outputs(aux_path: Path, log_path: Path) -> tuple[dict[str, Anchor], PageInfo | None, dict[str, BoxDims], dict[str, MarkDims]]:
    anchors = parse_aux(aux_path)
    page_info = parse_log_for_pageinfo(log_path)
    boxes = parse_log_for_boxes(log_path)
    marks = parse_log_for_marks(log_path)
    return anchors, page_info, boxes, marks
