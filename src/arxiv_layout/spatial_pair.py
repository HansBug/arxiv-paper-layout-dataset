"""Shared helpers for deciding whether a paper's bbox layout is "clean
enough" for training, by checking spatial pairing between every
body label (fig / table / algorithm / listing) and its matching
caption label.

Three scripts depend on this — ``export_yolo.py`` (paper-level filter
during export), ``run_corpus_pipeline.py`` (per-paper broadcast
qualification), ``corpus_snapshot.py`` (Monitor B 3-column subset
table) — so the logic lives here rather than duplicated.

The two predicates below encode the two accepted training subsets:

- :func:`paper_passes_spatial_pairing` — default, N:1 containment.
  Accepts the common subfigure pattern: one ``fig_cap`` holding
  multiple ``fig`` bboxes is fine.
- :func:`paper_passes_strict_1to1` — stricter. Same spatial checks
  PLUS per-page count equality for each active pair.

Both predicates ignore pairs whose members aren't in
``active_classes``; they also require that at least one active pair
is non-empty *somewhere* in the paper (otherwise there's nothing to
learn).
"""

from __future__ import annotations

from typing import Iterable, Iterator


# Canonical class ordering — shared with ``scripts/export_yolo.py``.
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

# Named subsets used across the codebase for reporting and filtering.
# Keep the names short ("8", "6", "4") because they're keyed into dicts
# that get printed in dashboards and broadcast lines.
CLASS_SUBSETS: dict[str, tuple[str, ...]] = {
    "8": CLASSES,
    # 6-label drops the ``listing`` pair (NOT ``algorithm``). arxiv
    # papers almost never wrap code in a ``listing`` float, so
    # ``listing_cap`` is structurally always 0 and keeping ``listing``
    # in the subset would just reject every paper with a stray
    # ``lstlisting`` block. ``algorithm`` still matters for ML-flavour
    # papers where pseudocode is integral.
    "6": ("fig", "fig_cap", "table", "table_cap", "algorithm", "algorithm_cap"),
    "4": ("fig", "fig_cap", "table", "table_cap"),
}

# Body/caption pairs used for every spatial check. A subset export
# drops pairs whose members aren't both in the active set.
CAPTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("fig", "fig_cap"),
    ("table", "table_cap"),
    ("algorithm", "algorithm_cap"),
    ("listing", "listing_cap"),
)


def _body_mostly_inside_cap(
    body_xywh: Iterable[float],
    cap_xywh: Iterable[float],
    thresh: float,
) -> bool:
    """``intersection_area / body_area >= thresh``.

    A body bbox that sits cleanly inside its cap bbox — or overshoots
    it by a small amount — counts as contained. The asymmetry is
    deliberate: we never divide by cap_area, because a big cap holding
    a small body is *still* a valid pair (that's exactly the N:1
    subfigure case).
    """
    xa, ya, wa, ha = body_xywh
    xc, yc, wc, hc = cap_xywh
    ix0 = max(xa, xc)
    iy0 = max(ya, yc)
    ix1 = min(xa + wa, xc + wc)
    iy1 = min(ya + ha, yc + hc)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    body_area = max(1e-9, wa * ha)
    return (inter / body_area) >= thresh


def _pair_per_page_bboxes(
    annotations: dict,
    active_classes: tuple[str, ...],
) -> Iterator[tuple[str, str, list, list]]:
    """Yield ``(body_kind, cap_kind, bodies_xywh, caps_xywh)`` per page,
    per active pair. Pairs whose both members aren't in
    ``active_classes`` are skipped. Pages with neither body nor cap
    are skipped.
    """
    active = set(active_classes)
    active_pairs = [
        (b, c) for b, c in CAPTION_PAIRS if b in active and c in active
    ]
    if not active_pairs:
        return
    kind_by_cat = {
        c["id"]: c["name"] for c in annotations.get("categories", [])
    }
    per_page: dict[int, dict[str, list]] = {}
    for item in annotations.get("annotations", []):
        kind = kind_by_cat.get(item["category_id"])
        if kind is None:
            continue
        per_page.setdefault(item["image_id"], {}).setdefault(
            kind, []
        ).append(item["bbox"])
    for page_kinds in per_page.values():
        for body_kind, cap_kind in active_pairs:
            bodies = page_kinds.get(body_kind, [])
            caps = page_kinds.get(cap_kind, [])
            if not bodies and not caps:
                continue
            yield body_kind, cap_kind, bodies, caps


def paper_passes_spatial_pairing(
    annotations: dict,
    active_classes: tuple[str, ...],
    iou_thresh: float = 0.9,
) -> bool:
    """Relaxed spatial-pair filter (DEFAULT training-subset gate).

    For every active body/cap pair, on every page:
    - Every body bbox must be mostly contained in some cap bbox on
      that page. An orphan body (no parent cap) rejects the paper.
    - Every cap bbox must have at least one body mostly contained in
      it. An empty cap (no child body) rejects the paper.
    - At least one (body, cap) pair must be non-empty *somewhere*,
      otherwise there's nothing to learn.

    Unlike :func:`paper_passes_strict_1to1`, per-page counts do NOT
    have to match: one ``fig_cap`` enclosing multiple ``fig`` bboxes
    (the common subfigure pattern) is accepted.
    """
    any_pair_nonempty = False
    for body_kind, cap_kind, bodies, caps in _pair_per_page_bboxes(
        annotations, active_classes
    ):
        for bb in bodies:
            if not any(
                _body_mostly_inside_cap(bb, cb, iou_thresh) for cb in caps
            ):
                return False
        for cb in caps:
            if not any(
                _body_mostly_inside_cap(bb, cb, iou_thresh) for bb in bodies
            ):
                return False
        if bodies and caps:
            any_pair_nonempty = True
    return any_pair_nonempty


def paper_passes_strict_1to1(
    annotations: dict,
    active_classes: tuple[str, ...],
    iou_thresh: float = 0.9,
) -> bool:
    """Strict 1:1 + spatial validity — cleanest (smallest) subset.

    Same as :func:`paper_passes_spatial_pairing` plus: on every page,
    ``count(body) == count(cap)`` for each active pair.
    """
    any_pair_nonempty = False
    for body_kind, cap_kind, bodies, caps in _pair_per_page_bboxes(
        annotations, active_classes
    ):
        if len(bodies) != len(caps):
            return False
        for bb in bodies:
            if not any(
                _body_mostly_inside_cap(bb, cb, iou_thresh) for cb in caps
            ):
                return False
        for cb in caps:
            if not any(
                _body_mostly_inside_cap(bb, cb, iou_thresh) for bb in bodies
            ):
                return False
        any_pair_nonempty = True
    return any_pair_nonempty


def count_kinds(annotations: dict) -> dict[str, int]:
    """Per-class instance count for a COCO-style ``annotations.json``."""
    kind_by_cat = {
        c["id"]: c["name"] for c in annotations.get("categories", [])
    }
    out: dict[str, int] = {}
    for item in annotations.get("annotations", []):
        name = kind_by_cat.get(item["category_id"])
        if name is not None:
            out[name] = out.get(name, 0) + 1
    return out


def pages_label_stats(
    annotations: dict,
    active_classes: tuple[str, ...],
) -> tuple[int, int]:
    """Return ``(pages_total, pages_without_active_labels)`` for a paper.

    ``pages_without_active_labels`` counts every image in ``images``
    that has zero annotations whose class is in ``active_classes`` —
    including images with annotations of *other* classes (those are
    still "background" from the active subset's POV). This is the
    correct count for YOLO negative samples under a restricted class
    set.
    """
    active = set(active_classes)
    all_image_ids = {img["id"] for img in annotations.get("images", [])}
    kind_by_cat = {
        c["id"]: c["name"] for c in annotations.get("categories", [])
    }
    active_image_ids: set[int] = set()
    for item in annotations.get("annotations", []):
        kind = kind_by_cat.get(item["category_id"])
        if kind in active:
            active_image_ids.add(item["image_id"])
    pages_total = len(all_image_ids)
    pages_without = pages_total - len(active_image_ids & all_image_ids)
    return pages_total, pages_without


def spatial_pair_qualification(
    annotations: dict,
    iou_thresh: float = 0.9,
) -> dict[str, bool]:
    """Check spatial-pair eligibility against the three canonical
    subsets ("8", "6", "4"). Returns a dict mapping subset-name to
    True/False so callers can fold it into short tags like
    ``sp=[8:Y 6:Y 4:N]``.
    """
    return {
        name: paper_passes_spatial_pairing(
            annotations, classes, iou_thresh
        )
        for name, classes in CLASS_SUBSETS.items()
    }
