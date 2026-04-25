#!/usr/bin/env python3
"""Build + send a Monitor-B Feishu card from the live corpus state.

Reads ``runs/corpus/state.json``, recomputes the 3-column subset
table the same way ``corpus_snapshot.py`` does, derives a coarse
health verdict (monotonicity / archive coverage / fail clustering /
recent OK growth) plus a recommendation and a search-direction hint,
and POSTs everything as a Feishu schema-2.0 interactive card via
``scripts/notify_feishu.py``.

Usage::

    python3 scripts/feishu_corpus_b.py
    python3 scripts/feishu_corpus_b.py --root runs/corpus --dry-run

The dry-run mode prints the assembled card JSON to stdout without
hitting the webhook — useful when iterating on the rendering.

Designed to be called manually after each Monitor-B ``SNAPSHOT``
notification arrives. Doesn't itself poll; the schedule is whatever
the operator (or the conversation agent) decides.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arxiv_layout.corpus import ARXIV_ARCHIVES  # noqa: E402
from arxiv_layout.spatial_pair import (  # noqa: E402
    CLASS_SUBSETS,
    CLASSES,
    count_kinds,
    pages_label_stats,
    paper_passes_spatial_pairing,
)


def _fmt(n: int | str) -> str:
    if isinstance(n, int):
        return f"{n:,}"
    return str(n)


def _compute_subsets(state: dict, root: Path) -> dict[str, dict]:
    """Same logic as corpus_snapshot._compute_subsets but inlined so
    this script has no soft dependency."""
    per_subset = {
        name: {
            "papers_pass": 0,
            "pages_total": 0,
            "pages_without_labels": 0,
            "kinds": {c: 0 for c in CLASSES},
        }
        for name in CLASS_SUBSETS
    }
    for rec in state.get("papers", {}).values():
        if rec.get("status") != "ok":
            continue
        ws = rec.get("workspace") or ""
        if not ws:
            continue
        ann_path = Path(ws) / "dataset" / "annotations.json"
        if not ann_path.is_file():
            continue
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        kh = count_kinds(ann)
        for name, classes in CLASS_SUBSETS.items():
            if not paper_passes_spatial_pairing(ann, classes):
                continue
            bucket = per_subset[name]
            bucket["papers_pass"] += 1
            pt, pw = pages_label_stats(ann, classes)
            bucket["pages_total"] += pt
            bucket["pages_without_labels"] += pw
            for cls, n in kh.items():
                if cls in bucket["kinds"]:
                    bucket["kinds"][cls] += n
    return per_subset


def _table_element(subsets: dict[str, dict]) -> dict:
    """Convert the SUBSETS dict into a Feishu v2 ``table`` element."""
    names = list(CLASS_SUBSETS.keys())
    rows: list[dict[str, str]] = []
    rows.append(
        {
            "metric": "papers_pass",
            **{f"c{n}": _fmt(subsets[n]["papers_pass"]) for n in names},
        }
    )
    rows.append(
        {
            "metric": "pages_total",
            **{f"c{n}": _fmt(subsets[n]["pages_total"]) for n in names},
        }
    )
    rows.append(
        {
            "metric": "pages_no_label",
            **{f"c{n}": _fmt(subsets[n]["pages_without_labels"]) for n in names},
        }
    )
    for cls in CLASSES:
        cells = {"metric": cls}
        for n in names:
            if cls in CLASS_SUBSETS[n]:
                cells[f"c{n}"] = _fmt(subsets[n]["kinds"][cls])
            else:
                cells[f"c{n}"] = "—"
        rows.append(cells)
    return {
        "tag": "table",
        "page_size": len(rows),
        "row_height": "low",
        "freeze_first_column": True,
        "header_style": {
            "text_align": "center",
            "text_size": "normal",
            "background_style": "grey",
            "text_color": "default",
            "bold": True,
        },
        "columns": [
            {"name": "metric", "display_name": "指标 / kind",
             "data_type": "text", "horizontal_align": "left", "width": "auto"},
            *[
                {"name": f"c{n}", "display_name": f"{n}-label",
                 "data_type": "text", "horizontal_align": "right", "width": "auto"}
                for n in names
            ],
        ],
        "rows": rows,
    }


def _health_lines(state: dict, subsets: dict[str, dict]) -> list[str]:
    """Coarse health signals — each line either ✅ / ⚠️ / ❌ + reason."""
    lines: list[str] = []
    pp = [subsets[n]["papers_pass"] for n in ("8", "6", "4")]
    if pp[0] <= pp[1] <= pp[2]:
        lines.append(
            f"✅ 单调性 `8 ⊆ 6 ⊆ 4` 成立（{pp[0]} ≤ {pp[1]} ≤ {pp[2]}）"
        )
    else:
        lines.append(
            f"❌ 单调性破裂：`8 ⊆ 6 ⊆ 4` 应单调不减，实测 {pp}"
        )

    archives_hist = state.get("stats", {}).get("archive_histogram", {}) or {}
    covered = sum(1 for a in ARXIV_ARCHIVES if a in archives_hist)
    if covered == len(ARXIV_ARCHIVES):
        if archives_hist:
            top_a = max(archives_hist.items(), key=lambda kv: kv[1])
            bot_a = min(archives_hist.items(), key=lambda kv: kv[1])
            spread = top_a[1] / max(1, bot_a[1])
            lines.append(
                f"✅ archive {covered}/{len(ARXIV_ARCHIVES)}，"
                f"top `{top_a[0]}={top_a[1]}` · bot `{bot_a[0]}={bot_a[1]}`，"
                f"spread **{spread:.2f} : 1**"
            )
    else:
        untouched = [a for a in ARXIV_ARCHIVES if a not in archives_hist]
        lines.append(
            f"⚠️ archive 仅 {covered}/{len(ARXIV_ARCHIVES)}，未触碰：{untouched}"
        )

    fails_by_cat: dict[str, int] = {}
    for paper in state.get("papers", {}).values():
        if paper.get("status") != "failed":
            continue
        cat = paper.get("primary_category") or "?"
        fails_by_cat[cat] = fails_by_cat.get(cat, 0) + 1
    top_fail_cats = sorted(fails_by_cat.items(), key=lambda kv: -kv[1])[:3]
    stats = state.get("stats", {}) or {}
    pf = stats.get("papers_failed", 0)
    pt = stats.get("papers_ok", 0) + pf
    fr = (pf / pt * 100.0) if pt else 0.0
    fail_str = " · ".join(f"`{k}={v}`" for k, v in top_fail_cats)
    if fr <= 30:
        lines.append(
            f"✅ fail rate {fr:.1f}%（健康阈值 ≤ 30%），失败 top: {fail_str}"
        )
    elif fr <= 40:
        lines.append(
            f"⚠️ fail rate {fr:.1f}% 偏高，失败 top: {fail_str}"
        )
    else:
        lines.append(
            f"❌ fail rate {fr:.1f}% 异常，失败 top: {fail_str}（请检查 driver / arxiv 连通性）"
        )

    return lines


# label-bias policy — per user instruction 2026-04-25:
# label balance (table & algorithm catching up to figure) outranks
# archive balance. Map "which kind is starving" -> "which archive
# tends to produce it" so we can suggest a force_next_archive that
# pushes the bottom kind, even if it skews archive coverage.
LABEL_RICH_ARCHIVES: dict[str, list[str]] = {
    "table":     ["stat", "econ", "q-fin", "q-bio", "eess"],
    "algorithm": ["cs"],  # ML/DS subcategories of cs are pseudocode-heavy
    "listing":   ["cs"],
}


def _starving_kinds(stats: dict) -> list[tuple[str, int, float]]:
    """Return (kind, count, ratio_to_figure) for body kinds that are
    far smaller than figure. Sorted scarcest first."""
    kh = stats.get("labels_by_kind") or {}
    fig = kh.get("figure", 0)
    if fig <= 0:
        return []
    out: list[tuple[str, int, float]] = []
    for kind in ("algorithm", "table", "listing"):
        n = kh.get(kind, 0)
        ratio = fig / max(1, n)
        # Only flag if the body kind trails figure by more than 4x.
        # 4x is a coarse "this is now an asymmetric class" threshold.
        if ratio > 4:
            out.append((kind, n, ratio))
    out.sort(key=lambda kv: -kv[2])  # scarcest (highest ratio) first
    return out


def _suggest_lines(state: dict, subsets: dict[str, dict]) -> list[str]:
    out: list[str] = []
    ctl: dict = {}
    try:
        ctl = json.loads(
            (Path(state.get("__root__", ".")) / "control.json").read_text(
                encoding="utf-8"
            )
        )
    except Exception:  # noqa: BLE001
        pass
    skip = sorted(ctl.get("skip_primary_cats") or [])
    force = ctl.get("force_next_archive")

    starving = _starving_kinds(state.get("stats") or {})
    if starving:
        kind, n, ratio = starving[0]
        archives_for_kind = LABEL_RICH_ARCHIVES.get(kind, [])
        out.append(
            f"⚖️ **label balance 优先**：`{kind}={n}` 仅为 figure 的 "
            f"1/{ratio:.0f}。建议 `force_next_archive` 走 "
            f"{archives_for_kind} 中的一个把 `{kind}` 拉起来。"
            f"（用户 2026-04-25 指示：label balance > archive balance）"
        )

    if force:
        out.append(
            f"`force_next_archive='{force}'` 当前生效；如果该 archive 已经"
            f"为目标 kind 贡献 ≥10 篇，可以切换到 starving-kind 列表里下一个 archive。"
        )
    elif not starving:
        # Archive coverage already 20/20 and all kinds reasonably balanced.
        archives_hist = state.get("stats", {}).get("archive_histogram", {}) or {}
        untouched = [a for a in ARXIV_ARCHIVES if a not in archives_hist]
        if untouched:
            out.append(
                f"⚠️ 未触碰 archive：{untouched}，建议下一轮 "
                f"`force_next_archive` 设为其中一个。"
            )
        else:
            out.append("无需干预。")

    if skip:
        out.append(f"`skip_primary_cats={skip}` 维持。")
    return out


def _direction_lines(state: dict) -> list[str]:
    """Search-direction hint with label-bias override."""
    stats = state.get("stats") or {}
    archives_hist = stats.get("archive_histogram") or {}
    sorted_a = sorted(archives_hist.items(), key=lambda kv: kv[1])
    bottom3_str = (
        " · ".join(f"`{a}={n}`" for a, n in sorted_a[:3]) if sorted_a else "(?)"
    )

    starving = _starving_kinds(stats)
    if starving:
        kind, n, ratio = starving[0]
        target = LABEL_RICH_ARCHIVES.get(kind, ["?"])[0]
        return [
            f"⚖️ **label-bias active**：`{kind}` 仅 {n}（figure 的 1/{ratio:.0f}）→ "
            f"主推 archive `{target}`。",
            f"BalancedQueryStrategy 默认 least-covered 是：{bottom3_str}，"
            f"但被 label-bias 覆盖；archive 均衡性此阶段适度让位（用户授权）。",
        ]
    return [
        f"BalancedQueryStrategy 自动选 least-covered。当前最少：{bottom3_str}。"
    ]


def build_elements(state: dict, root: Path) -> tuple[list[dict], dict[str, Any]]:
    stats = state.get("stats", {}) or {}
    subsets = _compute_subsets(state, root)

    pf = stats.get("papers_failed", 0)
    po = stats.get("papers_ok", 0)
    pt = po + pf
    pages_total = stats.get("pages_total", 0)
    pages_with = stats.get("pages_with_labels", 0)
    pages_neg = max(0, pages_total - pages_with)
    neg_pct = (pages_neg / pages_total * 100.0) if pages_total else 0.0
    fr = (pf / pt * 100.0) if pt else 0.0
    total_labels = sum((stats.get("labels_by_kind") or {}).values())

    archives_hist = stats.get("archive_histogram", {}) or {}
    covered = sum(1 for a in ARXIV_ARCHIVES if a in archives_hist)

    snapshot_md = (
        "**📊 当前状况 (SNAPSHOT)**\n\n"
        f"- **papers**：{_fmt(pt)} total · **{_fmt(po)} ok** · "
        f"{_fmt(pf)} fail（fail rate **{fr:.1f}%**）\n"
        f"- **pages**：{_fmt(pages_total)} total，"
        f"{_fmt(pages_with)} 有标注，**{_fmt(pages_neg)} 纯负样本**"
        f"（{neg_pct:.1f}%）\n"
        f"- **total_labels**：{_fmt(total_labels)}\n"
        f"- **archive coverage**：**{covered}/{len(ARXIV_ARCHIVES)}** ✅"
    )

    state["__root__"] = str(root)  # for _suggest_lines control.json lookup

    elements = [
        {"tag": "markdown", "content": snapshot_md},
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": "**🧮 三档子集（spatial-pair N:1 with containment, IoU≥0.9）**",
        },
        _table_element(subsets),
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": "**🩺 健康度**\n\n"
            + "\n".join(f"- {l}" for l in _health_lines(state, subsets)),
        },
        {
            "tag": "markdown",
            "content": "**💡 建议**\n\n"
            + "\n".join(f"- {l}" for l in _suggest_lines(state, subsets)),
        },
        {
            "tag": "markdown",
            "content": "**🔭 后续搜索倾向**\n\n"
            + "\n".join(f"- {l}" for l in _direction_lines(state)),
        },
    ]

    summary = {
        "papers_total": pt,
        "papers_ok": po,
        "papers_failed": pf,
        "subsets_papers_pass": [subsets[n]["papers_pass"] for n in ("8", "6", "4")],
    }
    return elements, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/corpus"))
    parser.add_argument(
        "--title",
        default=None,
        help="override card title; default auto-generated from state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="emit assembled card JSON instead of POSTing",
    )
    args = parser.parse_args()

    state = json.loads((args.root / "state.json").read_text(encoding="utf-8"))
    elements, summary = build_elements(state, args.root)

    title = args.title or (
        f"corpus 播报 @ {summary['papers_total']} papers / "
        f"{summary['papers_ok']} OK"
    )

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "notify_feishu.py"),
        "--kind", "B",
        "--title", title,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(
        cmd, input=json.dumps(elements, ensure_ascii=False), text=True
    )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
