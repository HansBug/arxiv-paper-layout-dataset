"""End-to-end pipeline for a single arxiv paper source tree."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _tex_env() -> dict[str, str]:
    """TeX reads max_print_line from the environment; crank it up so the log
    doesn't wrap at 79 chars and truncate our ARXIVLAYOUT-* lines."""
    env = os.environ.copy()
    env.setdefault("max_print_line", "65536")
    env.setdefault("error_line", "254")
    env.setdefault("half_error_line", "238")
    return env

from .extractor import parse_compile_outputs
from .injector import MultiFileInjector
from .render import (
    bbox_pt_to_px,
    page_heights_pt,
    render_pdf_pages,
    resolve_labels,
    union_span_with_bodies,
)
from .visualize import draw_labels_on_image


ENGINE_FLAGS = {
    "pdflatex": "-pdf",
    "xelatex": "-xelatex",
    "lualatex": "-lualatex",
}


@dataclass
class PaperOutcome:
    paper_id: str
    ok: bool
    reason: str = ""
    pages: int = 0
    labels: int = 0
    labels_by_kind: dict[str, int] = field(default_factory=dict)
    workspace: str | None = None


def _score_tex_file(path: Path, src_root: Path) -> int:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return -10**9
    if "\\documentclass" not in content or "\\begin{document}" not in content:
        return -10**9
    name = path.name.lower()
    rel = path.relative_to(src_root).as_posix().lower()
    bonuses = {"main.tex": 1000, "paper.tex": 900, "ms.tex": 800, "manuscript.tex": 780}
    score = path.stat().st_size + bonuses.get(name, 0)
    score -= len(path.relative_to(src_root).parts) * 25
    for bad in ("template", "sample", "supplement", "appendix"):
        if bad in rel:
            score -= 400
    return score


def find_main_tex(src_root: Path) -> Path | None:
    scored: list[tuple[int, Path]] = []
    for tex in src_root.rglob("*.tex"):
        if not tex.is_file():
            continue
        s = _score_tex_file(tex, src_root)
        if s > -10**9:
            scored.append((s, tex))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def pick_engine(tex: Path, original_engine: str | None) -> list[str]:
    content = tex.read_text(encoding="utf-8", errors="ignore").lower()
    if any(tok in content for tok in ("fontspec", "xecjk", "xetex", "\\setmainfont", "ctex")):
        seq = ["xelatex", "lualatex", "pdflatex"]
    elif "luatex" in content or "luacode" in content:
        seq = ["lualatex", "xelatex", "pdflatex"]
    else:
        seq = ["pdflatex", "xelatex", "lualatex"]
    if original_engine in seq:
        seq = [original_engine] + [e for e in seq if e != original_engine]
    return seq


def run_latexmk(tex_name: str, cwd: Path, engine: str, timeout: int = 900) -> tuple[bool, str]:
    cmd = [
        "latexmk",
        ENGINE_FLAGS[engine],
        "-interaction=nonstopmode",
        "-file-line-error",
        "-halt-on-error",
        tex_name,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
            timeout=timeout,
            check=False,
            env=_tex_env(),
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    pdf = cwd / (Path(tex_name).stem + ".pdf")
    return (proc.returncode == 0 and pdf.exists()), (proc.stdout or "")


def run_direct(tex_name: str, cwd: Path, engine: str, runs: int = 3, timeout: int = 900) -> tuple[bool, str]:
    cmd = [
        engine,
        "-interaction=nonstopmode",
        "-file-line-error",
        tex_name,
    ]
    stdout_total = []
    for _ in range(runs):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="ignore",
                timeout=timeout,
                check=False,
                env=_tex_env(),
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        stdout_total.append(proc.stdout or "")
        if proc.returncode != 0:
            break
    pdf = cwd / (Path(tex_name).stem + ".pdf")
    return pdf.exists(), "\n".join(stdout_total)


def compile_paper(work_root: Path, tex_name: str, original_engine: str | None) -> tuple[bool, str, str | None]:
    engines = pick_engine(work_root / tex_name, original_engine)
    for engine in engines:
        ok, out = run_latexmk(tex_name, work_root, engine)
        if ok:
            return True, out, engine
        # fallback: direct engine (needed if .bbl is prebuilt & latexmk can't re-bibtex)
        ok2, out2 = run_direct(tex_name, work_root, engine)
        if ok2:
            return True, out2, engine
    return False, out, None


def copy_source_tree(src_root: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src_root, dest)
    # wipe latex build side-effects
    for pattern in ("*.aux", "*.bbl", "*.blg", "*.log", "*.out", "*.fls", "*.fdb_latexmk"):
        for p in dest.rglob(pattern):
            try:
                p.unlink()
            except OSError:
                pass


def process_paper(paper_src: Path, work_root: Path, dpi: int = 200, original_engine: str | None = None) -> PaperOutcome:
    paper_id = paper_src.name
    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    workspace = work_root / paper_id
    src_dest = workspace / "src"
    pages_dir = workspace / "pages"
    qc_dir = workspace / "qc"
    out_dir = workspace / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        copy_source_tree(paper_src / "src", src_dest)
    except FileNotFoundError:
        return PaperOutcome(paper_id=paper_id, ok=False, reason="missing src/")

    main_tex = find_main_tex(src_dest)
    if main_tex is None:
        return PaperOutcome(paper_id=paper_id, ok=False, reason="no main tex found")

    # Inject (across all .tex files -- fragments included via \input / \include)
    multi = MultiFileInjector()
    multi.inject_tree(src_dest)
    multi.manifest.save(workspace / "manifest.json")

    # Compile
    ok, log_output, engine = compile_paper(src_dest, main_tex.name, original_engine)
    log_path = main_tex.with_suffix(".log")
    # if latexmk/engine didn't write a .log (eg early crash), dump captured stdout
    if not log_path.exists():
        log_path.write_text(log_output or "", encoding="utf-8")
    else:
        # append engine output so ARXIVLAYOUT lines from stdout are available
        log_path.write_text(
            log_path.read_text(encoding="utf-8", errors="ignore") + "\n" + (log_output or ""),
            encoding="utf-8",
        )

    if not ok:
        return PaperOutcome(paper_id=paper_id, ok=False, reason=f"compile failed ({engine})", workspace=str(workspace))

    # Parse anchors from .aux and page geometry / box / mark dims from .log
    aux_path = main_tex.with_suffix(".aux")
    anchors, page_info, boxes, marks = parse_compile_outputs(aux_path, log_path)
    if page_info is None:
        return PaperOutcome(paper_id=paper_id, ok=False, reason="no PAGEINFO in log", workspace=str(workspace))
    if not anchors:
        return PaperOutcome(paper_id=paper_id, ok=False, reason="no zref anchors in aux", workspace=str(workspace))

    # Render PDF pages
    pdf_path = main_tex.with_suffix(".pdf")
    pages_meta = render_pdf_pages(pdf_path, pages_dir, dpi=dpi)

    # Resolve labels
    labels = resolve_labels(
        multi.manifest,
        anchors,
        boxes,
        marks,
        page_info,
        page_heights_pt(pages_meta),
    )
    labels = union_span_with_bodies(labels, multi.manifest)

    # Clip to page bounds (in pt) before writing
    page_rects = {p["abspage"]: (p["width_pt"], p["height_pt"]) for p in pages_meta}
    cleaned = []
    kind_counts: dict[str, int] = {}
    for lab in labels:
        if lab.page not in page_rects:
            continue
        w_pt, h_pt = page_rects[lab.page]
        lab.bbox_pt = lab.bbox_pt.clipped(w_pt, h_pt)
        if lab.bbox_pt.area() < 1.0:
            continue
        cleaned.append(lab)
        kind_counts[lab.kind] = kind_counts.get(lab.kind, 0) + 1

    # Emit per-page COCO-style JSON (paper-local)
    coco = {
        "info": {"paper_id": paper_id, "dpi": dpi, "engine": engine},
        "categories": [
            {"id": idx, "name": kind}
            for idx, kind in enumerate(sorted({lab.kind for lab in cleaned}))
        ],
        "images": [],
        "annotations": [],
    }
    kind_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    for p in pages_meta:
        coco["images"].append(
            {
                "id": p["abspage"],
                "file_name": Path(p["image_path"]).name,
                "width": p["width_px"],
                "height": p["height_px"],
            }
        )
    next_ann_id = 1
    for lab in cleaned:
        b_px = bbox_pt_to_px(lab.bbox_pt, dpi)
        w = max(0.0, b_px.x1 - b_px.x0)
        h = max(0.0, b_px.y1 - b_px.y0)
        coco["annotations"].append(
            {
                "id": next_ann_id,
                "image_id": lab.page,
                "category_id": kind_to_id[lab.kind],
                "bbox": [b_px.x0, b_px.y0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "label_id": lab.label_id,
            }
        )
        next_ann_id += 1

    (out_dir / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")

    # Generate QC visualizations
    by_page: dict[int, list] = {}
    for lab in cleaned:
        by_page.setdefault(lab.page, []).append(lab)
    for p in pages_meta:
        lbls = by_page.get(p["abspage"], [])
        draw_labels_on_image(
            Path(p["image_path"]),
            lbls,
            dpi=dpi,
            out_path=qc_dir / f"page_{p['abspage']:03d}.png",
        )

    return PaperOutcome(
        paper_id=paper_id,
        ok=True,
        pages=len(pages_meta),
        labels=len(cleaned),
        labels_by_kind=kind_counts,
        workspace=str(workspace),
    )
