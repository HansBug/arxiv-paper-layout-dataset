# AGENTS.md

Guidance for LLM agents (Claude, Codex, Cursor, etc.) hacking on this repo.

## What the pipeline does

Given an arXiv LaTeX source tree, the pipeline:

1. Injects `zref-savepos` / TikZ anchors around every `figure`, `table`,
   `algorithm`, `listing` (and the atomic `\includegraphics` / `tabular` /
   `tikzpicture` / pseudo-code body).
2. Compiles with a TeX Live toolchain (see
   `/home/zhangshaoang/texlive-arxiv-validation/README.md` on the reference
   dev host).
3. Reads anchor coordinates from the compiled `.aux`, page geometry and
   box dimensions from the `.log`.
4. Projects everything into PyMuPDF page-pixel space and emits COCO-style
   annotations + QC overlays per page.

## Files that matter

```
src/arxiv_layout/
  injector.py     # regex-based LaTeX rewriter. Macros \alxwrap, \alxmark.
  extractor.py    # .aux + .log parsing. Anchor / BoxDims / MarkDims / PageInfo.
  render.py       # sp->pt->pixel projection; cap<->body union; COCO emission.
  visualize.py    # PIL-drawn bboxes on top of PyMuPDF page rasters.
  pipeline.py     # end-to-end per-paper driver (copy src, inject, compile, render).
scripts/
  build_dataset.py  # CLI entrypoint.
tests/              # golden-output regression tests (see below).
```

## Ground rules when editing

- **Prefer first-hand TeX data over Python heuristics.** If you need a new
  dimension, emit it from the injected macro via `\typeout{ARXIVLAYOUT-...}`
  or encode it in `\zref@newlabel`. Don't add column-count or layout guessing
  logic in Python.
- **`\alxmark` must not call `\leavevmode`.** That forces an empty first line
  in vertical mode and pushes the anchor below the vbox by `\topskip`, which
  shows up as a ~10pt slack at the top of every `*_cap` box.
- **`\alxwrap` uses `\setbox`+`\copy` not TikZ overlay.** TikZ's
  `remember picture, overlay` trick does NOT reliably capture position via
  `\zsavepos` (the overlay bypasses the regular whatsit shipout path).
- **X extent is always `\the\hsize` at mark time.** That's the exact
  horizontal budget TeX gave the float. Do not approximate from
  `\columnwidth` in Python.
- **Don't rely on `\AtEndDocument`.** Some journal classes (A&A's `aa.cls`)
  strip end-document hooks. Emit `ARXIVLAYOUT-PAGEINFO` from
  `\AtBeginDocument` instead — paper geometry is fixed by then.
- **Use `.aux` (not log typeouts) for anchor coordinates.** `\zposx`/
  `\zposy` only return values after the `.aux` has been written AND read
  back in the *next* run; extracting `\zref@newlabel{…}{\posx{…}\posy{…}
  \abspage{…}}` directly from the `.aux` is cycle-free.
- **Convert sp -> PDF pt with the 72.27/72 factor.** TeX uses 72.27 pt/in
  internally, the PDF user-space unit is 72 pt/in. Dividing by 65536 alone
  gives you *TeX* pt; flipping those against a PyMuPDF-derived page height
  (which is in PDF pt) drifts every y-coord by ~0.4% of its value — visible
  at the bottom of a letter page as a 1-3 pt cut-off of bottom rules.
  `extractor.SP_PER_PT` already embeds the ratio; use it and you're fine.
- **Always set `\alx@current@alg@id`, even for un-instrumented floats.**
  Every `\begin{figure}` / `\begin{table}` we *don't* emit a cap for still
  sets `\alx@current@alg@id` to a sentinel id so the shared
  `\@makecaption` hook doesn't write anchor coords to a *previous*
  float's id. Otherwise a subsequent `\caption` inside an un-instrumented
  wrapper can clobber the caption bottom of the previous figure.
- **Hook bodies defined at top-level `\def`, not inside `\AtBeginDocument`.**
  `\AtBeginDocument`'s accumulator leaves `##` un-reduced on the first
  pass, so `##1` in the source becomes `##1` in the stored
  `\float@makebox` body, turning `\hsize=##1` into `\hsize=0`. Put the
  hook installer in its own top-level macro and call *that* from
  `\AtBeginDocument`.

## Regression tests

```bash
pytest -q tests/
```

Each golden paper has a snapshot JSON in `tests/golden/<paper_id>.json`
(a subset of `annotations.json` covering `categories` + a stable summary:
count per kind, page number per label, rounded bbox). Tests re-run the
pipeline on a fixture source tree, compare against the snapshot, and fail
if labels move by more than 2px at 200dpi.

Re-baseline after an intentional bbox change:

```bash
python scripts/regen_golden.py --papers <paper_id> ...
```

## Things not to regress

- Multi-column `*_cap` width = column width when float stays in one column,
  text width for `figure*`/`table*` (driven by `\the\hsize` — not a Python
  rule).
- Algorithm / listing body = only the pseudocode / code, NOT the
  caption / title. Top/bot marks are placed just outside the `\caption{…}`
  arg, deterministically regardless of whether the caption is at top or
  bottom of the float.
- `\input`-d fragments get their floats instrumented too (see
  `MultiFileInjector.inject_tree`).
- All 20 papers from
  `/home/zhangshaoang/texlive-arxiv-validation/runs/full-20260424-132920`
  must still compile + produce ≥ the previous label counts; the regression
  test pins this.

## Where the dev data lives

- Source arXiv trees: `/home/zhangshaoang/texlive-arxiv-validation/runs/full-20260424-132920/<paper>/src/`
  (20 strictly-validated papers from the TeX Live validation run).
- Build outputs: `runs/v2_validated/<paper>/` — `src/` (injected tex), `pages/` (PDF
  rasters), `qc/` (overlayed PNGs), `dataset/annotations.json`.
- Sample QC covers: `samples/*.png` — stable representative pages per
  layout family; please keep this set up to date when you add new kinds.

## Useful one-liners

```bash
# Run the pipeline on one paper
python3 scripts/build_dataset.py \
  --papers 2604.21800v1_mathph_variance_geometry_codes --limit 1

# Inspect bbox vs body containment
python3 scripts/qc_check.py --workdir runs/v2_validated
```
