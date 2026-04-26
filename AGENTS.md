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
  injector.py       # regex-based LaTeX rewriter. Macros \alxwrap, \alxmark.
  extractor.py      # .aux + .log parsing. Anchor / BoxDims / MarkDims / PageInfo.
  render.py         # sp->pt->pixel projection; cap<->body union; COCO emission.
  visualize.py      # PIL-drawn bboxes on top of PyMuPDF page rasters.
  pipeline.py       # end-to-end per-paper driver (copy src, inject, compile, render).
  spatial_pair.py   # body/cap spatial-pair predicates. Used by export filter,
                    #   corpus snapshot, and the driver's per-paper sp tag.
                    #   CLASS_SUBSETS = {"4","6","8"} also lives here.
  corpus.py         # CorpusState + BalancedQueryStrategy + slug helpers for
                    #   the continuous-crawl pipeline.

scripts/
  build_dataset.py        # one-shot CLI driver over an existing source tree.
  run_corpus_pipeline.py  # continuous arxiv crawler: BQS picks (archive, year),
                          #   downloads e-prints, runs pipeline, writes
                          #   state.json atomically. control.json is a hot-reload
                          #   intervention hook (skip_primary_cats /
                          #   skip_archive_query / force_next_archive).
  corpus_snapshot.py      # one-shot health snapshot from state.json: SUBSETS
                          #   table, fail reasons, archive coverage. Used by
                          #   the slow-monitor loop.
  corpus_stats.py         # human-readable archive/year/category histograms.
  export_yolo.py          # corpus -> Ultralytics YOLO. Roboflow-style layout,
                          #   --kinds + --mode for cap-only/box-only/both
                          #   ablations, parallel emit, --neg-ratio cap, full
                          #   dataset card (README + analysis/*.png +
                          #   dataset_meta.json + train_recommended.yaml +
                          #   manifest.sha256). See ## Export to YOLO below.
  fetch_arxiv_catalog.py  # bulk arxiv metadata across all 20 archives -> Parquet.
  fetch_test_papers.py    # downloads ~32 algorithm/listing-rich seed papers.
  feishu_corpus_b.py      # optional: pushes Monitor B's snapshot as a Feishu
                          #   v2 interactive card via webhook.
  notify_feishu.py        # underlying Feishu schema-2.0 card POST helper.
  regen_golden.py         # rewrite tests/golden/<paper>.json from the current
                          #   pipeline output (use after intentional bbox change).
  qc_check.py             # quick sanity scan: bbox-vs-body containment, etc.

tests/                    # golden-output regression tests (see below).
```

## Where the live corpus lives

```
runs/corpus/
  state.json             # source of truth. {papers: {arxiv_id: record}, stats}
  control.json           # hot-reload intervention hook (read every BQS pick)
  workspaces/            # per-paper output (same shape as runs/v2_validated)
    <slug>/dataset/annotations.json
    <slug>/pages/page_NNN.png
    <slug>/qc/page_NNN.png
  driver.log             # human-readable run log
```

`state.json` is written atomically (tmp + rename), so SIGINT / OOM /
preemption never corrupt the corpus. Re-launching `run_corpus_pipeline.py`
resumes from `state.json` and skips already-logged papers (OK or fail).

## Export to YOLO

`scripts/export_yolo.py` turns either the seed
`runs/v2_validated/` + `runs/v2_extra/` or the live
`runs/corpus/workspaces/` into a Roboflow-style YOLO dataset. The script
is fully documented in `README.md` (## Export to Ultralytics YOLO format)
and has its own self-explanatory `--help`. Key surface for agents:

- `--kinds figure,table[,algorithm[,listing]]` + `--mode both|box-only|cap-only`
  decide what classes land in the YOLO output. The paper-level
  spatial-pair filter still uses the full `(body, cap)` pair set in
  every mode, so `box-only` / `cap-only` is purely an output projection
  — it does NOT weaken the structural sanity check.
- `--sample N` + `--sample-strategy balanced|class-balanced|by-archive|random`
  + `--neg-ratio` produce small smoke-test slices. `--neg-ratio` also
  works in full (no-sample) mode and downsamples negatives to hit the
  requested final-dataset ratio (recommended `0.3` for training).
- `--workers N` parallelises both candidate enumeration (per-paper
  spatial-pair check + label gen) and image emit (per-page resize +
  JPEG encode). Default `min(16, cpu_count)`. PIL releases the GIL on
  heavy ops so threads scale near-linearly.
- Every export writes `data.yaml` (Roboflow `../train/images` style,
  no `path:` field), `train_recommended.yaml`, `dataset_meta.json`,
  `manifest.sha256`, `README.md`, and `analysis/*.png`. Disable any
  with the matching `--no-…` flag.
- Splits stay deterministic across runs: filename stem
  (`<arxiv_id>__page_<NNN>`) hashed by `sha256 % 10` →
  `0-7 train / 8 val / 9 test`. Sampling and `--neg-ratio` never move
  a page across splits.

When extending export functionality, the canonical extension points
are `_collect_candidates` (per-paper filter + label gen),
`_sample_candidates` (selection strategies), `_emit_candidates`
(per-page write + stats accumulation), and `write_dataset_card`
(plots + README). The dataset card readers live in `_save_*` and
`_write_readme` / `_write_dataset_meta` / `_write_train_yaml`.

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
