# arxiv-paper-layout-dataset

Build object-detection training data for paper-page layouts straight out of
arXiv LaTeX sources. Classes are

- `fig` / `fig_cap` — a figure's image region, and the whole figure float
  (image + caption).
- `table` / `table_cap` — a table's `tabular` grid, and the whole float.
- `algorithm` / `algorithm_cap` — a pseudocode block, and the float with its
  title / caption.
- `listing` / `listing_cap` — a `lstlisting` code block, and the enclosing
  `listing` float with its caption, when one exists.

The pipeline doesn't rely on PDF rasterisation tricks to guess where things
are — it asks TeX directly for the coordinates, then projects them onto
PyMuPDF page-pixel space.

<p align="center">
  <img src="samples/01_IEEE_figure-star_spans_two_columns.png" width="30%" />
  <img src="samples/05_PhysRev_full-width_multipanel.png" width="30%" />
  <img src="samples/08_algorithm2e_block.png" width="30%" />
</p>

## Requirements

- Linux with a full TeX Live install (`texlive-full` is enough). Tested on
  Ubuntu 24.04 with TeX Live 2023.
- `latexmk`, `biber`, `python3-pygments`, `inkscape`, `gnuplot`, `graphviz`,
  `librsvg2-bin`, `pdf2svg`, `fonts-noto`, `fonts-noto-cjk` — the classic
  arXiv-friendly set.
- Python 3.10+ with `pymupdf` and `Pillow`:

  ```bash
  pip install pymupdf pillow
  ```

## Inputs

Each paper must be laid out as

```
<paper_id>/
  src/
    main.tex           # or any *.tex with \begin{document}
    figure1.pdf
    ...
  downloads/           # optional, holds the original arxiv source blob
```

The [`texlive-arxiv-validation`](https://github.com/HansBug/texlive-arxiv-validation)
validator writes exactly this layout; you can point
`--source-root` at its `runs/` folder.

## Running the pipeline

```bash
# 20 validated arXiv papers, default DPI 200
python3 scripts/build_dataset.py \
  --source-root /path/to/validated-arxiv-runs \
  --work-root   runs/v1

# one paper only
python3 scripts/build_dataset.py \
  --source-root /path/to/validated-arxiv-runs \
  --papers 2604.21800v1_mathph_variance_geometry_codes
```

CLI flags:

| flag             | default                                                                    | meaning                              |
|------------------|----------------------------------------------------------------------------|--------------------------------------|
| `--source-root`  | `/home/zhangshaoang/texlive-arxiv-validation/runs/full-20260424-132920`    | directory of `<paper>/src/...` trees |
| `--work-root`    | `<repo>/runs/v1`                                                           | where outputs land                   |
| `--dpi`          | `200`                                                                      | rasterisation DPI                    |
| `--limit N`      | —                                                                          | process first N papers only          |
| `--papers a b …` | —                                                                          | only process named paper directories |
| `--engine-hint`  | —                                                                          | pin the TeX engine (`pdflatex` etc.) |

## Outputs

For every paper:

```
runs/v1/<paper_id>/
  src/                   # injected TeX + all compile products
  pages/page_NNN.png     # clean page rasters
  qc/page_NNN.png        # same page with labelled bboxes overlaid
  dataset/
    annotations.json     # COCO-style labels
  manifest.json          # mapping label_id -> float_id -> anchor names
```

`annotations.json` follows the MS-COCO schema (`images`, `categories`,
`annotations`), and `annotations[*].bbox` is `[x, y, w, h]` in **image
pixels** at the chosen DPI. Each annotation has an extra `label_id`
field (`fig_3`, `fig_cap_3`, `algorithm_1`, …) for debugging.

`runs/v1/summary.json` aggregates per-paper success, page count, label
count and class breakdown.

## How it works (short version)

1. **Inject anchors** into every `.tex`:
   - `\alxwrap{ID}{payload}` — `\setbox` the payload, log its
     `\wd/\ht/\dp` + current `\the\hsize`, drop two `\zsavepos` anchors
     (left-baseline / right-baseline). Gives exact 2-D corners.
   - `\alxmark{ID}` — bare `\zsavepos` + typeout of the current `\hsize`.
     Bracketing this around a float body gives a Y span + an exact column
     width, no column-count heuristics needed.
2. **Compile** with the same recipe the TeX Live arxiv validator uses
   (latexmk, fall back to pdflatex/xelatex/lualatex, fall back to direct
   engine if a prebuilt `.bbl` confuses latexmk).
3. **Extract** coordinates from the generated `.aux`
   (`\zref@newlabel{name}{\posx{…}\posy{…}\abspage{…}}`). Trusting the
   `.aux` avoids the two-run TeX cycle that typeouts need.
4. **Project** sp → pt → PyMuPDF pixels; union `*_cap` with body boxes so
   it covers all `\includegraphics` inside multi-panel figures.
5. **Visualise** overlays onto the page raster so a human can sanity-check.

A longer write-up aimed at contributors / LLM agents lives in
[`AGENTS.md`](AGENTS.md).

## Testing

Tests compare a fingerprint (sorted labels + integer-pixel bboxes) of
every cached `runs/v1/<paper_id>/dataset/annotations.json` against the
committed snapshot under `tests/golden/`. The pipeline itself takes
minutes per paper, so tests never re-run it — populate `runs/v1/` first:

```bash
# one-time (takes a while)
python3 scripts/build_dataset.py

# then
pytest -q tests/
```

The default tolerance is 2 pixels per coordinate; override via
`ALX_BBOX_TOL=<int> pytest`. Tests `skip` a paper whose workspace hasn't
been built yet.

If a change *intentionally* improves precision, re-baseline the
affected papers:

```bash
python3 scripts/regen_golden.py --papers <paper_id>
```

### Adding more coverage (algorithm / listing cases)

Extra papers with `algorithm` / `algorithm2e` / `lstlisting` blocks can be
pulled straight from arXiv via:

```bash
python3 scripts/fetch_test_papers.py          # downloads into runs/test_papers_src
python3 scripts/build_dataset.py \
  --source-root runs/test_papers_src \
  --work-root   runs/v1_extra
python3 scripts/regen_golden.py               # with ALX_RUNS_ROOT=runs/v1_extra
```

## Known limitations

- `subfigure` / `subcaption` sub-labels are not emitted separately; only
  the parent `fig_cap` covers the whole float.
- `minted` code blocks aren't recognised (use the `listings` package).
- Figures whose only content is a `tikzpicture` use the picture's declared
  bounding box, which may undershoot arrow heads / labels that TikZ draws
  outside that box.
- We require the paper to compile cleanly with the local TeX Live; truly
  broken arXiv uploads won't produce labels.

## License

[MIT](LICENSE).
