"""arxiv-paper-layout-dataset

Build object-detection training data (figure / figure_cap / table / table_cap /
algorithm / algorithm_cap / listing / listing_cap) from arXiv LaTeX sources.

Approach:
1. Inject zref-savepos / TikZ remember-picture anchors into the original .tex
2. Compile with the existing texlive toolchain (latexmk + multiple engines)
3. Parse the generated .aux for anchor coordinates (sp, abspage)
4. Rasterize the PDF with PyMuPDF and project anchor coords into pixels
5. Emit a COCO-style dataset + per-image visualizations for QC
"""
