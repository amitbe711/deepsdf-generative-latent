# Report

LaTeX source for the final report. Authored to compile on **Overleaf** out of
the box (standard packages only); a git-synced copy is kept here.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload this folder to Overleaf and set `main.tex` as the main document.

## Figures and tables

`main.tex` reads its numbers and plots from `report/figures/`, which is
populated from a completed grid run:

```bash
# from Final_Project/code/
python scripts/make_figures.py --input outputs/grid --figures ../report/figures
```

This writes `results_table.tex` (\input by the report) plus
`degradation_generation.png`, `degradation_reconstruction.png`,
`loss_curves.png`, and `gallery.png`.

## Files

- `main.tex`               - the report (Intro & Related Work, Method, Evaluation & Results, Conclusion).
- `references.bib`         - bibliography.
- `proposal_appendix.tex`  - approved proposal, reproduced as an appendix.
- `figures/`               - generated tables and plots (from `make_figures.py`).
