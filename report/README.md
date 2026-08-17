# Report

LaTeX source for the final report. Two-column ACL 2023 format; the style files
are vendored here so the build has no external dependencies beyond a TeX engine.

## Build

```bash
cd report
tectonic main.tex     # -> main.pdf
```

Tectonic downloads any missing packages on first run and invokes BibTeX itself,
so this single command is the whole build (no separate `bibtex` + rerun cycle).
Install it with `brew install tectonic` if needed.

`pdflatex` also works if you have a full TeX distribution, but it needs the
classic four-pass sequence:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Overleaf works too: upload this folder and set `main.tex` as the main document.

## Figures and tables

`main.tex` reads its numbers and plots from `report/figures/`, which is populated
from a completed grid run:

```bash
# from Final_Project/code/
python scripts/make_figures.py --input outputs/shapenet_scoped --figures ../report/figures
```

This writes `results_table.tex` (`\input` by the report) plus
`degradation_generation.png`, `degradation_reconstruction.png`,
`loss_curves.png`, and `gallery.png`. It reads `summary.json` from the run
directory, so re-running it after adding grid cells refreshes everything.

The experiment grid is intentionally ragged: one latent dimension carries the
N-sweep and any other D appears at a single N as an ablation. `make_figures.py`
handles this by plotting only the sweep dimension in the degradation figures and
reporting the ablation as a separate block in the results table.

## Finishing the report once the grid has run

Everything except the empirical claims is written. Remaining slots are marked with
a loud red `[TODO: ...]` in the compiled PDF (the `\TODO` macro in `main.tex`), so
nothing can be handed in half-filled by accident.

```bash
# 1. numbers + plots (overwrites the placeholder table and the placeholder PNGs)
cd code && python scripts/make_figures.py \
    --input outputs/shapenet_scoped --figures ../report/figures

# 2. fill in every \TODO in main.tex
grep -n "TODO" ../report/main.tex

# 3. rebuild and confirm no TODO survives
cd ../report && tectonic main.tex
python -c "from pypdf import PdfReader; \
  t='\n'.join(p.extract_text() for p in PdfReader('main.pdf').pages); \
  print('remaining TODOs:', t.count('TODO'))"

# 4. package
cd ../code && python scripts/make_submission.py --name AmitBenbenishti_final
```

The figures currently in `figures/` are deliberate placeholders that say so on
their face; step 1 replaces them. They are placeholders rather than the older
synthetic-data plots so that no synthetic number can be mistaken for a ShapeNet
result.

## Files

- `main.tex`               - the report (Intro & Related Work, Method, Evaluation & Results, Conclusion).
- `references.bib`         - bibliography.
- `proposal_appendix.tex`  - approved proposal, reproduced as an appendix.
- `acl2023.sty`            - ACL 2023 style, vendored (from the ACL template).
- `acl_natbib.bst`         - ACL natbib BibTeX style, vendored.
- `figures/`               - generated tables and plots (from `make_figures.py`).
