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
# from Final_Project/code/, against a completed run directory
python scripts/make_figures.py \
    --input outputs/shapenet_scoped \
    --figures ../report/figures \
    --gallery-cell N10_D16
```

This writes `results_table.tex` (`\input` by the report) plus
`degradation_generation.png`, `degradation_reconstruction.png`,
`loss_curves.png`, and `gallery.png`. It reads `summary.json` from the run
directory, so re-running it after adding grid cells refreshes everything.

The paper includes two of the four figures — the generation curves and the
gallery. The reconstruction curves and loss curves are still generated (both are
useful when checking a run) but their content is two sentences in the text, which
is not worth a figure in a 4-page paper.

`--gallery-cell N10_D16` matters: the gallery defaults to the largest-N cell, but
the report shows N=10 because that is where the DDPM's valid-ratio failure is
visible as an empty row, and the caption describes that cell specifically.

The experiment grid is intentionally ragged: one latent dimension carries the
N-sweep and any other D appears at a single N as an ablation. `make_figures.py`
handles this by plotting only the sweep dimension in the degradation figures and
reporting the ablation as a separate block in the results table.

## Status

The report is complete: 4 pages of body plus references and the proposal
appendix (6 total). Every number comes from the `shapenet_scoped` run (three
N-sweep cells at D=16 plus the D=32 ablation at N=50), and no `\TODO`
placeholders remain. To confirm after any edit:

```bash
grep -c '\\TODO{' main.tex     # expect 0
tectonic main.tex              # expect no "Overfull \hbox" warnings
```

## Files

- `main.tex`               - the report (Intro & Related Work, Method, Evaluation & Results, Conclusion).
- `references.bib`         - bibliography.
- `proposal_appendix.tex`  - approved proposal, reproduced as an appendix.
- `acl2023.sty`            - ACL 2023 style, vendored (from the ACL template).
- `acl_natbib.bst`         - ACL natbib BibTeX style, vendored.
- `figures/`               - generated tables and plots (from `make_figures.py`).
