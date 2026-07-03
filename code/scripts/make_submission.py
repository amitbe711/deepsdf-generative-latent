"""Assemble the final submission zip.

Produces ``Final_Project/submission/<name>.zip`` containing:
  * ``report/``  - LaTeX source + figures, and ``main.pdf`` if it has been
    compiled (e.g. exported from Overleaf into ``report/``),
  * ``code/``    - the implementation with requirements.txt and README,
    excluding caches, logs, virtualenvs and bulky ``outputs/`` artifacts.

Usage:
    python scripts/make_submission.py --name AmitBenbenishti_<ID>
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Paths/patterns to skip when zipping the code folder.
_SKIP_DIRS = {"__pycache__", ".pytest_cache", "outputs", "figures_smoke", ".ipynb_checkpoints"}
_SKIP_SUFFIXES = {".pyc", ".log"}

# Only these report artifacts are packaged (avoids bundling unrelated files that
# may sit in report/). main.pdf is added separately when present.
_REPORT_FILES = ("main.tex", "references.bib", "proposal_appendix.tex", "README.md")


def _should_skip(path: Path) -> bool:
    if any(part in _SKIP_DIRS for part in path.parts):
        return True
    if path.suffix in _SKIP_SUFFIXES:
        return True
    if path.name == ".DS_Store":
        return True
    return False


def add_tree(zf: zipfile.ZipFile, root: Path, arc_prefix: str) -> int:
    count = 0
    for file in sorted(root.rglob("*")):
        if file.is_dir() or _should_skip(file.relative_to(root)):
            continue
        zf.write(file, arcname=f"{arc_prefix}/{file.relative_to(root)}")
        count += 1
    return count


def add_report(zf: zipfile.ZipFile, report_dir: Path) -> int:
    """Package only whitelisted report source + generated figures (+ main.pdf)."""
    count = 0
    for name in _REPORT_FILES:
        path = report_dir / name
        if path.exists():
            zf.write(path, arcname=f"report/{name}")
            count += 1
    figures = report_dir / "figures"
    if figures.exists():
        for file in sorted(figures.rglob("*")):
            if file.is_file():
                zf.write(file, arcname=f"report/figures/{file.relative_to(figures)}")
                count += 1
    pdf = report_dir / "main.pdf"
    if pdf.exists():
        zf.write(pdf, arcname="report/main.pdf")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="submission")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]  # Final_Project/
    code_dir = project_root / "code"
    report_dir = project_root / "report"
    out_dir = project_root / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{args.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        n_code = add_tree(zf, code_dir, "code")
        n_report = add_report(zf, report_dir)

    pdf = report_dir / "main.pdf"
    print(f"Wrote {zip_path} ({n_code} code files, {n_report} report files).")
    if not pdf.exists():
        print(
            "[note] report/main.pdf not found. Compile the report on Overleaf and\n"
            "       place main.pdf in Final_Project/report/, then re-run this script\n"
            "       so the PDF is included in the submission zip."
        )


if __name__ == "__main__":
    main()
