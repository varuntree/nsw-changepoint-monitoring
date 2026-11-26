"""Compile the repository appendix (appendix/*.tex) into PDFs: appendix/method-details.pdf, simulation.pdf, opal.pdf.

Each file is the LaTeX source of one chapter-style appendix. It is compiled on its own, in a temporary folder, with
the tables it inputs (tables/ and repo-tables/) and the works it cites (appendix/refs.bib). Chapters are lettered so
that their sections, tables and equations cannot be mistaken for those of the thesis: M (method details),
S (simulation) and O (Opal). References to the thesis are given in the text by their numbers in the thesis.

Needs tectonic (https://tectonic-typesetting.github.io); the first run downloads the LaTeX packages it uses.
Usage: python scripts/appendix_pdf.py        (make appendix-pdf)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "appendix"
TITLE = ("Detecting Real Changes in Daily Public Data Streams: Evaluating Online Changepoint Methods on NSW Opal "
         "Train Travel and Electricity Demand")
AUTHORS = "Yihan Zhang, Sihan Wang and Varun Prasad Gopisetti"
DOCS = {"method-details": ("M", "Method details"), "simulation": ("S", "Additional simulation results"),
        "opal": ("O", "Additional Opal results")}

PREAMBLE = r"""\documentclass[11pt,a4paper,oneside]{report}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,array,longtable}
\usepackage{xcolor,colortbl}
\usepackage[round,authoryear]{natbib}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\linespread{1.15}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false}
\captionsetup[table]{position=top,skip=5pt}
\setlength{\LTcapwidth}{\textwidth}
\setcounter{secnumdepth}{2}
\hypersetup{pdftitle={%(name)s: repository appendix to the thesis}, pdfauthor={%(authors)s},
  pdfsubject={%(title)s}, pdfkeywords={changepoint detection, online monitoring, NSW Opal, electricity demand},
  pdfcreator={LaTeX (tectonic)}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\footnotesize Repository appendix: %(name)s}
\fancyhead[R]{\footnotesize \url{https://github.com/varuntree/nsw-changepoint-monitoring}}
\fancyfoot[C]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\footnotesize\thepage}\renewcommand{\headrulewidth}{0pt}}
\renewcommand{\thechapter}{%(letter)s}
\makeatletter
\renewcommand{\chapter}{\thispagestyle{plain}\global\@topnum\z@\@afterindentfalse\secdef\@chapter\@schapter}
\makeatother
\begin{document}
\begin{center}\small
Repository appendix to the thesis\\[2pt]
\textit{%(title)s}\\[2pt]
%(authors)s. Supervisor: Bao Khue Tran.\\
DATA5925 Data Science Project, Group 5, School of Mathematics and Statistics, UNSW Sydney, November 2026.\\[4pt]
Sections, tables and equations numbered M, S or O are in these repository appendices; all other numbers refer to the thesis.
\end{center}
\vspace{-1.5cm}
"""

END = r"""
\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
"""


def build(stem, letter, name, tex_bin):
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        # the appendix files input ../tables/ and ../repo-tables/, so keep that layout
        (t / "appendix").mkdir()
        shutil.copytree(ROOT / "tables", t / "tables")
        shutil.copytree(ROOT / "repo-tables", t / "repo-tables")
        shutil.copy(APP / "refs.bib", t / "appendix" / "refs.bib")
        shutil.copy(APP / f"{stem}.tex", t / "appendix" / f"{stem}.tex")
        doc = (PREAMBLE % dict(name=name, authors=AUTHORS, title=TITLE, letter=letter)
               + "\\input{%s}\n" % stem + END)
        (t / "appendix" / f"{stem}-main.tex").write_text(doc)
        env = dict(os.environ, SOURCE_DATE_EPOCH="1795996800")   # 30 November 2026, for a fixed PDF date
        r = subprocess.run([tex_bin, "--keep-logs", f"{stem}-main.tex"], cwd=t / "appendix", env=env,
                           capture_output=True, text=True)
        log = (t / "appendix" / f"{stem}-main.log")
        text = log.read_text(errors="replace") if log.exists() else ""
        if r.returncode != 0:
            sys.exit(f"{stem}: tectonic failed\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        problems = [line for line in text.splitlines()
                    if ("undefined" in line.lower() and ("reference" in line.lower() or "citation" in line.lower()))
                    or line.startswith("Overfull \\hbox")]
        shutil.copy(t / "appendix" / f"{stem}-main.pdf", APP / f"{stem}.pdf")
        print(f"wrote appendix/{stem}.pdf" + (f"  ({len(problems)} warnings: {problems[:3]})" if problems else ""))


def main():
    tex_bin = shutil.which("tectonic")
    if not tex_bin:
        sys.exit("tectonic not found: install it from https://tectonic-typesetting.github.io")
    for stem, (letter, name) in DOCS.items():
        build(stem, letter, name, tex_bin)


if __name__ == "__main__":
    main()
