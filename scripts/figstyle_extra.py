"""Shared style for the four figures drawn from the result files for the final text of the thesis
(Figures 4.2, 6.1, 6.3 and 8.1: scripts/fig_worked_example.py, fig_three_cases.py, fig_episode_catches.py,
fig_quiet_or_sensitive.py).

Same palette and sizes as the other thesis figures (scripts/figstyle.py): Inter, 9 pt at the printed width, grey
for data, red for alarms and thresholds, green dashed for scored boundaries. The Inter TTF files are in
scripts/fonts/; without them the figures fall back to DejaVu Sans.
Usage in a figure script:  REPO, FIG = figstyle_extra.paths()   (the repository root and its figures/ folder;
set FIGDIR to write the figures somewhere else).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent      # scripts/
ROOT = HERE.parent
INK, MUTED, GRID = "#1b2330", "#6b7486", "#e6e9ef"
GREY, LIGHT, PALE = "#a3abb9", "#d3d8e1", "#eef1f6"
BLUE = "#1f5fae"
RED = "#c8372d"
GREEN = "#2b8a3e"
TW = 6.3     # text width in inches (A4, 2.5 cm margins)
FS = 9.0     # every label is at least this size at the printed width


def paths():
    """The repository root and the folder the figures are written to (figures/, or $FIGDIR)."""
    fig = Path(os.environ.get("FIGDIR", ROOT / "figures"))
    fig.mkdir(parents=True, exist_ok=True)
    return ROOT, fig


def setup():
    fam = "DejaVu Sans"
    for f in (HERE / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
        fam = "Inter"
    plt.rcParams.update({
        "font.family": fam, "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlesize": FS + 0.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.titlepad": 5,
        "axes.labelsize": FS, "axes.labelcolor": INK, "axes.edgecolor": GREY, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.fontsize": FS, "legend.frameon": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.3,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white", "mathtext.default": "regular",
    })


def save(fig, folder: Path, name: str):
    fig.savefig(folder / f"{name}.pdf", metadata={"CreationDate": None})   # reproducible PDFs
    fig.savefig(folder / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)
