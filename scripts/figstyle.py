"""One visual style for every thesis figure.

Font: Inter (put the Inter TTF files in scripts/fonts/), embedded as TrueType in the PDFs; DejaVu Sans if
the files are missing.
Figures are drawn at the width they are printed in the thesis, so a font size here is the printed size.

Palette:
  ink / muted / grid       text, axes and light gridlines
  data grey                daily series and anything that is "just data"
  BLUE                     our recommended monitor (Holt-Winters with FOCuS or EWMA) and nothing else
  RED                      alarms and nothing else
  GREEN, dashed            scored episodes and their boundaries
  DET[...]                 one fixed colour per detector family, the same in every figure;
                           capped versions dashed with open markers, FOCuS-AR dotted, variance versions
                           share their family's colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONTS = Path(__file__).resolve().parent / "fonts"

INK, MUTED, GRID = "#1b2330", "#6b7486", "#e6e9ef"
GREY, LIGHT, PALE = "#a3abb9", "#d3d8e1", "#eef1f6"
BLUE, BLUE_L = "#1f5fae", "#b9cfea"
RED = "#c8372d"
GREEN, GREEN_L = "#2b8a3e", "#e3f1e6"

# detector families: validated as a set (dataviz validate_palette.js, light, all pairs: CVD dE >= 13,
# normal-vision dE >= 23, contrast >= 3:1); CPM is ink. Blue and red stay out of this set.
ORANGE, TEAL, VIOLET = "#d4602a", "#00928f", "#5b4bc4"
DET = {"CUSUM": ORANGE, "EWMA": TEAL, "FOCuS": VIOLET, "FOCuS-AR": VIOLET, "CPM": INK,
       "CUSUM-r": ORANGE, "EWMA-r": TEAL, "FOCuS-r": VIOLET,
       "CUSUM-var": ORANGE, "EWMA-var": TEAL, "FOCuS-var": VIOLET}
DLS = {d: ((0, (3.2, 1.6)) if d.endswith("-r") else (0, (1, 1.3)) if d == "FOCuS-AR" else "-") for d in DET}
DMK = {"CUSUM": "o", "EWMA": "s", "FOCuS": "^", "FOCuS-AR": "v", "CPM": "D",
       "CUSUM-r": "o", "EWMA-r": "s", "FOCuS-r": "^", "CUSUM-var": "o", "EWMA-var": "s", "FOCuS-var": "^"}
DNAME = {"CUSUM": "CUSUM", "EWMA": "EWMA", "FOCuS": "FOCuS", "FOCuS-AR": "FOCuS-AR", "CPM": "CPM",
         "CUSUM-r": "capped CUSUM", "EWMA-r": "capped EWMA", "FOCuS-r": "capped FOCuS",
         "CUSUM-var": "variance CUSUM", "EWMA-var": "variance EWMA", "FOCuS-var": "variance FOCuS"}
DNAME2 = {k: v.replace(" ", "\n", 1) for k, v in DNAME.items()}  # two-line version for column heads
DNAME2["FOCuS-AR"] = "FOCuS-\nAR"   # keeps it apart from the FOCuS column head

# adjustments: names used in the thesis (HW and the yearly memory are allowed in short form in figures, like tables)
ANAME = {"oracle": "true pattern removed", "raw": "no forecast", "A": "same-weekday difference",
         "B": "median of four", "HW": "Holt-Winters", "YM": "yearly memory",
         "MSTL_p": "MSTL refitted on the past", "REG_p": "regression refitted on the past",
         "STL": "retrospective STL", "MSTL": "retrospective MSTL", "REG": "retrospective regression"}
ASHORT = {"oracle": "true pattern", "raw": "none", "A": "same-weekday diff.", "B": "median of four", "HW": "Holt-Winters", "YM": "yearly memory",
          "MSTL_p": "MSTL (past)", "REG_p": "regression (past)", "STL": "STL (retro.)", "MSTL": "MSTL (retro.)",
          "REG": "regression (retro.)"}

TW = 6.3            # \textwidth in inches (A4, 2.5 cm margins)
MIN_PT = 9.0        # smallest text allowed once the figure sits at its printed width
FS = 9.0            # base size; panel labels FS+0.5


def setup():
    fam = "DejaVu Sans"
    if FONTS.exists():
        for f in FONTS.glob("*.ttf"):
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
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "legend.fontsize": FS, "legend.frameon": False, "legend.handlelength": 1.8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white", "mathtext.default": "regular",
    })


def ygrid(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def print_pt(fig, print_w):
    """Smallest text (pt) once the figure is scaled to its printed width (inches) in the thesis."""
    from matplotlib.text import Text
    r = fig.canvas.get_renderer()
    w = fig.get_tightbbox(r).width + 2 * plt.rcParams["savefig.pad_inches"]
    scale = print_w / w
    sizes = [(t.get_fontsize() * scale, t.get_text()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_window_extent(r).width > 0]
    return (min(sizes) if sizes else (np.inf, "")), scale, w


def save(fig, out_dir: Path, name: str, print_w: float = TW, png=True):
    fig.canvas.draw()
    (pt, txt), scale, w = print_pt(fig, print_w)
    flag = "  <-- below 9 pt" if pt < MIN_PT - 0.05 else ""
    print(f"  {name}: drawn {w:.2f} in, printed {print_w:.2f} in (x{scale:.2f}); smallest text {pt:.1f} pt "
          f"({txt[:24]!r}){flag}")
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None})  # reproducible PDFs
    if png:
        fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def boundary(ax, d, **kw):
    """A scored boundary: green dashed vertical line."""
    kw = {"color": GREEN, "lw": 0.9, "ls": (0, (3, 2)), "zorder": 1, **kw}
    ax.axvline(d, **kw)


def alarm_marks(ax, xs, ys, **kw):
    """Alarms: red downward triangles."""
    kw = {"marker": "v", "ms": 6, "color": RED, "ls": "", "zorder": 5, "mec": "white", "mew": 0.5, **kw}
    ax.plot(xs, ys, **kw)


def det_line(ax, x, y, d, label=None, **kw):
    capped = d.endswith("-r")
    kw = {"color": DET[d], "ls": DLS[d], "marker": DMK[d], "ms": 4, "lw": 1.4,
          "mfc": "white" if capped else DET[d], "mec": DET[d], "mew": 1.0, **kw}
    return ax.plot(x, y, label=label, **kw)
