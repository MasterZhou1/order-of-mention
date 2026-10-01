"""Shared paths, fonts, base style, and layout helpers for the paper's figure scripts.

Importing this module (before any figure is drawn):
  - selects the non-interactive Agg backend;
  - registers the Liberation Sans fonts found in paper/fonts/ or in $FIG_FONT_DIR (the paper's figures use
    Liberation Sans; without it matplotlib falls back to DejaVu Sans and text widths change slightly);
  - applies the base rcParams that every figure script then refines with its own style.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
FIGURES = PAPER / "figures"
TEXTWIDTH_IN = 5.5  # printed text width of the paper, in inches

for _dir in [PAPER / "fonts"] + ([Path(os.environ["FIG_FONT_DIR"])] if os.environ.get("FIG_FONT_DIR") else []):
    for _font in sorted(_dir.glob("*.ttf")):
        font_manager.fontManager.addfont(str(_font))

# Base style; each figure script overrides most of it.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.4, "axes.titlesize": 8.8, "axes.labelsize": 8.2,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "#B8C0C6", "axes.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#D5DBE0", "legend.fancybox": False,
    "legend.borderpad": 0.35, "legend.handlelength": 1.8, "axes.labelcolor": "#263848", "text.color": "#263848",
    "xtick.color": "#263848", "ytick.color": "#263848", "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# The open-axes style shared by most result figures.
INK, MUTED, GREY, LIGHT = "#1F2A36", "#5E6B78", "#9AA3AC", "#C9CED3"
ADDRESS, ADDRESS2, CONTENT = "#0F4D92", "#42949E", "#B64342"   # model colors: Qwen blue, (teal), Gemma red
OPEN_STYLE = {
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.2, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#3A4652", "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": .8, "ytick.major.width": .8, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.labelsize": 7.8, "legend.frameon": False,
    "legend.fontsize": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
}


def apply_open_style():
    plt.rcParams.update(OPEN_STYLE)


def load(name):
    return json.loads((DATA / name).read_text())


def save(fig, name, dpi, **kwargs):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=dpi, **kwargs)


def floor_text(fig, printed_in=TEXTWIDTH_IN, floor_pt=5.8):
    """Raise every text whose printed size would fall below floor_pt when the canvas is printed printed_in wide.
    Call after the figure is drawn and before saving."""
    need = floor_pt / (printed_in / fig.get_figwidth())
    raised = 0
    fig.canvas.draw()  # tick labels are created lazily; draw once so every one exists
    for ax in fig.axes:
        for name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, name, None)
            labels = list(axis.get_ticklabels()) if axis is not None else []
            if labels and min(t.get_fontsize() for t in labels) < need:
                axis.set_tick_params(labelsize=need)
                raised += 1
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_fontsize() < need:
            t.set_fontsize(need)
            raised += 1
    return raised


# ------------------------------------------------------------------ layout check
def _tbox(t, r):
    return Text.get_window_extent(t, r)


def _skip(artist):
    """Reference artists that check() ignores (labels such as '_band'); unlabeled artists ('_child0') are data."""
    lab = str(artist.get_label())
    return lab.startswith("_") and not lab.startswith("_child")


def _points(ax):
    from matplotlib.collections import PathCollection
    pts = [c.get_offset_transform().transform(c.get_offsets()) for c in ax.collections
           if isinstance(c, PathCollection) and not _skip(c) and len(c.get_offsets())]
    return np.vstack(pts) if pts else np.empty((0, 2))


def _segments_hit(xy, bb):
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        ts = np.linspace(0, 1, 60)
        px, py = x0 + (x1 - x0) * ts, y0 + (y1 - y0) * ts
        if np.any((px >= bb.x0) & (px <= bb.x1) & (py >= bb.y0) & (py <= bb.y1)):
            return True
    return False


def check(fig, name):
    """Warn when a label inside an axes touches a line, bar, point, legend, or another label (a layout check only;
    it does not change the figure). Texts with gid 'tile' and artists labeled '_band' are exempt."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        axbb = ax.get_window_extent(r)
        for t in ax.texts:
            if t.get_gid() == "tile" or not t.get_text():
                continue
            bb = _tbox(t, r)
            if not axbb.overlaps(bb) or bb.x1 <= axbb.x0 or bb.y0 >= axbb.y1:
                continue
            bb = Bbox.from_extents(bb.x0 - 5, bb.y0 - 5, bb.x1 + 5, bb.y1 + 5)
            for ln in ax.lines:
                if _skip(ln):
                    continue
                xy = ax.transData.transform(np.column_stack([ln.get_xdata(orig=False), ln.get_ydata(orig=False)]).astype(float))
                if len(xy) >= 2 and _segments_hit(xy, bb):
                    bad.append((t.get_text(), "line"))
            for p in ax.patches:
                if p.get_label() not in ("_band", "_tile") and p.get_window_extent(r).overlaps(bb):
                    bad.append((t.get_text(), "bar"))
            P = _points(ax)
            if len(P) and np.any((P[:, 0] >= bb.x0) & (P[:, 0] <= bb.x1) & (P[:, 1] >= bb.y0) & (P[:, 1] <= bb.y1)):
                bad.append((t.get_text(), "point"))
        inside = [t for t in ax.texts if t.get_text() and t.get_gid() != "tile" and t.get_fontweight() != "bold"
                  and axbb.overlaps(_tbox(t, r))]
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                if _tbox(inside[i], r).overlaps(_tbox(inside[j], r)):
                    bad.append((inside[i].get_text() + " / " + inside[j].get_text(), "text on text"))
        lg = ax.get_legend()
        if lg is not None:
            lb = lg.get_window_extent(r)
            lb = Bbox.from_extents(lb.x0 - 2, lb.y0 - 2, lb.x1 + 2, lb.y1 + 2)
            for ln in ax.lines:
                if _skip(ln):
                    continue
                xy = ax.transData.transform(np.column_stack([ln.get_xdata(orig=False), ln.get_ydata(orig=False)]).astype(float))
                if _segments_hit(xy, lb) or (len(xy) == 1 and lb.contains(*xy[0])):
                    bad.append(("legend", "line"))
            P = _points(ax)
            if len(P) and np.any((P[:, 0] >= lb.x0) & (P[:, 0] <= lb.x1) & (P[:, 1] >= lb.y0) & (P[:, 1] <= lb.y1)):
                bad.append(("legend", "scatter point"))
            for t in ax.texts:
                if t.get_text() and _tbox(t, r).overlaps(lb):
                    bad.append(("legend / " + t.get_text(), "text"))
        labs = [lab for lab in ax.get_xticklabels() if lab.get_text()]
        boxes = [lab.get_window_extent(r) for lab in labs]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i].overlaps(boxes[j]):
                    bad.append((labs[i].get_text() + " / " + labs[j].get_text(), "tick labels"))
    if bad:
        print(f"{name}: layout warning: {sorted(set(bad))}")
