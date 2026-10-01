"""Figure 3 (top row, query-organization) and Figure 12 (bottom row, query-controls), drawn on one canvas.

Figure 3: test-word-set query states at each model's main block in the first two principal directions of the four
fact-address templates, colored by the queried fact's order of mention. Figure 12: the cosine between the measured
person x relation interaction and the one predicted by fact addresses (Lemma E.1). The bottom row of the canvas also
draws two behavior panels (rearranged facts, the eats vector under other relations; reported in Table 2 and
Figure 9) that fall outside both saved crops; they are kept so that the canvas, and hence both crops, are unchanged.
Reads paper/data/query_geometry.json, transfer.json, relations.json.
"""
import numpy as np
from common import check, floor_text, load, save
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

G = load("query_geometry.json")

INK, MUTED, GREY, LIGHT = "#1F2A36", "#5E6B78", "#9AA3AC", "#C9CED3"
ADDRESS, ADDRESS2, CONTENT = "#0F4D92", "#42949E", "#B64342"   # model colors: Qwen blue, (teal), Gemma red
# Non-model encodings of the depth figure, kept apart from the model colors: the address / ordinal vector in the
# ordinal-position purple of the query-state figure, whole-state address in teal, content in the answer rust of Figure 1.
STAGE_ADDR, STAGE_CTX, STAGE_CONTENT, BAND = "#5B3F94", "#42949E", "#C0703A", "#EEEAF6"
POS = ["#2E1F5E", "#5B3F94", "#8E72BF", "#BBA8DC"]           # 1st -> 4th fact: one sequential hue, apart from the model colors
PERSON = ["#6B4C9A", "#A48CC9", "#4D4D4D", "#A8A8A8"]
PERSON_MARK = ["o", "s", "^", "D"]
MODELS = [("qwen7b", "Qwen2.5-7B"), ("gemma4b", "Gemma-3-4B"), ("llama8b", "Llama-3.1-8B")]
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.2, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#3A4652", "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": .8, "ytick.major.width": .8, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.labelsize": 7.8, "legend.frameon": False,
    "legend.fontsize": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
})


def title(fig, x, y, letter, text):
    fig.text(x, y, letter, fontsize=9.2, fontweight="bold", va="bottom")
    fig.text(x + .022, y, text, fontsize=8.4, va="bottom")


def kde_contours(ax, pts, color, levels=(.5, .85)):
    """Highest-density contours of a 2-D Gaussian KDE (numpy only)."""
    lo, hi = pts.min(0), pts.max(0)
    pad = (hi - lo) * .6 + 1e-9
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 120),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], 120))
    cov = np.cov(pts.T) * len(pts) ** (-2 / 6) + np.eye(2) * 1e-8
    inv = np.linalg.inv(cov)
    d = np.stack([gx.ravel(), gy.ravel()], 1)[:, None, :] - pts[None, :, :]
    dens = np.exp(-.5 * np.einsum("npi,ij,npj->np", d, inv, d)).sum(1).reshape(gx.shape)
    flat = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    thresholds = sorted(flat[np.searchsorted(cum, lv)] for lv in levels)
    ax.contourf(gx, gy, dens, levels=[thresholds[-1], dens.max() * 1.01], colors=[color], alpha=.13, zorder=1)
    ax.contour(gx, gy, dens, levels=thresholds, colors=[color], linewidths=[.6, .9], alpha=.85, zorder=2)


def projection(ax, key, by="position", arrows=True):
    pr = G["models"][key]["projection"]
    by = {"position": "order_of_mention"}.get(by, by)
    xy = np.array([p["xy"] for p in pr["points"]])
    t2 = np.array(pr["template_xyz"])[:, :2]
    scale = np.linalg.norm(t2[1] - t2[0])          # unit: distance between the fact-1 and fact-2 templates
    xy = xy / scale
    template = t2 / scale
    labels = np.array([p[by] for p in pr["points"]])
    if by == "order_of_mention":
        if arrows:
            ctx = {}
            for p, (x, y) in zip(pr["points"], xy):
                ctx.setdefault(p["context"], {})[p["order_of_mention"]] = (x, y)
            for c in ctx.values():
                if 0 in c and 1 in c:
                    ax.annotate("", xy=c[1], xytext=c[0], zorder=1.5,
                                arrowprops=dict(arrowstyle="-", color=GREY, lw=.35, alpha=.35))
        for k in range(4):
            kde_contours(ax, xy[labels == k], POS[k])
            ax.scatter(*xy[labels == k].T, s=3.5, color=POS[k], alpha=.45, lw=0, zorder=3, rasterized=True)
        ax.annotate("", xy=template[1], xytext=template[0], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=9, shrinkA=6, shrinkB=6))
        for k in range(4):
            ax.scatter(*template[k], s=62, color=POS[k], edgecolor="white", lw=1.1, zorder=7, label="_template")
            ax.text(*template[k], str(k + 1), color="white" if k < 2 else INK, fontsize=6.8, fontweight="bold", ha="center",
                    va="center", zorder=8, gid="tile")
    else:
        for k in range(4):
            ax.scatter(*xy[labels == k].T, s=6, color=PERSON[k], marker=PERSON_MARK[k], alpha=.7, lw=0, zorder=3,
                       rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    cf = pr.get("captured_fraction")
    vx, vy = xy[:, 0].var(), xy[:, 1].var()
    if cf is not None:
        ax.set_xlabel(f"PC 1 ({100 * cf * vx / (vx + vy):.0f}% of variance)", fontsize=6.6, labelpad=1)
        ax.set_ylabel(f"PC 2 ({100 * cf * vy / (vx + vy):.0f}%)", fontsize=6.6, labelpad=1)
    ax.tick_params(labelsize=6.2, length=2, pad=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.5))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return template


# ---------------------------------------------------------------- Figure 3
fig = plt.figure(figsize=(7.0, 3.35))
for j, (key, name) in enumerate(MODELS):
    ax = fig.add_axes([.075 + j * .335, .55, .21, .33])
    template = projection(ax, key, "position")
    if j == 0:   # reading guide, drawn once
        mid = (template[0] + template[1]) / 2
        ax.text(mid[0] - .1, mid[1] - .1, "predicted\nfact 1→2", fontsize=6.8, color=INK, ha="right", va="center")
    m = G["models"][key]
    c = {x["block"]: x for x in m["curves"]}[m["main_block"]]
    title(fig, .01 + j * .335, .93, "abc"[j], f"{name}, block {m['main_block']}")
    fig.text(.01 + j * .335 + .022, .925, f"fact-address $R^2$ {c['fact_address_r2']:.2f}    person $R^2$ {c['person_r2']:.2f}".replace(" -", " \u2212"),
             fontsize=6.8, color=MUTED, ha="left", va="top")

LLAMA = "#D2A13B"
MODEL3 = (("qwen7b", ADDRESS, "Qwen"), ("gemma4b", CONTENT, "Gemma"), ("llama8b", LLAMA, "Llama"))
MARK3 = {"qwen7b": "o", "gemma4b": "^", "llama8b": "s"}

# e: the lemma as a prediction
ax = fig.add_axes([.03, .15, .24, .23])
fig.text(.01, .41, "Predicted interaction", fontsize=8.4, va="bottom")  # single-panel appendix figure: no letter
grid = np.linspace(-.55, 1.0, 300)
top = 0
nulls = []
for key, color, name in MODEL3:
    m = G["models"][key]
    lem = m["interaction"]
    v = np.array(lem["cosines"])
    bw = 1.06 * v.std() * len(v) ** (-1 / 5)
    dens = np.exp(-.5 * ((grid[:, None] - v[None]) / bw) ** 2).sum(1) / (len(v) * bw * np.sqrt(2 * np.pi))
    ax.fill_between(grid, dens, color=color, alpha=.12, lw=0)
    ax.plot(grid, dens, color=color, lw=1.4)
    top = max(top, dens.max())
    nulls += lem["relabeled"]
null = float(np.mean(nulls))
ax.axvline(null, color=GREY, lw=1.0, ls=(0, (3, 2)), label="_null")
ax.text(null - .05, top * 2.75, f"shuffled\n{null:.2f}".replace("-", "\u2212"), color=MUTED, fontsize=6.2, ha="right", va="top",
        linespacing=1.05)
ax.set(xlim=(-.55, 1.0), ylim=(0, top * 2.9), xticks=[0, .5, 1], yticks=[])
ax.spines["left"].set_visible(False)
ax.set_xlabel("cosine: measured vs predicted\nperson×relation interaction", fontsize=7.2)
ax.legend(handles=[Line2D([0], [0], color=c, lw=1.4, label=n) for k, c, n in MODEL3], loc="upper right", fontsize=6.2,
          handlelength=1.2, frameon=False, borderaxespad=.2)

# f: behavior when facts are rearranged
ax = fig.add_axes([.39, .15, .22, .23])
title(fig, .35, .41, "b", "Rearranged facts")
T = load("transfer.json")
for key, color, name in MODEL3:
    for change, ls, short in (("person", "-", "person"), ("relation", (0, (3, 2)), "relation")):
        ys = [100 * T["models"][key]["rearranged"][change][arm]["mean"]
              for arm in ("original_pair", "new_order_of_mention", "recalibrated")]
        ax.plot(range(3), ys, color=color, ls=ls, lw=1.4, marker=MARK3[key], ms=3.2,
                mfc=color if short == "person" else "white", mew=1)
ax.set(ylim=(-3, 103), xlim=(-.25, 2.25), yticks=[0, 50, 100], xticks=range(3))
ax.set_xticklabels(["original\npair", "new order\nof mention", "re-\ncalibrated"], fontsize=6.4)
ax.set_ylabel("Designated answers (%)", fontsize=7.2)

# g: the eats-calibrated change applied under other verbs
GEN = load("relations.json")
ax = fig.add_axes([.74, .15, .25, .23])
title(fig, .725, .41, "c", "\u2018eats\u2019 vector, other relations")
targets = [t for t in GEN["models"]["qwen7b"]["relations"] if t != "food"]
short = {"pet": "owns", "city": "visited", "job": "works\nas", "instrument": "plays\ninstr.", "sport": "plays\nsport"}
for j, (key, color, name) in enumerate(MODEL3):
    m = GEN["models"][key]
    for i, t in enumerate(targets):
        x = i + (j - 1) * .24
        ax.plot([x - .11, x - .01], [100 * m["matrix"][t][t]["mean"]] * 2, color=INK, lw=.9, zorder=2)
        # eats change rescaled to the receiving verb's own direction length (equal-norm, the fair comparison)
        ax.scatter([x + .05], [100 * m["eats_vector_equal_norm"][t]["mean"]], s=16, marker=MARK3[key], color=color, zorder=3, edgecolors="white", lw=.4)
ax.axhline(0, color=LIGHT, lw=.6, zorder=0)
ax.set(ylim=(-3, 103), xlim=(-.5, len(targets) - .5), yticks=[0, 50, 100], xticks=range(len(targets)))
ax.set_xticklabels([short[t] for t in targets], fontsize=5.8, linespacing=.95)
ax.set_ylabel("Designated answers (%)", fontsize=7.2, labelpad=1)

# one legend row for the bottom panels
fig.legend(handles=[Line2D([0], [0], color=c, lw=1.6, marker=MARK3[k], ms=3.2, label=n) for k, c, n in MODEL3] +
                   [Line2D([0], [0], color=MUTED, lw=1.4, label="person change"),
                    Line2D([0], [0], color=MUTED, lw=1.4, ls=(0, (3, 2)), label="relation change"),
                    Line2D([0], [0], marker="_", ls="none", color=INK, ms=6, mew=1.1, label="own-relation vector")],
           loc="lower center", bbox_to_anchor=(.655, -.03), ncol=6, fontsize=6.4, handlelength=1.6, columnspacing=.8)

floor_text(fig, 5.500)
check(fig, "query-organization")
# Main text: the top row (a-c). Appendix: the bottom row, relabeled a-c (query-controls). Same canvas, so the
# printed sizes are those checked above.
from matplotlib.transforms import Bbox
W, H = fig.get_size_inches()
for name, box in (("query-organization", Bbox.from_extents(0, .468 * H, W, H + .02)),
                  ("query-controls", Bbox.from_extents(0, .005 * H, .30 * W, .458 * H))):
    save(fig, name, dpi=300, bbox_inches=box)
print("SAVED query-organization, query-controls")
