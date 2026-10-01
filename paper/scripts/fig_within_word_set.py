"""Figure 7: inside one word set, the query state moves from the named person to the fact address to the
requested content.

Held-out R^2 within each of the 16 calibration word sets (templates fit on the other 15 contexts of the same word
set), by block, for the queried fact's order of mention, the queried person and the requested content; shading is
the shared-address window of Figure 5 (>= 10% designated answers for the ordinal vector).
Reads paper/data/within_word_set.json and depth_curves.json.
"""
from common import check, floor_text, load, save
import matplotlib.pyplot as plt

W = load("within_word_set.json")
E2 = load("depth_curves.json")
INK, MUTED, GREY, LIGHT = "#1F2A36", "#5E6B78", "#7F8994", "#C9CED3"
ADDRESS, CONTENT = "#5B3F94", "#C0703A"   # non-model encodings, as in Figure 5 (address purple, content rust)
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.2, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#3A4652", "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": .8, "ytick.major.width": .8, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.labelsize": 7.8, "legend.frameon": False,
    "legend.fontsize": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
})
MODELS = [("qwen7b", "Qwen2.5-7B"), ("gemma4b", "Gemma-3-4B"), ("llama8b", "Llama-3.1-8B")]
SERIES = [("person", "queried person", GREY, (0, (3, 2)), 1.4),
          ("fact_address", "fact address", ADDRESS, "-", 1.7),
          ("content", "requested content", CONTENT, "-", 1.7)]


def window(key):
    rows = sorted([c for c in E2["models"][key]["curves"] if c["edit"] == "ordinal_vector"], key=lambda c: c["block"])
    active = [c["block"] for c in rows if c["designated"]["mean"] >= .10]
    return min(active), max(active)


fig = plt.figure(figsize=(7.0, 2.35))
for j, (key, name) in enumerate(MODELS):
    m = W["models"][key]
    ax = fig.add_axes([.07 + j * .318, .2, .25, .6])
    fig.text(.07 + j * .318 - .05, .86, "abc"[j], fontsize=9.2, fontweight="bold", va="bottom")
    fig.text(.07 + j * .318 - .028, .86, name, fontsize=8.4, va="bottom")
    lo, hi = window(key)
    ax.axvspan(lo - .5, hi + .5, color="#EEEAF6", lw=0, zorder=0, label="_band")
    xs = [c["block"] for c in m["curves"]]
    for field, label, color, ls, lw in SERIES:
        ax.plot(xs, [c[field] for c in m["curves"]], color=color, ls=ls, lw=lw, label=label, solid_capstyle="round")
    ax.axhline(0, color=LIGHT, lw=.6, zorder=0)
    ax.set(ylim=(-.2, 1.05), yticks=[0, .5, 1], xlim=(-.7, m["blocks"] - .3), xlabel="Transformer block")
    ticks = list(range(0, m["blocks"], 8)) + ([m["blocks"] - 1] if (m["blocks"] - 1) % 8 > 3 else [])
    ax.set_xticks(ticks)
    if j == 0:
        ax.set_ylabel("Within-word-set $R^2$")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.04), ncol=3, handlelength=2.2,
                   columnspacing=1.6)
floor_text(fig, 5.5)
check(fig, "within-world")
save(fig, "within-world", dpi=280, bbox_inches="tight", pad_inches=.03)
print("SAVED within-world")
