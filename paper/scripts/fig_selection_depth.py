"""Figure 5 (top, selection-depth) and Figure 13 (bottom row, depth-geometry), drawn on one canvas.

Figure 5: (a) one Gemma-3-4B receiver/donor example and, for every model, the stage of each block (shared address:
the ordinal vector selects >= 10%; whole-state address: only donor states select >= 10%; content: the donor's first
token in >= 50%); (b-d) designated answers through depth for the ordinal vector, a whole donor state, and the
receiver's own state. Figure 13: fact-address and person R^2 and answer alignment through depth.
Reads paper/data/query_geometry.json, transfer.json, depth_curves.json, scale_sweep.json.
"""
import numpy as np
from common import check, floor_text, load, save
import matplotlib.pyplot as plt

G = load("query_geometry.json")
D = load("transfer.json")
E2 = load("depth_curves.json")

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


# ---------------------------------------------------------------- Figure 4
# a: the three tests, defined on one example and summarized as a stage map over relative depth for all models.
fig = plt.figure(figsize=(7.0, 6.0))
from matplotlib.patches import FancyBboxPatch, Rectangle
ex = D["example"]
rec, don = ex["receiver"], ex["donor"]
assert ex["question_fact"] == 0 and ex["designated_fact"] == 1 and ex["model_key"] == "gemma4b"


def output(block, edit):
    rows = [r for r in ex["outputs"] if r["block"] == block and r["edit"] == edit]
    assert len(rows) == 1 and not rows[0]["truncated"]
    return rows[0]["answer"]


fig.text(.005, .995, "a", fontsize=9.2, fontweight="bold", va="top")
fig.text(.027, .995, "Three outcomes through depth: one example, and where each succeeds", fontsize=8.4, va="top")
ax = fig.add_axes([0, .87, 1, .095]); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
for x0, who, color, ctx, q, asks in ((0, "Receiver", INK, rec["context"], rec["question"], "fact 1"),
                                     (.505, "Donor", INK, don["context"], don["question"], "fact 2")):
    ax.add_patch(FancyBboxPatch((x0, 0), .495, 1, boxstyle="round,pad=0,rounding_size=.03", facecolor="#F6F8F9",
                                edgecolor="#C9D2D9", lw=.6, label="_band"))
    facts = ctx.split("\n")
    ax.text(x0 + .01, .8, f"{who}, asks about {asks}:", fontsize=6.8, color=color, fontweight="bold", va="center")
    ax.text(x0 + .228, .8, q, fontsize=6.8, fontweight="bold", va="center")
    for k in range(2):
        ax.text(x0 + .01, .47 - k * .3, f"fact {k + 1}", fontsize=6.4, color=color, va="center")
        ax.text(x0 + .06, .47 - k * .3, facts[k], fontsize=6.8, va="center")
    ax.text(x0 + .485, .17, f"(+{len(facts) - 2} more)", fontsize=6.2, color=MUTED, va="center", ha="right")

# Stage map: rows are the three tests (each with one strip per model), columns relative depth.
tests = [("ordinal vector", "add $v$ from other contexts", ("ordinal_vector", "designated"), STAGE_ADDR,
          f"\u2192 {output(22, 'ordinal_vector')}, its fact 2 (block 22)"),
         ("donor \u2192 receiver's fact", "copy the donor's state", ("donor_state", "designated"),
          STAGE_CTX, f"\u2192 {output(22, 'donor_state')}, its own fact 2 (block 22)"),
         ("donor \u2192 donor's first token", "copy the donor's state", ("donor_state", "donor_first_token"),
          STAGE_CONTENT, f"\u2192 {output(33, 'donor_state')} (block 33)")]
assert output(22, "ordinal_vector") == rec["designated_answer"] == output(22, "donor_state")
assert output(33, "donor_state").split()[0] == don["answer"].split()[0]
ax = fig.add_axes([0, .66, 1, .18]); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
X0, X1, strip, gap = .565, .98, .17, .04
names3 = [("qwen7b", "Qwen2.5-7B"), ("gemma4b", "Gemma-3-4B"), ("llama8b", "Llama-3.1-8B")]
for i, (name, how, (arm, metric), color, eg) in enumerate(tests):
    y = .93 - i * .3
    ax.text(.005, y, name, fontsize=7.2, color=color, fontweight="bold", va="top")
    ax.text(.005, y - .13, how, fontsize=6.6, color=MUTED, va="top")
    ax.text(.18, y - .13, eg, fontsize=6.6, color=color, va="top")
# Level of representation per block: the three tests grouped into stages. Core models: every block; the Qwen
# scale sweep (8 donors x 32 receivers): every second block, each sampled block colored up to the next one.
SW = load("scale_sweep.json")
sm = fig.add_axes([0, .625, 1, .245]); sm.axis("off"); sm.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
sm.text(X0, .915, "stage by block", fontsize=7.0, color=INK, va="bottom")
for dx, lab, col in ((.112, "shared address", STAGE_ADDR), (.222, "whole-state address", STAGE_CTX), (.37, "content", STAGE_CONTENT)):
    sm.text(X0 + dx, .915, lab, fontsize=6.2, color=col, va="bottom")


def stage(ft, md, dt):
    if ft is not None and ft >= .5:
        return STAGE_CONTENT
    if md >= .10:
        return STAGE_ADDR
    if dt >= .10:
        return STAGE_CTX
    return "white"


rows = []
for key, short in names3:
    d = E2["models"][key]
    get = lambda b, edit, m, d=d: next((c[m]["mean"] for c in d["curves"] if c["block"] == b and c["edit"] == edit), None)  # noqa: E731
    blocks = sorted({c["block"] for c in d["curves"]})
    cells = [(b, b + 1, stage(get(b, "donor_state", "donor_first_token"),
                              get(b, "ordinal_vector", "designated"), get(b, "donor_state", "designated")))
             for b in blocks]
    rows.append((short, d["blocks"], cells, set(blocks)))
SWEEP = [("qwen1_5b", "Qwen2.5-1.5B"), ("qwen3b", "Qwen2.5-3B"), ("qwen14b", "Qwen2.5-14B"), ("qwen32b", "Qwen2.5-32B"),
         ("qwen3_8b", "Qwen3-8B")]
for key, short in SWEEP:
    m = SW["models"][key]; c = m["depth"]; nb = m["blocks"]
    run = sorted(int(b) for b in c["ordinal_vector"])
    cells = []
    for a, b in zip(run, run[1:] + [nb]):
        cells.append((a, b, stage(c["donor_state_first_token"].get(str(a)), c["ordinal_vector"][str(a)],
                                  c["donor_state"][str(a)])))
    rows.append((short, nb, cells, None))
h, pitch = .072, .094
for r, (short, nb, cells, sampled) in enumerate(rows):
    y = .89 - (r + 1) * pitch - (.035 if r >= len(names3) else 0)
    sm.text(X0 - .006, y + h * .5, short, fontsize=6.0, color=MUTED, ha="right", va="center")
    w = (X1 - X0) / nb
    covered = set()
    for a, b, col in cells:
        sm.add_patch(Rectangle((X0 + a * w, y), (b - a) * w - w * .08, h, facecolor=col, ec="#E3E7EA", lw=.25, label="_band"))
        covered.update(range(a, b))
    for b in range(nb):
        if b not in covered:
            sm.add_patch(Rectangle((X0 + b * w, y), w * .92, h, facecolor="white", ec="#EEF1F3", lw=.25, hatch="////",
                                   label="_band"))
ybot = .89 - len(rows) * pitch - .035
sm.plot([X0 - .09, X1], [.89 - len(names3) * pitch - .012] * 2, color="#E3E7EA", lw=.6, label="_band")
for f, lab in ((0, "input"), (.5, "relative depth (block / last block)"), (1, "output")):
    sm.text(X0 + f * (X1 - X0), ybot - .015, lab, fontsize=6.4, color=MUTED, ha="center", va="top")

series = [("own_state", "designated", "own state (ceiling)", "#B9BFC5", "-", 2.4),
          ("ordinal_vector", "designated", "ordinal vector", STAGE_ADDR, "-", 1.6),
          ("donor_state", "designated", "donor → receiver's fact", STAGE_CTX, "-", 1.6),
          ("donor_state", "donor_first_token", "donor → donor's first token", STAGE_CONTENT, "-", 1.6)]
geo = [("fact_address_r2", "fact-address $R^2$", STAGE_ADDR, "-", 1.7),
       ("answer", "answer alignment", STAGE_CONTENT, "-", 1.7),
       ("person_r2", "person $R^2$", GREY, (0, (3, 2)), 1.0)]


def window(d):
    rows = sorted([c for c in d["curves"] if c["edit"] == "ordinal_vector"], key=lambda c: c["block"])
    active = [c["block"] for c in rows if c["designated"]["mean"] >= .10]
    return min(active), max(active)


for j, (key, name) in enumerate(MODELS):
    d = E2["models"][key]; g = G["models"][key]
    lo_w, hi_w = window(d)
    x0 = .07 + j * .318
    ticks = list(range(0, d["blocks"], 8)) + ([d["blocks"] - 1] if (d["blocks"] - 1) % 8 > 3 else [])
    ax = fig.add_axes([x0, .37, .25, .185])
    title(fig, x0 - .05, .565, "bcd"[j], f"{name}: answers")
    ax.axvspan(lo_w - .5, hi_w + .5, color=BAND, lw=0, zorder=0, label="_band")
    for arm, metric, label, color, ls, lw in series:
        rows = sorted([c for c in d["curves"] if c["edit"] == arm], key=lambda c: c["block"])
        x = np.array([c["block"] for c in rows]); y = np.array([c[metric]["mean"] for c in rows]) * 100
        lo = np.array([c[metric]["ci"][0] for c in rows]) * 100; hi = np.array([c[metric]["ci"][1] for c in rows]) * 100
        if arm != "own_state":
            ax.fill_between(x, lo, hi, color=color, alpha=.12, lw=0)
        ax.plot(x, y, color=color, lw=lw, ls=ls, label=label, solid_capstyle="round")
    ax.set(ylim=(-3, 103), yticks=[0, 50, 100], xlim=(-.7, d["blocks"] - .3))
    ax.set_xticks(ticks); ax.set_xlabel("Transformer block", labelpad=1)
    if j == 0:
        ax.set_ylabel("Answers (%)")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(.5, .612), ncol=4, fontsize=7.2, handlelength=2,
                   columnspacing=1.4)
    ax = fig.add_axes([x0, .045, .25, .17])
    title(fig, x0 - .05, .235, "abc"[j], f"{name}")
    ax.axvspan(lo_w - .5, hi_w + .5, color=BAND, lw=0, zorder=0, label="_band")
    xb = [c["block"] for c in g["curves"]]
    for field, label, color, ls, lw in geo:
        if field == "answer":
            xs = [c["block"] for c in g["answer_alignment"]]; ys = [c["cosine"] for c in g["answer_alignment"]]
        else:
            xs = xb; ys = [c[field] for c in g["curves"]]
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=label, solid_capstyle="round")
    ax.axhline(0, color=LIGHT, lw=.6, zorder=0)
    ax.set(ylim=(-.3, .95), yticks=[0, .4, .8], xlim=(-.7, d["blocks"] - .3), xlabel="Transformer block")
    ax.set_xticks(ticks)
    if j == 0:
        ax.set_ylabel("$R^2$ / alignment")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(.5, .275), ncol=3, fontsize=7.2, handlelength=2,
                   columnspacing=1.4)

floor_text(fig, 5.500)
check(fig, "selection-depth")
from matplotlib.transforms import Bbox
W6, H6 = fig.get_size_inches()
save(fig, "selection-depth", dpi=300, bbox_inches=Bbox.from_extents(0, .302 * H6, W6, H6 + .02))
save(fig, "depth-geometry", dpi=300, bbox_inches=Bbox.from_extents(0, -.05, W6, .296 * H6))
print("SAVED selection-depth, depth-geometry")
