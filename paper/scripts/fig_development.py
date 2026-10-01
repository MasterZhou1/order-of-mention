"""Figure 6: the fact address during pretraining (OLMo-2-7B checkpoints, paper/data/olmo2_checkpoints.json).

(a) Stage by block for each checkpoint, as Figure 5a: shared address (ordinal vector >= 10 points over the unedited
answer), whole-state address (only donor states), content (donor's first token >= 50%); 8 donors x 32 receivers at
every second block. (b) Geometry against training tokens: peak fact-address R^2 over blocks (leave-one-word-set-out),
and the new-name and reversed-order cosines at block 16 (the released model's rule-chosen block). (c) Behaviour at block 16:
designated answers for the ordinal vector, a random direction and the ceiling, and clean accuracy, all scored by the
single food an answer names (base checkpoints answer in sentences). `final` is the released model after stage-2
mid-training.
"""
from common import check, floor_text, load, save
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEV = load("olmo2_checkpoints.json")["checkpoints"]
INK, MUTED, GREY, LIGHT = "#1F2A36", "#5E6B78", "#9AA3AC", "#C9CED3"
ADDR, CTX, CONTENT = "#5B3F94", "#42949E", "#C0703A"
STAGE = {"shared_address": ADDR, "whole_state_address": CTX, "content": CONTENT, "none": "white"}
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.0, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#3A4652", "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "axes.labelsize": 7.4,
    "legend.frameon": False, "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
ORDER = [("step1000", "5B"), ("step4000", "17B"), ("step10000", "42B"), ("step25000", "105B"), ("step50000", "210B"),
         ("step99000", "416B"), ("step200000", "839B"), ("step400000", "1.7T"), ("step928646", "3.9T"), ("main", "final")]
FINAL_X = 7.5e12   # display position of the released model, right of the last stage-1 checkpoint


def tokens(key):
    t = DEV[key].get("tokens_seen")
    return FINAL_X if t is None else t


fig = plt.figure(figsize=(7.0, 2.3))

# a: stage map
sm = fig.add_axes([0, .06, .40, .82]); sm.axis("off"); sm.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
fig.text(.005, .97, "a", fontsize=9.0, fontweight="bold", va="top")
fig.text(.027, .97, "Stage by block through training", fontsize=8.2, va="top")
X0, X1 = .19, .99
for dx, lab, col in ((.02, "shared address", ADDR), (.30, "whole-state address", CTX), (.66, "content", CONTENT)):
    sm.text(X0 + dx, .955, lab, fontsize=6.4, color=col, va="center")
pitch, h = .088, .066
for r, (key, short) in enumerate(ORDER):
    y = .90 - (r + 1) * pitch
    sm.text(X0 - .015, y + h / 2, short, fontsize=6.4, color=MUTED, ha="right", va="center")
    st = DEV[key]["stage_by_block"]
    nb = DEV[key]["blocks"]
    run = sorted(int(b) for b in st)
    w = (X1 - X0) / nb
    for a, b in zip(run, run[1:] + [nb]):
        sm.add_patch(Rectangle((X0 + a * w, y), (b - a) * w - w * .1, h, facecolor=STAGE[st[str(a)]], ec="#E3E7EA",
                               lw=.25, label="_band"))
sm.text(X0 - .015, .955, "tokens", fontsize=6.2, color=MUTED, ha="right", va="center")
for f, lab in ((0, "input"), (.5, "relative depth"), (1, "output")):
    sm.text(X0 + f * (X1 - X0), .90 - (len(ORDER) + 1) * pitch + .045, lab, fontsize=6.4, color=MUTED, ha="center", va="top")


def xaxis(ax, ymin=0, ymax=1):
    ax.set_xscale("log")
    ax.set_xlim(3e9, 1.3e13)
    ax.set_xticks([1e10, 1e11, 1e12, FINAL_X])
    ax.set_xticklabels(["10B", "100B", "1T", "final"])
    ax.minorticks_off()
    ax.set_xlabel("training tokens")
    ax.axvline(5.3e12, ymin=ymin, ymax=ymax, color=LIGHT, lw=.6, ls=(0, (2, 2)))


def series(metric):
    xs, ys = [], []
    for key, _ in ORDER:
        v = metric(DEV[key])
        if v is not None:
            xs.append(tokens(key)); ys.append(v)
    return xs, ys


def at16(m, part):
    return m["at_block_16"].get(part)


# b: geometry
ax = fig.add_axes([.47, .24, .22, .58])
fig.text(.435, .97, "b", fontsize=9.0, fontweight="bold", va="top")
fig.text(.457, .97, "From person to order of mention", fontsize=8.2, va="top")
lines = [(lambda m: m["peak_fact_address_r2"], "fact-address $R^2$", ADDR, "-", "o"),
         (lambda m: at16(m, "invariance")["new_names_cosine"], "cos, new names", INK, "-", "s"),
         (lambda m: at16(m, "invariance")["reversed_order_cosine"],
          "cos, reversed order", INK, (0, (3, 2)), "D")]
for f, lab, col, ls, mk in lines:
    xs, ys = series(f)
    ax.plot(xs, ys, color=col, ls=ls, lw=1.4, marker=mk, ms=3.0, label=lab)
ax.axhline(0, color=LIGHT, lw=.6)
ax.set(ylim=(-1.1, 1.1), yticks=[-1, -.5, 0, .5, 1])
xaxis(ax, ymin=.86)
for x, y, text, col in ((6.0e11, .2, "fact-address $R^2$", ADDR), (3.0e11, 1.0, "cos, new names", INK),
                       (3.0e11, -.95, "cos, reversed order", INK)):
    ax.text(x, y, text, fontsize=6.4, color=col, ha="center", va="center")


# c: behaviour at block 16, by food mention
ax = fig.add_axes([.775, .24, .22, .58])
fig.text(.74, .97, "c", fontsize=9.0, fontweight="bold", va="top")
fig.text(.762, .97, "Transfer at block 16", fontsize=8.2, va="top")


def mention(arm):
    return lambda m: 100 * at16(m, "by_content")[arm]["designated"]


for arm, lab, col, ls, mk in (("ceiling", "ceiling", GREY, "-", None), ("ordinal_vector", "ordinal vector", ADDR, "-", "o"),
                              ("random", "random", GREY, (0, (1, 1.5)), None), ("unedited", "unedited", CONTENT, (0, (3, 2)), None)):
    xs, ys = series(mention(arm))
    ax.plot(xs, ys, color=col, ls=ls, lw=1.4 if arm != "ceiling" else 2.0, marker=mk, ms=3.0, label=lab)
ax.set(ylim=(0, 138), yticks=[0, 50, 100])
ax.set_ylabel("designated answers (%)")
xaxis(ax, ymax=.72)
ax.legend(loc="upper center", ncol=2, fontsize=6.2, handlelength=1.6, columnspacing=.8, borderaxespad=.1)

floor_text(fig, 5.5)
check(fig, "development")
save(fig, "development", dpi=280, bbox_inches="tight", pad_inches=.03)
print("SAVED development")
