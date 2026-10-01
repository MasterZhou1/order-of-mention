"""Figure 1(b): the setting, as one real Qwen2.5-7B run, drawn in Figure 1's palette.

Row 1: the same facts with two different questions are two separate inputs; the block-20 states at the last prompt
token give Delta = h2 - h1, and its average over calibration contexts is v. Row 2: a context with a new verb, nouns,
and names is run twice with the same question, once as is and once with v added at block 20; the two answers are
the archived greedy answers. Prompts are shown without native chat delimiters, first two of four facts.
"""
from common import check, floor_text, load, save
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

EX = load("teaser_example.json")
INK, BLUE, ORANGE, GREY = "#294D70", "#548F9C", "#C88858", "#8A96A3"
ADDRESS = "#5B3F94"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.2, "text.color": INK,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})

cal, rec = EX["calibration_example"], EX["receiver"]
assert EX["model_key"] == "qwen7b" and EX["block"] == 20
assert EX["unedited"]["answer"].strip() == "sloth" and EX["edited"]["answer"].strip() == "penguin"
assert EX["edited"]["direction"] == "0>1"   # v is the change from asking about fact 1 to asking about fact 2
facts_cal, facts_rec = cal["context"].split("\n"), rec["context"].split("\n")
assert rec["question"].startswith("Which animal does " + facts_rec[0].split(" owns")[0])


def arrow(ax, a, b, color=INK, lw=1.0, ms=8, style="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, connectionstyle=style,
                                 shrinkA=0, shrinkB=0, label="_band"))


def box(ax, x, y, w, h, edge="#C9D2D9", face="#F6F8F9", lw=.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.012", facecolor=face,
                                edgecolor=edge, lw=lw, label="_band"))


def chip(ax, x, y, k, r=.0105):
    """An order-of-mention chip outside the prompt box (as in panel a)."""
    aspect = 2.45 / (7.55 * .94)
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r / aspect, facecolor="white", edgecolor=ADDRESS, lw=.9, label="_band",
                         clip_on=False))
    ax.text(x, y, str(k), fontsize=6.4, fontweight="bold", color=ADDRESS, ha="center", va="center", gid="tile",
            clip_on=False)


def facts(ax, x, y, w, h, lines, accent, tag, question=None, edge="#C9D2D9", lw=.7):
    """The real prompt (first two of four facts) in a box; order-of-mention chips outside it, as in panel a."""
    bx = x + .016
    box(ax, bx, y, w - .016, h, edge=edge, lw=lw)
    ax.text(bx + .008, y + h - .045, tag, fontsize=6.6, color=accent, va="center", gid="tile")
    for k in range(2):
        yy = y + h - .115 - k * .075
        chip(ax, x - .002, yy, k + 1)
        ax.text(bx + .008, yy, lines[k], fontsize=7.2, va="center", gid="tile")
    ax.text(bx + .008, y + h - .115 - 2 * .075, f"(+{len(lines) - 2} more facts)", fontsize=6.6, color=GREY,
            va="center", gid="tile")
    if question:
        ax.text(bx + .008, y + .045, question, fontsize=7.2, fontweight="bold", va="center", gid="tile")


def llm(ax, x0, yc, color, w=.075, h=.13, n=7, tap=4):
    """A model drawn as a stack of blocks; the tapped block is colored."""
    y0 = yc - h / 2
    box(ax, x0, y0, w, h, edge="#C3CDD5", face="#F2F5F7")
    gap = w / (n + 1)
    for k in range(n):
        cx = x0 + gap * (k + 1)
        ax.add_patch(Rectangle((cx - gap * .3, y0 + h * .15), gap * .6, h * .7, lw=0,
                               color=color if k == tap else "#D4DCE2", label="_band"))
    return x0 + gap * (tap + 1)


fig = plt.figure(figsize=(7.55, 2.45))
side = fig.add_axes([0, 0, .055, 1]); side.axis("off"); side.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
for (y0, y1, color, lab) in ((.52, .96, BLUE, "calibration\ncontexts"), (.0, .44, ORANGE, "a completely\nnew context")):
    side.add_patch(Rectangle((.62, y0), .2, y1 - y0, color=color, lw=0, label="_band"))
    side.text(.3, (y0 + y1) / 2, lab, rotation=90, fontsize=7.4, fontweight="semibold", color=color, ha="center",
              va="center", linespacing=1.05)
ax = fig.add_axes([.06, 0, .94, 1]); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
ax.yaxis.set_label_coords(.5, .5)   # keep the (empty) label box away from the side strip
side.text(.02, .995, "b", fontsize=9.5, fontweight="bold", va="top")

# Row 1: two inputs that differ only in the question; their states at block 20 give Delta.
ax.text(.025, .975, "(1) Measure the change between two questions", fontsize=8.2, fontweight="bold", va="top")
facts(ax, .0, .53, .345, .36, facts_cal, BLUE, "calibration context (eats, other names)")
qs = [(cal["questions"][0], "asks about fact 1", .83, "$h_1$"), (cal["questions"][1], "asks about fact 2", .61, "$h_2$")]
for q, lab, yc, h in qs:
    arrow(ax, (.347, .71), (.357, yc), color=GREY, lw=.8, ms=6)
    box(ax, .36, yc - .045, .295, .09, edge=BLUE, face="white", lw=.6)
    ax.text(.367, yc, "+ " + q, fontsize=7.0, va="center")
    ax.text(.5075, yc + .07, lab, fontsize=6.6, color=BLUE, ha="center", va="center")
    arrow(ax, (.657, yc), (.667, yc), ms=6)
    llm(ax, .669, yc, BLUE, w=.06, h=.1)
    arrow(ax, (.731, yc), (.741, yc), color=BLUE, ms=6)
    ax.text(.744, yc, h, fontsize=8.2, color=BLUE, va="center")
ax.text(.699, .515, "LLM, block 20", fontsize=6.4, color=GREY, va="center", ha="center")
ax.plot([.771, .779, .779, .771], [.83, .83, .61, .61], color=BLUE, lw=.7)
arrow(ax, (.779, .72), (.79, .72), color=BLUE, ms=6)
box(ax, .792, .6, .203, .24, edge=BLUE, face="white", lw=.8)
ax.text(.8935, .775, r"$\Delta=h_2-h_1$", fontsize=8.4, color=BLUE, ha="center", va="center")
ax.text(.8935, .675, "ordinal vector $v$ =\naverage $\\Delta$ over contexts", fontsize=7.0, color=INK, ha="center", va="center",
        linespacing=1.15)

# Row 2: a new context, the same question run twice: as is, and with v added once at block 20.
ax.text(.025, .455, "(2) Add $v$ once in a new context", fontsize=8.2, fontweight="bold", va="top")
facts(ax, .0, .01, .345, .38, facts_rec, ORANGE, "new relation, new contents, new names", question=rec["question"],
      edge=ORANGE, lw=1.6)
rows = [(.25, True, "penguin", "fact 2", ORANGE), (.08, False, "sloth", "fact 1", INK)]
for yc, edited, word, fact, color in rows:
    arrow(ax, (.347, .165), (.395, yc), color=GREY, lw=.8, ms=6)
    llm(ax, .398, yc, ORANGE if edited else "#D4DCE2", w=.07, h=.1)
    arrow(ax, (.47, yc), (.495, yc), ms=6)
    box(ax, .498, yc - .063, .235, .135, edge=color, face="white", lw=1.0 if edited else .7)
    ax.text(.506, yc + .025, "with $v$" if edited else "without $v$", fontsize=6.6, color=color if edited else GREY,
            va="center")
    ax.text(.506, yc - .023, f"answers {word}", fontsize=7.4, color=color, va="center",
            fontweight="bold" if edited else "normal")
    ax.text(.727, yc - .023, f"({fact})", fontsize=6.6, color=color if edited else GREY, va="center", ha="right")
# v enters the edited run at block 20: down from the v box, left above the row, down into the model.
ax.plot([.8935, .8935, .433], [.6, .39, .39], color=BLUE, lw=1.1, solid_joinstyle="round")
arrow(ax, (.433, .39), (.433, .305), color=BLUE, lw=1.1)
ax.text(.6, .425, "add $v$ at block 20", fontsize=6.8, color=BLUE, va="bottom", ha="center")
ax.text(.76, .165, "$v$ was measured on $eats$\nwith other names; here the\nrelation, contents, and\nnames are all new.",
        fontsize=6.6, color=GREY, va="center", linespacing=1.2)

floor_text(fig, 5.170)
check(fig, "access-teaser")
save(fig, "access-teaser", dpi=280, bbox_inches="tight", pad_inches=.02)
print("SAVED access-teaser")
