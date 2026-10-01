"""Figure 1(a), drawn in the layout, boxes, fonts and palette of Figure 1(b) (fig_access_teaser.py).

One context of three facts and a question about Bob goes into a small bot (the language model; panel b draws the
model as a stack of blocks because it reads hidden states). What the question state holds is not the name but the
place of Bob's fact, drawn as order-of-mention chips with chip 2 filled; the answer is the second fact's food.
Illustrative, no data.
"""
from common import check, floor_text, save
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

# the palette of fig_access_teaser.py, plus the address purple of Figures 3-6
INK, BLUE, ORANGE, GREY = "#294D70", "#548F9C", "#C88858", "#8A96A3"
ADDRESS = "#5B3F94"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.2, "text.color": INK,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})

W_IN, H_IN = 7.55, 1.45
fig = plt.figure(figsize=(W_IN, H_IN))


def arrow(ax, a, b, color=INK, lw=1.0, ms=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 label="_band"))


def box(ax, x, y, w, h, edge="#C9D2D9", face="#F6F8F9", lw=.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.012", facecolor=face,
                                edgecolor=edge, lw=lw, label="_band"))


# side strip, as in panel b
side = fig.add_axes([0, 0, .055, 1]); side.axis("off"); side.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
side.add_patch(Rectangle((.62, .06), .2, .80, color=ADDRESS, lw=0, label="_band"))
side.text(.3, .46, "one\ncontext", rotation=90, fontsize=7.4, fontweight="semibold", color=ADDRESS, ha="center",
          va="center", linespacing=1.05)
side.text(.02, .995, "a", fontsize=9.5, fontweight="bold", va="top")
ax = fig.add_axes([.06, 0, .94, 1]); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
ax.yaxis.set_label_coords(.5, .5)   # keep the (empty) label box away from the side strip, as in panel b
ASPECT = H_IN / (W_IN * .94)

ax.text(.025, .975, "Latently, the model first finds which fact the question is about, then reads that fact", fontsize=8.2,
        fontweight="bold", va="top")


def chip(x, y, k, filled, r=.011, size=8.5):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r / ASPECT, facecolor=ADDRESS if filled else "white", edgecolor=ADDRESS,
                         lw=.9, zorder=5, label="_band"))
    ax.text(x, y, str(k), fontsize=size, fontweight="bold", color="white" if filled else ADDRESS, ha="center",
            va="center", zorder=6, gid="tile")


# the context, in the facts box of panel b, facts numbered by order of mention
box(ax, .03, .05, .27, .72)
ax.text(.165, .82, "real prompt (facts and question)", fontsize=8.5, color=GREY, ha="center", va="center")
lines = ["Alice eats an apple.", "Bob eats a pear.", "Carol eats a plum."]
for k, line in enumerate(lines):
    y = .66 - k * .14
    chip(.012, y, k + 1, filled=False)
    ax.text(.04, y, line, fontsize=8.5, va="center", gid="tile")
ax.text(.04, .14, "What does Bob eat?", fontsize=8.5, fontweight="bold", va="center", gid="tile")

# a small bot: the language model
cx, cy = .36, .42
arrow(ax, (.302, cy), (.322, cy), color=GREY, lw=.8, ms=6)
box(ax, cx - .032, cy - .14, .064, .28, edge=INK, face="#F2F5F7", lw=.9)
for dx in (-.013, .013):
    ax.add_patch(Ellipse((cx + dx, cy + .035), .011, .011 / ASPECT, color=INK, lw=0, label="_band"))
ax.add_patch(Arc((cx, cy - .03), .028, .065, theta1=205, theta2=335, color=INK, lw=1.0, label="_band"))
ax.plot([cx, cx], [cy + .14, cy + .20], color=INK, lw=.9, label="_band")
ax.add_patch(Ellipse((cx, cy + .215), .011, .011 / ASPECT, color=ADDRESS, lw=0, label="_band"))
ax.text(cx, cy - .21, "LLM", fontsize=8.5, color=GREY, ha="center", va="center")

# its thought, in two steps: which fact, then what that fact says
for (x, y, r) in ((.402, .60, .0045), (.415, .66, .0075)):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r / ASPECT, facecolor="#F3F0F9", edgecolor=ADDRESS, lw=.8,
                         label="_band"))
ax.text(.445, .74, "latent, not a written chain of thought", fontsize=8.5, color=ADDRESS, va="center", style="italic")
ax.add_patch(FancyBboxPatch((.43, .06), .345, .78, boxstyle="round,pad=0,rounding_size=.03", facecolor="#F3F0F9",
                            edgecolor=ADDRESS, lw=.9, label="_band"))
ax.text(.445, .58, "step 1", fontsize=8.5, color=GREY, va="center")
ax.text(.505, .58, "the 2nd fact", fontsize=8.5, color=ADDRESS, fontweight="bold", va="center")
for k in range(3):
    chip(.66 + k * .032, .58, k + 1, filled=k == 1)
ax.text(.445, .40, "step 2", fontsize=8.5, color=GREY, va="center")
ax.text(.505, .40, "read it: \u201cBob eats a pear.\u201d", fontsize=8.5, color=INK, va="center", gid="tile")
name = ax.text(.445, .19, "find \u201cBob\u201d", fontsize=8.5, color=GREY, va="center")
fig.canvas.draw()
bb = name.get_window_extent().transformed(ax.transData.inverted())
ax.plot([bb.x0 - .003, bb.x1 + .003], [.19, .19], color=GREY, lw=.9, label="_band")
ax.text(bb.x1 + .012, .19, "not by the name", fontsize=8.5, color=GREY, va="center")

# the answer, in the answer box of panel b
arrow(ax, (.778, cy), (.80, cy), ms=6)
box(ax, .803, cy - .12, .19, .24, edge=ORANGE, face="white", lw=1.0)
ax.text(.813, cy, "answers pear", fontsize=8.5, color=ORANGE, va="center", fontweight="bold")
ax.text(.985, cy, "(fact 2)", fontsize=8.5, color=ORANGE, va="center", ha="right")

floor_text(fig, 5.170)
check(fig, "order-cartoon")
save(fig, "order-cartoon", dpi=280, bbox_inches="tight", pad_inches=.02)
print("SAVED order-cartoon")

