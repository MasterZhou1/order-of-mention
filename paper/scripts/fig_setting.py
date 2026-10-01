"""Figure 2(a): the problem in the notation of Definition 3.1, with the opening example drawn as name tiles and
"loves" arrows (blue for kept, orange for changed).

Left: a context states two facts in order; each order of mention i (chip) has a binding identity beta_C(i); the
question "Whom does Carol love?" asks about order of mention p_C(q) = 2, and its answer is the content of beta_C(2).
Right: three operations on the opening example and what each changes. Swapping Bob and David changes the binding
identities and keeps the orders of mention; stating the facts in the other order changes the orders of mention and
keeps the binding identities; the passive paraphrase keeps both. Illustrative, no data.
"""
from common import check, floor_text, save
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

INK, GREY, LIGHT = "#294D70", "#8A96A3", "#C9D2D9"
ADDRESS, CONTENT = "#5B3F94", "#C0703A"
BLUE, ORANGE = "#548F9C", "#C88858"          # kept / changed, as in the original loves panels
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.5, "text.color": INK,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})

W_IN, H_IN = 7.55, 1.62
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
ASPECT = H_IN / W_IN
FS = 8.6


def rbox(x, y, w, h, edge=LIGHT, face="#F6F8F9", lw=.8, r=.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=face,
                                edgecolor=edge, lw=lw, label="_band"))


def chip(x, y, k, color=ADDRESS, r=.0095, filled=False):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r / ASPECT, facecolor=color if filled else "white", edgecolor=color,
                         lw=.9, zorder=5, label="_band"))
    ax.text(x, y, str(k), fontsize=7.6, fontweight="bold", color="white" if filled else color, ha="center",
            va="center", zorder=6, gid="tile")


def tile(x, y, name):
    ax.text(x, y, name, ha="center", va="center", fontsize=FS, zorder=4, gid="tile",
            bbox=dict(boxstyle="round,pad=.22", facecolor="#F0F4F5", edgecolor="#CFD9DE", lw=.6))


def edge(a, b, color, label=None, label_dy=.075, cross=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, shrinkA=17, shrinkB=17, lw=1.2,
                                 color=color, zorder=3, label="_band"))
    if label:
        ax.text((a[0] + b[0]) / 2, a[1] + label_dy, label, ha="center", va="center", fontsize=FS, color=color)


ax.text(.004, .985, "a", fontsize=9.2, fontweight="bold", va="top")
ax.text(.026, .985, "Stated facts, order of mention, and binding identity (Definition 3.1)", fontsize=8.6, va="top")

# ---- left: the objects of Definition 3.1
rbox(.02, .05, .365, .78)
ax.text(.03, .76, "context $C$", fontsize=FS, color=GREY, va="center")
ax.text(.367, .76, "binding identity $\\beta_C(i)$", fontsize=FS, color=CONTENT, va="center", ha="right")
for k, (sent, beta) in enumerate((("Alice loves Bob.", "(Alice, loves, Bob)"), ("Carol loves David.", "(Carol, loves, David)"))):
    y = .62 - k * .145
    chip(.04, y, k + 1, filled=k == 1)
    ax.text(.056, y, sent, fontsize=FS, va="center", gid="tile")
    ax.text(.367, y, beta, fontsize=FS, color=CONTENT, va="center", ha="right", gid="tile")
ax.plot([.03, .365], [.40, .40], color=LIGHT, lw=.6, label="_band")
ax.text(.03, .315, "question $q$: Whom does Carol love?", fontsize=FS, fontweight="bold", va="center", gid="tile")
ax.text(.03, .165, "$p_C(q)=2$", fontsize=FS, color=ADDRESS, va="center", gid="tile")
ax.add_patch(FancyArrowPatch((.105, .165), (.13, .165), arrowstyle="-|>", mutation_scale=7, color=INK, lw=.8,
                             label="_band"))
ax.text(.136, .165, "answer: David, from $\\beta_C(2)$", fontsize=FS, color=CONTENT, va="center", gid="tile")

# ---- right: the opening example, three operations (name tiles and loves arrows, chips = order of mention)
PANELS = [
    ("swap Bob and David", [("Alice", "David"), ("Carol", "Bob")], ORANGE, "loves", True,
     [("$\\beta_C$ changes", ORANGE), ("$p_C$ same", BLUE)]),
    ("reorder the facts", [("Carol", "David"), ("Alice", "Bob")], ORANGE, "loves", False,
     [("same facts", BLUE), ("$p_C$ changes", ORANGE)]),
    ("passive paraphrase", [("Bob", "Alice"), ("David", "Carol")], BLUE, "is loved by", False,
     [("$\\beta_C$ same", BLUE), ("$p_C$ same", BLUE)]),
]
Y1, Y2 = .60, .32
for j, (title, rows, color, pred, crossing, footer) in enumerate(PANELS):
    x0 = .41 + j * .197
    ax.text(x0 + .092, .80, title, fontsize=FS, ha="center", va="center", color=INK)
    lx, rx = x0 + .048, x0 + .152
    if crossing:   # swap: the names stay in place and the arrows cross
        for k, y in enumerate((Y1, Y2)):
            chip(x0 + .005, y, k + 1)
        tile(lx, Y1, "Alice"); tile(rx, Y1, "Bob"); tile(lx, Y2, "Carol"); tile(rx, Y2, "David")
        edge((lx, Y1), (rx, Y2), color); edge((lx, Y2), (rx, Y1), color)
        ax.text((lx + rx) / 2, (Y1 + Y2) / 2, pred, ha="center", va="center", fontsize=FS, color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=.6))
    else:
        for k, ((a, b), y) in enumerate(zip(rows, (Y1, Y2))):
            chip(x0 + .005, y, k + 1)
            tile(lx, y, a); tile(rx, y, b)
            if pred == "is loved by":   # Bob is loved by Alice: arrow from the lover, labelled with the passive
                edge((rx, y), (lx, y), color, label=pred, label_dy=.085)
            else:
                edge((lx, y), (rx, y), color, label=pred, label_dy=.085)
    ax.text(x0 + .092, .13, footer[0][0], fontsize=FS, color=footer[0][1], ha="right", va="center")
    ax.text(x0 + .102, .13, footer[1][0], fontsize=FS, color=footer[1][1], ha="left", va="center")

floor_text(fig, 5.17)
check(fig, "setting")
save(fig, "setting", dpi=280, bbox_inches="tight", pad_inches=.02)
print("SAVED setting")
