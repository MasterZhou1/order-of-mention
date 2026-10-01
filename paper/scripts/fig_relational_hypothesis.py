"""Figure 2(b-d): the ordinal addressing hypothesis as vector geometry, the measured Qwen2.5-7B states, and the
cosines of every model against the hypothesis's two predictions.

(b) Hypothesis: h(C, q) = c_C + a_{p(q)}; each context is a plane raised by its offset c_C, and switching the question
from the first- to the second-named person moves every context by the same a_2 - a_1 (reversed when the facts are
stated in the other order). (c) Measured Qwen2.5-7B states at block 20 in the same form. (d) Full-dimensional cosines
of the query change with v: new names (predicted +1) and reversed order (predicted -1), for the three main models
and eleven further models at their own main blocks.
Reads paper/data/invariance.json and paper/data/scale_sweep.json.
"""
import numpy as np
from common import INK, GREY, MUTED, apply_open_style, check, floor_text, load, save
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
# Condition colors, kept apart from the model colors.
SAME, REV = '#2E3B4E', '#8C6BB1'

apply_open_style()
INV = load('invariance.json')
SW = load('scale_sweep.json')['models']


def arrow(ax, a, b, color=INK, lw=1.0, ls='-', shrink=0, ms=8, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 shrinkA=shrink, shrinkB=shrink, connectionstyle=f'arc3,rad={rad}', label='_band'))


def swarm(vals, v_in, s_in, d_in):
    """Beeswarm offsets for points at vals: each point, in the given order, takes the smallest sideways offset at
    which it lies at least d_in inches from every point already placed. v_in, s_in: inches per data unit along the
    value axis and the spreading axis."""
    placed, out = [], []
    for v in vals:
        pv = v * v_in
        for k in range(200):
            ds = (k + 1) // 2 * d_in * .5 * (1 if k % 2 else -1)
            if all((pv - qv) ** 2 + (ds - qs) ** 2 >= d_in ** 2 for qv, qs in placed):
                break
        placed.append((pv, ds)); out.append(ds / s_in)
    return out


def theory(ax, w_in, h_in):
    """Full-dimensional cosines against the position hypothesis's two predictions, one dot per model and condition,
    in one column: dark for new names (predicted +1), purple for the reversed order (predicted -1), the colors of
    panel c; marker shape gives the model."""
    ax.axhline(0, color='#DDE2E6', lw=.6, zorder=0)
    # the three main models (new names: mean over 64 name pairs x 2 sentence orders), then eleven further models
    # at their own main site (scale sweep and base-vs-instruct control): small, filled instruct, open base
    pts = []
    for key, marker in [('qwen7b', 'o'), ('gemma4b', '^'), ('llama8b', 's')]:
        d = INV['models'][key]
        pts.append((float(np.mean(d['new_names_cosines'])), d['reversed_order_cosine'],
                    dict(marker=marker, ms=4.0, mec='white', mew=.4, zorder=4), True))
    others = [('qwen1_5b', True), ('qwen3b', True), ('qwen14b', True), ('qwen32b', True), ('qwen3_8b', True),
              ('llama3b_it', True), ('qwen1_5b_base', False), ('qwen3b_base', False), ('qwen3_8b_base', False),
              ('llama3b_base', False), ('gemma4b_pt', False)]
    for key, tuned in others:
        nb = SW[key]['invariance']
        pts.append((nb['new_names_cosine'], nb['reversed_order_cosine'],
                    dict(marker='o', ms=2.9, mew=.8, zorder=3), tuned))
    XLIM, YLIM_D = (-1, 1), (-1.2, 1.2)
    v_in, s_in = h_in / (YLIM_D[1] - YLIM_D[0]), w_in / (XLIM[1] - XLIM[0])
    for c, (pred, color, label, ly) in enumerate([(1.0, SAME, 'new names', .62), (-1.0, REV, 'reversed order', -.22)]):
        ax.plot([-.5, .5], [pred, pred], color=INK, lw=1.1, zorder=1)
        vals = [pt[c] for pt in pts]
        for (v, dx), pt in zip(zip(vals, swarm(vals, v_in, s_in, 3.5 / 72)), pts):
            filled = pt[3]
            ax.plot([dx], [v], ls='none', color=color, mfc=color if filled else 'white', **{k: w for k, w in pt[2].items()
                                                                                            if k != 'mec' or filled})
        ax.text(0, ly, label, fontsize=6.8, color=color, ha='center', va='center')
    ax.set(xlim=XLIM, ylim=YLIM_D)
    ax.set_xticks([])
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel(r'cosine with $v$', fontsize=7.2, labelpad=1)
    ax.spines['bottom'].set_visible(False)


def _field(ax, starts, arrows, colors, guides=None, axes_on=False):
    """One plane per context (its offset c_C sets the height); on it, the state for the first-named person and the
    displacement caused by changing the question to the second-named person."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if not axes_on:
        ax.set_axis_off()
    for (x, y, z), (dx, dy), c in sorted(zip(starts, arrows, colors), key=lambda t: t[0][2]):
        quad = [[(-.75, -.35, z), (.75, -.35, z), (.75, .35, z), (-.75, .35, z)]]
        ax.add_collection3d(Poly3DCollection(quad, facecolor=c, alpha=.06, edgecolor=c, lw=.35))
        ax.scatter([x], [y], [z], s=8, color=INK, depthshade=False)
        ax.quiver(x, y, z, dx, dy, 0, color=c, lw=1.3, arrow_length_ratio=.2)
    for gx in (guides or []):
        ax.plot([gx, gx], [0, 0], [-.1, 1.75], ls=(0, (2, 2)), lw=.7, color=GREY)
    ax.set(xlim=(-.75, .75), ylim=(-.35, .35), zlim=(-.1, 1.75)); ax.set_box_aspect((1, .55, 1.2))
    ax.view_init(elev=17, azim=-62)
    if not axes_on:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def hypothesis3d(ax):
    """Hypothesis: every context shares the addresses a_1, a_2; its plane is lifted by c_C; the query moves every
    context by the same a_2 - a_1 (or its negative when the two facts are reversed)."""
    zs = np.linspace(0, 1.6, 6)
    rev = np.array([False, True, False, True, False, True])
    a1, a2 = -.45, .45
    starts = [((a2 if r else a1), 0.0, z) for r, z in zip(rev, zs)]
    arrows = [((a1 - a2 if r else a2 - a1), 0.0) for r in rev]
    _field(ax, starts, arrows, [REV if r else SAME for r in rev], guides=(a1, a2))
    ax.figure.canvas.draw()
    from mpl_toolkits.mplot3d import proj3d
    for x, lab in ((a1, 'fact 1 ($a_1$)'), (a2, 'fact 2 ($a_2$)')):
        px, py, _ = proj3d.proj_transform(x, 0, -.25, ax.get_proj())
        ax.text2D(px, py, lab, fontsize=7.6, color=INK, ha='center', va='top', transform=ax.transData)


def real3d(ax, E):
    """Measured Qwen states in the same form: 8 real contexts, each on its own plane at its measured context offset
    (top direction of calibration offsets, stretched for display); contexts at fixed quantiles of that offset."""
    pool = [(c, SAME) for c in E if c['condition'] == 'new_names' and c['assignment'] == 0] + \
           [(c, REV) for c in E if c['condition'] == 'reversed' and c['assignment'] == 0]
    picks = []
    for kind_col in (SAME, REV):
        cs = sorted([t for t in pool if t[1] == kind_col], key=lambda t: t[0]['offset'][2])
        picks += [cs[int(q * (len(cs) - 1))] for q in (.1, .37, .63, .9)]
    zs = np.array([c['offset'][2] for c, _ in picks]); zscale = 1.6 / (zs.max() - zs.min())
    starts = [(c['q1'][0], c['q1'][1], (c['offset'][2] - zs.min()) * zscale) for c, _ in picks]
    arrows = [(c['q2'][0] - c['q1'][0], c['q2'][1] - c['q1'][1]) for c, _ in picks]
    first = np.mean([c['q1'][0] if col == SAME else c['q2'][0] for c, col in picks])
    second = np.mean([c['q2'][0] if col == SAME else c['q1'][0] for c, col in picks])
    _field(ax, starts, arrows, [col for _, col in picks], guides=(first, second), axes_on=True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor('#E3E7EA')
        axis._axinfo['grid']['color'] = '#EEF1F3'
    ax.set_xticks([-.5, 0, .5]); ax.set_yticks([-.3, 0, .3])
    zt = np.array([-.1, 0, .1])
    ax.set_zticks((zt - zs.min()) * zscale); ax.set_zticklabels([f'{z:+.1f}'.replace('+0.0', '0').replace('-', '\u2212') for z in zt])
    ax.tick_params(labelsize=5.8, pad=0)
    ax.set_xlabel('ordinal vector\n($v$)', fontsize=6.4, labelpad=2)
    ax.set_ylabel('which people\nare named', fontsize=6.4, labelpad=1)
    ax.set_zlabel('context offset ($c_C$)', fontsize=6.4, labelpad=-5)
    ax.figure.canvas.draw()
    from mpl_toolkits.mplot3d import proj3d
    for x, lab in ((first, 'fact 1'), (second, 'fact 2')):
        px, py, _ = proj3d.proj_transform(x, 0, -.25, ax.get_proj())
        ax.text2D(px, py, lab, fontsize=7.0, color=INK, ha='center', va='top', transform=ax.transData)
    return zscale


fig = plt.figure(figsize=(7.0, 2.7))
fig.text(.0, .97, r'$\mathbf{b}$  Hypothesis', fontsize=8.0, va='top')
hypothesis3d(fig.add_axes([.0, .08, .33, .74], projection='3d'))
fig.text(.015, .9, "plane: one context, raised by its offset $c_C$\ndot: question about person A\narrow: switch to person B (purple: facts reversed)",
         fontsize=6.6, color=MUTED, va='top', linespacing=1.2)
fig.text(.015, .0, r"same arrow $a_2-a_1$ everywhere; flipped if reversed", fontsize=6.8, color=INK,
         va='bottom')
fig.text(.36, .97, r'$\mathbf{c}$  Measured: Qwen2.5-7B, block 20', fontsize=8.0, va='top')
_axb = fig.add_axes([.36, .08, .31, .74], projection='3d'); _axb.set_facecolor('none')
real3d(_axb, INV['qwen7b_states_3d'])
fig.text(.375, .9, "each plane: two test contexts, one offset level\ndark: calibration order; purple: facts reversed", fontsize=6.6, color=MUTED,
         va='top', linespacing=1.2)
ax = fig.add_axes([.8, .2, .17, .56])
theory(ax, .17 * 7.0, .56 * 2.7)
fig.text(.715, .97, r'$\mathbf{d}$  All models', fontsize=8.0, va='top')  # same baseline as b, c
fig.legend(handles=[Line2D([0], [0], marker='o', ls='none', color=MUTED, ms=4, label='Qwen'),
                    Line2D([0], [0], marker='^', ls='none', color=MUTED, ms=4, label='Gemma'),
                    Line2D([0], [0], marker='s', ls='none', color=MUTED, ms=3.6, label='Llama'),
                    Line2D([0], [0], marker='_', ls='none', color=INK, ms=7, mew=1.1, label='prediction'),
                    Line2D([0], [0], marker='o', ls='none', color=MUTED, ms=2.9, label='other instruction-tuned'),
                    Line2D([0], [0], marker='o', ls='none', mfc='white', mec=MUTED, ms=2.9, label='base')],
           loc='upper left', bbox_to_anchor=(.72, .92), ncol=2, fontsize=6.6, handlelength=1.0, columnspacing=.9,
           handletextpad=.3)
floor_text(fig, 5.500)
check(fig, 'relational-hypothesis')
save(fig, 'relational-hypothesis', dpi=260, bbox_inches='tight', pad_inches=.03)
plt.close(fig)
print('SAVED relational-hypothesis')
