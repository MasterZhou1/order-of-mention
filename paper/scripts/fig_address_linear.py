"""Figure 4: fact addresses are low-rank, and later addresses crowd together.

(a) Qwen2.5-7B query-dependent states of held-out contexts in the 3-dim address subspace span{a_p - a_1} fit on
    calibration word sets (gray segments join the four states of one context), with the four fact addresses.
(b) Addresses a_1..a_8 of eight-fact lists in their first two principal directions (units: ||a_2 - a_1||).
(c) Distance between neighbouring addresses along the list, for four, six, and eight facts.
(d) Distance from each address to its nearest other address against transfer to that fact (all models, lengths,
    positions); line: least-squares fit after removing each model's mean.
Reads paper/data/address_geometry.json.
"""
import numpy as np
from matplotlib.lines import Line2D
from common import INK, MUTED, ADDRESS, CONTENT, apply_open_style as apply, check, floor_text, load, save
import matplotlib.pyplot as plt
apply()
A = load('address_geometry.json')
O = A['spacing']
LLAMA = '#D2A13B'
MODELS = [('qwen7b', 'Qwen', ADDRESS, 'o'), ('gemma4b', 'Gemma', CONTENT, '^'), ('llama8b', 'Llama', LLAMA, 's')]

fig = plt.figure(figsize=(7.0, 2.35))

# a: the address subspace as geometry. Qwen's held-out contexts (test word sets; subspace fit on calibration word sets) in the 3-dim
# address subspace (the random-subspace share, 0.1%, is stated in the text).
POS = ["#2E1F5E", "#5B3F94", "#8E72BF", "#BBA8DC"]       # fact 1 -> 4, as in the query-state figure
fig.text(.005, .97, r'$\mathbf{a}$  Changes lie in a three-dimensional subspace (Qwen)', fontsize=8.0, va='top')
SP = A['qwen7b_subspace']
rng = np.random.default_rng(0)
show = rng.choice(len(SP['points']) // 4, 40, replace=False)
lim = .78


def simplex(ax, part, title, share, vertices=True):
    pts = np.array([q['xyz'] for q in part['points']]).reshape(-1, 4, 3)
    pos = np.array([q['order_of_mention'] for q in part['points']]).reshape(-1, 4)
    for c in show:                                          # per-context changes: segments between its 4 states
        order = np.argsort(pos[c])
        xyz = pts[c][order]
        ax.plot(xyz[[0, 1, 2, 3, 0, 2, 1, 3], 0], xyz[[0, 1, 2, 3, 0, 2, 1, 3], 1], xyz[[0, 1, 2, 3, 0, 2, 1, 3], 2],
                color='#9AA3AC', lw=.35, alpha=.55)
    for p in range(4):
        sel = pts[pos == p]
        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], s=2.5, color=POS[p], alpha=.55, depthshade=False, lw=0)
    if vertices:
        t = np.array(part['template_xyz'])
        for p in range(4):
            for q in range(p + 1, 4):
                ax.plot(*zip(t[p], t[q]), color=INK, lw=.9)
        for p in range(4):
            ax.scatter(*t[p], s=46, color=POS[p], edgecolors='white', lw=.8, depthshade=False, zorder=5)
            ax.text(*(t[p] * 1.6), f'fact {p + 1}', fontsize=6.4, color=INK, ha='center', va='center')
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=36, azim=-48)
    if vertices:   # the address subspace gets real axes: its three directions, in units of the mean address distance
        ticks = [-.5, 0, .5]
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
        for set_labels in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            set_labels(['\u22120.5', '0', '0.5'])
        ax.tick_params(labelsize=5.6, pad=-2)
        ax.set_xlabel('dir. 1', fontsize=6.2, labelpad=-7)
        ax.set_ylabel('dir. 2', fontsize=6.2, labelpad=-7)
        ax.set_zlabel('dir. 3', fontsize=6.2, labelpad=-8)
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):     # identical faint box in both panels: the common scale
        axis.pane.fill = False; axis.pane.set_edgecolor('#E3E7EA'); axis._axinfo['grid']['color'] = '#EEF1F3'
        axis.line.set_color('#D5DADF'); axis.line.set_linewidth(.5)
    return ax


main3d = simplex(fig.add_axes([-.03, .02, .34, .86], projection='3d'), SP, None, None)


# b: ordinal curve of eight addresses
ax = fig.add_axes([.465, .2, .14, .58])
fig.text(.425, .97, r'$\mathbf{b}$  Eight addresses', fontsize=8.0, va='top')
for key, name, color, marker in MODELS:
    xy = np.array(O[key]['lengths']['8']['pca_coords_unit'])[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.0, alpha=.8, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=[26 if i == 0 else 12 for i in range(8)], marker=marker, color=color,
               edgecolors='white', lw=.4, zorder=3)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('PC 1 (unit: fact 1–2 distance)', fontsize=6.6, labelpad=1)
ax.set_ylabel('PC 2', fontsize=6.6, labelpad=1)
ax.tick_params(labelsize=6.2)

# c: neighbouring distances shrink along the list
ax = fig.add_axes([.69, .2, .12, .58])
fig.text(.645, .97, r'$\mathbf{c}$  Addresses crowd', fontsize=8.0, va='top')
for key, name, color, marker in MODELS:
    for n, alpha, lw in (('4', .35, .9), ('6', .6, .9), ('8', 1.0, 1.3)):
        d = O[key]['lengths'][n]['adjacent_distance_unit']
        ax.plot(np.arange(1, len(d) + 1) + .5, d, color=color, alpha=alpha, lw=lw, marker=marker, ms=2.4)
ax.set(xlim=(1.2, 7.8), ylim=(0, 1.08), xticks=[1.5, 3.5, 5.5, 7.5], yticks=[0, .5, 1])
ax.set_xticklabels(['1–2', '3–4', '5–6', '7–8'], fontsize=6.4)
ax.set_xlabel('neighboring facts', fontsize=6.6, labelpad=1)
ax.set_ylabel('next-address distance\n(rel. to facts 1–2)', fontsize=6.8, labelpad=1)

# d: nearest-address distance predicts transfer
ax = fig.add_axes([.875, .2, .12, .58])
fig.text(.83, .97, r'$\mathbf{d}$  Spacing → transfer', fontsize=8.0, va='top')
xs, ys, ms = [], [], []
for key, name, color, marker in MODELS:
    x, y = [], []
    for n in ('4', '6', '8'):
        for dest in O[key]['lengths'][n]['destinations']:
            x.append(dest['nearest_distance_unit']); y.append(100 * dest['ordinal_vector'])
    ax.scatter(x, y, s=10, marker=marker, color=color, alpha=.8, edgecolors='none', zorder=3)
    xs.append(np.array(x)); ys.append(np.array(y))
r2 = A['spacing_vs_transfer']['r_squared']
ax.set_title(f'$R^2={r2:.2f}$', fontsize=6.6, color=MUTED, pad=2, loc='right')
ax.set(xlim=(0, 1.08), ylim=(-3, 100), xticks=[0, .5, 1], yticks=[0, 50, 100])
ax.set_xlabel('distance to nearest address', fontsize=6.6, labelpad=1)
ax.set_ylabel('Designated answers (%)', fontsize=6.8, labelpad=1)

fig.legend(handles=[Line2D([0], [0], marker=m, ls='none', color=c, ms=4, label=n) for _, n, c, m in MODELS] +
                   [],
           loc='lower center', bbox_to_anchor=(.71, -.07), ncol=3, fontsize=6.6, handletextpad=.3, columnspacing=1.0)
floor_text(fig, 5.500)
# Axes3D tick labels report stale window extents (e.g. inside panel c), so the neighbour-overlap check would flag
# phantom collisions; hide them only while check() runs (the 3D panel is inspected on the rendered image instead).
_3d = [t for ax in fig.axes if ax.name == '3d' for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()]
_vis = [t.get_visible() for t in _3d]
for t in _3d: t.set_visible(False)
check(fig, 'address-linear')
for t, v in zip(_3d, _vis): t.set_visible(v)
save(fig, 'address-linear', dpi=280, bbox_inches='tight', pad_inches=.03)
print('SAVED address-linear')
