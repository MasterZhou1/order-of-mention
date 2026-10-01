"""Figure 10: the calibrated change follows fact order, not token distance.

(a) Designated answers by length condition for the ordinal position change calibrated on uniform contexts
(filled, 95% word-set bootstrap CI) and on mixed-length contexts (open); grey tick: the receiver's own changed
state (ceiling); x: an equally long random direction. Conditions: none (uniform), a long prefix on fact 1-4,
on facts 2 and 4, filler sentences after facts 1 and 3, a filler block after all facts.
(b) Preregistered contrast on discordant cases, where token distance and fact order point to different
facts: ordinal-target rate minus distance-predicted rate, for absolute and relative distance.
Reads paper/data/token_distance.json.
"""
import numpy as np
from matplotlib.lines import Line2D
from common import INK, GREY, LIGHT, ADDRESS, CONTENT, apply_open_style as apply, check, floor_text, load, save
import matplotlib.pyplot as plt
apply()
A = load('token_distance.json')
MODELS = [('qwen7b', 'Qwen', ADDRESS, 'o'), ('gemma4b', 'Gemma', CONTENT, '^'), ('llama8b', 'Llama', '#D2A13B', 's')]
CONDS = A['conditions']
LABEL = {'uniform': 'uniform', 'long_fact_1': 'long\nfact 1', 'long_fact_2': 'long\nfact 2', 'long_fact_3': 'long\nfact 3',
         'long_fact_4': 'long\nfact 4', 'long_facts_2_4': 'long\nfacts 2, 4', 'filler_lines_after_1_3': 'filler lines\nafter 1, 3',
         'filler_paragraph': 'filler\nparagraph'}
OFF = np.linspace(-.2, .2, len(MODELS)) if len(MODELS) > 1 else [0.0]


def err(ax, x, m, color, mk, filled=True, ms=4.0):
    y = 100 * m['mean']; lo, hi = 100 * np.array(m['ci'])
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=color, ms=ms, lw=.9, capsize=0,
                mfc=color if filled else 'white', mec=color if not filled else 'white', mew=.8 if not filled else .5,
                zorder=3)


fig = plt.figure(figsize=(7.0, 2.3))
ax = fig.add_axes([.065, .22, .6, .6])
fig.text(.005, .93, r'$\mathbf{a}$  Designated answers by length condition', fontsize=8.2)
for j, (key, name, color, mk) in enumerate(MODELS):
    D = A['models'][key]['conditions']
    for i, c in enumerate(CONDS):
        x = i + OFF[j]
        err(ax, x - .05, D[c]['uniform_calibration'], color, mk, True)
        err(ax, x + .05, D[c]['mixed_calibration'], color, mk, False, ms=3.4)
        ax.plot([x - .08, x + .08], [100 * D[c]['ceiling']['mean']] * 2, color=GREY, lw=1.2, zorder=2)
        ax.scatter([x], [100 * D[c]['random_0']['mean']], marker='x', s=9, color=GREY, lw=.7, zorder=2)
ax.axvline(.5, color=LIGHT, lw=.6, zorder=0)
ax.set(xlim=(-.5, len(CONDS) - .5), ylim=(-4, 104), yticks=[0, 50, 100], xticks=range(len(CONDS)))
ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=6.4, linespacing=.95)
ax.set_ylabel('Designated answers (%)')
ax.legend(handles=[Line2D([0], [0], marker=mk, ls='none', color=c, ms=4, label=n) for _, n, c, mk in MODELS] +
                  [Line2D([0], [0], marker='o', ls='none', color=INK, ms=4, label='uniform calib.'),
                   Line2D([0], [0], marker='o', ls='none', mfc='white', mec=INK, ms=3.4, label='mixed calib.'),
                   Line2D([0], [0], color=GREY, lw=1.2, label='ceiling'),
                   Line2D([0], [0], marker='x', ls='none', color=GREY, ms=3.5, label='random')],
          loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=len(MODELS) + 4, fontsize=6.2, handletextpad=.2,
          columnspacing=.8, handlelength=1.2)

ax = fig.add_axes([.75, .22, .23, .6])
fig.text(.7, .93, r'$\mathbf{b}$  Order minus distance', fontsize=8.2)
for j, (key, name, color, mk) in enumerate(MODELS):
    pre = A['models'][key]['order_minus_distance']
    for i, which in enumerate(('absolute', 'relative')):
        x = i + OFF[j] * 1.3
        err(ax, x - .05, pre['uniform_calibration'][which], color, mk, True)
        err(ax, x + .05, pre['mixed_calibration'][which], color, mk, False, ms=3.4)
ax.axhline(0, color=GREY, lw=.8, ls=(0, (3, 2)), zorder=0)
ax.set(xlim=(-.5, 1.5), ylim=(-10, 100), yticks=[0, 50, 100], xticks=[0, 1])
ax.set_xticklabels(['absolute\ndistance', 'relative\ndistance'], fontsize=6.6, linespacing=.95)
ax.set_ylabel('Paired difference (points)', labelpad=1)

floor_text(fig, 5.500)
check(fig, 'address-controls')
save(fig, 'address-controls', dpi=280, bbox_inches='tight', pad_inches=.03)
print('SAVED address-controls')
