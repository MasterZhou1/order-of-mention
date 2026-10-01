"""Figure 8: the address has a fact coordinate and an argument-slot coordinate (two-place relations).

(a) Designated answers for calibrated changes of fact, argument slot, or both, applied within the same verb in active
contexts (dots, 95% word-set bootstrap CI); grey tick: the receiver's own changed state (ceiling); x: best of two
equally long random directions. (b) Held-out R^2 of query-centered states under five templates, one small
point per active verb and a large point for the verb mean. (c) Cosine of the slot-averaged fact-change
vectors and of the slot vectors between verbs (active, all ten pairs) and between the active and passive
voice of the same verb (five verbs).
Reads paper/data/two_place_relations.json.
"""
import numpy as np
from matplotlib.lines import Line2D
from common import GREY, LIGHT, ADDRESS, CONTENT, apply_open_style as apply, check, floor_text, load, save
import matplotlib.pyplot as plt
apply()
D = load('two_place_relations.json')
MODELS = [('qwen7b', 'Qwen', ADDRESS, 'o'), ('gemma4b', 'Gemma', CONTENT, '^'), ('llama8b', 'Llama', '#D2A13B', 's')]
VERBS = D['models']['qwen7b']['verbs']
OFF = (-.22, 0, .22)

fig = plt.figure(figsize=(7.0, 2.35))

# a: transfer of fact / slot / joint changes within the same verb, active contexts
ax = fig.add_axes([.065, .2, .27, .6])
fig.text(.005, .93, r'$\mathbf{a}$  Change fact, slot, or both', fontsize=8.2)
for j, (key, name, color, mk) in enumerate(MODELS):
    arms = D['models'][key]['active']
    for i, c in enumerate(('fact', 'slot', 'both')):
        x = i + OFF[j]
        w = arms['ordinal_vector'][c]
        y = 100 * w['mean']; lo, hi = 100 * np.array(w['ci'])
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=color, ms=4.2, lw=1, capsize=0, mec='white',
                    mew=.5, zorder=3)
        ax.plot([x - .08, x + .08], [100 * arms['ceiling'][c]['mean']] * 2, color=GREY, lw=1.3, zorder=2)
        rnd = 100 * max(arms['random_0'][c]['mean'], arms['random_1'][c]['mean'])
        ax.scatter([x], [rnd], marker='x', s=10, color=GREY, lw=.7, zorder=2)
ax.set(xlim=(-.5, 2.5), ylim=(-4, 100), yticks=[0, 50, 100], xticks=range(3))
ax.set_xticklabels(['fact', 'slot', 'both'])
ax.set_ylabel('Designated answers (%)')
ax.legend(handles=[Line2D([0], [0], color=GREY, lw=1.3, label='ceiling'),
                   Line2D([0], [0], marker='x', ls='none', color=GREY, ms=3.5, label='random')],
          loc='upper right', bbox_to_anchor=(1.0, 1.12), ncol=2, fontsize=6.2, handlelength=1.2, columnspacing=.8,
          handletextpad=.3)

# b: held-out R^2 by template
TEMPL = [('fact_x_slot', 'fact\n×slot'), ('fact_plus_slot', 'fact\n+slot'), ('fact', 'fact'), ('slot', 'slot'),
         ('person', 'person')]
ax = fig.add_axes([.41, .2, .3, .6])
fig.text(.365, .93, r'$\mathbf{b}$  Held-out $R^2$ by template', fontsize=8.2)
for j, (key, name, color, mk) in enumerate(MODELS):
    r2 = D['models'][key]['r2']
    for i, (t, _) in enumerate(TEMPL):
        vals = [r2['active_' + v][t] for v in VERBS]
        x = i + OFF[j]
        ax.scatter(np.full(len(vals), x), vals, s=6, color=color, alpha=.45, lw=0, marker=mk, zorder=2)
        ax.scatter([x], [np.mean(vals)], s=22, color=color, marker=mk, edgecolors='white', lw=.5, zorder=3)
ax.axhline(0, color=LIGHT, lw=.6, zorder=0)
ax.set(xlim=(-.5, len(TEMPL) - .5), ylim=(-.1, .95), yticks=[0, .4, .8], xticks=range(len(TEMPL)))
ax.set_xticklabels([l for _, l in TEMPL], fontsize=6.6, linespacing=.95, va='top')
ax.legend(handles=[Line2D([0], [0], marker=mk, ls='none', color=c, ms=4, label=n) for _, n, c, mk in MODELS],
          loc='upper right', ncol=3, fontsize=6.6, handletextpad=.2, columnspacing=.9, borderaxespad=.2)

# c: fact-change vs slot-vector cosine, across verbs and across voice
ax = fig.add_axes([.78, .2, .21, .6])
fig.text(.745, .93, r'$\mathbf{c}$  Shared across relations', fontsize=8.2)
act = ['active_' + v for v in VERBS]
for j, (key, name, color, mk) in enumerate(MODELS):
    m = D['models'][key]
    groups = [
        [m['fact_vector_cosine'][a][b] for i, a in enumerate(act) for b in act[i + 1:]],
        [m['slot_vector_cosine'][a][b] for i, a in enumerate(act) for b in act[i + 1:]],
    ]
    for i, vals in enumerate(groups):
        x = i + OFF[j] * .9
        ax.scatter(np.full(len(vals), x), vals, s=6, color=color, alpha=.45, lw=0, marker=mk, zorder=2)
        ax.scatter([x], [np.mean(vals)], s=22, color=color, marker=mk, edgecolors='white', lw=.5, zorder=3)
ax.axhline(0, color=LIGHT, lw=.6, zorder=0)
ax.set(xlim=(-.6, 1.6), ylim=(-.2, 1.05), yticks=[0, .5, 1], xticks=range(2))
ax.set_xticklabels(['fact', 'slot'], fontsize=6.6)
ax.set_ylabel('Cosine', labelpad=1)


floor_text(fig, 5.500, floor_pt=6.5)
check(fig, 'role-address')
save(fig, 'role-address', dpi=280, bbox_inches='tight', pad_inches=.03)
print('SAVED role-address')
