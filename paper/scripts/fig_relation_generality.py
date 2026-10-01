"""Figure 9: the query address is shared across relations (six relations x three models).

(a) Designated-answer rates by receiving relation: the change calibrated on the same relation (large filled),
on each of the five other relations (small open), and random directions (x). (b) Held-out R^2 of query states
under fact-position vs person templates, per relation. (c) Cosine between position-change templates fit
separately on each pair of relations, against the cosine between mismatched position changes.
Reads paper/data/relations.json.
"""
import numpy as np
from matplotlib.lines import Line2D
from common import INK, MUTED, GREY, LIGHT, ADDRESS, CONTENT, apply_open_style as apply, check, floor_text, load, save
import matplotlib.pyplot as plt
apply()
G = load('relations.json')
MODELS = [('qwen7b', 'Qwen', ADDRESS), ('gemma4b', 'Gemma', CONTENT), ('llama8b', 'Llama', '#D2A13B')]
MARK = {'qwen7b': 'o', 'gemma4b': '^', 'llama8b': 's'}
VERB = {'food': 'eats', 'pet': 'owns', 'city': 'visited', 'job': 'works\nas', 'instrument': 'plays\ninstr.',
        'sport': 'plays\nsport'}
doms = G['models']['qwen7b']['relations']

fig = plt.figure(figsize=(7.0, 2.45))

# a: transfer within vs across relations
ax = fig.add_axes([.06, .2, .47, .62])
fig.text(.005, .93, r'$\mathbf{a}$  Transfer by receiving relation', fontsize=8.2)
for j, (key, name, color) in enumerate(MODELS):
    m = G['models'][key]
    for i, t in enumerate(doms):
        x = i + (j - 1) * .24
        others = [m['matrix'][s][t]['mean'] * 100 for s in doms if s != t]
        ax.scatter(np.full(len(others), x), others, s=9, marker=MARK[key], facecolors='none', edgecolors=color, lw=.7, alpha=.8, zorder=2)
        ax.scatter([x], [m['matrix'][t][t]['mean'] * 100], s=26, marker=MARK[key], color=color, zorder=3, edgecolors='white', lw=.5)
        ax.scatter([x], [m['random'][t] * 100], s=10, marker='x', color=GREY, lw=.7, zorder=2)
ax.set(xlim=(-.5, len(doms) - .5), ylim=(-4, 100), yticks=[0, 50, 100], xticks=range(len(doms)))
ax.set_xticklabels([VERB[t] for t in doms], fontsize=6.6)
ax.set_ylabel('Designated answers (%)')
ax.legend(handles=[Line2D([0], [0], marker='o', ls='none', color=INK, ms=4.5, label='same relation'),
                   Line2D([0], [0], marker='o', ls='none', mfc='none', mec=INK, ms=3.5, label='other relation'),
                   Line2D([0], [0], marker='x', ls='none', color=GREY, ms=3.5, label='random')] +
                  [Line2D([0], [0], marker=MARK[k], ls='none', color=c, ms=4, label=n) for k, n, c in MODELS],
          loc='upper center', bbox_to_anchor=(.5, 1.13), ncol=3, fontsize=6.3, handletextpad=.2, columnspacing=.9)

# b: position vs person R^2
ax = fig.add_axes([.61, .2, .16, .62])
fig.text(.575, .93, r'$\mathbf{b}$  Held-out $R^2$', fontsize=8.2)
for j, (key, name, color) in enumerate(MODELS):
    m = G['models'][key]
    pos = [m['r2'][t]['fact_address'] for t in doms]; per = [m['r2'][t]['person'] for t in doms]
    ax.scatter(np.full(6, j) + np.linspace(-.18, .18, 6), pos, s=10, marker=MARK[key], color=color, zorder=3)
    ax.scatter(np.full(6, j) + np.linspace(-.18, .18, 6), per, s=10, marker=MARK[key], facecolors='none', edgecolors=color, lw=.7, zorder=3)
ax.axhline(0, color=LIGHT, lw=.6, zorder=0)
ax.set(xlim=(-.5, 2.5), ylim=(-.1, .85), xticks=range(3), yticks=[0, .4, .8])
ax.set_xticklabels([n for _, n, _ in MODELS], fontsize=6.6)
ax.text(2.45, .8, 'filled: fact\naddress\nopen: person', fontsize=6, color=MUTED, ha='right', va='top', linespacing=1.1)

# c: shared position changes across relations
ax = fig.add_axes([.83, .2, .17, .62])
fig.text(.795, .93, r'$\mathbf{c}$  Ordinal vectors across relations', fontsize=8.2)
for j, (key, name, color) in enumerate(MODELS):
    m = G['models'][key]
    same = [m['ordinal_vector_cosine'][s][t] for a, s in enumerate(doms) for t in doms[a + 1:]]
    mism = [m['mismatched_cosine'][s][t] for a, s in enumerate(doms) for t in doms[a:]]
    ax.scatter(np.full(len(same), j) + np.random.default_rng(j).uniform(-.15, .15, len(same)), same, s=6, marker=MARK[key], color=color,
               alpha=.8, zorder=3)
    ax.plot([j - .25, j + .25], [np.mean(mism)] * 2, color=GREY, lw=1.4, zorder=2)
ax.set(xlim=(-.5, 2.5), ylim=(0, 1.02), xticks=range(3), yticks=[0, .5, 1])
ax.set_xticklabels([n for _, n, _ in MODELS], fontsize=6.6)
ax.text(1, .015, 'gray: mismatched\norders of mention', fontsize=6, color=MUTED, ha='center', va='bottom', linespacing=1.1)

floor_text(fig, 5.500, floor_pt=6.5)
check(fig, 'relation-generality')
save(fig, 'relation-generality', dpi=280, bbox_inches='tight', pad_inches=.03)
print('SAVED relation-generality')
