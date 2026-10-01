"""Figure 11: transfer by destination position in lists of four, six, and eight facts.

One panel per model: designated-answer rate of the calibrated change by the destination fact's position, with
word-set bootstrap 95% intervals, for N = 4, 6, 8; the dashed line is the receiver's own changed state (ceiling).
Reads paper/data/longer_lists.json.
"""
import numpy as np
from matplotlib.lines import Line2D
from common import GREY, ADDRESS, apply_open_style as apply, check, floor_text, load, save
import matplotlib.pyplot as plt
apply()
L = load('longer_lists.json')
MODELS = [('qwen7b', 'Qwen2.5-7B'), ('gemma4b', 'Gemma-3-4B'), ('llama8b', 'Llama-3.1-8B')]
# one light-to-dark ramp per model, in the model's own hue (Qwen blue, Gemma red, Llama gold)
RAMPS = {'qwen7b': ('#A9C4E4', '#5B8CC6', ADDRESS), 'gemma4b': ('#E3AFAA', '#CC6E66', '#B8403A'),
         'llama8b': ('#EBD39C', '#DDB35C', '#B8861E')}
LENGTHS = ('4', '6', '8')
MARK = {'qwen7b': 'o', 'gemma4b': '^', 'llama8b': 's'}
KEY = ('#C9CED3', '#8E969D', '#4A5259')  # neutral ramp for the shared legend

fig = plt.figure(figsize=(7.0, 2.3))
for j, (key, name) in enumerate(MODELS):
    ax = fig.add_axes([.07 + j * .315, .2, .26, .62])
    ax.set_title(rf'$\mathbf{{{"abc"[j]}}}$  ' + name, fontsize=7.6, loc='left', pad=3)
    for n, color in zip(LENGTHS, RAMPS[key]):
        d = L['models'][key]['lengths'][n]
        pos = sorted(d['ordinal_vector'], key=int)
        x = np.array([int(p) + 1 for p in pos])
        y = np.array([100 * d['ordinal_vector'][p]['mean'] for p in pos])
        lo = np.array([100 * d['ordinal_vector'][p]['ci'][0] for p in pos])
        hi = np.array([100 * d['ordinal_vector'][p]['ci'][1] for p in pos])
        ax.fill_between(x, lo, hi, color=color, alpha=.18, lw=0)
        ax.plot(x, y, color=color, lw=1.5, marker=MARK[key], ms=2.8, label=f'{n} facts')
        ceil = [100 * d['ceiling'][p]['mean'] for p in pos]
        ax.plot(x, ceil, color=GREY, lw=.8, ls=(0, (2, 2)))
    ax.set(ylim=(-3, 103), xlim=(.6, 8.4), xticks=range(1, 9), yticks=[0, 50, 100])
    ax.set_xlabel("designated fact's order of mention", fontsize=7.2, labelpad=1)
    if j == 0:
        ax.set_ylabel('Designated answers (%)', fontsize=7.2)
fig.legend(handles=[Line2D([0], [0], color=c, lw=1.5, marker='o', ms=2.8, label=f'{n} facts') for n, c in zip(LENGTHS, KEY)]
           + [Line2D([0], [0], color=GREY, lw=.8, ls=(0, (2, 2)), label="ceiling")],
           loc='upper center', bbox_to_anchor=(.5, 1.02), ncol=4, fontsize=6.8, handlelength=1.8, columnspacing=1.2)
floor_text(fig, 5.500)
check(fig, 'longer-lists')
save(fig, 'longer-lists', dpi=280, bbox_inches='tight', pad_inches=.03)
print('SAVED longer-lists')
