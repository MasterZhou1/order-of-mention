"""Fig. 5: the three outcomes through depth, with the ceiling, and the stage of every block.

Usage: python plots/depth.py results/depth_*.json --out figures/depth.pdf
"""
import argparse
import json

import matplotlib.pyplot as plt

COLORS = {"shared_address": "#377EB8", "whole_state_address": "#CB7838", "content": "#4DAF4A", "none": "white"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = [json.load(open(path)) for path in args.results]
    fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 4.2), squeeze=False,
                             gridspec_kw=dict(height_ratios=[4, 0.5]))
    for col, run in enumerate(runs):
        blocks = run["per_block"]
        x = [b["relative_depth"] for b in blocks]
        ax = axes[0, col]
        for key, label, style in (("ordinal", "ordinal vector", "-"), ("donor_other", "donor state", "-"),
                                  ("donor_first_token", "donor's first token", "--"), ("ceiling", "ceiling", ":")):
            ax.plot(x, [100 * b[key] for b in blocks], style, label=label)
        if run["depth_window"]:
            last = blocks[-1]["block"]                      # the last block is always evaluated
            lo, hi = (b / last for b in run["depth_window"])
            ax.axvspan(lo, hi, color="0.9", zorder=0)
        ax.set_title(run["model"].split("/")[-1])
        ax.set_ylim(0, 100)
        ax.set_ylabel("designated answers (%)" if col == 0 else "")
        stages = axes[1, col]
        for b in blocks:
            stages.axvspan(b["relative_depth"] - .5 / len(blocks), b["relative_depth"] + .5 / len(blocks),
                           color=COLORS[b["stage"]])
        stages.set_yticks([])
        stages.set_xlabel("relative depth (block / last block)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
