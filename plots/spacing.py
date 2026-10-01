"""Fig. 4c-d and Prop. 3.9: link lengths of the address chain, and nearest-neighbor distance against transfer.

Pools the longer_lists.py outputs of several models (3 models x (4 + 6 + 8) = 54 addresses); R^2 of a line after
removing each model's mean distance and mean rate.
Usage: python plots/spacing.py results/longer_lists_*.json --out figures/spacing.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relgeom.geometry import centered_fit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = [json.load(open(path)) for path in args.results]
    fig, (links, scatter) = plt.subplots(1, 2, figsize=(8, 3.2))
    xs, ys, models = [], [], []
    for run in runs:
        name = run["model"].split("/")[-1]
        for n, entry in run["lengths"].items():
            links.plot(range(1, len(entry["links"]) + 1), entry["links"], marker="o", label=f"{name}, {n} facts")
            for point in entry["spacing_points"]:
                xs.append(point["nearest"])
                ys.append(point["transfer"])
                models.append(name)
        scatter.scatter([x for x, m in zip(xs, models) if m == name], [100 * y for y, m in zip(ys, models) if m == name],
                        label=name, s=14)
    fit = centered_fit(xs, ys, models)
    links.set_xlabel("link (fact p to p+1)")
    links.set_ylabel("length / ||a_2 - a_1||")
    scatter.set_xlabel("distance to nearest fact address (unit ||a_2 - a_1||)")
    scatter.set_ylabel("transfer to that fact (%)")
    scatter.set_title(f"R^2 within models = {fit['r2']:.2f}")
    links.legend(frameon=False, fontsize=6)
    scatter.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
