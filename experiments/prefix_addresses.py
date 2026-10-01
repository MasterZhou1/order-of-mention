"""Sec. 6.2, App. E.3: each fact address is fixed once its fact is read.

The eight-fact contexts of longer_lists.py (same word sets: 16 calibration + 64 test, 8 contexts each following a
Latin square: arrangement 0-7, content assignment = word-set index mod 8) are cut after their first k facts,
k = 2..8, and asked about each of their k facts. Templates a^(k)_p are computed within each prefix length from all
80 word sets and compared only through differences (which do not depend on centering).
  * stability: cos(a^(k)_p - a^(k)_1, a^(8)_p - a^(8)_1) and the length ratio, for facts before the prefix's last;
  * gaps g(k, q) = ||a^(k)_{q+1} - a^(k)_q|| / ||a^(k)_2 - a^(k)_1||;
  * leave-one-prefix-length-out prediction of the gaps, indexed by order of mention (q) or by distance from the
    end of the list (k - 1 - q), on gaps q = 1..k-2 (R^2).
"""
from collections import defaultdict

import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P
from relgeom.capture import stack_by_position


def loko_r2(gaps, index):
    """Predict each held-out prefix length's gaps by the mean of other lengths' gaps with the same index."""
    obs, pred = [], []
    for k in gaps:
        pool = defaultdict(list)
        for k2, row in gaps.items():
            if k2 != k:
                for q in range(1, k2 - 1):
                    pool[index(k2, q)].append(row[q])
        for q in range(1, k - 1):
            if pool[index(k, q)]:
                obs.append(gaps[k][q])
                pred.append(np.mean(pool[index(k, q)]))
    obs, pred = np.array(obs), np.array(pred)
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 8, args.seed)
    full = [M.fact_list(ws, i % 8, o, n=8) for i, ws in enumerate(sets) for o in range(8)]
    templates = {}
    for k in range(2, 9):
        ctxs = [M.prefix(c, k) for c in full]
        Y = G.center(stack_by_position(ctxs, P.states_at(lm, ctxs, [args.block], args.batch_size), 0).astype(np.float64))
        groups = np.array([c.word_set for c in ctxs])
        templates[k] = np.mean([Y[groups == g].mean(axis=0) for g in np.unique(groups)], axis=0)
    ref = templates[8][1:] - templates[8][0]
    stability, gaps = {}, {}
    for k, a in templates.items():
        u = np.linalg.norm(np.diff(a, axis=0), axis=1)
        gaps[k] = (u / u[0]).tolist()
        if k < 8:
            v = a[1:] - a[0]
            cos = [G.cosine(v[p - 1], ref[p - 1]) for p in range(1, k)]
            ratio = [float(np.linalg.norm(v[p - 1]) / np.linalg.norm(ref[p - 1])) for p in range(1, k)]
            stability[k] = dict(interior_cosine=cos[:-1], interior_length_ratio=ratio[:-1], last_cosine=cos[-1])
    interior = [c for s in stability.values() for c in s["interior_cosine"]]
    ratios = [r for s in stability.values() for r in s["interior_length_ratio"]]
    return dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
                experiment="prefix_addresses", stability=stability, gaps=gaps,
                median_interior_cosine=float(np.median(interior)), min_interior_cosine=float(np.min(interior)),
                max_length_change=float(np.max(np.abs(np.array(ratios) - 1))),
                gap_r2_by_order_of_mention=loko_r2(gaps, lambda k, q: q),
                gap_r2_by_distance_from_end=loko_r2(gaps, lambda k, q: k - 1 - q))


if __name__ == "__main__":
    P.cli(__doc__, run)
