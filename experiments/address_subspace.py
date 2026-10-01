"""Sec. 6.1, Fig. 4a, App. E.1: fact addresses span a shared low-rank subspace, and ordinal vectors compose.

Same contexts as order_of_mention.py, at the main block.
  * Share of held-out changes: A = span{a_p - a_1} (3-dim) from templates fit on the 16 calibration word sets;
    for each of the 128 test contexts and its six pairs of orders of mention, ||P_A Delta||^2 / ||Delta||^2,
    against the best 3-dim subspace of the held-out changes (in-sample bound) and 100 random 3-dim subspaces.
  * Composition (Cor. 3.7) on disjoint word sets: v_S1(p->p') + v_S2(p'->p'') against v_T(p->p''), 24 ordered
    triples, halves/quarters of the calibration word sets; null: 23 relabelings of the orders of mention.
  * Fig. 4a: templates fit on calibration word sets 0-7 and the held-out states of word sets 8-15 in A.
"""
import numpy as np

from relgeom import geometry as G, pipeline as P
from relgeom.capture import stack_by_position


def run(lm, args):
    cal, test = P.address_design(lm, args)
    Hc = stack_by_position(cal, P.states_at(lm, cal, [args.block], args.batch_size), 0).astype(np.float64)
    Ht = stack_by_position(test, P.states_at(lm, test, [args.block], args.batch_size), 0).astype(np.float64)
    groups = np.array([c.word_set for c in cal])
    share = G.subspace_share(Hc, Ht)
    share["relative_to_gap"] = (share["address"] - share["random"]) / (share["best"] - share["random"])
    composition = G.composition(Hc, groups)
    fit = np.isin(groups, sorted(set(groups))[:8])
    a = G.center(Hc[fit]).mean(axis=0)
    basis, centre, unit = G.address_basis(a), a.mean(axis=0), np.mean(
        [np.linalg.norm(a[p] - a[q]) for p in range(4) for q in range(p + 1, 4)])
    Y = G.center(Hc[~fit])
    figure = dict(templates=((a - centre) @ basis.T / unit).round(4), states=((Y - centre) @ basis.T / unit).round(4),
                  changes=np.stack([(Y[:, q] - Y[:, p]) @ basis.T / unit for p in range(4) for q in range(p + 1, 4)],
                                   axis=1).round(4), unit="mean distance between fact addresses")
    return dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
                experiment="address_subspace", share_of_heldout_changes=share, composition=composition,
                figure_4a=figure)


if __name__ == "__main__":
    P.cli(__doc__, run)
