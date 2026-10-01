"""Sec. 6.2, Fig. 4b-d, Fig. 11, App. E.2: later addresses crowd together, and the first is easiest to reach.

Word sets have eight people and eight contents; a list of N = 4, 6, 8 facts uses the first N of each, so the three
lengths share names and contents. Calibration: 16 word sets in all N x N arrangement x assignment cells; all N(N-1)
directed pairs are calibrated. Test: 64 word sets, each in its N cyclic arrangements with content assignment
(word-set index mod N). Transfer is reported by the designated fact's order of mention (serial-position curve).
Geometry: templates a_1..a_N from the calibration word sets (equal weight per word set); link lengths and each
address's nearest-neighbor distance in units of ||a_2 - a_1||; top principal directions of the eight addresses.
Prop. 3.9: nearest-neighbor distance against transfer to that fact (pool several models with plots/spacing.py).
"""
import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, steering as S
from relgeom.capture import stack_by_position


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 8, args.seed)
    lengths = {}
    for n in (4, 6, 8):
        cal = [M.fact_list(ws, a, o, n=n) for ws in P.by_split(sets, "calibration") for a in range(n) for o in range(n)]
        test = [M.fact_list(ws, i % n, o, n=n) for i, ws in enumerate(P.by_split(sets, "test")) for o in range(n)]
        h_cal = P.states_at(lm, cal, [args.block], args.batch_size)
        receivers = P.transfer_receivers(test, P.states_at(lm, test, [args.block], args.batch_size),
                                         S.mean_changes(cal, h_cal))
        records = P.run_edits(lm, receivers, args.block, by_content=args.by_content)
        Y = G.center(stack_by_position(cal, h_cal, 0).astype(np.float64))
        groups = np.array([c.word_set for c in cal])
        a = np.mean([Y[groups == g].mean(axis=0) for g in np.unique(groups)], axis=0)
        unit = np.linalg.norm(a[1] - a[0])
        centred = a - a.mean(axis=0)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        by_target = P.summarize(records, by=("condition", "target_position"))
        lengths[str(n)] = dict(
            summary=P.summarize(records), by_target_position=by_target, **G.spacing(a),
            pca_coordinates=(centred @ vt[:2].T / unit).round(4), pca_variance=(sv[:2] ** 2 / np.sum(sv ** 2)).tolist(),
            spacing_points=[dict(position=p, nearest=G.spacing(a)["nearest"][p],
                                 transfer=by_target[f"ordinal/{p}"]["designated"]["mean"]) for p in range(n)])
    return dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
                experiment="longer_lists", lengths=lengths)


if __name__ == "__main__":
    P.cli(__doc__, run)
