"""Sec. 3, Fig. 2c-d, App. B: invariance and reversal of the query change in two-fact contexts (Prop. 3.4).

Each context states two facts ("Ruby eats the tuna. Simon eats the guava.") in the active or passive form. Word
sets 0-7 calibrate (content set 0), word sets 8-71 are tested (content set 1). For arrangement o (o = 0: person A's
fact first), v_o is the mean of h(C, q_B) - h(C, q_A) over both content assignments and the calibration word sets.
Reported: the mean cosine of each test word set's change with v_o of the same arrangement (new names), and
cos(v_0, v_1) (reversal, predicted -1); plus the three display axes of Fig. 2c.
"""
import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, vocab

CALIBRATION, TEST = range(8), range(8, len(vocab.TWO_FACT_NAMES))


def changes(lm, args, form, sets, bank):
    """(word set, arrangement) -> mean over content assignments of h(q_B) - h(q_A); and the context offsets."""
    ctxs = [M.two_fact(i, bank, form, a, o) for i in sets for o in range(2) for a in range(2)]
    h = np.stack(P.states_at(lm, ctxs, [args.block], args.batch_size))[:, :, 0].astype(np.float64)
    delta = (h[:, 1] - h[:, 0]).reshape(len(sets), 2, 2, -1).mean(axis=2)      # (sets, arrangement, d)
    return delta, h.mean(axis=1), h


def run(lm, args):
    result = dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
                  experiment="invariance_reversal", forms={})
    for form in ("active", "passive"):
        cal, offsets, _ = changes(lm, args, form, CALIBRATION, 0)
        test, _, h_test = changes(lm, args, form, TEST, 1)
        v = cal.mean(axis=0)                                                    # v_0, v_1
        new_names = [[G.cosine(t[o], v[o]) for o in range(2)] for t in test]
        axes = G.two_fact_axes(v[0], v[1], offsets)
        scale = np.sqrt(np.mean(np.sum(v ** 2, axis=1)))
        plane = axes[:2]
        entry = dict(new_name_cosine=float(np.mean(new_names)), new_name_cosine_min=float(np.min(new_names)),
                     reversal_cosine=G.cosine(v[0], v[1]),
                     plane_share_of_test_changes=float(np.sum((test.reshape(-1, test.shape[-1]) @ plane.T) ** 2)
                                                       / np.sum(test ** 2)))
        if form == "active":
            # Fig. 2c: test states (centered by the mean calibration offset) on the three axes, units of ||v||.
            coords = (h_test - offsets.mean(axis=0)) @ axes.T / scale
            entry["display"] = dict(axes_units="||v|| (RMS of v_0, v_1)", test_state_coordinates=coords.round(4))
        result["forms"][form] = entry
    return result


if __name__ == "__main__":
    P.cli(__doc__, run)
