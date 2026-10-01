"""App. E.4, Table 3, Lemma E.1, Fig. 12: person and relation changes do not add; fact addresses predict how.

Contexts state four facts about two people and two relations; question cells (person e, relation r) -> 2e + r.
Joint versus summed changes: 16 calibration word sets in four arrangements (grouped by person or by relation, and
their reversals) x 4 content assignments; each change is the calibration mean for its pair of cells within the same
arrangement. Test: 64 word sets x 4 arrangements x 4 content assignments x 4 starting cells, scored against the
joint target (cell XOR 3): v_J; the same-start sum v_E + v_R; the sum rescaled to ||v_J||; random; the ceiling.
Interaction (Lemma E.1): templates fit on calibration contexts in the 12 even permutations of the facts; held-out
contexts in the 12 odd permutations plus the original order; cosine of I = y11 - y10 - y01 + y00 with
a_p(11) - a_p(10) - a_p(01) + a_p(00), at the main and the last block, against 23 relabelings of the orders.
"""
from itertools import permutations

import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, steering as S
from relgeom.capture import stack_by_position


def parity(order):
    return sum(order[i] > order[j] for i in range(4) for j in range(i + 1, 4)) % 2


def by_cells(ctx, s, t):
    return ctx.meta["layout"], s, t


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 4, args.seed, full_names=True, multiword=True)
    cal_sets, test_sets = P.by_split(sets, "calibration"), P.by_split(sets, "test")
    layouts = list(M.PERSON_RELATION_LAYOUTS.values())
    cal = [M.person_relation(ws, a, lay) for ws in cal_sets for a in range(4) for lay in layouts]
    v = S.mean_changes(cal, P.states_at(lm, cal, [args.block], args.batch_size), key=by_cells)
    test = [M.person_relation(ws, a, lay) for ws in test_sets for a in range(4) for lay in layouts]
    receivers, geometry = [], []
    for ctx, h in zip(test, P.states_at(lm, test, [args.block], args.batch_size)):
        lay = ctx.meta["layout"]
        for s in range(4):
            t, designated = s ^ 3, ctx.queries[s ^ 3].answer
            joint, person, relation = v[lay, s, t], v[lay, s, s ^ 2], v[lay, s, s ^ 1]
            summed = person + relation
            receivers.append(dict(ctx=ctx, q=s, target=t, layout=lay, arms=[
                ("unedited", S.NONE, designated), ("joint", S.add(joint), designated),
                ("sum", S.add(summed), designated),
                ("sum_rescaled", S.add(S.rescale(summed, np.linalg.norm(joint))), designated),
                ("random", S.add(S.random_direction(joint, (lay, s))), designated),
                ("ceiling", S.replace(h[t, 0]), designated)]))
    for lay in layouts:
        for s in range(4):
            j, e, r = v[lay, s, s ^ 3], v[lay, s, s ^ 2], v[lay, s, s ^ 1]
            geometry.append(dict(person_relation_cosine=G.cosine(e, r), sum_joint_cosine=G.cosine(e + r, j),
                                 interaction_share=float(np.linalg.norm(j - e - r) / np.linalg.norm(j))))
    records = P.run_edits(lm, receivers, args.block, by_content=args.by_content)
    # Lemma E.1: the measured interaction against its fact-address prediction.
    even = [o for o in permutations(range(4)) if parity(o) == 0]
    odd = [o for o in permutations(range(4)) if parity(o) == 1] + [(0, 1, 2, 3)]
    fit = [M.person_relation(ws, a, o) for ws in cal_sets for o in even for a in range(4)]
    held = [M.person_relation(ws, a, o) for ws in test_sets for o in odd for a in range(4)]
    blocks = [args.block, lm.n_blocks - 1]
    h_fit, h_held = P.states_at(lm, fit, blocks, args.batch_size), P.states_at(lm, held, blocks, args.batch_size)
    interaction = {}
    for i, b in enumerate(blocks):
        a = G.center(stack_by_position(fit, h_fit, i).astype(np.float64)).mean(axis=0)
        Y = G.center(np.stack([h[:, i] for h in h_held]).astype(np.float64))         # indexed by cell
        positions = np.array([[q.position for q in ctx.queries] for ctx in held])
        result = G.interaction_cosines(Y, positions, a)
        interaction[str(b)] = {k: result[k] for k in ("cosine", "null", "contexts")}
    return P.report(args, records, experiment="person_relation", interaction=interaction,
                    change_geometry={k: float(np.mean([g[k] for g in geometry])) for k in geometry[0]})


if __name__ == "__main__":
    P.cli(__doc__, run)
