"""App. D.3, Fig. 9 and App. E.1 (across relations): the same ordinal vector acts under every relation.

Six relations, each with its own contents: eats a dish, owns an animal, visited a city, works as a job, plays an
instrument, plays a sport. Per relation: 16 calibration word sets (all 16 arrangement x assignment cells) and 64 test
word sets (arrangements and content assignments 0 and 2); a word set keeps its four people across relations. Each
test receiver is edited once with the ordinal vector of every relation (native length), with the "eats" vector
rescaled to the receiving relation's own length, with two random directions of that length, and with the ceiling.
Geometry: fact-address and person templates (leave-one-word-set-out R^2); cosines between ordinal vectors of
relation pairs versus mismatched pairs of orders; principal angles between the relations' address subspaces.
"""
import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, steering as S
from relgeom.capture import stack_by_position


def run(lm, args):
    per = {}
    for rel in M.SIX_RELATIONS:
        sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 4, args.seed, full_names=True,
                           contents=M.CONTENT_POOLS[rel])
        cal = [M.fact_list(ws, a, o, relation=rel) for ws in P.by_split(sets, "calibration") for a in range(4) for o in range(4)]
        test = [M.fact_list(ws, a, o, relation=rel) for ws in P.by_split(sets, "test") for a in (0, 2) for o in (0, 2)]
        h_cal = P.states_at(lm, cal, [args.block], args.batch_size)
        per[rel] = dict(cal=cal, test=test, h_cal=h_cal, vectors=S.mean_changes(cal, h_cal),
                        h_test=P.states_at(lm, test, [args.block], args.batch_size))
    records = []
    for rel, d in per.items():
        receivers = []
        for ctx, h in zip(d["test"], d["h_test"]):
            for s in range(4):
                for t in range(4):
                    if s == t:
                        continue
                    key, designated = S.by_position(ctx, s, t), ctx.queries[t].answer
                    own = d["vectors"][key]
                    arms = [("unedited", S.NONE, designated), ("ceiling", S.replace(h[t, 0]), designated)]
                    arms += [(f"vector:{src}", S.add(per[src]["vectors"][key]), designated) for src in per]
                    arms.append(("eats_rescaled", S.add(S.rescale(per["eats"]["vectors"][key], np.linalg.norm(own))),
                                 designated))
                    arms += [(f"random_{i}", S.add(S.random_direction(own, (rel, key, i))), designated) for i in range(2)]
                    receivers.append(dict(ctx=ctx, q=s, target=t, relation=rel, arms=arms))
        records += P.run_edits(lm, receivers, args.block, by_content=args.by_content, strip_article=True)
    geometry, bases, templates = {}, {}, {}
    for rel, d in per.items():
        H = stack_by_position(d["cal"], d["h_cal"], 0).astype(np.float64)
        Y = G.center(H)
        groups = [c.word_set for c in d["cal"]]
        persons = np.array([[q.labels["person"] for q in sorted(c.queries, key=lambda q: q.position)] for c in d["cal"]])
        templates[rel] = Y.mean(axis=0)
        bases[rel] = G.address_basis(templates[rel])
        geometry[rel] = dict(lowo_fact_r2=G.lowo_r2(Y, np.tile(np.arange(4), (len(Y), 1)), groups),
                             lowo_person_r2=G.lowo_r2(Y, persons, groups))
    pairs = [(p, q) for p in range(4) for q in range(p + 1, 4)]
    rels = list(per)
    cross = {}
    for i, r1 in enumerate(rels):
        for r2 in rels[i + 1:]:
            a, b = templates[r1], templates[r2]
            cross[f"{r1}|{r2}"] = dict(
                same_pair_cosine=float(np.mean([G.cosine(a[q] - a[p], b[q] - b[p]) for p, q in pairs])),
                mismatched_cosine=float(np.mean([G.cosine(a[q] - a[p], b[y] - b[x]) for p, q in pairs
                                                 for x, y in pairs if (x, y) != (p, q)])),
                principal_cosines=G.principal_cosines(bases[r1], bases[r2]).tolist())
    return P.report(args, records, by=("relation", "condition"), experiment="relation_generality",
                    geometry=geometry, cross_relation=cross)


if __name__ == "__main__":
    P.cli(__doc__, run)
