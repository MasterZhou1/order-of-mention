"""Sec. 4, App. C.3, Fig. 8: queries over two-place facts address a fact and an argument slot.

Each context states three facts of one two-place relation among six people ("Alice loves Bob. Carol loves David.
Emma loves Frank."), for loves, trusts, helps, visits, pays. A question asks for one argument given the other:
"Whom does Alice love?" (slot o) or "Who loves Bob?" (slot s). A change moves to another fact (same slot), to the
other argument of the same fact, or both. Calibration: 16 word sets x 3 arrangements x 2 bindings per relation;
test: 64 word sets, one context each. Edits: the calibration mean of the same relation, the fact+slot additive
prediction, the means of the other four relations, two random directions, and the ceiling. Templates: fact-by-slot,
additive (fact + slot), fact-only, slot-only, and person (which of the six names is the answer).
"""
import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, steering as S, vocab


def cell(q):
    return q.labels["fact"], q.labels["slot"]


def by_cells(ctx, s, t):
    return cell(ctx.queries[s]) + cell(ctx.queries[t])


def fit_templates(contexts, states):
    """Templates of the query-dependent state by (fact, slot), and the additive fact + slot fit."""
    Y = G.center(np.stack([h[:, 0] for h in states]).astype(np.float64))
    labels = np.empty((len(contexts), 6), dtype=object)
    for i, c in enumerate(contexts):
        for j, q in enumerate(c.queries):
            labels[i, j] = cell(q)
    T = G.templates(Y, labels)
    F = {p: np.mean([T[p, r] for r in "os"], axis=0) for p in range(3)}
    R = {r: np.mean([T[p, r] for p in range(3)], axis=0) for r in "os"}
    return Y, labels, T, {(p, r): F[p] + R[r] for p in range(3) for r in "os"}


def run(lm, args):
    sets, _ = M.draw_word_sets({"calibration": 16, "test": 64}, 6, args.seed)
    used = {n for ws in sets for n in ws.names}
    names_reserve = [n for n in vocab.FEMININE + vocab.MASCULINE if n not in used]
    M.qualify(sets, names_reserve, lm.first_token, field="names")
    cal_sets, test_sets = P.by_split(sets, "calibration"), P.by_split(sets, "test")
    per = {}
    for verb in M.TWO_PLACE:
        cal = [M.two_place(ws, verb, o, b) for ws in cal_sets for o in range(3) for b in range(2)]
        test = [M.two_place(ws, verb, i % 3, (i // 3) % 2) for i, ws in enumerate(test_sets)]
        h_cal, h_test = P.states_at(lm, cal, [args.block], args.batch_size), P.states_at(lm, test, [args.block], args.batch_size)
        per[verb] = dict(cal=cal, test=test, h_cal=h_cal, h_test=h_test, v=S.mean_changes(cal, h_cal, key=by_cells))
        per[verb]["Y"], per[verb]["labels"], per[verb]["T"], per[verb]["additive"] = fit_templates(cal, h_cal)
    records, geometry = [], {}
    for verb, d in per.items():
        receivers = []
        for ctx, h in zip(d["test"], d["h_test"]):
            for s in range(6):
                for t in range(6):
                    if s == t:
                        continue
                    (ps, rs), (pt, rt) = cell(ctx.queries[s]), cell(ctx.queries[t])
                    change = "slot" if ps == pt else "fact" if rs == rt else "joint"
                    key, designated = by_cells(ctx, s, t), ctx.queries[t].answer
                    own = d["v"][key]
                    arms = [("unedited", S.NONE, designated), ("within", S.add(own), designated),
                            ("additive", S.add(d["additive"][pt, rt] - d["additive"][ps, rs]), designated),
                            ("ceiling", S.replace(h[t, 0]), designated)]
                    arms += [(f"random_{i}", S.add(S.random_direction(own, (verb, key, i))), designated) for i in range(2)]
                    arms += [(f"cross:{u}", S.add(per[u]["v"][key]), designated) for u in per if u != verb]
                    receivers.append(dict(ctx=ctx, q=s, target=t, verb=verb, change=change, arms=arms))
        records += [dict(r, condition=r["condition"].split(":")[0]) for r in
                    P.run_edits(lm, receivers, args.block, by_content=args.by_content, strip_article=True)]
        Yt, lt, _, _ = fit_templates(d["test"], d["h_test"])
        fact = {p: np.mean([d["T"][p, r] for r in "os"], axis=0) for p in range(3)}
        slot = {r: np.mean([d["T"][p, r] for p in range(3)], axis=0) for r in "os"}
        person_c = np.array([[q.labels["person"] for q in c.queries] for c in d["cal"]])
        person_t = np.array([[q.labels["person"] for q in c.queries] for c in d["test"]])
        geometry[verb] = dict(
            fact_by_slot_r2=G.r2(Yt, lt, d["T"]), additive_r2=G.r2(Yt, lt, d["additive"]),
            fact_only_r2=G.r2(Yt, np.array([[x[0] for x in row] for row in lt]), fact),
            slot_only_r2=G.r2(Yt, np.array([[x[1] for x in row] for row in lt]), slot),
            person_r2=G.r2(Yt, person_t, G.templates(d["Y"], person_c)))
    cross = {}
    verbs = list(per)
    for i, u in enumerate(verbs):
        for w in verbs[i + 1:]:
            Tu, Tw = per[u]["T"], per[w]["T"]
            slot_cos = G.cosine(*[np.mean([T[p, "s"] - T[p, "o"] for p in range(3)], axis=0) for T in (Tu, Tw)])
            fact_cos = np.mean([G.cosine(*[np.mean([T[q, r] - T[p, r] for r in "os"], axis=0) for T in (Tu, Tw)])
                                for p in range(3) for q in range(p + 1, 3)])
            cross[f"{u}|{w}"] = dict(fact_change_cosine=float(fact_cos), slot_change_cosine=slot_cos)
    return P.report(args, records, by=("change", "condition"), experiment="two_place_relations",
                    geometry=geometry, cross_relation=cross)


if __name__ == "__main__":
    P.cli(__doc__, run)
