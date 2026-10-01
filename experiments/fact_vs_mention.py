"""Sec. 4-5, App. C.2: the address belongs to the fact, not to the name (coordinated subjects).

One of four facts has two subjects ("Alice and Bob eat the apple. Carol eats the pear. ..."), so after it a person's
name order m exceeds the order of their fact f by one. Calibration: ordinary four-fact lists of 16 word sets (all 16
cells): fact-address templates a_p and the twelve ordinal vectors. Test: 64 word sets of five people and four
multiword contents, the coordinated fact at each of the four orders, content assignments 0 and 2, every question.
  * geometry: for questions with f != m and m <= 4, is y closer to a_f than to a_m in the address subspace?
  * behavior: the fact-order vector v^{f->t} (scored on the content of fact t) versus the name-order vector
    v^{m->t} (scored on the content its order predicts, the fact of the t-th name), on targets t after the
    coordinated fact; receivers whose two orders agree; random directions; the ceiling.
"""
import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, scoring, steering as S, vocab
from relgeom.capture import stack_by_position


def run(lm, args):
    exclude = [c for c in vocab.FOODS if " and " in f" {c} " or "eat" in c]
    sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 5, args.seed, n_contents=4, multiword=True, exclude=exclude)
    cal = [M.fact_list(ws, a, o, n=4) for ws in P.by_split(sets, "calibration") for a in range(4) for o in range(4)]
    h_cal = P.states_at(lm, cal, [args.block], args.batch_size)
    a = G.center(stack_by_position(cal, h_cal, 0).astype(np.float64)).mean(axis=0)
    v = S.mean_changes(cal, h_cal)
    rms = float(np.sqrt(np.mean([np.linalg.norm(x) ** 2 for x in v.values()])))
    basis = np.linalg.qr((a[1:] - a[0]).T)[0].T
    test = [M.coordinated(ws, c, b) for ws in P.by_split(sets, "test") for c in range(4) for b in (0, 2)]
    h_test = P.states_at(lm, test, [args.block], args.batch_size)
    closer, closer_full, advantage, receivers = [], [], [], []
    for ctx, h in zip(test, h_test):
        y = h[:, 0].astype(np.float64) - h[:, 0].mean(axis=0)
        c = ctx.meta["coordinated"]
        facts = [ctx.queries[m].labels["fact"] for m in range(5)]
        for m, q in enumerate(ctx.queries):
            f = facts[m]
            if f != m and m < 4:
                yb, A = y[m] @ basis.T, a @ basis.T
                closer.append(dict(word_set=ctx.word_set, x=int(np.sum((yb - A[f]) ** 2) < np.sum((yb - A[m]) ** 2))))
                closer_full.append(dict(word_set=ctx.word_set, x=int(np.sum((y[m] - a[f]) ** 2) < np.sum((y[m] - a[m]) ** 2))))
                advantage.append(dict(word_set=ctx.word_set, x=G.cosine(yb, A[f]) - G.cosine(yb, A[m])))
            content = [ctx.candidates[k] for k in range(4)]            # content of fact k
            arms = [("unedited", S.NONE, q.answer)]
            arms += [(f"random_{i}", S.add(S.random_direction(v[0, 1], (ctx.word_set, c, m, i), rms)), q.answer)
                     for i in range(2)]
            for t in range(4):
                if t != f:
                    arms.append((f"{'aligned' if f == m else 'fact_order'}:{t}", S.add(v[f, t]), content[t]))
                    arms.append((f"ceiling:{t}", S.replace(h[facts.index(t), 0]), content[t]))
                if f != m and m < 4 and t != m:
                    arms.append((f"name_order:{t}", S.add(v[m, t]), content[facts[t]]))
            receivers.append(dict(ctx=ctx, q=m, fact=f, mention=m, coordinated=c, arms=arms))
    records = []
    for r in P.run_edits(lm, receivers, args.block, by_content=args.by_content):
        condition, _, t = r["condition"].partition(":")
        records.append(dict(r, condition=condition, target=int(t) if t else None))
    # Receivers whose name order exceeds their fact order, targets after the coordinated fact: there the fact-order
    # and name-order readings name different contents.
    contrast = {cond: scoring.bootstrap([r for r in records if r["condition"] == cond and r["fact"] != r["mention"]
                                         and r["mention"] < 4 and r["target"] > r["coordinated"]], "designated")
                for cond in ("fact_order", "name_order")}
    return P.report(args, records, experiment="fact_vs_mention", contrast_after_coordinated_fact=contrast,
                    geometry=dict(closer_to_fact_address=scoring.bootstrap(closer, "x"),
                                  closer_full_space=scoring.bootstrap(closer_full, "x"),
                                  cosine_advantage=scoring.bootstrap(advantage, "x")))


if __name__ == "__main__":
    P.cli(__doc__, run)
