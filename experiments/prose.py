"""Sec. 5, App. G.3, Table 7: fact addresses do not depend on the list format.

Each prose context is one paragraph stating the four facts with four different verb frames (none uses "eats"),
two or three unrelated sentences between facts, and an optional opening and closing sentence. The ordinal vector
calibrated on the main-text lists (the 8 calibration word sets of steering_transfer.py) is applied unchanged to
paragraphs of 64 new word sets (all 16 cells), each also rendered as a list. To separate format from word sets,
16 further word sets are rendered both as paragraphs and as lists for calibration.
Geometry: leave-one-word-set-out fact-address R^2 on paragraphs; cosines between prose and list templates and
ordinal vectors; share of prose query-dependent states in the 3-dim span of the main-text fact addresses.
"""
import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, steering as S
from relgeom.capture import stack_by_position


def cells(sets, render):
    return [render(ws, a, o) for ws in sets for a in range(4) for o in range(4)]


def run(lm, args):
    main = P.word_sets(lm, {"calibration": 8, "test": 64}, 4, args.list_seed, multiword=True)
    exclude = {x for ws in main for x in ws.contents + ws.names}
    sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 4, args.seed, multiword=True, exclude=exclude)
    as_prose = lambda ws, a, o: M.prose(ws, a, o, args.seed)  # noqa: E731
    lists_main = cells(P.by_split(main, "calibration"), M.fact_list)
    lists_cal, prose_cal = (cells(P.by_split(sets, "calibration"), r) for r in (M.fact_list, as_prose))
    states = {name: P.states_at(lm, ctxs, [args.block], args.batch_size)
              for name, ctxs in (("list_main", lists_main), ("list_same_sets", lists_cal), ("prose", prose_cal))}
    vectors = {name: S.mean_changes(ctxs, states[name]) for name, ctxs in
               (("list_main", lists_main), ("list_same_sets", lists_cal), ("prose", prose_cal))}
    records = []
    for fmt, render in (("prose", as_prose), ("list", M.fact_list)):
        test = cells(P.by_split(sets, "test"), render)
        h_test = P.states_at(lm, test, [args.block], args.batch_size)
        receivers = []
        for ctx, h in zip(test, h_test):
            for s, t in P.all_pairs(ctx):
                key, designated = S.by_position(ctx, s, t), ctx.queries[t].answer
                arms = [("unedited", S.NONE, designated), ("ceiling", S.replace(h[t, 0]), designated)]
                arms += [(f"vector_{name}", S.add(v[key]), designated) for name, v in vectors.items()]
                arms += [(f"random_{i}", S.add(S.random_direction(vectors["list_main"][key], (key, i))), designated)
                         for i in range(2)]
                receivers.append(dict(ctx=ctx, q=s, target=t, format=fmt, target_position=key[1], arms=arms))
        records += P.run_edits(lm, receivers, args.block, by_content=args.by_content)
        if fmt == "prose":
            Yq = G.center(stack_by_position(test, h_test, 0).astype(np.float64))
    Y = {name: G.center(stack_by_position(ctxs, states[name], 0).astype(np.float64)) for name, ctxs in
         (("list_main", lists_main), ("list_same_sets", lists_cal), ("prose", prose_cal))}
    a = {name: y.mean(axis=0) for name, y in Y.items()}
    fact = np.tile(np.arange(4), (len(prose_cal), 1))
    span = G.address_basis(a["list_main"])
    X = Yq.reshape(-1, Yq.shape[-1])
    geometry = dict(
        prose_lowo_fact_r2=G.lowo_r2(Y["prose"], fact, [c.word_set for c in prose_cal]),
        prose_lowo_person_r2=G.lowo_r2(Y["prose"], np.array([[q.labels["person"] for q in sorted(
            c.queries, key=lambda q: q.position)] for c in prose_cal]), [c.word_set for c in prose_cal]),
        template_cosine={n: float(np.mean([G.cosine(a["prose"][p], a[n][p]) for p in range(4)])) for n in a},
        ordinal_vector_cosine={n: float(np.mean([G.cosine(vectors["prose"][k], vectors[n][k]) for k in vectors[n]]))
                               for n in vectors},
        prose_share_in_list_span=float(np.sum((X @ span.T) ** 2) / np.sum(X ** 2)),
        random_share=float(np.mean([np.sum((X @ R.T) ** 2) / np.sum(X ** 2) for R in G.random_bases(X.shape[1])])))
    return P.report(args, records, by=("format", "condition"), experiment="prose", geometry=geometry,
                    by_target_position=P.summarize(records, by=("format", "condition", "target_position")))


def options(p):
    p.add_argument("--list-seed", type=int, default=0, help="seed of the main-text lists (steering_transfer.py)")
    p.set_defaults(seed=1)


if __name__ == "__main__":
    P.cli(__doc__, run, options)
