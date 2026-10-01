"""Table 1B (Sec. 5, App. D.1): the facts define the ordinal vector.

Every direction is scaled to one common length L, the root-mean-square length of the twelve ordinal vectors
calibrated with the facts (8 word sets x 4 arrangements x 4 content assignments). On 64 further word sets
(arrangements 0 and 2, content assignments 0 and 2, person changes s -> s XOR 1; 1,024 cases per row) we compare:
  ordinal         the calibrated ordinal vector;
  question_alone  h(q_t) - h(q_s) computed on the receiver's two questions without any facts;
  names_same      the same difference after "Names, in order: ..." listing the people in the order of their facts;
  names_shuffled  the same list with the designated person and the person of the rival fact swapped (the rival
                  is the earliest order of mention other than the source and the designated one);
  random_0/1      Gaussian directions; ceiling: the receiver's own state for the designated question.
"""
import numpy as np

from relgeom import materials as M, pipeline as P, steering as S


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 8, "test": 64}, 4, args.seed, full_names=True, multiword=True)
    cal = [M.fact_list(ws, a, o) for ws in P.by_split(sets, "calibration") for a in range(4) for o in range(4)]
    vectors = S.mean_changes(cal, P.states_at(lm, cal, [args.block], args.batch_size))
    L = float(np.sqrt(np.mean([np.linalg.norm(v) ** 2 for v in vectors.values()])))
    test = [M.fact_list(ws, a, o) for ws in P.by_split(sets, "test") for o in (0, 2) for a in (0, 2)]
    h_test = P.states_at(lm, test, [args.block], args.batch_size)
    variants = []                                   # per test context: question-only and the two rosters
    for ctx in test:
        ws = next(w for w in sets if w.id == ctx.word_set)
        variants.append({"question_alone": M.no_facts(ctx)})
        for s in range(4):
            t = s ^ 1
            ps, pt = ctx.queries[s].position, ctx.queries[t].position
            rival = min(p for p in range(4) if p not in (ps, pt))
            same = [ws.names[k] for k in ctx.meta["order"]]
            shuffled = list(same)
            shuffled[pt], shuffled[rival] = shuffled[rival], shuffled[pt]
            variants[-1][f"names_same:{s}"] = M.no_facts(ctx, same)
            variants[-1][f"names_shuffled:{s}"] = M.no_facts(ctx, shuffled)
    flat = [c for v in variants for c in v.values()]
    h_flat = iter(P.states_at(lm, flat, [args.block], args.batch_size))
    h_var = [{k: next(h_flat) for k in v} for v in variants]
    receivers = []
    for ctx, h, hv in zip(test, h_test, h_var):
        for s in range(4):
            t, designated = s ^ 1, ctx.queries[s ^ 1].answer
            key = S.by_position(ctx, s, t)
            arms = [("unedited", S.NONE, designated), ("ordinal", S.add(S.rescale(vectors[key], L)), designated)]
            for name, states in (("question_alone", hv["question_alone"]), ("names_same", hv[f"names_same:{s}"]),
                                 ("names_shuffled", hv[f"names_shuffled:{s}"])):
                arms.append((name, S.add(S.rescale(states[t, 0].astype(np.float64) - states[s, 0], L)), designated))
            arms += [(f"random_{i}", S.add(S.random_direction(vectors[key], (key, i), L)), designated) for i in range(2)]
            arms.append(("ceiling", S.replace(h[t, 0]), designated))
            receivers.append(dict(ctx=ctx, q=s, target=t, arms=arms))
    records = P.run_edits(lm, receivers, args.block, by_content=args.by_content)
    return P.report(args, records, experiment="no_facts", common_length=L)


if __name__ == "__main__":
    P.cli(__doc__, run)
