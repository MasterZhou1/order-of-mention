"""App. D.4, Fig. 10: the ordinal vector follows order of mention, not token distance.

Test contexts of 64 word sets are rendered under eight length conditions with the same arrangements (all four) and
content assignment (0): uniform facts; one fact lengthened by a neutral opening clause (four conditions); facts 2
and 4 lengthened; filler lines between facts; a filler paragraph before the question. Ordinal vectors are calibrated
on uniform contexts of 16 other word sets (primary) or on randomly lengthened ones (secondary).
For an edit p -> p' on a question about fact s, with d(.) the token distance of a fact's content from the last prompt
token and d_ref[p] the mean distance of order p in uniform calibration contexts, the absolute-distance rule predicts
the fact nearest d_ref[p'], the relative rule the fact nearest d(s) + d_ref[p'] - d_ref[p] (ties: nearer the
question). Contrast, on edits where the distance-predicted fact differs from s and from the designated fact:
answers naming the designated fact minus answers naming the distance-predicted fact.
"""
import random

import numpy as np

from relgeom import geometry as G, materials as M, pipeline as P, scoring, steering as S
from relgeom.capture import stack_by_position


def distances(lm, ctx, q):
    """Token distance from the last prompt token to the last token of each fact's content, by person."""
    prompt = lm.render(ctx.body(q))
    enc = lm.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    last, base = len(enc["input_ids"]) - 1, prompt.index(ctx.facts)
    out = {}
    for query in ctx.queries:
        start = base + ctx.facts.index(f"eats the {query.answer}.") + len("eats the ")
        end = start + len(query.answer)
        token = max(i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end and b > start)
        out[query.labels["person"]] = last - token
    return out


def nearest(dist, value):
    return min(dist, key=lambda k: (abs(dist[k] - value), dist[k]))


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 16, "test": 64}, 4, args.seed, full_names=True, multiword=True)
    cal_sets = P.by_split(sets, "calibration")
    uniform = [M.lengthened(ws, a, o, "uniform") for ws in cal_sets for a in range(4) for o in range(4)]
    mixed = [M.lengthened(ws, a, o, "mixed", random.Random(f"{args.seed}|{ws.id}|{a}|{o}"))
             for ws in cal_sets for a in range(4) for o in range(4)]
    h_uniform = P.states_at(lm, uniform, [args.block], args.batch_size)
    v_uniform = S.mean_changes(uniform, h_uniform)
    v_mixed = S.mean_changes(mixed, P.states_at(lm, mixed, [args.block], args.batch_size))
    ref = np.zeros(4)
    for ctx in uniform:
        d = distances(lm, ctx, 0)
        for q in ctx.queries:
            ref[q.position] += d[q.labels["person"]] / len(uniform)
    test = [M.lengthened(ws, 0, o, cond) for ws in P.by_split(sets, "test") for cond in M.LENGTH_CONDITIONS
            for o in range(4)]
    h_test = P.states_at(lm, test, [args.block], args.batch_size)
    receivers, token_labels = [], []
    for ctx, h in zip(test, h_test):
        for s in range(4):                          # question index = person index in fact_list contexts
            d = distances(lm, ctx, s)
            if s == 0:                              # token-distance template label of every question (App. D.4)
                token_labels.append([min(range(4), key=lambda p: (abs(d[q.labels["person"]] - ref[p]), -p))
                                     for q in sorted(ctx.queries, key=lambda q: q.position)])
            for t in range(4):
                if t == s:
                    continue
                ps, pt = S.by_position(ctx, s, t)
                designated = ctx.queries[t].answer
                arms = [("unedited", S.NONE, designated), ("ordinal_uniform", S.add(v_uniform[ps, pt]), designated),
                        ("ordinal_mixed", S.add(v_mixed[ps, pt]), designated),
                        ("random", S.add(S.random_direction(v_uniform[ps, pt], (ps, pt))), designated),
                        ("ceiling", S.replace(h[t, 0]), designated)]
                rules = dict(absolute=nearest(d, ref[pt]), relative=nearest(d, d[s] + ref[pt] - ref[ps]))
                receivers.append(dict(ctx=ctx, q=s, target=t, length_condition=ctx.meta["condition"], arms=arms,
                                      **rules, **{f"{k}_answer": ctx.queries[v].answer for k, v in rules.items()}))
    records = P.run_edits(lm, receivers, args.block, by_content=args.by_content)
    contrasts = {}
    for rule in ("absolute", "relative"):
        for condition in ("ordinal_uniform", "ordinal_mixed"):
            rows = [dict(word_set=r["word_set"], d=r["designated"] - int(
                        scoring.normalize(r["answer"]) == scoring.normalize(r[f"{rule}_answer"])))
                    for r in records if r["condition"] == condition and r[rule] not in (r["q"], r["target"])]
            contrasts[f"{condition}/{rule}"] = scoring.bootstrap(rows, "d")
    # Held-out query states: fact-address template versus the token-distance template (both fit on uniform
    # calibration contexts and scored on the test contexts).
    Yu = G.center(stack_by_position(uniform, h_uniform, 0).astype(np.float64))
    tmpl = G.templates(Yu, np.tile(np.arange(4), (len(uniform), 1)))
    Yt = G.center(stack_by_position(test, h_test, 0).astype(np.float64))
    geometry = dict(fact_address_r2=G.r2(Yt, np.tile(np.arange(4), (len(test), 1)), tmpl),
                    token_distance_r2=G.r2(Yt, np.array(token_labels), tmpl), reference_distance=ref.tolist())
    return P.report(args, records, by=("length_condition", "condition"), experiment="token_distance",
                    contrasts=contrasts, geometry=geometry)


if __name__ == "__main__":
    P.cli(__doc__, run)
