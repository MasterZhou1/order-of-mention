"""App. D.2, Table 2: when facts move, the ordinal vector must follow their new orders of mention.

Contexts state four facts about two people and two relations ("Alice eats the apple. Alice carries the pear. Bob
eats the plum. Bob carries the peach."); a question asks about one cell (person, relation). Calibration: 8 word sets
in the four grouped arrangements (by person, by relation, and their reversals) x 4 content assignments. Test: 64
word sets in two held-out arrangements, (1,4,2,3) and (3,2,4,1) in person-major numbering, x 4 content assignments
x 4 starting cells, for person changes (cell XOR 2) and relation changes (cell XOR 1).
  original_pair  the person-major calibration mean for the same pair of questions;
  new_order      the calibration mean for the pair of orders of mention the two questions now occupy;
  recalibrated   estimated on calibration word sets rendered in the held-out arrangement;
  no_facts       the question-alone change, rescaled to the original-pair length; random: same length; ceiling.
"""
from relgeom import materials as M, pipeline as P, steering as S

HELD = ((0, 3, 1, 2), (2, 1, 3, 0))


def by_cells(ctx, s, t):
    return ctx.meta["layout"], s, t


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 8, "test": 64}, 4, args.seed, multiword=True)
    cal_sets = P.by_split(sets, "calibration")
    grouped = [M.person_relation(ws, a, lay) for ws in cal_sets for a in range(4) for lay in M.PERSON_RELATION_LAYOUTS.values()]
    held = [M.person_relation(ws, a, lay) for ws in cal_sets for a in range(4) for lay in HELD]
    h_grouped = P.states_at(lm, grouped, [args.block], args.batch_size)
    h_held = P.states_at(lm, held, [args.block], args.batch_size)
    person_major = [i for i, c in enumerate(grouped) if c.meta["layout"] == (0, 1, 2, 3)]
    original = S.mean_changes([grouped[i] for i in person_major], [h_grouped[i] for i in person_major], key=by_cells)
    new_order = S.mean_changes(grouped, h_grouped)
    recalibrated = S.mean_changes(held, h_held, key=by_cells)
    bare = [M.no_facts(grouped[i]) for i in person_major]
    question_only = S.mean_changes(bare, P.states_at(lm, bare, [args.block], args.batch_size), key=by_cells)
    test = [M.person_relation(ws, a, lay) for ws in P.by_split(sets, "test") for a in range(4) for lay in HELD]
    receivers = []
    for ctx, h in zip(test, P.states_at(lm, test, [args.block], args.batch_size)):
        for s in range(4):
            for change, t in (("person", s ^ 2), ("relation", s ^ 1)):
                designated, fixed = ctx.queries[t].answer, original[(0, 1, 2, 3), s, t]
                arms = [("unedited", S.NONE, designated), ("original_pair", S.add(fixed), designated),
                        ("new_order", S.add(new_order[S.by_position(ctx, s, t)]), designated),
                        ("recalibrated", S.add(recalibrated[by_cells(ctx, s, t)]), designated),
                        ("no_facts", S.add(S.rescale(question_only[(0, 1, 2, 3), s, t], float((fixed @ fixed) ** .5))),
                         designated),
                        ("random", S.add(S.random_direction(fixed, (s, t))), designated),
                        ("ceiling", S.replace(h[t, 0]), designated)]
                receivers.append(dict(ctx=ctx, q=s, target=t, change=change, layout=ctx.meta["layout"], arms=arms))
    records = P.run_edits(lm, receivers, args.block, by_content=args.by_content)
    return P.report(args, records, by=("change", "condition"), experiment="rearranged_facts")


if __name__ == "__main__":
    P.cli(__doc__, run)
