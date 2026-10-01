"""App. A: the rule that fixes a model's main block on separate block-selection word sets.

Ordinal vectors are calibrated on 8 word sets; 8 further block-selection word sets are edited in arrangements
{0, 2} x content assignments {0, 2} with the person changes 0->1, 1->0, 2->3, 3->2. For every second block, the
criterion is the ordinal vector's gain over the unedited answer, minus the random direction's gain, minus any rise
in invalid answers; the rule finds the best block and takes the earliest block within two points of it. If no block
reaches two points, the block nearest Qwen2.5-7B's relative depth (20 of 28) is used.
"""
import numpy as np

from relgeom import materials as M, pipeline as P, scoring, steering as S

PAIRS = ((0, 1), (1, 0), (2, 3), (3, 2))


def choose(criteria, n_blocks):
    grid = sorted(criteria)
    best = max(criteria.values())
    if best >= .02:
        return min(b for b in grid if criteria[b] >= best - .02), "best_within_2pp_earliest"
    target = 20 / 28 * n_blocks
    return min(grid, key=lambda b: (abs(b - target), b)), "fallback_relative_depth"


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": 8, "selection": 8}, 4, args.seed, multiword=True)
    cal = [M.fact_list(ws, a, o) for ws in P.by_split(sets, "calibration") for a in range(4) for o in range(4)]
    sel = [M.fact_list(ws, a, o) for ws in P.by_split(sets, "selection") for a in (0, 2) for o in (0, 2)]
    grid = list(range(0, lm.n_blocks, 2))
    h_cal = P.states_at(lm, cal, grid, args.batch_size)
    base = P.run_edits(lm, [dict(ctx=c, q=s, target=t, arms=[("unedited", S.NONE, c.queries[t].answer)])
                            for c in sel for s, t in PAIRS], grid[0], by_content=args.by_content)
    unedited = {(r["word_set"], r["case"]): r for r in base}
    criteria, profile = {}, {}
    for i, b in enumerate(grid):
        v = S.mean_changes(cal, h_cal, block_index=i)
        receivers = []
        for c in sel:
            for s, t in PAIRS:
                key = S.by_position(c, s, t)
                receivers.append(dict(ctx=c, q=s, target=t, arms=[
                    ("ordinal", S.add(v[key]), c.queries[t].answer),
                    ("random", S.add(S.random_direction(v[key], (key, "random"))), c.queries[t].answer)]))
        records = P.run_edits(lm, receivers, b, by_content=args.by_content)
        for r in records:                                          # paired with the unedited answer of the case
            clean = unedited[r["word_set"], r["case"]]
            r["gain"], r["invalid_gain"] = r["designated"] - clean["designated"], r["invalid"] - clean["invalid"]
        rate = {cond: {m: scoring.bootstrap([r for r in records if r["condition"] == cond], m)["mean"]
                       for m in ("gain", "invalid_gain", "designated")} for cond in ("ordinal", "random")}
        criteria[b] = rate["ordinal"]["gain"] - rate["random"]["gain"] - max(rate["ordinal"]["invalid_gain"], 0)
        profile[b] = dict(rate, criterion=criteria[b])
        print(b, round(criteria[b], 3))
    block, rule = choose(criteria, lm.n_blocks)
    return dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, experiment="select_block",
                block=block, rule=rule, relative_depth=block / (lm.n_blocks - 1), profile=profile,
                unedited_accuracy=float(np.mean([r["original"] for r in base])))


if __name__ == "__main__":
    P.cli(__doc__, run, lambda p: p.set_defaults(seed=2, block=-1))
