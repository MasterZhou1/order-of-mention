"""Table 1A (Sec. 5, Prop. 3.6): an ordinal vector from other contexts selects the designated fact.

Ordinal vectors v^{p->p'} are calibrated on 8 word sets (all four arrangements x four content assignments) and
added once to question q_C(p) of contexts built from 64 new word sets. Each test word set crosses four
arrangements, four content assignments and twelve directed changes: 192 cases per word set. Controls: two
random directions of the same length, and the ceiling (the receiver's own state for the designated question).
"""
from relgeom import materials as M, pipeline as P, steering as S


def contexts(sets):
    return [M.fact_list(ws, a, o) for ws in sets for a in range(4) for o in range(4)]


def calibrate(lm, args, sets):
    cal = contexts(P.by_split(sets, "calibration"))
    return S.mean_changes(cal, P.states_at(lm, cal, [args.block], args.batch_size))


def run(lm, args):
    sets = P.word_sets(lm, {"calibration": args.calibration, "test": args.test}, 4, args.seed, multiword=True)
    vectors = calibrate(lm, args, sets)
    test = contexts(P.by_split(sets, "test"))
    receivers = P.transfer_receivers(test, P.states_at(lm, test, [args.block], args.batch_size), vectors)
    records = P.run_edits(lm, receivers, args.block, by_content=args.by_content)
    return P.report(args, records, experiment="transfer", by_target_position=P.summarize(
        records, by=("condition", "target_position")))


def options(p):
    p.add_argument("--calibration", type=int, default=8)
    p.add_argument("--test", type=int, default=64)


if __name__ == "__main__":
    P.cli(__doc__, run, options)
