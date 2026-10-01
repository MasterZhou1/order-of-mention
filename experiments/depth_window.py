"""Sec. 7.1, Fig. 5, Table 1C, App. F.1: where the fact address is shared (the depth window).

Receivers: test word sets in one arrangement, content assignments 0 and 2, the four person changes s -> s XOR 1.
Donors: calibration word sets in the same arrangement and content assignment. At each block:
  ordinal     add the ordinal vector re-estimated at that block from the 16 calibration word sets;
  ceiling     replace the receiver's state by its own state for the designated question;
  donor_other replace it by a donor's state for the other question t (Table 1C, first row);
  donor_same  (main and last block) replace it by the donor's state for the receiver's own question s.
Outcomes: the receiver's designated fact (complete answer), the donor's complete answer, and the donor's first
answer token. A block is classified as content (donor first token >= 50%), else shared address (ordinal vector
>= 10%), else whole-state address (donor states >= 10%); blocks are classified one by one. The depth window spans
the first to the last block where the ordinal vector selects at least 10%.
Table 1C: `--blocks main`. Scale sweep and checkpoints: `--blocks even --donors 8 --receivers 32`.
"""
from relgeom import pipeline as P, scoring, steering as S
from relgeom.scoring import normalize


def blocks_for(mode, n_blocks, main):
    if mode == "all":
        return list(range(n_blocks))
    if mode == "even":
        return sorted(set(range(0, n_blocks, 2)) | {main, n_blocks - 1})
    return sorted({main, n_blocks - 1})


def stage(rates, gain=False):
    """Stage of one block; with gain=True the address thresholds apply to the gain over the unedited answer
    (base checkpoints can name the designated content without any edit)."""
    base = rates["unedited"] if gain else 0.0
    if rates["donor_first_token"] >= .5:
        return "content"
    if rates["ordinal"] - base >= .10:
        return "shared_address"
    if rates["donor_other"] - base >= .10:
        return "whole_state_address"
    return "none"


def run(lm, args):
    cal, test = P.address_design(lm, args)
    donors = sorted({c.word_set for c in cal})[:args.donors]
    test = [c for c in test if c.word_set in sorted({t.word_set for t in test})[:args.receivers]]
    blocks = blocks_for(args.blocks, lm.n_blocks, args.block)
    h_cal = P.states_at(lm, cal, blocks, args.batch_size)
    h_test = P.states_at(lm, test, blocks, args.batch_size)
    donor = {(c.word_set, c.meta["assignment"]): (c, h) for c, h in zip(cal, h_cal)
             if c.word_set in donors and c.meta["arrangement"] == 0}
    first = {}
    records = [dict(r, depth=None, donor=None) for r in P.run_edits(lm, [
        dict(ctx=ctx, q=s, target=s ^ 1, arms=[("unedited", S.NONE, ctx.queries[s ^ 1].answer)])
        for ctx in test for s in range(4)], args.block, by_content=args.by_content)]
    for i, b in enumerate(blocks):
        vectors = S.mean_changes(cal, h_cal, block_index=i)
        receivers, answers = [], []
        for ctx, h in zip(test, h_test):
            for s in range(4):
                t, designated = s ^ 1, ctx.queries[s ^ 1].answer
                arms = [("ordinal", S.add(vectors[S.by_position(ctx, s, t)]), designated),
                        ("ceiling", S.replace(h[t, i]), designated)]
                donor_answers = [None, None]
                for d in donors:
                    dctx, dh = donor[d, ctx.meta["assignment"]]
                    arms.append((f"donor_other:{d}", S.replace(dh[t, i]), designated))
                    donor_answers.append(dctx.queries[t].answer)
                    if b in (args.block, lm.n_blocks - 1):
                        arms.append((f"donor_same:{d}", S.replace(dh[s, i]), designated))
                        donor_answers.append(dctx.queries[s].answer)
                receivers.append(dict(ctx=ctx, q=s, target=t, arms=arms))
                answers += donor_answers
        for r, donor_answer in zip(P.run_edits(lm, receivers, b, by_content=args.by_content), answers):
            condition, _, d = r["condition"].partition(":")
            if donor_answer is not None and donor_answer not in first:
                first[donor_answer] = lm.first_token(donor_answer)
            records.append(dict(r, condition=condition, depth=b, donor=d or None,
                                donor_complete=int(donor_answer is not None
                                                   and normalize(r["answer"]) == normalize(donor_answer)),
                                donor_first_token=int(donor_answer is not None
                                                      and r["first_token"] == first[donor_answer])))
    summary = scoring.summarize(records, by=("depth", "condition"), two_way=lambda r: r["donor"] is not None,
                                metrics=("designated", "donor_complete", "donor_first_token", "invalid"))
    per_block = []
    for b in blocks:
        rates = dict(unedited=summary["None/unedited"]["designated"]["mean"],
                     ordinal=summary[f"{b}/ordinal"]["designated"]["mean"],
                     donor_other=summary[f"{b}/donor_other"]["designated"]["mean"],
                     donor_first_token=summary[f"{b}/donor_other"]["donor_first_token"]["mean"],
                     ceiling=summary[f"{b}/ceiling"]["designated"]["mean"])
        per_block.append(dict(block=b, relative_depth=b / (lm.n_blocks - 1), **rates,
                              stage=stage(rates), stage_gain=stage(rates, gain=True)))
    active = [e["block"] for e in per_block if e["ordinal"] >= .10]
    out = dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
               experiment="depth_window", donors=donors, per_block=per_block, summary=summary,
               depth_window=[min(active), max(active)] if active else None)
    if args.save_records:
        out["records"] = records
    return out


def options(p):
    p.add_argument("--blocks", choices=("all", "even", "main"), default="all")
    p.add_argument("--donors", type=int, default=16)
    p.add_argument("--receivers", type=int, default=64)


if __name__ == "__main__":
    P.cli(__doc__, run, options)
