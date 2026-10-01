"""Shared command-line options and the calibrate -> edit -> score loop used by the experiment scripts."""
import argparse
import json
from pathlib import Path

from . import materials as M
from .capture import capture_contexts
from .models import LM, main_block
from .scoring import outcome, summarize
from . import steering as S


def parser(doc):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--revision", default=None, help="checkpoint branch or commit (OLMo-2 development)")
    p.add_argument("--prompt-format", choices=("chat", "plain"), default="chat")
    p.add_argument("--block", type=int, default=None, help="edited block (default: the model's main block)")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="random seed for drawing word sets")
    p.add_argument("--by-content", action="store_true",
                   help="score the single stated content an answer names (base checkpoints answer in sentences)")
    p.add_argument("--save-records", action="store_true", help="also store every scored answer")
    return p


def setup(args):
    lm = LM(args.model, args.revision, args.prompt_format)
    if args.block is None:
        args.block = main_block(args.model, args.prompt_format)
        if args.block is None:
            raise SystemExit("No registered main block for this model; run experiments/select_block.py first")
    return lm


def word_sets(lm, counts, n_people, seed, **kwargs):
    """Draw the experiment's word sets and make each set's answers start with distinct tokens."""
    sets, reserve = M.draw_word_sets(counts, n_people, seed, **kwargs)
    cache = {}

    def first(content):
        if content not in cache:
            cache[content] = lm.first_token(content)
        return cache[content]

    return M.qualify(sets, reserve, first)


def by_split(sets, split):
    return [ws for ws in sets if ws.split == split]


def address_design(lm, args, calibration=16, test=64):
    """The four-fact design behind Table 1C, Figs. 3-5 and App. C.1/E.1: calibration word sets in all four
    arrangements x four content assignments; test word sets in one arrangement and content assignments 0 and 2."""
    sets = word_sets(lm, {"calibration": calibration, "test": test}, 4, args.seed, full_names=True, multiword=True)
    cal = [M.fact_list(ws, a, o) for ws in by_split(sets, "calibration") for a in range(4) for o in range(4)]
    tst = [M.fact_list(ws, a, 0) for ws in by_split(sets, "test") for a in (0, 2)]
    return cal, tst


def states_at(lm, contexts, blocks, batch_size=8):
    return capture_contexts(lm, contexts, blocks, batch_size)


def run_edits(lm, receivers, block, by_content=False, strip_article=False):
    """Generate and score. receivers: dicts with ctx, q (source question index), arms [(condition, edit,
    designated content)], and any extra fields to keep; one batched generation per receiver prompt."""
    records = []
    for i, r in enumerate(receivers):
        ctx, q = r["ctx"], r["q"]
        prompt = lm.encode(ctx.body(q))
        results = S.generate(lm, prompt, r.get("block", block), [edit for _, edit, _ in r["arms"]])
        extra = {k: v for k, v in r.items() if k not in ("ctx", "arms")}
        for (condition, _, designated), result in zip(r["arms"], results):
            score = outcome(result, ctx.candidates, designated, ctx.queries[q].answer, by_content, strip_article)
            records.append(dict(extra, word_set=ctx.word_set, condition=condition, case=r.get("case", i),
                                first_token=result["first_token"], **score))
    return records


def transfer_arms(vector, tag, designated, own_state=None, n_random=2, length=None):
    """Unedited, ordinal vector, equally long random directions, and the ceiling (receiver's own state for q')."""
    arms = [("unedited", S.NONE, designated), ("ordinal", S.add(vector), designated)]
    arms += [(f"random_{i}", S.add(S.random_direction(vector, (tag, i), length)), designated) for i in range(n_random)]
    if own_state is not None:
        arms.append(("ceiling", S.replace(own_state), designated))
    return arms


def all_pairs(ctx):
    n = len(ctx.queries)
    return [(s, t) for s in range(n) for t in range(n) if s != t]


def transfer_receivers(contexts, states, vectors, pairs=all_pairs, block_index=0, **arm_options):
    """One receiver per (test context, question s, designated question t): edit q_C(p) with v^{p -> p'}."""
    receivers = []
    for ctx, h in zip(contexts, states):
        for s, t in pairs(ctx):
            key = S.by_position(ctx, s, t)
            receivers.append(dict(ctx=ctx, q=s, target=t, source_position=key[0], target_position=key[1],
                                  arms=transfer_arms(vectors[key], key, ctx.queries[t].answer, h[t, block_index],
                                                     **arm_options)))
    return receivers


def report(args, records, by=("condition",), **extra):
    """Result dict: run settings, word-set bootstrap summaries, and optionally every scored answer."""
    out = dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
               summary=summarize(records, by=by), **extra)
    if getattr(args, "save_records", False):
        out["records"] = records
    return out


def cli(doc, run, options=None):
    """Parse the shared options (plus `options(parser)`), load the model, run, and write the JSON result."""
    p = parser(doc)
    if options:
        options(p)
    args = p.parse_args()
    save(args.out, run(setup(args), args))


def save(path, result):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))} | {"out": str(path)}))
