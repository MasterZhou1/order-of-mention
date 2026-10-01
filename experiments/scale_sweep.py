"""App. G.1-G.2, Tables 5-6, Fig. 2d (small dots): the same tests across model scale and base weights.

For one model and prompt format: the main block by the rule of App. A (unless registered or given), the transfer
test of Table 1A, the cosines of Prop. 3.4, the query-state templates of App. C.1, and the depth curves at every
second block with 8 donors x 32 receivers. Base-versus-instruct rows use `--prompt-format plain`
([BOS] + instruction, facts and question + "\\nAnswer:"); Gemma-3-4B base answers in sentences, so add --by-content.
Models: Qwen/Qwen2.5-{1.5B,3B,14B,32B}-Instruct, Qwen/Qwen3-8B (non-thinking), meta-llama/Llama-3.2-3B-Instruct,
and the base weights Qwen/Qwen2.5-{1.5B,3B}, Qwen/Qwen3-8B-Base, google/gemma-3-4b-pt, meta-llama/Llama-3.2-3B.
"""
import copy

from experiments import depth_window, invariance_reversal, order_of_mention, select_block, steering_transfer
from relgeom import pipeline as P


def sweep(lm, args, depth=True, geometry=True):
    """Run the per-model battery; returns one dict (used by olmo_development.py as well)."""
    result = dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, n_blocks=lm.n_blocks)
    selection_args = copy.copy(args)
    selection_args.seed = 2                         # block-selection word sets (select_block.py default)
    selection = select_block.run(lm, selection_args)
    result["rule_block"], result["rule"] = selection["block"], selection["rule"]
    if args.block is None or args.reselect:
        args.block = selection["block"]
    result["block"] = args.block
    sub = copy.copy(args)
    sub.seed, sub.calibration, sub.test = 0, 8, 64
    transfer = steering_transfer.run(lm, sub)
    result["transfer"] = transfer["summary"]
    result["cosines"] = {k: {m: v for m, v in f.items() if m != "display"}
                         for k, f in invariance_reversal.run(lm, sub)["forms"].items()}
    if geometry:
        curves = order_of_mention.run(lm, sub)["curves"]
        result["geometry_main_block"] = curves[args.block]
        result["peak_fact_r2"] = max((c["lowo_fact_r2"], c["block"]) for c in curves)
    if depth:
        sub.blocks, sub.donors, sub.receivers, sub.by_content = "even", 8, 32, False
        d = depth_window.run(lm, sub)
        result["depth"] = dict(per_block=d["per_block"], depth_window=d["depth_window"])
    return result


def run(lm, args):
    return sweep(lm, args, depth=not args.no_depth)


def options(p):
    p.add_argument("--reselect", action="store_true", help="use the rule's block even if one is registered")
    p.add_argument("--no-depth", action="store_true", help="skip the depth curves (as for base-weight rows)")


if __name__ == "__main__":
    p = P.parser(__doc__)
    options(p)
    args = p.parse_args()
    lm = P.LM(args.model, args.revision, args.prompt_format)
    if args.block is None:
        args.block = P.main_block(args.model, args.prompt_format)
    P.save(args.out, run(lm, args))
