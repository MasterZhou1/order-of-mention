"""Sec. 7.2, Fig. 6, Table 4, App. F.2: the fact address forms during pretraining.

Ten checkpoints of allenai/OLMo-2-1124-7B, from 5B training tokens to the released model (after stage-2
mid-training), loaded through Hugging Face `revision`. Prompts use the plain format with the tokenizer's
<|endoftext|> as the first token (a document start in OLMo-2's packed pretraining data). Each checkpoint runs the
battery of scale_sweep.py: its own rule-chosen block (reported), and at the released model's block 16 the transfer
test (answers scored by the single stated content they name), the cosines of Prop. 3.4, the query-state geometry at
every block, and the depth curves at every second block (8 donors x 32 receivers, complete answers). The stages of
Fig. 6a apply the address thresholds to the gain over the unedited answer (`stage_gain` in the depth output).
"""
import copy
import gc

import torch

from experiments.scale_sweep import sweep
from relgeom import pipeline as P
from relgeom.models import LM

REPO = "allenai/OLMo-2-1124-7B"
# (branch, pinned commit, training tokens); step 99000 stands in for step 101000, whose branch lacks tokenizer files.
CHECKPOINTS = [
    ("stage1-step1000-tokens5B", "ea9a026581be70db65a9463c0b8a28ae39d79301", "5B"),
    ("stage1-step4000-tokens17B", "83cfe2642301eb146308aafddd40047d5f66202d", "17B"),
    ("stage1-step10000-tokens42B", "35f054c4c53d49a7e198ce6e76588b1acab93455", "42B"),
    ("stage1-step25000-tokens105B", "6c4a7f3f99166f55d4b3a355f6504415d9604758", "105B"),
    ("stage1-step50000-tokens210B", "e6ecf9bf3025bc8bcf4346413f7913194077ba2a", "210B"),
    ("stage1-step99000-tokens416B", "4f6810f2ce1802470ef14468273fe931fe191c0d", "416B"),
    ("stage1-step200000-tokens839B", "45d6a453ddeb07f3db7b8dbba878f3cec2e0da11", "839B"),
    ("stage1-step400000-tokens1678B", "1fdb9ee53a860a4ec10889401f5ef2255cdf88b3", "1.7T"),
    ("stage1-step928646-tokens3896B", "c0371f4281bf2376207646c6b62ddc6c442c7577", "3.9T"),
    ("main", "7df9a82518afdecae4e8c026b27adccc8c1f0032", "final"),
]
FIXED_BLOCK = 16


def main():
    p = P.parser(__doc__)
    p.add_argument("--checkpoints", nargs="*", default=None, help="subset of token labels, e.g. 5B 42B final")
    p.set_defaults(model=REPO, prompt_format="plain", block=FIXED_BLOCK, by_content=True)
    for action in p._actions:
        if action.dest == "model":
            action.required = False
    args = p.parse_args()
    results = []
    for branch, commit, tokens in CHECKPOINTS:
        if args.checkpoints and tokens not in args.checkpoints:
            continue
        run_args = copy.copy(args)
        run_args.revision, run_args.block, run_args.reselect = commit, FIXED_BLOCK, False
        lm = LM(REPO, commit, "plain")
        entry = sweep(lm, run_args)
        entry.update(branch=branch, tokens=tokens)
        results.append(entry)
        P.save(args.out, dict(experiment="olmo_development", repo=REPO, checkpoints=results))
        del lm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
