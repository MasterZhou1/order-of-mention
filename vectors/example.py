"""Add the ordinal vector v^{1->2} once at the main block and watch the answer move to the second fact.

    python vectors/example.py --model qwen        # or gemma, llama (needs a GPU; FP32 weights)

The context, names, contents and even the relation differ from the calibration lists ("eats"), as in Figure 1(b).
The vector is added to the output of the main block at the last prompt token, once, while the prompt is processed;
decoding then continues greedily (relgeom/steering.py).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vectors"))

from load_vectors import ordinal_vector  # noqa: E402
from relgeom.models import LM  # noqa: E402
from relgeom.steering import NONE, add, generate  # noqa: E402

INSTRUCTION = "Read the sentences. Answer the question with only the animal, in lowercase, with no explanation."
FACTS = ["Duncan Hardy owns a sloth.", "Catherine Eaton owns a penguin.", "Carl Francis owns a scorpion.",
         "Dorothy Glover owns an emu."]
QUESTION = "Which animal does Duncan Hardy own?"            # asks about the first fact


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("qwen", "gemma", "llama"), default="qwen")
    args = p.parse_args()
    v, block, model_id = ordinal_vector(args.model, 1, 2)
    lm = LM(model_id)                                        # chat template, FP32, as in the paper
    ids = lm.encode("\n\n".join([INSTRUCTION, "\n".join(FACTS), QUESTION]))
    plain, edited = generate(lm, ids, block, [NONE, add(v)])
    print(f"{model_id}, block {block}")
    print(f"  without v: {plain['text']}")
    print(f"  with v^(1->2): {edited['text']}   (the second fact's animal is 'penguin')")


if __name__ == "__main__":
    main()
