"""Load the released ordinal vectors.

Each vectors/<model>.npz holds the ordinal vectors of Table 1A at the model's main block:
  ordinal_vectors  float16 [4, 4, hidden]; ordinal_vectors[p, q] = v^{p+1 -> q+1}, the mean change of the query state
                   when the question moves from the fact mentioned (p+1)-th to the fact mentioned (q+1)-th
                   (zero on the diagonal), over 8 calibration word sets x 4 arrangements x 4 content assignments
                   (128 query changes per pair) of four-fact "eats" lists
  block            main block: the vector is added to the output of decoder block `block` (zero-based)
  model_id         Hugging Face model id
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = {"qwen": "qwen2.5-7b-instruct.npz", "gemma": "gemma-3-4b-it.npz", "llama": "llama-3.1-8b-instruct.npz"}


def load(model):
    """model: 'qwen', 'gemma' or 'llama'. Returns (vectors [4, 4, hidden] float32, block, Hugging Face id)."""
    with np.load(HERE / FILES[model]) as f:
        return f["ordinal_vectors"].astype(np.float32), int(f["block"]), str(f["model_id"])


def ordinal_vector(model, source=1, target=2):
    """v^{source -> target} (1-based orders of mention), its block, and the model id."""
    vectors, block, model_id = load(model)
    return vectors[source - 1, target - 1], block, model_id


if __name__ == "__main__":
    for name in FILES:
        v, block, model_id = load(name)
        print(f"{model_id:36s} block {block:2d}  shape {tuple(v.shape)}  ||v^(1->2)|| = {np.linalg.norm(v[0, 1]):.2f}")
