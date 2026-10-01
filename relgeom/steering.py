"""Ordinal vectors, once-only edits of the query state, and donor-to-receiver state transfer.

Definition 3.2, Eq. (1): v_l^{p->p'} = E_{C in C_cal} [h_l(C, q_C(p')) - h_l(C, q_C(p))], estimated as the mean over
calibration word sets of each word set's mean change (equal weight per word set).
Eq. (2): the receiver's query state at block l is edited once, at the last prompt token while the prompt is
processed; every later block and position is recomputed and greedy decoding continues from there.
"""
import hashlib
from collections import defaultdict

import numpy as np
import torch

from .models import MAX_NEW_TOKENS


def by_position(ctx, s, t):
    """Default pooling key of a query change: the directed pair of orders of mention (p, p')."""
    return ctx.queries[s].position, ctx.queries[t].position


def mean_changes(contexts, states, block_index=0, key=by_position, pairs=None):
    """{key: mean over word sets of the per-word-set mean of h(C, q_t) - h(C, q_s)} for all ordered pairs."""
    per_set = defaultdict(lambda: defaultdict(list))
    for ctx, h in zip(contexts, states):
        n = len(ctx.queries)
        for s, t in (pairs(ctx) if pairs else [(s, t) for s in range(n) for t in range(n) if s != t]):
            k = key(ctx, s, t)
            if k is not None:
                per_set[k][ctx.word_set].append(h[t, block_index].astype(np.float64) - h[s, block_index])
    return {k: np.mean([np.mean(v, axis=0) for v in sets.values()], axis=0) for k, sets in per_set.items()}


def rescale(vector, length):
    norm = np.linalg.norm(vector)
    return vector * (length / norm) if norm > 1e-10 else np.zeros_like(vector)


def random_direction(like, tag, length=None):
    """A fixed Gaussian direction (determined by `tag`) with the length of `like` (or `length`)."""
    state = int.from_bytes(hashlib.sha256(str(tag).encode()).digest()[:4], "big")
    draw = np.random.default_rng(state).standard_normal(like.shape[-1])
    return rescale(draw, np.linalg.norm(like) if length is None else length)


def add(vector):
    return ("add", vector)


def replace(state):
    return ("replace", state)


NONE = ("none", None)


def _hidden(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _with_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    return hidden


@torch.inference_mode()
def generate(lm, prompt_ids, block, edits, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy answers of one prompt under each edit in `edits` (batched; one row per edit).

    An edit is ("none", None), ("add", v) or ("replace", h). It acts on the output of block `block` at the last
    prompt token during prefill only; decoding steps are not edited.
    """
    n = len(edits)
    width = lm.hidden_size
    values = np.stack([np.zeros(width) if v is None else np.asarray(v, dtype=np.float64) for _, v in edits])
    values = torch.tensor(values, dtype=torch.float32)
    is_replace = torch.tensor([mode == "replace" for mode, _ in edits])
    state = {"prefill": True}

    def hook(_module, _inputs, output):
        if not state["prefill"]:
            return output
        state["prefill"] = False
        hidden = _hidden(output)
        edited = hidden.clone()
        v = values.to(hidden.device, hidden.dtype)
        mask = is_replace.to(hidden.device)[:, None]
        edited[:, -1] = torch.where(mask, v, hidden[:, -1] + v)
        return _with_hidden(output, edited)

    ids = torch.tensor([list(prompt_ids)] * n, device=lm.device)
    plain = lm.prompt_format == "plain"
    stops = sorted(lm.eos_ids | lm.newline_ids) if plain else sorted(lm.eos_ids)
    extra = {"begin_suppress_tokens": sorted(lm.newline_ids)} if plain else {}
    handle = lm.layers[block].register_forward_hook(hook)
    try:
        generated = lm.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=stops,
                                      pad_token_id=lm.pad_id, temperature=None, top_p=None, top_k=None, **extra)
    finally:
        handle.remove()
    results = []
    for row in generated[:, len(prompt_ids):].tolist():
        stop = next((i for i, token in enumerate(row) if token in stops), None)
        tokens = row[:stop] if stop is not None else row
        text = lm.tokenizer.decode(tokens, skip_special_tokens=True)
        if plain:
            text = text.split("\n", 1)[0]
        results.append(dict(text=text.strip(), first_token=row[0] if row else None,
                            truncated=stop is None and len(row) >= max_new_tokens))
    return results
