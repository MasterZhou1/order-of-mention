"""Query states h_l(C, q): the residual stream at the last prompt token after Transformer block l."""
from collections import defaultdict

import numpy as np
import torch


def _hidden(output):
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.inference_mode()
def last_token_states(lm, prompts, blocks=None, batch_size=8):
    """States of the last token of each prompt (token-id lists) at `blocks` -> float32 (n, len(blocks), d).

    Prompts are batched only with prompts of equal length, so no padding changes positions.
    """
    blocks = list(range(lm.n_blocks)) if blocks is None else list(blocks)
    out = np.empty((len(prompts), len(blocks), lm.hidden_size), dtype=np.float32)
    by_length = defaultdict(list)
    for i, ids in enumerate(prompts):
        by_length[len(ids)].append(i)
    captured = {}

    def hook(block):
        def fn(_module, _inputs, output):
            captured[block] = _hidden(output)[:, -1].float().cpu().numpy()
        return fn

    handles = [lm.layers[b].register_forward_hook(hook(b)) for b in blocks]
    try:
        for indices in by_length.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                ids = torch.tensor([prompts[i] for i in chunk], device=lm.device)
                lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                out[chunk] = np.stack([captured[b] for b in blocks], axis=1)
    finally:
        for handle in handles:
            handle.remove()
    return out


def capture_contexts(lm, contexts, blocks=None, batch_size=8):
    """Query states of every question of every context: list of arrays (n_questions, n_blocks, d)."""
    prompts, owner = [], []
    for c, ctx in enumerate(contexts):
        for q in range(len(ctx.queries)):
            prompts.append(lm.encode(ctx.body(q)))
            owner.append(c)
    states = last_token_states(lm, prompts, blocks, batch_size)
    result, start = [], 0
    for ctx in contexts:
        result.append(states[start:start + len(ctx.queries)])
        start += len(ctx.queries)
    return result


def stack_by_position(contexts, states, block_index=None):
    """float32 (n_contexts, n_facts, [n_blocks,] d) with questions re-indexed by the queried fact's order of mention.

    Requires every context to ask exactly one question per order of mention.
    """
    rows = []
    for ctx, h in zip(contexts, states):
        order = np.argsort([q.position for q in ctx.queries])
        if sorted(q.position for q in ctx.queries) != list(range(len(ctx.queries))):
            raise ValueError("stack_by_position needs one question per order of mention")
        h = h[order]
        rows.append(h if block_index is None else h[:, block_index])
    return np.stack(rows)
