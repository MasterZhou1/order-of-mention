"""Sec. 4, Fig. 3, App. C.1 (Fig. 7) and Fig. 13: query states are organized by the queried fact's order of mention.

Four-fact contexts, states at every block. Calibration: 16 word sets x 4 arrangements x 4 content assignments
(256 contexts); test: 64 word sets x 2 content assignments in one arrangement (128 contexts).
  * fact-address template a_p (mean query-dependent state by order of mention) versus person template b_e
    (by the queried person, the k-th name of each word set): leave-one-word-set-out R^2 on calibration contexts,
    and R^2 of test contexts under calibration templates;
  * within one word set (names fixed): templates for person, fact address and requested content, each fit on
    the other 15 contexts of the word set;
  * alignment of each query-dependent state with its own answer's first-token output direction (test contexts);
  * Fig. 3 projection at the main block: plane of the top two principal directions of the templates fit on
    calibration word sets 0-7; points are the held-out calibration word sets 8-15.
"""
import numpy as np

from relgeom import geometry as G, pipeline as P
from relgeom.capture import stack_by_position


def labels(contexts, kind):
    """Per context and order of mention p: the queried person (k-th name) or the requested content."""
    out = []
    for ctx in contexts:
        by_pos = sorted(ctx.queries, key=lambda q: q.position)
        out.append([q.labels["person"] if kind == "person" else q.answer for q in by_pos])
    return np.array(out, dtype=object)


def run(lm, args):
    cal, test = P.address_design(lm, args)
    blocks = list(range(lm.n_blocks))
    Hc = stack_by_position(cal, P.states_at(lm, cal, blocks, args.batch_size))
    Ht = stack_by_position(test, P.states_at(lm, test, blocks, args.batch_size))
    fact_c, fact_t = np.tile(np.arange(4), (len(cal), 1)), np.tile(np.arange(4), (len(test), 1))
    person_c, person_t, content_c = labels(cal, "person"), labels(test, "person"), labels(cal, "content")
    groups = np.array([c.word_set for c in cal])
    first = {c: lm.first_token(c) for ctx in test for c in ctx.candidates}
    tokens = sorted(set(first.values()))
    rows = dict(zip(tokens, lm.output_directions(tokens)))          # g * W_U[t] of each first answer token
    U = np.stack([[rows[first[q.answer]] for q in sorted(ctx.queries, key=lambda q: q.position)] for ctx in test])
    curves = []
    for b in blocks:
        Yc, Yt = G.center(Hc[:, :, b].astype(np.float64)), G.center(Ht[:, :, b].astype(np.float64))
        fact_tmpl, person_tmpl = G.templates(Yc, fact_c), G.templates(Yc, person_c)
        curves.append(dict(
            block=b,
            lowo_fact_r2=G.lowo_r2(Yc, fact_c, groups), lowo_person_r2=G.lowo_r2(Yc, person_c, groups),
            test_fact_r2=G.r2(Yt, fact_t, fact_tmpl), test_person_r2=G.r2(Yt, person_t, person_tmpl),
            within_word_set=G.within_word_set_r2(Yc, dict(fact=fact_c, person=person_c, content=content_c), groups),
            answer_alignment=G.answer_alignment(Yt, U)))
        print(curves[-1]["block"], round(curves[-1]["lowo_fact_r2"], 3), round(curves[-1]["lowo_person_r2"], 3))
    # Fig. 3: projection at the main block.
    Y = G.center(Hc[:, :, args.block].astype(np.float64))
    fit = np.isin(groups, sorted(set(groups))[:8])
    a = np.stack([G.templates(Y[fit], fact_c[fit])[p] for p in range(4)])
    centre = a.mean(axis=0)
    basis = np.linalg.svd(a - centre, full_matrices=False)[2][:2]
    shown = (Y[~fit] - centre) @ basis.T
    unit = np.linalg.norm(a[1] - a[0])
    projection = dict(templates=((a - centre) @ basis.T / unit).round(4), points=(shown / unit).round(4),
                      point_fact=fact_c[~fit], point_person=person_c[~fit].astype(int),
                      variance_share=[float(np.sum(shown[..., k] ** 2) / np.sum((Y[~fit] - centre) ** 2))
                                      for k in range(2)], unit="||a_2 - a_1||")
    return dict(model=args.model, revision=args.revision, prompt_format=args.prompt_format, block=args.block,
                experiment="order_of_mention", curves=curves, main=curves[args.block], projection=projection)


if __name__ == "__main__":
    P.cli(__doc__, run)
