"""Geometry of query states: fact-address templates, held-out R^2, the address subspace, composition,
spacing of fact addresses, and the person-relation interaction.

Conventions. H has shape (n_contexts, n_questions, d). The query-dependent state is
y(C, q) = h(C, q) - (1/n) sum_{q'} h(C, q') (Sec. 3). A template averages y over calibration contexts by a
label of the question (the queried fact's order of mention, the queried person, ...).
"""
from collections import defaultdict
from itertools import permutations

import numpy as np


def center(H):
    return H - H.mean(axis=1, keepdims=True)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def templates(Y, labels):
    """{label: mean of y over all (context, question) with that label}; labels has shape (n, n_questions)."""
    sums, counts = {}, defaultdict(int)
    for c in range(len(Y)):
        for q in range(Y.shape[1]):
            k = labels[c][q]
            sums[k] = sums.get(k, 0) + Y[c, q]
            counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}


def r2(Y, labels, tmpl):
    """Held-out variance explained, 1 - sum ||y - a_label||^2 / sum ||y||^2 (unseen labels predict zero)."""
    sse = sst = 0.0
    for c in range(len(Y)):
        for q in range(Y.shape[1]):
            pred = tmpl.get(labels[c][q], 0.0)
            sse += float(np.sum((Y[c, q] - pred) ** 2))
            sst += float(np.sum(Y[c, q] ** 2))
    return 1.0 - sse / sst


def lowo_r2(Y, labels, groups):
    """Leave-one-word-set-out R^2: templates fit on the other word sets, sums pooled over held-out sets."""
    groups, labels = np.asarray(groups), np.asarray(labels, dtype=object)
    sse = sst = 0.0
    for g in np.unique(groups):
        tmpl = templates(Y[groups != g], labels[groups != g])
        test, lab = Y[groups == g], labels[groups == g]
        sst_g = float(np.sum(test ** 2))
        sse += (1.0 - r2(test, lab, tmpl)) * sst_g
        sst += sst_g
    return 1.0 - sse / sst


def within_word_set_r2(Y, keys, groups):
    """App. C.1: inside one word set, leave one context out, fit templates on the other contexts of that set.

    keys: {name: labels array}; must include "fact". Also scores "fact+<name>", where <name>'s template is
    fit on the residuals of the fact-address template (sequential two-way fit).
    """
    groups = np.asarray(groups)
    sse, sst = defaultdict(float), 0.0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for h in idx:
            train = [i for i in idx if i != h]
            fits = {name: templates(Y[train], np.asarray(lab, dtype=object)[train]) for name, lab in keys.items()}
            fact = fits["fact"]
            resid = np.stack([[Y[i, q] - fact[keys["fact"][i][q]] for q in range(Y.shape[1])] for i in train])
            second = {name: templates(resid, np.asarray(lab, dtype=object)[train])
                      for name, lab in keys.items() if name != "fact"}
            for q in range(Y.shape[1]):
                y = Y[h, q]
                sst += float(y @ y)
                for name, lab in keys.items():
                    sse[name] += float(np.sum((y - fits[name].get(lab[h][q], 0.0)) ** 2))
                base = fact.get(keys["fact"][h][q], 0.0)
                for name in second:
                    sse["fact+" + name] += float(np.sum((y - base - second[name].get(keys[name][h][q], 0.0)) ** 2))
    return {name: 1.0 - v / sst for name, v in sse.items()}


# ---------------------------------------------------------------- the address subspace (Sec. 6.1, App. E.1)
def address_basis(a, k=None):
    """Orthonormal basis (k, d) of A = span{a_p - a_1}, from the SVD of the centered template matrix."""
    a = np.asarray(a)
    k = len(a) - 1 if k is None else k
    return np.linalg.svd(a - a.mean(axis=0), full_matrices=False)[2][:k]


def share(X, B):
    """Per-row share of squared norm inside the row space of orthonormal B."""
    return np.sum((X @ B.T) ** 2, axis=-1) / np.maximum(np.sum(X ** 2, axis=-1), 1e-30)


def random_bases(d, k=3, draws=100, state=0):
    rng = np.random.default_rng(state)
    return [np.linalg.qr(rng.standard_normal((d, k)))[0].T for _ in range(draws)]


def pairwise_changes(H):
    """All within-context changes h(C, p') - h(C, p) for p < p' -> (n * n_pairs, d)."""
    n = H.shape[1]
    return np.concatenate([H[:, q] - H[:, p] for p in range(n) for q in range(p + 1, n)])


def subspace_share(H_cal, H_test, k=None, state=0):
    """Mean share of held-out changes in A (fit on calibration), versus the best and random k-dim subspaces."""
    a = center(H_cal).mean(axis=0)
    B = address_basis(a, k)
    X = pairwise_changes(H_test)
    best = np.linalg.svd(X, full_matrices=False)[2][:len(B)]
    rand = [share(X, R).mean() for R in random_bases(X.shape[1], len(B), state=state)]
    return dict(address=float(share(X, B).mean()), best=float(share(X, best).mean()),
                random=float(np.mean(rand)), changes=len(X))


def principal_cosines(B1, B2):
    """Cosines of the principal angles between two subspaces (descending); the last is the largest angle's."""
    return np.linalg.svd(B1 @ B2.T, compute_uv=False)


def composition(H, groups, word_sets=None):
    """App. E.1: v_S1(p->p') + v_S2(p'->p'') against v_T(p->p'') on disjoint word sets.

    The word sets are split into halves X, Y; one half into quarters. Cosines and relative errors are
    averaged over the 24 ordered triples and the four assignments of halves and quarters; the null relabels
    the orders of mention of the summands by the 23 non-identity permutations.
    """
    sets = sorted(set(groups)) if word_sets is None else list(word_sets)
    half = len(sets) // 2
    X, Y = sets[:half], sets[half:]
    schemes = [(X, Y[:len(Y) // 2], Y[len(Y) // 2:]), (X, Y[len(Y) // 2:], Y[:len(Y) // 2]),
               (Y, X[:len(X) // 2], X[len(X) // 2:]), (Y, X[len(X) // 2:], X[:len(X) // 2])]
    groups = np.asarray(groups)
    mean = {}

    def v(sel, p, q):
        key = tuple(sel)
        if key not in mean:
            mean[key] = H[np.isin(groups, sel)].mean(axis=0)
        return mean[key][q] - mean[key][p]

    n = H.shape[1]
    triples = [(p, q, r) for p in range(n) for q in range(n) for r in range(n) if len({p, q, r}) == 3]
    perms = [pm for pm in permutations(range(n)) if pm != tuple(range(n))]
    cos, err, null = [], [], []
    for p, q, r in triples:
        for target, first, second in schemes:
            t = v(target, p, r)
            s = v(first, p, q) + v(second, q, r)
            cos.append(cosine(t, s))
            err.append(float(np.linalg.norm(s - t) / np.linalg.norm(t)))
            for pm in perms:
                if (pm[p], pm[r]) != (p, r):
                    null.append(cosine(t, v(first, pm[p], pm[q]) + v(second, pm[q], pm[r])))
    return dict(cosine=float(np.mean(cos)), relative_error=float(np.mean(err)), null_cosine=float(np.mean(null)))


# ---------------------------------------------------------------- spacing (Sec. 6.2, Prop. 3.9)
def spacing(a):
    """Link lengths ||a_{p+1} - a_p|| and nearest-neighbor distances, in units of ||a_2 - a_1||."""
    a = np.asarray(a)
    unit = np.linalg.norm(a[1] - a[0])
    dist = np.linalg.norm(a[:, None] - a[None], axis=-1) / unit
    np.fill_diagonal(dist, np.inf)
    return dict(links=[float(np.linalg.norm(a[p + 1] - a[p]) / unit) for p in range(len(a) - 1)],
                nearest=[float(x) for x in dist.min(axis=1)])


def centered_fit(x, y, model):
    """R^2 of a least-squares line after removing each model's mean of x and of y (Fig. 4d)."""
    x, y, model = map(np.asarray, (x, y, model))
    for m in np.unique(model):
        x = np.where(model == m, x - x[model == m].mean(), x)
        y = np.where(model == m, y - y[model == m].mean(), y)
    slope = float(x @ y / (x @ x))
    return dict(slope=slope, r2=1.0 - float(np.sum((y - slope * x) ** 2) / np.sum(y ** 2)))


# ---------------------------------------------------------------- person-relation interaction (Lemma E.1)
def interaction_cosines(Y, positions, a):
    """Cosine of the measured four-cell contrast I = y11 - y10 - y01 + y00 with its fact-address prediction.

    Y: (n, 4, d) query-dependent states indexed by cell (e, r) -> 2e + r; positions: (n, 4) order of mention
    of each cell; a: (4, d) fact-address templates. Returns the mean cosine and the mean over the 23
    relabelings of the orders of mention (null).
    """
    perms = [pm for pm in permutations(range(4)) if pm != (0, 1, 2, 3)]
    obs, null = [], []
    for y, pos in zip(Y, positions):
        interaction = y[3] - y[2] - y[1] + y[0]

        def predict(p):
            return a[p[pos[3]]] - a[p[pos[2]]] - a[p[pos[1]]] + a[p[pos[0]]]

        obs.append(cosine(interaction, predict(range(4))))
        null.append(np.mean([cosine(interaction, predict(pm)) for pm in perms
                             if np.linalg.norm(predict(pm)) > 1e-9]))
    return dict(cosine=float(np.mean(obs)), null=float(np.mean(null)), contexts=len(obs), values=obs)


# ---------------------------------------------------------------- output alignment (App. F.1)
def answer_alignment(Y, U):
    """Mean cosine between each y(C, q) and its own answer's centered output direction.

    U: (n, n_questions, d) output directions g * W_U[first answer token]; centered over the context's answers.
    """
    U = U - U.mean(axis=1, keepdims=True)
    return float(np.mean([cosine(Y[c, q], U[c, q]) for c in range(len(Y)) for q in range(Y.shape[1])]))


# ---------------------------------------------------------------- two-fact coordinates (App. B)
def two_fact_axes(v0, v1, offsets):
    """Axes of Fig. 2c: u1 ~ v0 - v1 (order of mention), u2 = (v0 + v1)/2 orthogonal to u1 (named people),
    u3 = top principal direction of the calibration context offsets after projecting out u1, u2."""
    u1 = (v0 - v1) / np.linalg.norm(v0 - v1)
    u2 = (v0 + v1) / 2
    u2 = u2 - (u2 @ u1) * u1
    u2 /= np.linalg.norm(u2)
    O = offsets - offsets.mean(axis=0)
    O = O - np.outer(O @ u1, u1) - np.outer(O @ u2, u2)
    u3 = np.linalg.svd(O, full_matrices=False)[2][0]
    return np.stack([u1, u2, u3])
