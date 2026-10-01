"""Complete-answer scoring, first-token scoring, and word-set bootstrap intervals (App. A)."""
import re
from collections import defaultdict

import numpy as np

BOOTSTRAP_DRAWS = 5000


def normalize(text, strip_article=False):
    """Lowercase, strip, drop final sentence punctuation (and optionally one leading article)."""
    text = re.sub(r"[.!?]+$", "", str(text).strip().lower()).rstrip()
    return re.sub(r"^(a|an|the)\s+", "", text) if strip_article else text


def stated_content(text, contents):
    """The single stated content an answer names, or None (answers of base checkpoints are sentences).

    A content is named when it occurs as a whole-word phrase in the answer; answers naming zero or several
    contents of the context are unscorable and count as failures.
    """
    found = [c for c in contents if re.search(r"(?<![a-z])" + re.escape(c.lower()) + r"(?![a-z])", text.lower())]
    longest = [c for c in found if not any(c != o and c.lower() in o.lower() for o in found)]
    return longest[0] if len(longest) == 1 else None


def outcome(result, contents, designated, original, by_content=False, strip_article=False):
    """Classify one answer: designated / original / other (another stated content) / invalid / truncated.

    Invalid and truncated answers stay in every denominator, as do cases whose unedited answer is wrong.
    """
    if by_content:                      # stated-content scoring also reads truncated sentences
        named = stated_content(result["text"], contents)
        truncated = 0
    else:
        named = None if result["truncated"] else {normalize(c, strip_article): c for c in contents}.get(
            normalize(result["text"], strip_article))
        truncated = int(result["truncated"])
    return dict(designated=int(named == designated), original=int(named == original),
                other=int(named is not None and named not in (designated, original)),
                invalid=int(not truncated and named is None), truncated=truncated, answer=result["text"])


def _interval(samples):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def bootstrap(records, metric, cluster="word_set", state=0):
    """Mean over word sets of the per-word-set rate, with a 95% percentile interval over word sets."""
    groups = defaultdict(list)
    for r in records:
        groups[r[cluster]].append(float(r[metric]))
    per = np.array([np.mean(v) for _, v in sorted(groups.items())])
    draws = np.random.default_rng(state).integers(len(per), size=(BOOTSTRAP_DRAWS, len(per)))
    return dict(mean=float(per.mean()), ci=_interval(per[draws].mean(axis=1)), clusters=len(per), n=len(records))


def bootstrap_two_way(records, metric, rows="donor", cols="word_set", state=0):
    """Donor conditions: donor and receiver word sets are resampled independently."""
    rs = sorted({r[rows] for r in records})
    cs = sorted({r[cols] for r in records})
    total, count = np.zeros((len(rs), len(cs))), np.zeros((len(rs), len(cs)))
    for r in records:
        i, j = rs.index(r[rows]), cs.index(r[cols])
        total[i, j] += float(r[metric])
        count[i, j] += 1
    cell = total / np.maximum(count, 1)
    rng = np.random.default_rng(state)
    samples = [cell[np.ix_(rng.integers(len(rs), size=len(rs)), rng.integers(len(cs), size=len(cs)))].mean()
               for _ in range(BOOTSTRAP_DRAWS)]
    return dict(mean=float(cell.mean()), ci=_interval(samples), n=len(records))


def summarize(records, by=("condition",), metrics=("designated", "original", "invalid", "truncated"),
              two_way=None):
    """{group label: {metric: bootstrap summary}} for every combination of the `by` fields."""
    groups = defaultdict(list)
    for r in records:
        groups["/".join(str(r[k]) for k in by)].append(r)
    out = {}
    for label, rows in sorted(groups.items()):
        if two_way and two_way(rows[0]):
            out[label] = {m: bootstrap_two_way(rows, m) for m in metrics}
        else:
            out[label] = {m: bootstrap(rows, m) for m in metrics}
    return out


def paired_difference(records, left, right, metric="designated", key=("word_set", "case")):
    """Word-set bootstrap of the paired difference between two conditions on the same cases."""
    table = defaultdict(dict)
    for r in records:
        if r["condition"] in (left, right):
            table[tuple(r[k] for k in key)][r["condition"]] = (r[metric], r["word_set"])
    diffs = [dict(word_set=v[left][1], d=v[left][0] - v[right][0]) for v in table.values() if left in v and right in v]
    return bootstrap(diffs, "d")
