"""Exact decisions for transferring a source ROC working point by target rank."""
from fractions import Fraction

import numpy as np

from . import thresholds


def score_groups(scores):
    if scores.dtype != np.float64 or scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected nonempty finite float64 B0 margins')
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    # Positive sets are suffixes of the distinct values, never partial ties.
    cuts = [float('inf'), *[float(cut) for cut in values[-2::-1]], -float('inf')]
    return inverse, counts, cuts


def source_roc(scores, labels):
    inverse, counts, cuts = score_groups(scores)
    if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected one binary validation label per source margin')
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    bins = np.bincount(inverse[labels == 1], minlength=len(counts))
    tp = np.r_[0, np.cumsum(bins[::-1])]
    fp = np.r_[0, np.cumsum((counts - bins)[::-1])]
    return dict(positives=positives, negatives=negatives, points=[
        dict(tp=int(t), fp=int(f), cut=thresholds.encode_threshold(c))
        for t, f, c in zip(tp, fp, cuts, strict=True)
    ])


def choose(roc, prior):
    """Use exact weighted counts and pick the strictest maximizing point."""
    if roc is None or not 0 <= prior <= 1:
        raise ValueError('A two-class ROC and a probability are required')
    np_, nn = roc['positives'], roc['negatives']
    a, b = prior.numerator, prior.denominator
    best = max(roc['points'], key=lambda row: (
        a * row['tp'] * nn + (b-a) * (nn-row['fp']) * np_, -(row['tp']+row['fp'])))
    tpr, fpr = Fraction(best['tp'], np_), Fraction(best['fp'], nn)
    return best, prior*tpr + (1-prior)*fpr


def quantile(scores, rate):
    if not 0 <= rate <= 1:
        raise ValueError('Positive action fraction outside [0, 1]')
    _, counts, cuts = score_groups(scores)
    n = len(scores)
    totals = [int(v) for v in np.r_[0, np.cumsum(counts[::-1])]]
    idx = min(range(len(totals)), key=lambda i: (abs(totals[i]*rate.denominator-rate.numerator*n), totals[i]))
    return cuts[idx], totals[idx]


def classify(scores, cut):
    return thresholds.decisions(scores, float(cut))


def transfer(roc, prior, a_margin, h_margin):
    point, rate = choose(roc, prior)
    cut, npos = quantile(a_margin, rate)
    return classify(h_margin, cut), dict(p=str(prior), q=str(rate), selected=point, a_positive=npos,
                                         a_cut=thresholds.encode_threshold(cut))
