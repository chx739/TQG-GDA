"""Evaluation of frozen predictions. Class columns always follow source class_values."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
import torch


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _probabilities(value):
    p = _array(value)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2 or p.dtype.kind not in 'fiu':
        raise ValueError('probabilities must be a nonempty [graphs, classes] array')
    p = p.astype(np.float64, copy=False)
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-8)):
        raise ValueError('invalid class probabilities')
    return p


def _log_probabilities(p, supplied):
    if supplied is None:
        with np.errstate(divide='ignore'):
            return np.log(p)
    log_p = _array(supplied)
    if log_p.shape != p.shape or log_p.dtype.kind != 'f':
        raise ValueError('log_probabilities must be floating and match probabilities')
    log_p = log_p.astype(np.float64, copy=False)
    if (np.isnan(log_p).any() or np.isposinf(log_p).any() or (log_p > 1e-6).any()
            or not np.allclose(logsumexp(log_p, axis=1), 0., rtol=0., atol=1e-6)
            or not np.allclose(np.exp(log_p), p, rtol=1e-6, atol=1e-15)):
        raise ValueError('log_probabilities and probabilities disagree')
    return log_p


def _ovr_log_odds(log_p, k):
    # Subtract log mass of the other classes, which survives even when the
    # positive probability rounds to 1 and its normalized log rounds to 0.
    others = np.arange(log_p.shape[1]) != k
    return log_p[:, k] - logsumexp(log_p[:, others], axis=1)


def _classes(value, n, classes):
    y = _array(value)
    if y.shape != (n,) or y.dtype.kind not in 'iu' or (y < 0).any() or (y >= classes).any():
        raise ValueError('expected one integer source class index per graph')
    return y.astype(np.int64, copy=False)


def _auc(positive, score):
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(score, method='average')
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(probabilities, labels, *, decisions=None, log_probabilities=None):
    """BA and macro OvR AUROC require all source classes to have support.

    Report an absent class's recall/AUROC as None, keep per-class results, and
    use zero F1 when a class has neither labels nor positive decisions (R7).
    Optional log probabilities retain saturated tail differences for NLL and
    OvR ranking. Without them exact zeros in the true class give infinite NLL.
    """
    p = _probabilities(probabilities)
    log_p = _log_probabilities(p, log_probabilities)
    n, c = p.shape
    y = _classes(labels, n, c)
    decisions = log_p.argmax(1) if decisions is None else _classes(decisions, n, c)
    table = np.bincount(c*y+decisions, minlength=c*c).reshape(c, c)
    true, predicted = table.sum(axis=1), table.sum(axis=0)
    correct = table.diagonal()
    recall = [float(ok/k) if k else None for ok, k in zip(correct, true)]
    f1 = 2*correct / np.maximum(true+predicted, 1)
    auc = [_auc(y == k, _ovr_log_odds(log_p, k)) for k in range(c)]
    return dict(n=n, accuracy=float(correct.sum()/n),
                balanced_accuracy=float(np.mean(recall)) if true.all() else None,
                macro_f1=float(f1.mean()), auroc=float(np.mean(auc)) if None not in auc else None,
                nll=float(-log_p[np.arange(n), y].mean()),
                class_counts=true.tolist(), predicted_counts=predicted.tolist(),
                per_class_recall=recall, per_class_f1=f1.tolist(), per_class_auroc=auc,
                confusion_matrix=table.tolist())


def equal_positive_corrections(reference_probs, candidate_probs, labels, *, positive_count=None, ids=None,
                               reference_log_probabilities=None, candidate_log_probabilities=None):
    """Compare top-q binary actions, q defaults to the candidate's argmax positives.

    Like R7, match the reference's decision count to the candidate's without
    consulting labels. Break score ties by graph ID (numeric for TU IDs).
    """
    reference = _probabilities(reference_probs)
    candidate = _probabilities(candidate_probs)
    if reference.shape != candidate.shape or reference.shape[1] != 2:
        raise ValueError('equal-positive comparison requires aligned binary probabilities')
    ref_log = _log_probabilities(reference, reference_log_probabilities)
    cand_log = _log_probabilities(candidate, candidate_log_probabilities)
    n = len(candidate)
    y = _classes(labels, n, 2)
    q = int(cand_log.argmax(1).sum()) if positive_count is None else positive_count
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 0 <= q <= n:
        raise ValueError('positive_count must be an integer from zero to graph count')
    ids = list(range(n)) if ids is None else list(ids)
    if len(ids) != n or len(set(map(str, ids))) != n:
        raise ValueError('IDs must be unique and match graph count')
    tie_key = [str(i) for i in ids]
    if all(s.isdecimal() and str(int(s)) == s for s in tie_key):
        tie_key = [int(s) for s in tie_key]
    actions = []
    for log_p in (ref_log, cand_log):
        score = _ovr_log_odds(log_p, 1)
        ranking = sorted(range(n), key=lambda i: (-score[i], tie_key[i]))
        action = np.zeros(n, dtype=np.int64)
        action[ranking[:q]] = 1
        actions.append(action)
    before, after = actions
    corrected = int(((before != y) & (after == y)).sum())
    harmed = int(((before == y) & (after != y)).sum())
    return dict(n=n, positive_count=int(q), changed=int((before != after).sum()),
                corrected=corrected, harmed=harmed, net_corrections=corrected-harmed,
                net_correction_rate=(corrected-harmed)/n)
