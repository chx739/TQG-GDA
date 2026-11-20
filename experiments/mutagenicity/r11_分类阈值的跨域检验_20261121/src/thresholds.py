"""Exact-correct-count thresholds with registered strict comparison and reference ties."""
import hashlib
from fractions import Fraction

import numpy as np


def decisions(scores, threshold):
    return np.asarray(scores,dtype=np.float64) > threshold


def choose(scores, labels, reference):
    x = np.asarray(scores,dtype=np.float64)
    y = np.asarray(labels)
    if (x.ndim!=1 or not len(x) or x.shape!=y.shape or not np.isfinite(x).all()
            or not np.isin(y,[0,1]).all() or np.isnan(reference)):
        raise ValueError('Threshold fitting requires a nonempty finite binary-labeled margin vector')
    order = np.argsort(x,kind='stable')
    x, y = x[order],y[order].astype(np.int64)
    unique = np.unique(x)
    with np.errstate(over='ignore',under='ignore'):
        candidates = np.unique(np.r_[unique,np.nextafter(unique,-np.inf),np.float64(reference)])
    # Predict one to begin with. Crossing m reclassifies every tied x=m
    # together, since all logits equal to threshold are assigned class zero.
    prefix = np.r_[0,np.cumsum(1-2*y,dtype=np.int64)]
    correct = y.sum()+prefix[np.searchsorted(x,candidates,side='right')]
    maximum = int(correct.max())
    tied = candidates[correct==maximum]
    def key(t):
        if t==reference:
            return 0,t
        if not np.isfinite(t) or not np.isfinite(reference):
            return float('inf'),t
        difference = abs(Fraction.from_float(float(t))-Fraction.from_float(float(reference)))
        return difference,t
    winner = float(min(tied,key=key))
    return dict(threshold=winner,correct=maximum,n=len(x),candidate_count=len(candidates))


def encode_threshold(value):
    return value if np.isfinite(value) else ('-inf' if value<0 else '+inf')


def subset_ids(setting, ids, n):
    return sorted(ids,key=lambda i:(hashlib.sha256(f'r11-v1|{setting}|{int(i)}'.encode('utf-8')).digest(),i))[:n]


def h_upper_bounds(pools):
    """Per-domain and shared empirical optima, with exact equal-domain weights."""
    from fractions import Fraction
    from math import lcm

    margins = np.unique(np.concatenate([x for x,_ in pools]))
    with np.errstate(over='ignore',under='ignore'):
        candidates = np.unique(np.r_[margins,np.nextafter(margins,-np.inf),0.])
    counts, individual = [],[]
    for x,y in pools:
        y = np.asarray(y,dtype=np.int64)
        order = np.argsort(x,kind='stable')
        prefix = np.r_[0,np.cumsum(1-2*y[order],dtype=np.int64)]
        correct = y.sum()+prefix[np.searchsorted(x[order],candidates,side='right')]
        counts.append(correct)
        individual.append(int(correct.max()))
    sizes = [len(x) for x,_ in pools]
    normalizer = lcm(*sizes)
    joint = sum(values*(normalizer//n) for values,n in zip(counts,sizes))
    separate = sum(Fraction(c,n) for c,n in zip(individual,sizes))/len(sizes)
    shared = Fraction(int(joint.max()),normalizer*len(sizes))
    return individual,separate,shared
