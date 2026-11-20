"""Exact cellwise source quotas and unlabelled mid-CDF transport."""
from fractions import Fraction
from hashlib import sha256

import numpy as np


def anchor_name(setting):
    return 'Size' if setting == 'proteins_edge' else 'WL'


def anchor_cells(setting):
    return (0,1,0,1) if anchor_name(setting)=='Size' else (0,0,1,1)


def candidates(g, y, anchor):
    if not len(g):
        return dict(n=0,k=None,candidates=[])
    order = np.argsort(-g,kind='stable')
    y_int = np.asarray(y[order],dtype=np.int64)
    score = len(y)-int(y_int.sum())
    choices = [(0,score)]
    gs = g[order]
    for k,delta in enumerate(np.cumsum(2*y_int-1),1):
        if k==len(gs) or gs[k-1]!=gs[k]:
            choices.append((k,score+int(delta)))
    best = min(choices,key=lambda pair:(-pair[1],len(y)-pair[0] if anchor else pair[0]))[0]
    return dict(n=len(y),k=best,candidates=[list(pair) for pair in choices])


def fit(g, z, y, anchors):
    return [candidates(g[z==cell],y[z==cell],anchor) for cell,anchor in enumerate(anchors)]


def rank_numerators(pool, query):
    ordered = np.sort(pool)
    # Twice the count below the query, plus the count exactly equal.
    return np.searchsorted(ordered,query,side='left')+np.searchsorted(ordered,query,side='right')


def compare(num, den, quota, anchor):
    n,k = quota['n'],quota['k']
    if not n:
        return anchor
    if k==0 or k==n:
        return int(k>0)
    left,right = int(num)*n,den*(n-k)
    return int(left>right or (left==right and anchor))


def predict(quotas, source, reference, held, anchors, *, sign_mix=False):
    v_g,v_z = source
    a_g,a_z = reference
    g,z = held
    output = np.empty(len(g),dtype=np.int64)
    signs = a_g>0.
    for cell,anchor in enumerate(anchors):
        q = quotas[cell]
        mask = z==cell
        if not q['n'] or q['k'] in (0,q['n']):
            output[mask] = compare(0,1,q,anchor)
            continue
        a_cell = a_z==cell
        n_a = int(a_cell.sum())
        u = g[mask]
        if not n_a or not sign_mix:
            pool = a_g[a_cell] if n_a else v_g[v_z==cell]
            numerator = rank_numerators(pool,u)
            output[mask] = [compare(i,2*len(pool),q,anchor) for i in numerator]
            continue
        # The target cell supplies mixture counts, and each sign layer
        # supplies the CDF of all A scores with that sign.
        by_sign = []
        for sign in (False,True):
            sign_mask = signs==sign
            weight = int((a_cell&sign_mask).sum())
            if weight:
                layer = a_g[sign_mask]
                by_sign.append([Fraction(weight*int(i),2*n_a*len(layer))
                                for i in rank_numerators(layer,u)])
        output[mask] = [compare(r.numerator,r.denominator,q,anchor)
                        for r in (sum(row,Fraction(0)) for row in zip(*by_sign,strict=True))]
    return output


def count_matched(g, positives):
    if not 0<=positives<=len(g) or not np.isfinite(g).all():
        raise ValueError('Invalid fixed positive count or B2 margin')
    num = np.zeros(len(g),dtype=np.int64)
    den = np.ones_like(num)
    if positives==len(g):
        num.fill(1)
    elif positives:
        cut = np.sort(g)[-positives]
        above = g>cut
        tied = g==cut
        chance = Fraction(positives-int(above.sum()),int(tied.sum()))
        num[above] = 1
        num[tied],den[tied] = chance.numerator,chance.denominator
    return num,den


def half_split(setting, target, ids):
    order = sorted(range(len(ids)),key=lambda i:(
        sha256(f'r12-rank-half|{setting}|{target}|{int(ids[i])}'.encode('utf-8')).digest(),
        int(ids[i])))
    halves = np.empty(len(ids),dtype=np.int64)
    halves[order] = np.arange(len(ids))%2
    return halves


def concordance(g, z, y, strata):
    groups = [(g[(z==c)&(y==0)],g[(z==c)&(y==1)]) for c in range(strata)]
    pairs = credit = covered_0 = covered_1 = 0
    for neg,pos in groups:
        if not len(neg) or not len(pos):
            continue
        pairs += len(neg)*len(pos)
        covered_0 += len(neg)
        covered_1 += len(pos)
        credit += int(rank_numerators(neg,pos).sum())
    return dict(pairs=pairs,credit_twice=credit,covered_0=covered_0,covered_1=covered_1)


def majority(z, y, query, anchors):
    choices = np.array(anchors,dtype=np.int64)
    for cell,anchor in enumerate(anchors):
        labels = y[z==cell]
        delta = 2*int(labels.sum())-len(labels)
        choices[cell] = int(delta>0 or (delta==0 and anchor))
    return choices[query]


def supervised(g,z,y,halves,anchors):
    finer = 2*z+(g>0.)
    binary_anchors = [a for a in anchors for _ in range(2)]
    result = {m:np.empty(len(g),dtype=np.int64) for m in ('HalfRank','HalfFour','HalfEight')}
    for ref_half in (0,1):
        ref = halves==ref_half
        query = ~ref
        source = (g[ref],z[ref])
        quotas = fit(*source,y[ref],anchors)
        result['HalfRank'][query] = predict(quotas,source,source,(g[query],z[query]),anchors)
        result['HalfFour'][query] = majority(z[ref],y[ref],z[query],anchors)
        result['HalfEight'][query] = majority(finer[ref],y[ref],finer[query],binary_anchors)
    return result
