"""Exact source templates, size and hash ranges, and prior-odds updates."""
from fractions import Fraction as F
from hashlib import sha256

import numpy as np
from scipy.special import logit

RULES = ('SizeRange','Point','HashRange')


def size_cut(sizes):
    vals,counts = np.unique(sizes,return_counts=True)
    if len(vals)<2:
        return None
    i = int(np.argmin(np.abs(2*np.cumsum(counts)[:-1]-len(sizes))))
    return int(vals[i])


def source_groups(ids,y,u,edges,setting,source):
    cut = size_cut(edges)
    low = np.ones(len(ids),dtype=bool) if cut is None else edges<=cut
    hashed = low.copy()
    if cut is not None:
        hashed[:] = False
        for b in range(1,5):
            for c in (0,1):
                positions = np.flatnonzero((u==b)&(y==c))
                ordered = sorted(positions,key=lambda i:(
                    sha256(f'r12-reader|{setting}|{source}|{int(ids[i])}'.encode('utf-8')).hexdigest(),int(ids[i])))
                hashed[ordered[:int(low[positions].sum())]] = True
    return cut,low,hashed


def dot(counts,weights):
    return sum(F(int(n))*w for n,w in zip(counts,weights,strict=True))


def readout_mean(counts,weights):
    n = int(sum(counts))
    return None if n==0 else dot(counts,weights)/n


def four_cells(x):
    return np.bincount(x,minlength=4).tolist()


def count_vector(u,x):
    return [four_cells(x[u==b]) for b in range(1,5)]


def source_tables(ids,y,u,x,edges,setting,source):
    if not (len(ids)==len(y)==len(u)==len(x)==len(edges)) or not len(ids):
        raise ValueError('Malformed V inputs')
    if not np.isin(y,[0,1]).all() or not np.isin(u,[1,2,3,4]).all() or not np.isin(x,[0,1,2,3]).all():
        raise ValueError('Malformed V labels or ULT codes')
    cut,low,hashed = source_groups(ids,y,u,edges,setting,source)
    counts = [count_vector(u[y==c],x[y==c]) for c in (0,1)]
    group_counts = {}
    for name,mask in (('SizeRange',low),('HashRange',hashed)):
        groups = [mask] if cut is None else [mask,~mask]
        group_counts[name] = [[count_vector(u[(y==c)&part],x[(y==c)&part]) for c in (0,1)]
                              for part in groups]
    return dict(cut=cut,counts=counts,groups=group_counts),low,hashed


def templates(table):
    bins = []
    for b in range(4):
        counts = [table['counts'][c][b] for c in (0,1)]
        n0,n1 = (sum(row) for row in counts)
        p0 = F(2*n1+1,2*(n0+n1+1))
        w,means,intervals,missing = None,None,{},{}
        if n0 and n1:
            w = [F(a,n1)-F(z,n0) for z,a in zip(*counts,strict=True)]
            means = [readout_mean(row,w) for row in counts]
            if means[1]-means[0]!=sum(v*v for v in w):
                raise ArithmeticError('Source mean contrast does not agree with the squared vote gap')
            for rule,groups in table['groups'].items():
                bounds,empty = [],[]
                for c in (0,1):
                    observed = [readout_mean(g[c][b],w) for g in groups]
                    no_cell = any(v is None for v in observed)
                    bounds.append([min(w),max(w)] if no_cell else [min(observed),max(observed)])
                    empty.append(no_cell)
                intervals[rule],missing[rule] = bounds,empty
        bins.append(dict(bin=b+1,counts=counts,n0=n0,n1=n1,p0=str(p0),
                         w=None if w is None else [str(x) for x in w],
                         means=None if means is None else [str(x) for x in means],
                         ranges={r:[[str(lo),str(hi)] for lo,hi in ls] for r,ls in intervals.items()},
                         missing=missing))
    return bins


def interval(ranges,u):
    (l0,h0),(l1,h1) = ranges
    lo,hi = F(0),F(1)
    # Intersect the two affine half-spaces with the valid probability segment.
    for intercept,slope,bound in ((l0,l1-l0,u),(-h0,h0-h1,-u)):
        if slope>0:
            hi = min(hi,(bound-intercept)/slope)
        elif slope<0:
            lo = max(lo,(bound-intercept)/slope)
        elif intercept>bound:
            return None
    return None if lo>hi else (lo,hi)


def clip(value,lo,hi):
    return min(hi,max(lo,value))


def shift(p,p0,n):
    if p==p0 or not n:
        return 0.
    floor = F(1,2*(n+1))
    return float(logit(float(clip(p,floor,1-floor)))-logit(float(clip(p0,floor,1-floor))))


def fit_rules(source,acounts):
    fitted = []
    for b,counts in zip(source,acounts,strict=True):
        p0 = F(b['p0'])
        n = sum(counts)
        w = None if b['w'] is None else [F(x) for x in b['w']]
        u = None if w is None else readout_mean(counts,w)
        common = 'empty_A' if not n else 'missing_V_class' if w is None else 'zero_contrast' if not any(w) else None
        rules = {}
        for rule in RULES:
            p,seg,reason = p0,None,common
            if common is None and rule=='Point':
                m0,m1 = (F(v) for v in b['means'])
                p = clip((u-m0)/(m1-m0),F(0),F(1))
            elif common is None:
                bounds = [[F(x) for x in z] for z in b['ranges'][rule]]
                seg = interval(bounds,u)
                if seg is None:
                    reason = 'infeasible'
                else:
                    p = clip(p0,*seg)
            rules[rule] = dict(p=str(p),feasible=None if seg is None else [str(v) for v in seg],
                               reason=reason,shift=shift(p,p0,n))
        fitted.append(dict(bin=b['bin'],n=n,counts=counts,u=None if u is None else str(u),rules=rules))
    return fitted


def margins(anchor,h_bins,rules,name):
    step = np.array([b['rules'][name]['shift'] for b in rules],dtype=np.float64)
    return anchor+step[h_bins-1]
