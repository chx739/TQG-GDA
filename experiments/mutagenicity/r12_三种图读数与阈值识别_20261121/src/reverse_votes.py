"""Exact structural neighbors and four-cell voting rules for R12."""
from fractions import Fraction as F
from itertools import combinations_with_replacement

import numpy as np

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import discrete_types

CELLS = range(4)
TABLES = tuple(tuple((number >> z) & 1 for z in CELLS) for number in range(16))
LOCAL_ARMS = ('Reverse','Direct','PermMean','RowPerm')
TABLE_ARMS = (*LOCAL_ARMS,'Other1','Other2')


def anchor_number(setting):
    # z = 2*WL + Size. The mask's least significant bit is cell z=0.
    return 10 if setting == 'proteins_edge' else 12


def edge_histogram(graph, width):
    """Count simple undirected edges by unordered endpoint node types."""
    nodes = discrete_types(graph.x,width,len(graph.x),'node')
    vocab = {pair:i for i,pair in enumerate(combinations_with_replacement(range(width),2))}
    counts = np.zeros(len(vocab)+1,dtype=np.int64)
    edges = set()
    for a,b in graph.edge_index.T.tolist():
        if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
            raise ValueError('Graph edge references an unknown node')
        if a != b:
            edges.add((min(a,b),max(a,b)))
    for a,b in edges:
        pair = (min(nodes[a],nodes[b]),max(nodes[a],nodes[b]))
        counts[vocab[pair]] += 1
    counts[-1] = int(not edges)
    return counts


def votes(wl, size):
    wl,size = np.asarray(wl),np.asarray(size)
    if (wl.ndim!=1 or wl.shape!=size.shape or not np.isin(wl,[0,1]).all()
            or not np.isin(size,[0,1]).all()):
        raise ValueError('Expected paired binary WL and Size decisions')
    return 2*wl.astype(np.int64)+size.astype(np.int64)


def tally(z, y=None):
    if y is None:
        return np.bincount(z,minlength=4).tolist()
    return np.bincount(2*z+y,minlength=8).reshape(4,2).tolist()


def reverse_weights(v, a, z, neighbors):
    """For each V graph, sum the exact k-neighbor weights by the four A cells."""
    if (not len(v) or v.ndim!=2 or a.ndim!=2 or v.shape[1]!=a.shape[1]
            or not len(a) or not 1<=neighbors or z.shape!=(len(a),)
            or not np.issubdtype(v.dtype,np.integer) or not np.issubdtype(a.dtype,np.integer)
            or np.any(v<0) or np.any(a<0) or not np.isin(z,[0,1,2,3]).all()):
        raise ValueError('Expected nonempty integer graph histograms and four-cell votes')
    # Bound all pairwise L1 numerators before using NumPy int64 multiplication.
    vmax, amax = (max(sum(map(int,row)) for row in data) for data in (v,a))
    if 2*vmax*amax > np.iinfo(np.int64).max:
        raise OverflowError('Edge count products exceed int64')
    v, a = (counts.astype(np.int64,copy=False) for counts in (v,a))
    dv,da = v.sum(axis=1,dtype=np.int64),a.sum(axis=1,dtype=np.int64)
    if min(dv.min(),da.min()) < 1:
        raise ValueError('An edgeless graph needs its dedicated mass')
    n_a = list(map(int,da))
    k = min(neighbors,len(a))
    result = []
    for hist,n in zip(v,dv,strict=True):
        numerators = np.abs(hist * da[:,None] - a * n).sum(axis=1)
        # TV differs by the same positive factor 1/(2*n) for all A
        # candidates in this row, so it cancels when ordering distances.
        distances = [F(int(num),den) for num,den in zip(numerators,n_a,strict=True)]
        cutoff = sorted(distances)[k-1]
        nearer = [0]*4
        tied = [0]*4
        for vote,distance in zip(z,distances,strict=True):
            if distance < cutoff:
                nearer[vote] += 1
            elif distance == cutoff:
                tied[vote] += 1
        remaining = k-sum(nearer)
        boundary = sum(tied)
        result.append(tuple(F(near*boundary+tie*remaining,k*boundary)
                            for near,tie in zip(nearer,tied,strict=True)))
    return result


def brier(weights, labels, *, row_permutation=False):
    n = len(labels)
    prevalence = F(sum(map(int,labels)),n)
    targets = [prevalence]*n if row_permutation else list(map(int,labels))
    # Under permutation, the expected squared loss is (p-prevalence)^2
    # plus the variance of the original binary labels.
    variance = prevalence*(1-prevalence) if row_permutation else F(0)
    return [
        sum(((sum(w for w,t in zip(row,table,strict=True) if t)-y)**2
             for row,y in zip(weights,targets,strict=True)), F(0))/n + variance
        for table in TABLES
    ]


def choose(losses, anchor):
    return sorted(range(len(TABLES)),
                  key=lambda j:(losses[j],(j^anchor).bit_count(),j))


def record(losses, anchor):
    order = choose(losses,anchor)
    best,second = order[:2]
    return dict(table=best,second_table=second,brier=[str(x) for x in losses],
                gap_exact=str(losses[second]-losses[best]))


def fit(v_z, y, a_z, weights, anchor):
    counts = tally(a_z)
    n = len(a_z)
    direct = [tuple(F(int(z==j)) for j in CELLS) for z in v_z]
    perm = tuple(F(nz,n) for nz in counts)
    return dict(a_counts=counts,v_counts=tally(v_z,y),fits={
        'Reverse':record(brier(weights,y),anchor),
        'Direct':record(brier(direct,y),anchor),
        'PermMean':record(brier([perm]*len(y),y),anchor),
        'RowPerm':record(brier(weights,y,row_permutation=True),anchor),
    })


def count_matched(v_counts, h_counts, positives):
    """Randomize uniformly across all H graphs on the same exact score."""
    rates = [F(2*positive+1,2*(negative+positive+1)) for negative,positive in v_counts]
    q = [F(0)]*4
    remaining = positives
    if not 0<=remaining<=sum(h_counts):
        raise ValueError('Requested positive mass is outside H')
    for rate in sorted(set(rates),reverse=True):
        cells = [j for j in CELLS if rates[j]==rate]
        size = sum(h_counts[j] for j in cells)
        if size:
            taken = min(size,remaining)
            for j in cells:
                q[j] = F(taken,size)
            remaining -= taken
    return q


def apply(local, donors, h_counts, anchor):
    tables = dict(Anchor=anchor,**{arm:local['fits'][arm]['table'] for arm in LOCAL_ARMS})
    tables.update((arm,row['fits']['Reverse']['table'])
                  for arm,row in zip(('Other1','Other2'),donors,strict=True))
    policies = {name:TABLES[table] for name,table in tables.items()}
    positive = sum(n*pred for n,pred in zip(h_counts,policies['Reverse'],strict=True))
    policies['CountMatched'] = count_matched(local['v_counts'],h_counts,positive)
    return dict(tables=tables,h_counts=h_counts,
                policies={name:[str(q) for q in q_row] for name,q_row in policies.items()},
                positive_counts={name:str(sum(n*q for n,q in zip(h_counts,q_row,strict=True)))
                                 for name,q_row in policies.items()})
