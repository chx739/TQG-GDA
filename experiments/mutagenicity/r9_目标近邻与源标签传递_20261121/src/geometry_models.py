"""Source logistic models with D-variance-matched WL reference features."""
from hashlib import sha256

import numpy as np
from scipy import sparse
from scipy.special import logit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from . import wl

BASES = ('WL', 'Size', 'Base')
AUGMENTED = ('TargetFull', 'SourceFull', 'TargetDiag')
MARGINS = (*BASES, *AUGMENTED)
METHODS = (*MARGINS, 'EqualPositive')
EPS_VARIANCE = 1e-14


def select_references(ids, m, setting, source):
    return sorted(ids, key=lambda i: (
        sha256(f'r9-geometry-v1|{setting}|{source}|{i}'.encode('utf-8')).hexdigest(), i))[:m]


def reference_features(rows, query, refs):
    similarities = wl.kernel(rows, query, refs)
    return (similarities - similarities.mean(axis=1, keepdims=True)) / np.sqrt(len(refs))


def column_variance(x):
    """Population variance, including implicit sparse zeros."""
    if not sparse.issparse(x):
        return x.var(axis=0)
    x = x.tocsc()
    mean = np.asarray(x.mean(axis=0)).ravel()
    nnz = np.diff(x.indptr)
    columns = np.repeat(np.arange(x.shape[1]), nnz)
    centered = (x.data - mean[columns])**2
    return (np.bincount(columns, weights=centered, minlength=x.shape[1])
            + (x.shape[0]-nnz)*mean**2) / x.shape[0]


def matrix(f, z, extra=None, scale=None):
    if extra is None:
        return sparse.hstack((f,z), format='csr')
    half = np.sqrt(0.5)
    return sparse.hstack((f*half, extra*(scale*half), z), format='csr')


def width_scale(wl_variance, additional_variance):
    if min(wl_variance, additional_variance) <= EPS_VARIANCE:
        return None
    return float(np.sqrt(wl_variance / additional_variance))


def reference_blocks(rows, ids, vocab, f, target, source):
    # Coordinates absent from D have zero fitted diagonal coefficient under L2.
    # Full similarities still use every coordinate shared by query and reference.
    target_sd = np.sqrt(column_variance(wl.sparse_rows(rows,target,vocab)))
    return dict(TargetFull=reference_features(rows,ids,target),
                SourceFull=reference_features(rows,ids,source),
                TargetDiag=f.multiply(target_sd).tocsr())


def equal_positive(ids, base, count):
    order = np.lexsort((ids,-base))
    result = np.zeros(len(ids),dtype=np.int64)
    result[order[:count]] = 1
    return result


def _block_inputs(features, ids, vocabulary, mean, scale):
    rows = {i:record[0] for i,record in features.items()}
    f = wl.sparse_rows(rows,ids,vocabulary)
    z = (np.stack([features[i][1] for i in ids])-mean)/scale
    return rows, f, z


def _fit(name, md, mc, yd, yc, solver, strengths):
    fitted = structural.fit_candidates(md,mc,yd,yc,dict(C=strengths,solver=solver))
    return dict(name=name, selected_C=fitted['selected_C'], validation=fitted['validation'],
                coefficients=structural.select_model(fitted))


def fit(features, role, yd, yc, solver):
    ids = role['roles']
    d,c,u = (ids[name] for name in ('D','C','U'))
    if any(y.shape!=(len(keys),) or not np.isin(y,[0,1]).all()
           for y,keys in ((yd,d),(yc,c))):
        raise ValueError('Expected one authorized binary label per D or C graph')
    m = min(len(c),len(u))
    target,source = (select_references(keys,m,role['setting'],role['source']) for keys in (u,c))
    model = dict(target=target, source=source, m=m)
    if len(set(yd))==1:
        positives = int(yd.sum())
        p = (positives+.5)/(len(yd)+1)
        return dict(**model, status='single_class_D', constant=float(logit(p)), selected_base_C=None)

    vocab = wl.vocabulary({i:features[i][0] for i in d},d)
    sizes = np.stack([features[i][1] for i in d])
    mean,scale = sizes.mean(axis=0),sizes.std(axis=0)
    scale[np.ptp(sizes,axis=0)==0.] = 1.
    ids_fit = [*d,*c]
    rows,f,z = _block_inputs(features,ids_fit,vocab,mean,scale)
    n = len(d)
    candidates = {name:_fit(name,x[:n],x[n:],yd,yc,solver,structural.STRENGTHS)
                  for name,x in (('WL',f),('Size',z),('Base',matrix(f,z)))}
    base_c = candidates['Base']['selected_C']
    wl_variance = float(column_variance(f[:n]).sum())
    extra_blocks = reference_blocks(rows,ids_fit,vocab,f,target,source)
    geometry = {}
    for name, extra in extra_blocks.items():
        v = float(column_variance(extra[:n]).sum())
        a = width_scale(wl_variance,v)
        geometry[name] = dict(D_extra_variance=v, scale=a, fallback=a is None,
                              D_matched_variance=wl_variance if a is None else (wl_variance+a*a*v)/2)
        if a is None:
            continue
        x = matrix(f,z,extra,a)
        candidates[name] = _fit(name,x[:n],x[n:],yd,yc,solver,[base_c])
    return dict(**model, status='ok', vocabulary=[[d,h.hex()] for d,h in vocab],
                size_mean=mean.tolist(),size_scale=scale.tolist(),selected_base_C=base_c,
                D_wl_variance=wl_variance, geometry=geometry, candidates=candidates)


def predict(model, features, ids):
    if model['status']=='single_class_D':
        return {name:np.full(len(ids),model['constant']) for name in MARGINS}
    vocab = {(d,bytes.fromhex(h)):i for i,(d,h) in enumerate(model['vocabulary'])}
    rows,f,z = _block_inputs(features,ids,vocab,model['size_mean'],model['size_scale'])
    fitted = model['candidates']
    base = {name:scores(x,fitted[name]['coefficients'])
            for name,x in (('WL',f),('Size',z),('Base',matrix(f,z)))}
    blocks = reference_blocks(rows,ids,vocab,f,model['target'],model['source'])
    for name, v in blocks.items():
        a = model['geometry'][name]['scale']
        base[name] = base['Base'].copy() if a is None else scores(matrix(f,z,v,a),fitted[name]['coefficients'])
    return base


def decisions(model, features, ids):
    margins = predict(model,features,ids)
    actions = {name:structural.decisions(z) for name,z in margins.items()}
    eid = np.asarray(ids,dtype=np.int64)
    actions['EqualPositive'] = equal_positive(eid,margins['Base'],int(actions['TargetFull'].sum()))
    return dict(e_ids=eid, **actions, **{f'margin_{m}':z for m,z in margins.items()})
