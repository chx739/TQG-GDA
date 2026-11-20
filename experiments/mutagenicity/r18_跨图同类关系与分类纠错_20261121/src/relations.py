"""Symmetric graph-pair features, calibrated products and fixed oriented votes."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl

PAIR_ARMS = ('Product','CalProduct','Unary','Full')
CLIP = 1e-12


def hashed(ids, tag, setting, source):
    return sorted(ids, key=lambda i: (sha256(f'{tag}|{setting}|{source}|{i}'.encode('utf-8')).hexdigest(), i))


def folds(ids, setting, source):
    ordered = hashed(ids,'r18-fold',setting,source)
    return [ordered[i::4] for i in range(4)]


def refs(c,u,setting,source):
    n = min(len(c),len(u))
    return {pool:hashed(ids,'r18-ref',setting,source)[:n] for pool,ids in (('C',c),('U',u))}


def signed_margin(a):
    return np.clip(np.tanh(a/2.),-1+CLIP,1-CLIP)


def product(left,right):
    return (1.+left*right)/2.


@dataclass
class Geometry:
    ids: dict
    layers: list
    log_size: np.ndarray

    @classmethod
    def build(cls,features):
        ids = list(features)
        layer_matrices = []
        for depth in range(3):
            rows = {i:{k:v for k,v in features[i][0].items() if k[0]==depth} for i in ids}
            layer_matrices.append(wl.sparse_rows(rows,ids,wl.vocabulary(rows,ids)))
        sizes = np.stack([features[i][1] for i in ids])
        return cls({i:j for j,i in enumerate(ids)},layer_matrices,sizes)

    def columns(self,left,right):
        i,j = ([self.ids[k] for k in ids] for ids in (left,right))
        cos = [3*np.asarray(x[i].multiply(x[j]).sum(axis=1)).ravel() for x in self.layers]
        return np.column_stack((*cos,np.abs(self.log_size[i]-self.log_size[j])))


def columns(geometry,left,right,ml,mr):
    unary = np.column_stack((ml+mr,ml*mr,np.abs(ml-mr)))
    return np.column_stack((unary,geometry.columns(left,right)))


def training_pairs(train_folds, z, y, geometry):
    """Only form D pairs whose two endpoints share the same unseen fold."""
    valid = [f for f in train_folds if len(f)>=2]
    total = sum(map(len,valid))
    table,prior,labels,weights = [],[],[],[]
    for fold in valid:
        i,j = np.triu_indices(len(fold),k=1)
        ids = np.asarray(fold,dtype=np.int64)
        left,right = ids[i],ids[j]
        ml,mr = (signed_margin(np.array([z[k] for k in ids])) for ids in (left,right))
        table.append(columns(geometry,left,right,ml,mr))
        prior.append(product(ml,mr))
        labels.extend(int(y[a]==y[b]) for a,b in zip(left,right,strict=True))
        weights.append(np.full(len(i),len(fold)/(total*len(i))))
    if not valid:
        return np.empty((0,8)),np.empty(0),np.empty(0,dtype=np.int64),np.empty(0)
    return np.vstack(table),np.concatenate(prior),np.asarray(labels,dtype=np.int64),np.concatenate(weights)


def _loss(matrix,y,weights,offset,theta,center,n):
    margin = offset+matrix@theta
    signed = (1-2*y)*margin
    delta = theta-center
    v = weights*np.logaddexp(0.,signed)
    gradient = matrix.T@(weights*(1-2*y)*expit(signed)) + delta/n
    return float(v.sum()+(delta@delta)/(2*n)),gradient


def _fit(matrix,y,weights,offset,center,n,options,*,constrained=False):
    def objective(theta):
        return _loss(matrix,y,weights,offset,theta,center,n)
    bounds = [(None,None),(0.,None)] if constrained else None
    kwargs = dict(jac=True,method='L-BFGS-B',bounds=bounds)
    opts = dict(maxiter=options['max_iterations'],gtol=options['gradient_tolerance'],
                ftol=options['function_tolerance'])
    result = minimize(objective,center,options=opts,**kwargs)
    retried = not result.success and str(result.message).startswith('ABNORMAL')
    if retried:
        opts['gtol'] = options['stationarity_tolerance']
        result = minimize(objective,result.x,options=opts,**kwargs)
    value,grad = objective(result.x)
    gradient = grad.copy()
    if constrained and result.x[1]==0.:
        gradient[1] = min(gradient[1],0.)
    error = float(np.max(abs(gradient)))
    if (not result.success or not np.isfinite(value) or not np.isfinite(error)
            or not np.isfinite(result.x).all() or error>options['stationarity_tolerance']
            or (constrained and result.x[1]<0.)):
        raise ArithmeticError(f'R18 optimizer failed its original-coordinate certificate: {result.message}, {error:g}')
    return dict(coefficients=result.x.tolist(),gradient_inf=error,objective=value,
                iterations=int(result.nit),retried=retried)


def calibrator(a,y,options):
    n = len(y)
    return _fit(np.column_stack((np.ones(n),a)),y,np.full(n,1/n),np.zeros(n),
                np.array([0.,1.]),n,options,constrained=True)


def calibrated(a,model):
    b,t = model['coefficients']
    return b+t*a


def relation_head(x,p,y,w,n,options):
    if not len(x):
        return dict(fallback='no_training_pairs',columns=x.shape[1])
    mean = (w@x)/w.sum()
    scale = np.sqrt(w@((x-mean)**2)/w.sum())
    scale[np.ptp(x,axis=0)==0.] = 1.
    design = np.column_stack((np.ones(len(x)),(x-mean)/scale))
    fit = _fit(design,y,w,logit(p),np.zeros(design.shape[1]),n,options)
    return dict(**fit,fallback=None,columns=x.shape[1],mean=mean.tolist(),scale=scale.tolist())


def relation_probability(p,x,model):
    if model['fallback'] is not None:
        return p.copy()
    design = np.column_stack((np.ones(len(x)),(x-model['mean'])/model['scale']))
    return expit(logit(p)+design@np.asarray(model['coefficients']))


def apply_vote(mi,mj,p,q):
    """Use an exact zero residual for an unchanged product probability."""
    return mi+np.mean((2*(q-p))*mj,axis=1)/(1+np.mean(mj**2))


def other_votes(mi,mj,q):
    s = 2*q-1
    perm = (mi+s.mean(axis=1)*mj.mean())/(1+np.mean(mj**2))
    relation_only = np.mean(s*mj,axis=1)
    pure = np.where(relation_only==0.,mi,relation_only)
    return perm,pure
