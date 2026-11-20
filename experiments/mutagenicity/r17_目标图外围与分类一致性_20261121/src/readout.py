"""Paired typed WL readouts and a sparse logistic consistency penalty."""
from collections import Counter
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from experiments.mutagenicity.r15_新局部结构与图类别_20261121.src import structure

ARMS = ('TargetPair','SourcePair','Shuffled')
TRACE_FLOOR = 1e-14


def views(graph, setting):
    """Change counted centers while retaining all neighbors in each color."""
    nw,ew = structural.SETTINGS[setting]
    if setting=='proteins_edge':
        if graph['edge_attr'] is not None:
            raise ValueError('Expected the sealed untyped PROTEINS edges')
        graph = dict(graph,edge_attr=torch.ones((graph['edge_index'].shape[1],1),dtype=graph['x'].dtype))
    record = structure.encode(graph,dict(node_types=nw,edge_types=ew))
    full = wl.normalize(record.counts)
    centers = [i for i,adj in enumerate(record.neighbors) if len(set(adj)-{i})>=2]
    local = wl.normalize([Counter(layer[i] for i in centers) for layer in record.layers]) if centers else full
    return full,local,len(centers)


def select(ids, m, setting, source):
    return sorted(ids,key=lambda i:(sha256(f'r17-pair|{setting}|{source}|{i}'.encode('utf-8')).hexdigest(),i))[:m]


def differences(full, local, source, target, vocabulary):
    xd, xu = (wl.sparse_rows(full,ids,vocabulary) for ids in (source,target))
    ld, lu = (wl.sparse_rows(local,ids,vocabulary) for ids in (source,target))
    shuffled = np.roll(np.arange(len(target)),-1)
    return dict(TargetPair=xu-lu,SourcePair=xd-ld,Shuffled=xu-lu[shuffled])


def geometry(v):
    v = v.copy()
    v.eliminate_zeros()
    energy = np.asarray(v.multiply(v).sum(axis=1)).ravel()
    trace = float(energy.sum())
    return dict(T=trace,nonzero=int(np.count_nonzero(energy)),
                max_row_share=None if trace==0 else float(energy.max()/trace),
                normalized_trace=1. if trace>TRACE_FLOOR else None)


def objective(x, y, v, t, params):
    w, b = params[:-1], params[-1]
    margin = x@w+b
    signed = (1-2*y)*margin
    residual = (1-2*y)*expit(signed)
    loss = np.logaddexp(0.,signed).mean() + (w@w)/(2*len(y))
    grad = (x.T@residual+w)/len(y)
    if v is not None:
        shift = v@w
        loss += (shift@shift)/(2*t)
        grad += (v.T@shift)/t
    return float(loss),np.r_[grad,residual.mean()]


def certificate(x,y,v,t,model):
    params = np.r_[model['weights'],model['intercept']]
    loss,grad = objective(x,y,v,t,params)
    return loss,float(np.max(abs(grad)))


def fit(x,y,v,meta,base,options):
    t = meta['T']
    if t<=TRACE_FLOOR:
        return base,dict(fallback='zero_trace', gradient_inf=base['gradient_inf'],
                         objective=base['training_objective'],iterations=0,retried=False)
    initial = np.r_[base['weights'],base['intercept']]
    initial_value,_ = objective(x,y,v,t,initial)
    opts = dict(maxiter=options['max_iterations'],gtol=options['gradient_tolerance'],
                ftol=options['function_tolerance'])
    result = minimize(lambda p: objective(x,y,v,t,p),initial,jac=True,method='L-BFGS-B',options=opts)
    retried = not result.success and str(result.message).startswith('ABNORMAL')
    if retried:
        opts['gtol'] = options['stationarity_tolerance']
        result = minimize(lambda p: objective(x,y,v,t,p),result.x,jac=True,method='L-BFGS-B',options=opts)
    fitted = dict(weights=result.x[:-1].tolist(),intercept=float(result.x[-1]))
    value,error = certificate(x,y,v,t,fitted)
    if (not result.success or not np.isfinite(value) or not np.isfinite(error)
            or value>initial_value+1e-12 or error>options['stationarity_tolerance']):
        raise ArithmeticError(f'Paired readout fit failed original-coordinate certificate: {result.message}, {error:g}')
    return fitted,dict(fallback=None,iterations=int(result.nit),retried=retried,gradient_inf=error,objective=value)


def margins(model,e):
    """Replay signed logits using only the full-view E feature matrix."""
    if model['status']=='single_class_D':
        return {name:np.full(e.shape[0],model['margin']) for name in ('WL_C1',*ARMS)}
    return {name:scores(e,coefficients) for name,coefficients in model['heads'].items()}
