"""D-only source models and label-free capped kernel mean matching."""
from fractions import Fraction

import numpy as np
from scipy.special import expit, logit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import roc_rate, thresholds
from . import bridge, wl


ESTIMATORS = ('WeightedResidual','WLMean','UniformResidual','WeightedLabels','Neighbors8','SizeMean','CPrior')
METHODS = (*ESTIMATORS,'WL','Size')


def fit(features, ids, labels, optimizer):
    if not ids or labels.shape!=(len(ids),) or not np.isin(labels,[0,1]).all():
        raise ValueError('D requires one authorized binary label per source training graph')
    if len(set(labels))<2:
        p = Fraction(2*int(labels.sum())+1,2*(len(labels)+1))
        return dict(status='single_class_D',prior_exact=str(p),margin=float(logit(float(p))))

    rows = {i:features[i][0] for i in ids}
    vocabulary = wl.vocabulary(rows,ids)
    sizes = np.stack([features[i][1] for i in ids])
    mean,scale = sizes.mean(axis=0),sizes.std(axis=0)
    scale[np.ptp(sizes,axis=0)==0.] = 1.
    matrices = (wl.sparse_rows(rows,ids,vocabulary),(sizes-mean)/scale)
    # One fixed candidate. Reuse R10's same-objective numerical recovery
    # with D itself as the bookkeeping cohort. No selection step sees C.
    config = dict(C=[1.],solver=optimizer)
    wl_model,size_model = [structural.fit_candidates(x,x,labels,labels,config)['candidates']['1.0'] for x in matrices]
    return dict(status='ok',wl=wl_model,size=size_model,mean=mean.tolist(),scale=scale.tolist(),
                vocabulary=[[depth,color.hex()] for depth,color in vocabulary])


def margins(model, features, ids):
    if not ids:
        return np.empty(0,dtype=np.float64),np.empty(0,dtype=np.float64)
    if model['status']=='single_class_D':
        z = np.full(len(ids),model['margin'],dtype=np.float64)
        return z,z.copy()
    names = {(depth,bytes.fromhex(color)):i for i,(depth,color) in enumerate(model['vocabulary'])}
    rows = {i:features[i][0] for i in ids}
    size = np.stack([features[i][1] for i in ids])
    return (scores(wl.sparse_rows(rows,ids,names),model['wl']),
            scores((size-model['mean'])/model['scale'],model['size']))


def project_capped(values, cap):
    """Euclidean projection by the monotone sum of clipped coordinates."""
    lo,hi = float(values.min()-cap),float(values.max())
    for _ in range(55):
        mid = (lo+hi)/2
        if np.clip(values-mid,0.,cap).sum()>1.:
            lo = mid
        else:
            hi = mid
    weights = np.clip(values-(lo+hi)/2,0.,cap)
    if abs(weights.sum()-1.)>1e-12:
        raise ArithmeticError('Projection does not meet the capped-simplex constraint')
    return weights


def first_order_gap(gradient,w,cap):
    order = np.argsort(gradient,kind='stable')
    # Linear minimizer fills the cheapest coordinates to the cap.
    vertex = np.clip(1.-cap*np.arange(len(w)),0.,cap)
    return float(gradient[order]@(w[order]-vertex))


def match(gram, mean_inner, mean_norm2, cfg):
    """Solve in C coordinates, then certify first-order global optimality."""
    n = len(mean_inner)
    if (not n or gram.shape!=(n,n) or not np.isfinite(gram).all() or not np.isfinite(mean_inner).all()
            or not np.isfinite(mean_norm2) or mean_norm2<0.):
        raise ValueError('Invalid calibration kernel moments')
    cap = cfg['max_relative_weight']/n
    regularizer = cfg['regularization_per_calibration_graph']*n
    uniform = np.full(n,1./n)
    q = gram + regularizer*np.eye(n)
    c = mean_inner + regularizer*uniform
    lipschitz = np.abs(q).sum(axis=1).max()
    momentum = (np.sqrt(lipschitz)-np.sqrt(regularizer))/(np.sqrt(lipschitz)+np.sqrt(regularizer))
    opts = cfg['weight_solver']
    w = y = uniform
    gap = first_order_gap(2*(q@w-c),w,cap)
    iteration = 0
    while gap>opts['max_first_order_gap'] and iteration<opts['max_iterations']:
        candidate = project_capped(y-(q@y-c)/lipschitz,cap)
        y = candidate + momentum*(candidate-w)
        w = candidate
        gap = first_order_gap(2*(q@w-c),w,cap)
        iteration += 1
    if (not np.isfinite(gap) or gap<-1e-11 or gap>opts['max_first_order_gap']
            or not np.isfinite(w).all() or np.any(w<0.) or np.any(w>cap) or abs(w.sum()-1.)>1e-12):
        raise ArithmeticError(f'Uncertified capped mean matching after {iteration} iterations, gap={gap:g}')

    def distance(x):
        squared = float(x@gram@x-2*mean_inner@x+mean_norm2)
        if not np.isfinite(squared) or squared < -1e-10:
            raise ArithmeticError('Invalid squared WL mean distance')
        return float(np.sqrt(max(0.,squared)))

    before,after = distance(uniform),distance(w)
    return w,dict(distance_uniform=before,distance_matched=after,capped_count=int((w>=cap-1e-10).sum()),
                  effective_n=float(1/np.square(w).sum()),first_order_gap=gap,iterations=iteration,
                  cap=cap,regularizer=regularizer,mean_target_norm2=float(mean_norm2),
                  objective_uniform=before**2,objective_matched=after**2+regularizer*float(np.square(w-uniform).sum()))


def wl_geometry(features,c,u):
    rows = {i:features[i][0] for i in c+u}
    names = wl.vocabulary(rows,c+u)
    cm, um = (wl.sparse_rows(rows,ids,names) for ids in (c,u))
    return (cm@cm.T).toarray(),(um@cm.T).toarray(),um


def estimates(f_c,yc,f_u,size_u,w,neighbor_weights,*,constant=None):
    residual = yc-f_c
    base = float(f_u.mean())
    residual_prediction = base+float(w@residual)
    label_prediction = float(w@yc)
    model_gap = base-float(w@f_c)
    uniform_prediction = base+float(residual.mean())
    if constant is not None:
        # When f is constant, both residual corrections reduce to their direct labels.
        if abs(model_gap)>1e-12 or abs(residual_prediction-label_prediction)>1e-12:
            raise ArithmeticError('Constant WL scores did not reduce to weighted labels')
        if abs(uniform_prediction-float(yc.mean()))>1e-12:
            raise ArithmeticError('Uniform residual did not reduce to the C prior')
        base = float(constant)
        residual_prediction,uniform_prediction,model_gap = label_prediction,float(yc.mean()),0.
    identity_residual = residual_prediction-label_prediction-model_gap
    if abs(identity_residual)>1e-12:
        raise ArithmeticError('Weighted residual and direct-label identity failed')
    return dict(WeightedResidual=residual_prediction,WLMean=base,UniformResidual=uniform_prediction,
                WeightedLabels=label_prediction,Neighbors8=float((neighbor_weights@yc).mean()),
                SizeMean=float(size_u.mean()),CPrior=float(yc.mean())),model_gap,identity_residual


def clipped_rates(raw, *, source_prior, constant, has_u):
    rates = {name:Fraction.from_float(float(np.clip(value,0.,1.))) for name,value in raw.items()}
    if has_u:
        rates['CPrior'] = source_prior
        if constant is not None:
            rates.update(WLMean=constant,SizeMean=constant,UniformResidual=source_prior)
    elif constant is not None:
        rates = dict.fromkeys(ESTIMATORS,constant)
    return rates


def fitted_predictions(features, roles, yd, yc, cfg, optimizer):
    model = fit(features,roles['D'],yd,optimizer)
    c,u,e = (roles[name] for name in ('C','U','E'))
    wl_c,_ = margins(model,features,c)
    wl_u,sz_u = margins(model,features,u)
    wl_e,sz_e = margins(model,features,e)
    if not all(np.isfinite(x).all() for x in (wl_c,wl_u,wl_e,sz_u,sz_e)):
        raise FloatingPointError('D model produced nonfinite margins')
    zu,ze = sz_u>0.,sz_e>0.
    actions = {name:np.zeros(len(e),dtype=np.int64) for name in ESTIMATORS}
    actions.update(WL=(wl_e>0.).astype(np.int64),Size=ze.astype(np.int64))
    weights = np.zeros((2,len(c)),dtype=np.float64)
    groups = []
    gram,kernel,um = wl_geometry(features,c,u)
    constant = Fraction(model['prior_exact']) if model['status']=='single_class_D' else None
    f_u,f_c,p_u = (tuple(np.full(len(z),float(constant)) for z in (wl_u,wl_c,sz_u))
                 if constant is not None else tuple(map(expit,(wl_u,wl_c,sz_u))))
    source_prior = Fraction(int(yc.sum()),len(yc))
    neighbor, fallback = bridge.row_weights(kernel,c,cfg['neighbor_count'])
    for z in (0,1):
        selected_u,selected_e = zu==z,ze==z
        n_u,n_e = int(selected_u.sum()),int(selected_e.sum())
        if n_u:
            mean = np.asarray(um[selected_u].mean(axis=0)).ravel()
            weights[z],diagnostics = match(gram,kernel[selected_u].mean(axis=0),mean@mean,cfg)
            raw,model_gap,identity = estimates(f_c,yc,f_u[selected_u],p_u[selected_u],weights[z],neighbor[selected_u],
                                                   constant=constant)
            record = dict(size_action=z,status='fit',u_n=n_u,e_n=n_e,raw=raw,diagnostics=diagnostics,
                          weighted_model_gap=model_gap,decomposition_error=identity)
        else:
            raw = dict.fromkeys(ESTIMATORS,float(f_u.mean()))
            record = dict(size_action=z,status='fallback_WL',u_n=0,e_n=n_e,raw=raw,diagnostics=None,
                          weighted_model_gap=None,decomposition_error=None)
        rates = clipped_rates(raw,source_prior=source_prior,constant=constant,has_u=bool(n_u))
        thresholds_used = {}
        for name in ESTIMATORS:
            if n_u:
                cut,count = roc_rate.quantile(wl_u[selected_u],rates[name])
            else:
                cut,count = 0.,0
            thresholds_used[name] = dict(cut=thresholds.encode_threshold(cut),u_predicted_positive=count)
            actions[name][selected_e] = wl_e[selected_e]>cut
        groups.append(dict(**record,clipped={k:float(v) for k,v in rates.items()},
                           exact_rate={k:str(v) for k,v in rates.items()},thresholds=thresholds_used))
    arrays = dict(u_ids=np.asarray(u),e_ids=np.asarray(e),c_ids=np.asarray(c),
                  wl_u=wl_u,wl_e=wl_e,size_u=sz_u,size_e=sz_e,weights=weights,**actions)
    return model,dict(groups=groups,nearest_fallback_rows=fallback,calibration_positive=int(yc.sum()),
                      calibration_n=len(yc)),arrays
