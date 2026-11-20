"""Reproduce the frozen source anchors and construct four out-of-fold D scores."""
import numpy as np
from scipy.special import logit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import geometry_models as geom, wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural

MODELS = ('WL','Size','Base')


def select(model):
    if model['status']=='single_class_D':
        return 'Constant',None
    def candidate(m):
        fit = model['candidates'][m]
        c = fit['selected_C']
        validation = fit['validation']
        if [r['C'] for r in validation]!=structural.STRENGTHS or structural.choose(validation)!=c:
            raise PermissionError('The anchor candidates do not follow the R9 C selection')
        r = next(row for row in validation if row['C']==c)
        return -r['correct'], r['nll'], c, MODELS.index(m)
    name = min(MODELS,key=candidate)
    return name,model['candidates'][name]['selected_C']


def design(features, ids, model):
    vocab = {(d,bytes.fromhex(h)):i for i,(d,h) in enumerate(model['vocabulary'])}
    rows = {i:features[i][0] for i in ids}
    f = wl.sparse_rows(rows,ids,vocab)
    z = (np.stack([features[i][1] for i in ids])-model['size_mean'])/model['size_scale']
    return dict(WL=f,Size=z,Base=geom.matrix(f,z))


def predict(features,ids,model):
    if model['status']=='single_class_D':
        return {m:np.full(len(ids),model['constant']) for m in MODELS}
    return {m:scores(x,model['candidates'][m]['coefficients'])
            for m,x in design(features,ids,model).items()}


def chosen_margin(features,ids,model,choice):
    z = predict(features,ids,model)
    return z['WL' if choice[0]=='Constant' else choice[0]]


def design_from_D(features, ids):
    vocab = wl.vocabulary({i:features[i][0] for i in ids},ids)
    z = np.stack([features[i][1] for i in ids])
    average, scale = z.mean(axis=0),z.std(axis=0)
    scale[np.ptp(z,axis=0)==0.] = 1.
    return dict(vocabulary=[[d,h.hex()] for d,h in vocab],
                size_mean=average.tolist(),size_scale=scale.tolist())


def fit(features, train, cal, y_train, y_cal, options):
    if len(set(y_train))<2:
        p = (int(y_train.sum())+.5)/(len(y_train)+1)
        return dict(status='single_class_D',constant=float(logit(p)))
    transform = design_from_D(features,train)
    d,c = (design(features,ids,transform) for ids in (train,cal))
    branches = {}
    for name in MODELS:
        fitted = structural.fit_candidates(d[name],c[name],y_train,y_cal,dict(C=structural.STRENGTHS,solver=options))
        branches[name] = dict(selected_C=fitted['selected_C'],validation=fitted['validation'],
                              coefficients=structural.select_model(fitted))
    return dict(status='ok',**transform,candidates=branches)


def validate_cal(features,role,y_d,y_c,model):
    """Check D coordinates and C statistics of the selected frozen branches."""
    if model['status']=='single_class_D':
        p = (int(y_d.sum())+.5)/(len(y_d)+1)
        if len(set(y_d))!=1 or model['constant']!=float(logit(p)):
            raise PermissionError('Historical single-class fallback differs from current D')
        return
    if len(set(y_d))!=2 or any(model[k]!=v for k,v in design_from_D(features,role['D']).items()):
        raise PermissionError('Historical WL or size geometry differs from full D')
    z = predict(features,role['C'],model)
    for m in MODELS:
        fit_ = model['candidates'][m]
        c = fit_['selected_C']
        entry = next(r for r in fit_['validation'] if r['C']==c)
        if entry!=structural.validation_metrics(c,z[m],y_c):
            raise PermissionError(f'Historical C-selection statistics fail to reproduce for {m}')


def cross_fit(features, ids, fold_ids, y_d, y_c, options):
    d,c = ids['D'],ids['C']
    labels = dict(zip(d,y_d,strict=True))
    folds, z = [],{}
    for held in fold_ids:
        if not held:
            folds.append(dict(fit_ids=list(d),held_ids=[],model=None,choice=None))
            continue
        denied = set(held)
        train = [i for i in d if i not in denied]
        model = fit(features,train,c,np.asarray([labels[i] for i in train],dtype=np.int64),y_c,options)
        choice = select(model)
        value = chosen_margin(features,held,model,choice)
        z.update(zip(held,value,strict=True))
        folds.append(dict(fit_ids=train,held_ids=held,model=model,choice=list(choice)))
    if set(z)!=set(d):
        raise PermissionError('Out-of-fold scores must cover every D graph exactly once')
    return z,folds
