"""Authenticate, freeze and verify the registered R18 source-episode experiment."""
from dataclasses import dataclass
from fractions import Fraction
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import geometry_models, residual_inputs as data
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.src import screen as r17
from . import anchors, relations

MAIN = 'TargetFull'
OLD_ARMS = ('WL_selected','Size','Base','WL_C1')
BASE = 'Anchor'
HARD = ('EqualCount',)
VOTES = (MAIN,'TargetUnary','SourceFull','PermMean','RelationOnly')
MARGINS = (*OLD_ARMS,BASE,'CalAnchor')
ACTIONS = (*MARGINS,*VOTES,*HARD)
CONTROLS = tuple(m for m in ACTIONS if m!=MAIN)
GATE_CONTROLS = tuple(m for m in CONTROLS if m not in (BASE,'RelationOnly'))
META = ('executed-config.json','roles.json','source-provenance.json','baseline-provenance.json',
        'fit-label-access.json','action-bounds.json')
MINIMUM = Fraction(1,200)


@dataclass(frozen=True)
class Prepared:
    original: object
    models: dict
    lineage: dict


def parents(cfg):
    return r17.parents(cfg)


def members(roles):
    return [*META,*prior_run.inventory(roles,'predictions'),*(f'models/{key}.json' for key in roles)]


def prepare(repo,here,cfg):
    original = r17.prepare(repo,here,cfg)
    root = here/cfg['r9_dir']/cfg['geometry_attempt']
    seal = roc_transfer.stage(root,'OBS',cfg['geometry_obs_sha256'])
    models,lineage = {},{}
    for key in original.reference.roles:
        relative = f'models/{key}.json'
        source = json.loads(prior_run.read_member(root,seal['files'],relative))
        if source['status']=='ok':
            models[key] = {k:source[k] for k in ('status','vocabulary','size_mean','size_scale')}
            models[key]['candidates'] = {m:source['candidates'][m] for m in anchors.MODELS}
        elif source['status']=='single_class_D':
            models[key] = {k:source[k] for k in ('status','constant')}
        else:
            raise PermissionError(f'Historical source anchor was unavailable: {key}')
        lineage[key] = dict(**original.lineage[key],geometry_model_sha256=seal['files'][relative])
    return Prepared(original,models,lineage)


def geometry_for(prep,role):
    return relations.Geometry.build(prep.original.full[role['setting'],role['source']])


def old_predictions(prep,key,role):
    features = prep.original.full[role['setting'],role['source']]
    e = role['roles']['E']
    model = prep.models[key]
    from_checkpoint = anchors.predict(features,e,model)
    predicted = prep.original.prior[key]
    for m in anchors.MODELS:
        name = 'WL_selected' if m=='WL' else m
        np.testing.assert_array_equal(from_checkpoint[m],predicted[f'margin_{name}'])
    fixed = prep.original.baseline[key][1]
    return {**predicted,'WL_C1':fixed['WL'],'margin_WL_C1':fixed['wl_e']}


def fit_direction(prep,key,role,y_d,y_c):
    ids = role['roles']
    features = prep.original.full[role['setting'],role['source']]
    options = prep.original.archive.plan['solver']
    if any(y.shape!=(len(ids[p]),) or not np.isin(y,[0,1]).all() for y,p in ((y_d,'D'),(y_c,'C'))):
        raise ValueError('Invalid scoped source labels')
    anchor = prep.models[key]
    anchors.validate_cal(features,ids,y_d,y_c,anchor)
    choice = anchors.select(anchor)
    folds = relations.folds(ids['D'],role['setting'],role['source'])
    z,models = anchors.cross_fit(features,ids,folds,y_d,y_c,options)
    geometry = relations.Geometry.build({i:features[i] for i in ids['D']})
    x,p,y,w = relations.training_pairs(folds,z,dict(zip(ids['D'],y_d,strict=True)),geometry)
    n = len(ids['D'])
    model = dict(choice=list(choice),folds=models,D_pairs=len(y),D_oof=[float(z[i]) for i in ids['D']],
                 calibrator=relations.calibrator(np.asarray([z[i] for i in ids['D']]),y_d,options),
                 Unary=relations.relation_head(x[:,:3],p,y,w,n,options),
                 Full=relations.relation_head(x,p,y,w,n,options),
                 references=relations.refs(ids['C'],ids['U'],role['setting'],role['source']))
    return model, predict(prep,key,role,model,geometry_for(prep,role))


def pair_probabilities(geom, left, right, ml, mr, fitted):
    p = relations.product(ml,mr)
    x = relations.columns(geom,left,right,ml,mr)
    u = relations.relation_probability(p,x[:,:3],fitted['Unary'])
    f = relations.relation_probability(p,x,fitted['Full'])
    return p,u,f


def predict(prep,key,role,model,geom):
    old = old_predictions(prep,key,role)
    ids = old['e_ids']
    features = prep.original.full[role['setting'],role['source']]
    choice = tuple(model['choice'])
    a,cal = anchors.chosen_margin(features,ids,prep.models[key],choice), model['calibrator']
    mi = relations.signed_margin(a)
    aa = relations.calibrated(a,cal)
    mc = relations.signed_margin(aa)
    votes,archive = {},{}
    for pool,refs in model['references'].items():
        aj = anchors.chosen_margin(features,refs,prep.models[key],choice)
        mj = relations.signed_margin(aj)
        r = len(refs)
        p,unary,full = pair_probabilities(geom,np.repeat(ids,r),np.tile(refs,len(ids)),
                                         np.repeat(mi,r),np.tile(mj,len(ids)),model)
        p,unary,full = (arr.reshape(len(ids),r) for arr in (p,unary,full))
        own = relations.apply_vote(mi,mj,p,full)
        archive.update({f'{pool}_ids':np.asarray(refs,dtype=np.int64),f'{pool}_margin':aj,
                        f'{pool}_Full_probability':full})
        if pool=='U':
            votes[MAIN] = own
            votes['TargetUnary'] = relations.apply_vote(mi,mj,p,unary)
            votes['PermMean'],votes['RelationOnly'] = relations.other_votes(mi,mj,full)
            archive['U_Unary_probability'] = unary
        else:
            votes['SourceFull'] = own
    i,j = np.triu_indices(len(ids),k=1)
    prod,unary,full = pair_probabilities(geom,ids[i],ids[j],mi[i],mi[j],model)
    calprod = relations.product(mc[i],mc[j])
    x = {**old,**archive,BASE:(a>0).astype(np.int64),'CalAnchor':(aa>0).astype(np.int64),
         f'margin_{BASE}':a,'margin_CalAnchor':aa, 'pair_i':i.astype(np.int64),'pair_j':j.astype(np.int64),
         'pair_Product':prod,'pair_CalProduct':calprod,'pair_Unary':unary,'pair_Full':full}
    for name,value in votes.items():
        x[name] = (value>0).astype(np.int64)
        x[f'vote_{name}'] = value
    x['EqualCount'] = geometry_models.equal_positive(ids,a,int(x[MAIN].sum()))
    return x


def action_bounds(roles,predicted):
    rows,settings = [],{}
    for key,role in roles.items():
        x = predicted[key]
        r = {k:role[k] for k in data.IDENTITY}
        n = len(x['e_ids'])
        for m in CONTROLS:
            change = int(np.count_nonzero(x[MAIN]!=x[m]))
            rows.append(dict(**r,right=m,n=n,changed=change,upper_exact=str(Fraction(change,n))))
    for s in sorted({r['setting'] for r in rows}):
        by_arm = {}
        for m in CONTROLS:
            values = [r for r in rows if r['setting']==s and r['right']==m]
            if len(values)!=8:
                raise PermissionError("R18 requires all eight directions of each setting")
            bound = sum(Fraction(r['upper_exact']) for r in values)/len(values)
            by_arm[m] = dict(upper_exact=str(bound),upper_pp=float(100*bound),
                             max_positive_directions=sum(r['changed']>0 for r in values),
                             max_positive_groups=len({r['source'] for r in values if r['changed']}))
        base = by_arm[BASE]
        settings[s] = dict(comparisons=by_arm,can_advance_by_action_bounds=(
            Fraction(base['upper_exact'])>=MINIMUM and base['max_positive_groups']>=3
            and all(Fraction(by_arm[m]['upper_exact'])>0 for m in GATE_CONTROLS)))
    possible = settings['proteins_edge']['can_advance_by_action_bounds'] and any(
        settings[s]['can_advance_by_action_bounds'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=settings,per_block=rows,can_advance_by_action_bounds=possible)


def validate_predictions(x,role):
    e = x['e_ids']
    if e.dtype!=np.int64 or e.tolist()!=role['roles']['E']:
        raise PermissionError('R18 lost the authorized E order')
    for m in MARGINS:
        r17.check_array(e,x[m],x[f'margin_{m}'])
    for m in VOTES:
        r17.check_array(e,x[m],x[f'vote_{m}'])
    eq = geometry_models.equal_positive(e,x[f'margin_{BASE}'],int(x[MAIN].sum()))
    if x['EqualCount'].dtype!=np.int64 or not np.array_equal(x['EqualCount'],eq):
        raise PermissionError('R18 equal-count actions differ')
    i,j = np.triu_indices(len(e),k=1)
    if not np.array_equal(i,x['pair_i']) or not np.array_equal(j,x['pair_j']):
        raise PermissionError('R18 pair order differs from E upper triangle')
    shapes = {f'pair_{m}':i.shape for m in relations.PAIR_ARMS}
    for pool in ('C','U'):
        ids = x[f'{pool}_ids']
        original = role['roles'][pool]
        z = x[f'{pool}_margin']
        if (ids.dtype!=np.int64 or ids.ndim!=1 or len(np.unique(ids))!=len(ids)
                or not set(ids)<=set(original)
                or len(ids)!=min(len(role['roles']['C']),len(role['roles']['U']))
                or z.shape!=ids.shape or z.dtype!=np.float64 or not np.isfinite(z).all()):
            raise PermissionError(f'R18 {pool} references differ from authorized graph IDs')
        shapes[f'{pool}_Full_probability'] = (len(e),len(ids))
    shapes['U_Unary_probability'] = shapes['U_Full_probability']
    for name,shape in shapes.items():
        v = x[name]
        if v.shape!=shape or v.dtype!=np.float64 or not np.isfinite(v).all() or not np.all((0.<=v)&(v<=1.)):
            raise PermissionError(f'Invalid probability for {name}')


def freeze(out,cfg,prep):
    orig = prep.original
    roles = orig.reference.roles
    (out/'models').mkdir()
    (out/'predictions').mkdir()
    predictions = {}
    with data.ScopedLabels(orig.archive.by_setting,roles,'fit') as labels:
        for key,r in roles.items():
            with labels.direction(key):
                yd = labels.load(key,'D',r['roles']['D'])
                yc = labels.load(key,'C',r['roles']['C'])
            model,x = fit_direction(prep,key,r,yd,yc)
            validate_predictions(x,r)
            predictions[key] = x
            write_json(out/f'models/{key}.json',model)
            prior_run.save_predictions(out,f'predictions/{key}.npz',x)
            print(f"R18 {key} choice={model['choice']} train_pairs={model['D_pairs']} "
                  f"E_pairs={len(x['pair_i'])}",flush=True)
    data.validate_receipt(roles,labels.actual,'fit')
    for name,value in (('roles',roles),('fit-label-access',labels.actual),
                       ('action-bounds',action_bounds(roles,predictions)),
                       ('source-provenance',orig.reference.provenance),('baseline-provenance',prep.lineage)):
        write_json(out/f'{name}.json',value)
    return prior_run.seal(out,'OBS',members(roles),parents(cfg))


def validate_model(prep,key,role,model):
    d = role['roles']['D']
    fs = relations.folds(d,role['setting'],role['source'])
    features = prep.original.full[role['setting'],role['source']]
    if (model['choice']!=list(anchors.select(prep.models[key]))
            or model['references']!=relations.refs(role['roles']['C'],role['roles']['U'],role['setting'],role['source'])
            or model['D_pairs']!=sum(len(f)*(len(f)-1)//2 for f in fs)):
        raise PermissionError(f'R18 frozen choice or reference/fold count differs: {key}')
    replay = {}
    for values,f in zip(model['folds'],fs,strict=True):
        expected = set(f)
        training = [i for i in d if i not in expected]
        if values['fit_ids']!=training or values['held_ids']!=f:
            raise PermissionError('R18 OOF graph order or isolation changed')
        if f:
            m = values['model']
            choice = anchors.select(m)
            if values['choice']!=list(choice):
                raise PermissionError('R18 fold selection differs')
            if m['status']=='ok' and any(m[k]!=v for k,v in anchors.design_from_D(features,training).items()):
                raise PermissionError('R18 fold feature geometry changed')
            replay.update(zip(f,anchors.chosen_margin(features,f,m,choice),strict=True))
    np.testing.assert_array_equal(model['D_oof'],[replay[i] for i in d])
    mode = prep.original.archive.plan['solver']['stationarity_tolerance']
    if (not 0.<=model['calibrator']['gradient_inf']<=mode or model['calibrator']['coefficients'][1]<0.
            or any(m['fallback'] is None and not 0.<=m['gradient_inf']<=mode
                   for m in (model['Unary'],model['Full']))):
        raise ArithmeticError('R18 fitted stationarity certificate changed')


def observed(out,cfg,prep,obs):
    roles = prep.original.reference.roles
    raw = prior_run.verified_members(out,'OBS',obs,members(roles),parents(cfg))
    for name,expected in (('roles',roles),('source-provenance',prep.original.reference.provenance),
                          ('baseline-provenance',prep.lineage)):
        if json.loads(raw[f'{name}.json'])!=expected:
            raise PermissionError(f'R18 {name} differs from immutable R9')
    data.validate_receipt(roles,json.loads(raw['fit-label-access.json']),'fit')
    predictions,models = {},{}
    for key,r in roles.items():
        x = inputs.npz_member(raw[f'predictions/{key}.npz'])
        model = json.loads(raw[f'models/{key}.json'])
        validate_predictions(x,r)
        validate_model(prep,key,r,model)
        expected = predict(prep,key,r,model,geometry_for(prep,r))
        if set(x)!=set(expected) or any(v.dtype!=x[n].dtype or not np.array_equal(v,x[n]) for n,v in expected.items()):
            raise PermissionError(f'R18 E predictions and probabilities fail replay: {key}')
        predictions[key],models[key] = x,model
    if action_bounds(roles,predictions)!=json.loads(raw['action-bounds.json']):
        raise PermissionError('R18 label-free action bounds differ after reload')
    return predictions,models
