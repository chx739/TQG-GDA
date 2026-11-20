"""Run and seal the PROTEINS node attribute source-only screen."""
from dataclasses import dataclass
from fractions import Fraction
import json
from statistics import fmean

import numpy as np
import torch
from scipy.special import logit
from torch_geometric.data import Data

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    geometry_models, reference_geometry, residual_inputs as data, true_rate_replay)
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from . import source_attributes as attributes, structural

BASES = geometry_models.BASES
NEW = attributes.METHOD
METHODS = (*BASES,NEW)
META = ('executed-config.json','roles.json','input-verification.json','fit-label-access.json')


@dataclass(frozen=True)
class Inputs:
    roles: dict
    models: dict
    predictions: dict
    features: dict
    provenance: dict
    authority: object
    solver: dict


def parents(cfg):
    return {k:cfg[k] for k in ('geometry_obs_sha256','roles_obs_sha256')}


def members(roles):
    return [*META, *prior_run.inventory(roles,'predictions'),*(f'models/{key}.json' for key in roles)]


def load_geometry(here,cfg,reference):
    root = here/cfg['geometry_dir']/cfg['geometry_attempt']
    obs = roc_transfer.stage(root,'OBS',cfg['geometry_obs_sha256'])
    files = obs['files']
    if (obs['predecessors']!={'source_obs_sha256':cfg['roles_obs_sha256']}
            or set(files)!=set(reference_geometry.members(reference.roles))):
        raise PermissionError('Frozen R9 geometry is not bound to the complete original role registry')

    def read(name):
        return prior_run.read_member(root,files,name)

    for name,value in (('roles',reference.roles),('source-topology',reference.topology),
                       ('source-provenance',reference.provenance)):
        if json.loads(read(f'{name}.json'))!=value:
            raise PermissionError(f'R9 geometry {name} differs from its original authority')
    label_access = json.loads(read('fit-label-access.json'))
    data.validate_receipt(reference.roles,label_access,'fit')
    roles, models, predictions = {}, {}, {}
    keep = ['e_ids',*BASES,*(f'margin_{m}' for m in BASES)]
    for k,role in reference.roles.items():
        if role['setting']!=cfg['setting']:
            continue
        roles[k] = dict(role,roles={r:role['roles'][r] for r in ('D','C','E')})
        old_model = json.loads(read(f'models/{k}.json'))
        if old_model['status']=='ok':
            models[k] = {name:old_model[name] for name in ('status','vocabulary','size_mean','size_scale',
                                                           'selected_base_C')}
            models[k]['candidates'] = {m:old_model['candidates'][m] for m in BASES}
        elif old_model['status']=='single_class_D':
            models[k] = {name:old_model[name] for name in ('status','constant')}
        else:
            raise PermissionError(f'Unknown historical model status for {k}')
        x = inputs.npz_member(read(f'predictions/{k}.npz'),keep)
        validate_arrays(x,role['roles']['E'],BASES)
        predictions[k] = x
    if len(roles)!=8 or any(x['seed']!=cfg['seed'] for x in roles.values()):
        raise PermissionError('Expected all eight PROTEINS directions from seed 3405')
    return roles,models,predictions


def prepare(repo,here,cfg):
    if (cfg['setting'],cfg['seed'])!=('proteins_edge',3405):
        raise PermissionError('The source attribute screen is fixed to PROTEINS seed 3405')
    ref_cfg = dict(source_attempt=cfg['roles_attempt'],source_obs_sha256=cfg['roles_obs_sha256'])
    reference = reference_geometry.authenticate(here/cfg['geometry_dir'],ref_cfg)
    roles,models,predictions = load_geometry(here,cfg,reference)
    archive = data.Archive(repo,reference.plan,load_graphs=False)
    true_rate_replay.validate_e_roles(archive,reference)
    authority = archive.by_setting[cfg['setting']]
    graph_name = 'inputs/graphs.pt'
    graph_path = authority._member(graph_name,'INPUT')
    if (digest(graph_path)!=authority.receipt[graph_name] or
            authority.receipt[graph_name]!=reference.provenance[cfg['setting']][graph_name]):
        raise PermissionError('R5 graph container changed from the original role assignment')
    authority.graphs = torch.load(graph_path,mmap=True,weights_only=True,map_location='cpu')
    authority.graph_hashes = json.loads(authority._read('inputs/graph_hashes.json'))
    if len(authority.graphs)!=authority.count or len(authority.graph_hashes)!=authority.count:
        raise PermissionError('R5 graph inventory differs')
    owners,types,values,hashes = attributes.raw_columns(authority.plan,authority.count)
    features, graph_checks = {}, {}
    for source in authority.domains:
        key = f"{cfg['setting']}-M{source}-seed-{cfg['seed']}"
        top = reference.topology[key]
        s,v = top['source_ids'],top['validation_ids']
        approved = {i for row in roles.values() if row['source']==source for ids in row['roles'].values() for i in ids}
        if approved!=set(s+v):
            raise PermissionError('The two registered D/C/E direction grants differ from source S/V')
        graphs = data.source_graphs(repo,authority,source,s,v)
        feature = {}
        num_nodes = 0
        for i,record in graphs.items():
            median,n = attributes.graph_attributes(i,record,owners,types,values)
            num_nodes += n
            f,z = structural.features(Data(**record),cfg['setting'])
            feature[i] = f,z,median
        features[source] = feature
        graph_checks[key] = dict(graphs=len(feature),nodes=num_nodes,checked_within_type_permutation=len(feature))
    provenance = dict(raw_files=hashes,verified_graphs=graph_checks,authenticated_source_members=dict(authority.receipt),
                      label_records_sha256=authority.seal_files['INPUT']['inputs/label_records.json'],
                      parent_checksums=parents(cfg))
    return Inputs(roles,models,predictions,features,provenance,authority,archive.plan['solver'])


def validate_arrays(x, ids, methods):
    if set(x)!={'e_ids', *methods, *(f'margin_{m}' for m in methods)}:
        raise PermissionError('Unexpected frozen prediction inventory')
    if x['e_ids'].dtype!=np.int64 or x['e_ids'].tolist()!=ids:
        raise PermissionError('Frozen E graph order differs')
    for name in methods:
        z,a = x[f'margin_{name}'],x[name]
        if (z.dtype!=np.float64 or z.shape!=(len(ids),) or not np.isfinite(z).all()
                or a.dtype!=np.int64 or not np.array_equal(a,z>0.)):
            raise PermissionError(f'Malformed frozen E prediction: {name}')


def validate_baselines(old,style,design_c,design_e,yd,yc,pred):
    if old['status']=='single_class_D':
        p = (int(yd.sum())+.5)/(len(yd)+1)
        if len(set(yd))!=1 or old['constant']!=float(logit(p)):
            raise PermissionError('Historical constant differs from D class')
        if any(not np.array_equal(pred[f'margin_{m}'],np.full(len(pred['e_ids']),old['constant'])) for m in BASES):
            raise PermissionError('Historical constant fails to replay')
        return
    if (len(set(yd))!=2 or old['selected_base_C']!=old['candidates']['Base']['selected_C']
            or any(old[k]!=style[k] for k in ('vocabulary','size_mean','size_scale'))):
        raise PermissionError('Historical D coordinates or selected Base parameter differ')
    for m,fit in old['candidates'].items():
        c = fit['selected_C']
        control = fit['validation']
        c_selected = [v for v in control if v['C']==c]
        if ([v['C'] for v in control]!=structural.STRENGTHS or structural.choose(control)!=c
                or c_selected!=[structural.validation_metrics(c,scores(design_c[m],fit['coefficients']),yc)]
                or not np.array_equal(scores(design_e[m],fit['coefficients']),pred[f'margin_{m}'])):
            raise PermissionError(f'Historical {m} C selection or E prediction fails replay')


def freeze(output,cfg,prepared):
    (output/'models').mkdir()
    (output/'predictions').mkdir()
    with data.ScopedLabels({cfg['setting']:prepared.authority},prepared.roles,'fit') as port:
        for key,role in prepared.roles.items():
            ids = role['roles']
            with port.direction(key):
                yd = port.load(key,'D',ids['D'])
                yc = port.load(key,'C',ids['C'])
            features = prepared.features[role['source']]
            style = attributes.source_design(features,ids)
            d,c,e = (attributes.design_for(features,ids[part],style) for part in ('D','C','E'))
            old = prepared.predictions[key]
            validate_baselines(prepared.models[key],style,c,e,yd,yc,old)
            new = attributes.fit_new(d,c,yd,yc,prepared.solver)
            margin = attributes.predict_new(new,e[NEW])
            x = dict(old,**{NEW:structural.decisions(margin),f'margin_{NEW}':margin})
            validate_arrays(x,ids['E'],METHODS)
            write_json(output/f'models/{key}.json',dict(design=style,fit=new))
            prior_run.save_predictions(output,f'predictions/{key}.npz',x)
            print(f"R10 attribute {key} E={len(ids['E'])} C={new['selected_C']}",flush=True)
    data.validate_receipt(prepared.roles,port.actual,'fit')
    for name,value in (('fit-label-access',port.actual),('roles',prepared.roles),
                       ('input-verification',prepared.provenance)):
        write_json(output/f'{name}.json',value)
    return prior_run.seal(output,'OBS',members(prepared.roles),parents(cfg))


def observed(output,cfg,prepared,obs):
    raw = prior_run.verified_members(output,'OBS',obs,members(prepared.roles),parents(cfg))
    if (json.loads(raw['roles.json'])!=prepared.roles or
            json.loads(raw['input-verification.json'])!=prepared.provenance):
        raise PermissionError('Frozen roles or input checks differ from the registered scope')
    data.validate_receipt(prepared.roles,json.loads(raw['fit-label-access.json']),'fit')
    predictions, model_stats = {}, {}
    for key,row in prepared.roles.items():
        ids = row['roles']
        x = inputs.npz_member(raw[f'predictions/{key}.npz'])
        validate_arrays(x,ids['E'],METHODS)
        if any(not np.array_equal(x[s],val) for s,val in prepared.predictions[key].items()):
            raise PermissionError(f'Reused baseline changed in new OBS: {key}')
        record = json.loads(raw[f'models/{key}.json'])
        features = prepared.features[row['source']]
        fit = record['fit']
        if record['design']!=attributes.source_design(features,ids):
            raise PermissionError(f'Attribute standardization no longer matches D: {key}')
        if fit['status']=='ok' and (structural.choose(fit['validation'])!=fit['selected_C']
                                    or [v['C'] for v in fit['validation']]!=structural.STRENGTHS):
            raise PermissionError(f'Attribute model did not use all registered options: {key}')
        e = attributes.design_for(features,ids['E'],record['design'])
        if not np.array_equal(attributes.predict_new(fit,e[NEW]),x[f'margin_{NEW}']):
            raise PermissionError(f'Sealed E attribute logits cannot be replayed: {key}')
        predictions[key] = x
        model_stats[key] = dict(selected_C=fit['selected_C'],**record['design']['attribute'])
    return predictions, model_stats


def evaluate(x,y):
    if y.shape!=x['e_ids'].shape or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid E graph labels')
    scores_, pairs = {}, {}
    for m in METHODS:
        confusion = roc_score.confusion(x[m],y)
        tn,_,_,tp = confusion
        nll = float(np.logaddexp(0.,(1-2*y)*x[f'margin_{m}']).mean())
        scores_[m] = (Fraction(tn+tp,len(y)),roc_score.f1(confusion),nll,confusion)
    for base in BASES:
        count = true_rate_replay.paired_counts(x[NEW],x[base],y)
        a,f,nll,_ = scores_[NEW]
        b,g,l,_ = scores_[base]
        if a-b!=Fraction(count['net'],len(y)) or count['changed']!=count['corrected']+count['harmed']:
            raise ArithmeticError('Binary pair and classification deltas differ')
        pairs[base] = dict(**count,n=len(y),acc=a-b,f1=f-g,nll=nll-l)
    return scores_,pairs


def pair_columns(row):
    n = row['n']
    return dict(n=n,corrected=row['corrected'],harmed=row['harmed'],net=row['net'],
                changed=row['changed'],accuracy_exact=str(row['acc']),accuracy_pp=float(100*row['acc']),
                macro_f1_pp=float(100*row['f1']),nll_delta=row['nll'],disagreement_pp=100*row['changed']/n)


def exact_mean(values):
    values = tuple(values)
    return sum(values)/len(values)


def summarize(roles,scores_by_key,pairs_by_key):
    summaries, contrasts = [], []
    for source in (None,0,1,2,3):
        keys = [k for k,r in roles.items() if source is None or r['source']==source]
        if len(keys)!=(8 if source is None else 2):
            raise ArithmeticError('Missing registered source direction from summary')
        for m in METHODS:
            rows = [scores_by_key[k][m] for k in keys]
            a = exact_mean(r[0] for r in rows)
            summaries.append(dict(source=source,directions=len(keys),method=m,accuracy_exact=str(a),
                                  accuracy=float(a),macro_f1=float(exact_mean(r[1] for r in rows)),
                                  nll=fmean(r[2] for r in rows)))
        for base in BASES:
            rows = [pairs_by_key[k][base] for k in keys]
            gain = exact_mean(r['acc'] for r in rows)
            totals = {m:sum(r[m] for r in rows) for m in ('n','net','corrected','harmed','changed')}
            rates = {m+'_pp':float(100*exact_mean(Fraction(r[m],r['n']) for r in rows))
                     for m in ('corrected','harmed','changed')}
            contrasts.append(dict(source=source,directions=len(keys),left=NEW,right=base,**totals,**rates,
                                  accuracy_exact=str(gain),accuracy_pp=float(100*gain),
                                  macro_f1_pp=float(100*exact_mean(r['f1'] for r in rows)),
                                  nll_delta=fmean(r['nll'] for r in rows)))
    return summaries,contrasts


def gate(contrasts):
    overall = {r['right']:Fraction(r['accuracy_exact']) for r in contrasts if r['source'] is None}
    source = {str(r['source']):Fraction(r['accuracy_exact']) for r in contrasts
              if r['source'] is not None and r['right']=='Base'}
    count = sum(v>0 for v in source.values())
    clauses = dict(base_gain_at_least_half_pp=overall['Base']>=Fraction(1,200),
                   strictly_above_wl=overall['WL']>0,strictly_above_size=overall['Size']>0,
                   at_least_three_positive_source_groups=count>=3)
    return dict(advance_source_baseline=all(clauses.values()),clauses=clauses,positive_source_groups=count,
                overall_accuracy_gain_exact={k:str(v) for k,v in overall.items()},
                overall_accuracy_gain_pp={k:float(100*v) for k,v in overall.items()},
                source_group_vs_base_exact={k:str(v) for k,v in source.items()})


def score(output,cfg,prepared,obs,*,execution):
    predictions, stats = observed(output,cfg,prepared,obs)
    write_json(output/'score-execution.json',execution)
    endpoints, pairs, scores_by_key, pairs_by_key = [],[],{},{}
    with data.ScopedLabels({cfg['setting']:prepared.authority},prepared.roles,'replay_E') as port:
        for key,row in prepared.roles.items():
            with port.direction(key):
                y = port.load(key,'E',row['roles']['E'])
            ep,pa = evaluate(predictions[key],y)
            scores_by_key[key],pairs_by_key[key] = ep,pa
            ident = {k:row[k] for k in data.IDENTITY}
            for m,(a,f,nll,(tn,fp,fn,tp)) in ep.items():
                endpoints.append(dict(**ident,method=m,n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,positive=tp+fp,
                                      accuracy=float(a),macro_f1=float(f),nll=nll))
            for base,pair in pa.items():
                pairs.append(dict(**ident,left=NEW,right=base,**pair_columns(pair)))
    data.validate_receipt(prepared.roles,port.actual,'replay_E')
    ep,pa = summarize(prepared.roles,scores_by_key,pairs_by_key)
    tables = {'endpoints':endpoints,'paired-blocks':pairs,'summary':ep,'paired-summary':pa}
    for name,rows in tables.items():
        write_table(output/f'{name}.csv',rows)
    for name,value in (('e-label-access',port.actual),('gate',gate(pa)),('attribute-scaling',stats)):
        write_json(output/f'{name}.json',value)
    files = ['OBS.json','score-execution.json','e-label-access.json','gate.json','attribute-scaling.json',
             *[f'{name}.csv' for name in tables]]
    return prior_run.seal(output,'FINAL',files,dict(obs_sha256=obs))
