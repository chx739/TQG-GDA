"""Authenticate, fit, freeze and score the R17 source-episode screen."""
from dataclasses import dataclass
from fractions import Fraction
import json
from statistics import fmean

import numpy as np
from scipy.special import logit

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    geometry_models, reference_geometry, residual_inputs as data, true_rate_replay, weighted_residuals, wl)
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from . import readout

BASE = 'WL_C1'
MAIN = 'TargetPair'
AUGMENTED = (BASE,*readout.ARMS)
HISTORICAL = ('WL_selected','Size','Base')
MARGINS = (*AUGMENTED,*HISTORICAL)
MATCH = 'EqualPositive'
METHODS = (*MARGINS,MATCH)
CONTROLS = tuple(m for m in METHODS if m!=MAIN)
META = ('executed-config.json','roles.json','source-provenance.json','baseline-provenance.json',
        'fit-label-access.json','action-bounds.json')
MINIMUM = Fraction(1,200)


@dataclass(frozen=True)
class Prepared:
    reference: object
    archive: object
    full: dict
    baseline: dict
    prior: dict
    lineage: dict


def parents(cfg):
    return {k:cfg[k] for k in ('source_obs_sha256','geometry_obs_sha256')}


def members(roles):
    return [*META,*prior_run.inventory(roles,'predictions'),*(f'models/{key}.json' for key in roles)]


def check_array(ids,action,z):
    if (z.shape!=ids.shape or z.dtype!=np.float64 or not np.isfinite(z).all()
            or action.dtype!=np.int64 or not np.array_equal(action,z>0.)):
        raise PermissionError('Frozen E margins and actions disagree')


def load_baselines(here,cfg,reference):
    r9 = here/cfg['r9_dir']
    root = r9/cfg['source_attempt']
    other = r9/cfg['geometry_attempt']
    old = roc_transfer.stage(root,'OBS',cfg['source_obs_sha256'])
    geom = roc_transfer.stage(other,'OBS',cfg['geometry_obs_sha256'])
    files, more = old['files'],geom['files']
    if (geom['predecessors']!={'source_obs_sha256':cfg['source_obs_sha256']}
            or set(more)!=set(reference_geometry.members(reference.roles))
            or set(files)!=set(weighted_residuals.members(reference.roles))):
        raise PermissionError('R9 baselines do not share the complete original role authority')
    for name,value in (('roles',reference.roles),('source-topology',reference.topology),
                       ('source-provenance',reference.provenance)):
        if json.loads(prior_run.read_member(other,more,f'{name}.json'))!=value:
            raise PermissionError(f'R9 geometry {name} changed')
    data.validate_receipt(reference.roles,json.loads(prior_run.read_member(other,more,'fit-label-access.json')),'fit')
    baseline,prior,lineage = {},{},{}
    for key,role in reference.roles.items():
        model_name,pred_name = f'models/{key}.json',f'predictions/{key}.npz'
        fixed = json.loads(prior_run.read_member(root,files,model_name))
        vals = inputs.npz_member(prior_run.read_member(root,files,pred_name),['e_ids','wl_e','WL'])
        geom_actions = inputs.npz_member(prior_run.read_member(other,more,pred_name),
                                         ['e_ids','WL','Size','Base','margin_WL','margin_Size','margin_Base'])
        ids = np.asarray(role['roles']['E'],dtype=np.int64)
        for x in (vals,geom_actions):
            if x['e_ids'].dtype!=ids.dtype or not np.array_equal(x['e_ids'],ids):
                raise PermissionError(f'R9 baseline E order differs for {key}')
        check_array(ids,vals['WL'],vals['wl_e'])
        for m in ('WL','Size','Base'):
            check_array(ids,geom_actions[m],geom_actions[f'margin_{m}'])
        geom_actions['WL_selected'] = geom_actions.pop('WL')
        geom_actions['margin_WL_selected'] = geom_actions.pop('margin_WL')
        baseline[key],prior[key] = (fixed,vals),geom_actions
        lineage[key] = dict(C1_model_sha256=files[model_name],C1_predictions_sha256=files[pred_name],
                            geometry_predictions_sha256=more[pred_name])
    return baseline,prior,lineage


def prepare(repo,here,cfg):
    ref = reference_geometry.authenticate(here/cfg['r9_dir'],cfg)
    archive,full = reference_geometry.prepare(repo,ref)
    baseline,prior,lineage = load_baselines(here,cfg,ref)
    return Prepared(ref,archive,full,baseline,prior,lineage)


def fit_roles(roles):
    # Existing broker accepts exact grants only. Empty C grants are uncallable.
    return {key:dict(r,roles={'D':r['roles']['D'],'C':[]}) for key,r in roles.items()}


def validate_fit_reads(roles,actual):
    if actual!={key:{'D':r['roles']['D']} for key,r in roles.items()}:
        raise PermissionError('R17 fitting must parse only the current D labels')


def feature_matrices(prep,role,cache):
    ids = role['roles']
    setting,source = role['setting'],role['source']
    d,u,e = (ids[r] for r in ('D','U','E'))
    features = prep.full[setting,source]
    full = {i:features[i][0] for i in (*d,*u,*e)}
    m = min(len(d),len(u))
    s,t = (readout.select(pool,m,setting,source) for pool in (d,u))
    local,no_centers = {}, []
    for i in (*s,*t):
        stamp = setting,i
        if stamp not in cache:
            cache[stamp] = readout.views(prep.archive.by_setting[setting].graphs[i],setting)
        f,l,count = cache[stamp]
        if f!=full[i]:
            raise PermissionError(f'Full typed WL differs from the original R9 fingerprint: {stamp}')
        local[i] = l
        if count==0:
            no_centers.append(i)
    vocab = wl.vocabulary(full,d)
    train,held = (wl.sparse_rows(full,seq,vocab) for seq in (d,e))
    diffs = readout.differences(full,local,s,t,vocab)
    return (train,held,diffs), dict(source_refs=s,target_refs=t,vocabulary=[[j,c.hex()] for j,c in vocab],
                                   no_nonperipheral_centers=no_centers)


def fit_direction(prep,key,role,labels,cache):
    (train,held,v), info = feature_matrices(prep,role,cache)
    y = np.asarray(labels)
    if y.shape!=(train.shape[0],) or not np.isin(y,[0,1]).all():
        raise ValueError('Expected one binary label per D graph')
    y = y.astype(np.int64,copy=False)
    old,vals = prep.baseline[key]
    options = prep.archive.plan['solver']
    diagnostics = {n:readout.geometry(x) for n,x in v.items()}
    if len(set(y))==1:
        margin = float(logit((int(y.sum())+.5)/(len(y)+1)))
        if old['status']!='single_class_D' or margin!=old['margin']:
            raise PermissionError('Historical C=1 fallback does not match D')
        model = dict(status='single_class_D',margin=margin)
        for diag in diagnostics.values():
            diag.update(fallback='single_class_D',gradient_inf=None,objective=None,iterations=0,retried=False)
    else:
        if old['status']!='ok' or old['vocabulary']!=info['vocabulary']:
            raise PermissionError('Historical C=1 vocabulary differs from full D')
        base = old['wl']
        loss,error = readout.certificate(train,y,None,None,base)
        if not np.isclose(loss,base['training_objective'],rtol=1e-12,atol=1e-15) or error>options['stationarity_tolerance']:
            raise PermissionError('Historical C=1 WL does not solve the registered D objective')
        heads = {BASE:base}
        for name,diff in v.items():
            heads[name],notes = readout.fit(train,y,diff,diagnostics[name],base,options)
            diagnostics[name].update(notes)
        model = dict(status='ok',heads=heads,C1_gradient_inf=error)
    margins = readout.margins(model,held)
    if not np.array_equal(margins[BASE],vals['wl_e']):
        raise PermissionError('Historical C=1 E margins did not replay exactly')
    return dict(**info,**model,geometry=diagnostics), combine(prep.prior[key],margins)


def combine(prior,margins):
    x = {**prior,**{m:(z>0.).astype(np.int64) for m,z in margins.items()},
         **{f'margin_{m}':z for m,z in margins.items()}}
    x[MATCH] = geometry_models.equal_positive(x['e_ids'],margins[BASE],int(x[MAIN].sum()))
    return x


def mean(rows):
    rows = tuple(rows)
    return sum(rows)/len(rows)


def action_bounds(roles,predictions):
    bounds, per_block = {}, []
    for setting in sorted({r['setting'] for r in roles.values()}):
        ids = [k for k,row in roles.items() if row['setting']==setting]
        if len(ids)!=8:
            raise PermissionError('R17 requires all eight directions of each setting')
        arm = {}
        for name in CONTROLS:
            diffs = []
            for k in ids:
                x = predictions[k]
                changed = int((x[MAIN]!=x[name]).sum())
                n = len(x['e_ids'])
                diffs.append(Fraction(changed,n))
                per_block.append(dict(key=k,compared_to=name,n=n,changed=changed,
                                      accuracy_upper_pp=100*changed/n))
            max_positive = sum(v>0 for v in diffs)
            max_groups = len({roles[k]['source'] for k,v in zip(ids,diffs,strict=True) if v>0})
            upper = mean(diffs)
            arm[name] = dict(upper_exact=str(upper),upper_pp=float(100*upper),
                              max_positive_directions=max_positive,max_positive_source_groups=max_groups)
        possible = (Fraction(arm[BASE]['upper_exact'])>=MINIMUM
                    and arm[BASE]['max_positive_source_groups']>=3
                    and all(Fraction(arm[c]['upper_exact'])>0 for c in CONTROLS if c!=BASE))
        bounds[setting] = dict(comparisons=arm,possible_by_separate_pairwise_bounds=possible)
    possible = bounds['proteins_edge']['possible_by_separate_pairwise_bounds'] and any(
        bounds[s]['possible_by_separate_pairwise_bounds'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=bounds,cannot_advance_by_pairwise_bounds=not possible,per_block=per_block)


def freeze(out,cfg,prepared):
    roles = prepared.reference.roles
    (out/'models').mkdir()
    (out/'predictions').mkdir()
    cache,predicted = {},{}
    with data.ScopedLabels(prepared.archive.by_setting,fit_roles(roles),'fit') as labels:
        for key,role in roles.items():
            with labels.direction(key):
                y = labels.load(key,'D',role['roles']['D'])
            model,arrays = fit_direction(prepared,key,role,y,cache)
            validate_arrays(arrays,role['roles']['E'])
            predicted[key] = arrays
            write_json(out/f'models/{key}.json',model)
            prior_run.save_predictions(out,f'predictions/{key}.npz',arrays)
            print(f"R17 {key} m={len(model['target_refs'])}",flush=True)
    validate_fit_reads(roles,labels.actual)
    for name,obj in (('roles',roles),('source-provenance',prepared.reference.provenance),
                     ('baseline-provenance',prepared.lineage),('fit-label-access',labels.actual),
                     ('action-bounds',action_bounds(roles,predicted))):
        write_json(out/f'{name}.json',obj)
    return prior_run.seal(out,'OBS',members(roles),parents(cfg))


def validate_arrays(x,ids):
    if set(x)!={'e_ids',*METHODS,*(f'margin_{m}' for m in MARGINS)}:
        raise PermissionError('Unexpected R17 action inventory')
    if x['e_ids'].dtype!=np.int64 or x['e_ids'].tolist()!=ids:
        raise PermissionError('R17 prediction lost original E order')
    for m in MARGINS:
        check_array(x['e_ids'],x[m],x[f'margin_{m}'])
    if x[MATCH].dtype!=np.int64 or not np.array_equal(
            x[MATCH],geometry_models.equal_positive(x['e_ids'],x[f'margin_{BASE}'],int(x[MAIN].sum()))):
        raise PermissionError('R17 equal-positive actions differ from frozen WL rank')


def observed(out,cfg,prep,obs_sha):
    roles = prep.reference.roles
    raw = prior_run.verified_members(out,'OBS',obs_sha,members(roles),parents(cfg))
    for name,expected in (('roles',roles),('source-provenance',prep.reference.provenance),
                          ('baseline-provenance',prep.lineage)):
        if json.loads(raw[f'{name}.json'])!=expected:
            raise PermissionError(f'Frozen {name} no longer agrees with R9')
    validate_fit_reads(roles,json.loads(raw['fit-label-access.json']))
    arrays,models,cache = {},{},{}
    for key,role in roles.items():
        model = json.loads(raw[f'models/{key}.json'])
        (_,held,v),info = feature_matrices(prep,role,cache)
        if any(model[k]!=d for k,d in info.items()) or model['status']!=prep.baseline[key][0]['status']:
            raise PermissionError(f'Frozen reference identities or status changed: {key}')
        for name,d in v.items():
            geometry = model['geometry'][name]
            if any(geometry[k]!=b for k,b in readout.geometry(d).items()):
                raise PermissionError(f'Frozen penalty trace does not replay: {key}/{name}')
            if model['status']=='ok':
                expected_fallback = 'zero_trace' if geometry['T']<=readout.TRACE_FLOOR else None
                if (geometry['fallback']!=expected_fallback
                        or not 0.<=geometry['gradient_inf']<=prep.archive.plan['solver']['stationarity_tolerance']
                        or (expected_fallback and model['heads'][name]!=model['heads'][BASE])):
                    raise PermissionError(f'Frozen optimizer certificate does not match the penalty: {key}/{name}')
        x = inputs.npz_member(raw[f'predictions/{key}.npz'])
        validate_arrays(x,role['roles']['E'])
        margin = readout.margins(model,held)
        if not np.array_equal(margin[BASE],prep.baseline[key][1]['wl_e']):
            raise PermissionError(f'Reloaded C=1 WL differs from R9: {key}')
        expected = combine(prep.prior[key],margin)
        if any(not np.array_equal(x[k],val) for k,val in expected.items()):
            raise PermissionError(f'Frozen E prediction does not replay: {key}')
        arrays[key],models[key] = x,model
    bounds = action_bounds(roles,arrays)
    if json.loads(raw['action-bounds.json'])!=bounds:
        raise PermissionError('Label-free bounds do not replay')
    return arrays,models


def block_scores(role,x,y):
    n = len(x['e_ids'])
    if y.shape!=(n,) or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid held E labels')
    fields = weighted_residuals.id_fields(role)
    endpoints,pairs,stats,deltas = [],[],{},{}
    for m in METHODS:
        tn,fp,fn,tp = cell = roc_score.confusion(x[m],y)
        a,f = Fraction(tn+tp,n),roc_score.f1(cell)
        z = x.get(f'margin_{m}')
        nll = None if z is None else float(np.logaddexp(0.,(1-2*y)*z).mean())
        stats[m] = (a,f,nll)
        endpoints.append(dict(**fields,method=m,n=n,tn=tn,fp=fp,fn=fn,tp=tp,positive=fp+tp,
                              accuracy=float(a),macro_f1=float(f),nll=nll))
    main = stats[MAIN]
    for name in CONTROLS:
        p = true_rate_replay.paired_counts(x[MAIN],x[name],y)
        a,f = main[0]-stats[name][0],main[1]-stats[name][1]
        if a!=Fraction(p['net'],n) or p['changed']!=p['corrected']+p['harmed']:
            raise ArithmeticError('Paired accuracy does not agree with classification actions')
        loss = None if stats[name][2] is None else main[2]-stats[name][2]
        deltas[name] = (p,a,f,loss,n)
        pairs.append(dict(**fields,left=MAIN,right=name,n=n,**p,accuracy_exact=str(a),
                          accuracy_pp=float(100*a),macro_f1_pp=float(100*f),nll_delta=loss,
                          disagreement_pp=100*p['changed']/n))
    return endpoints,pairs,stats,deltas


def summarize(roles,stats,deltas):
    summary,contrast = [],[]
    for setting in sorted({r['setting'] for r in roles.values()}):
        for source in (None,0,1,2,3):
            keys = [key for key,r in roles.items() if r['setting']==setting
                    and (source is None or source==r['source'])]
            if len(keys)!=(8 if source is None else 2):
                raise PermissionError('R17 cannot omit a registered direction')
            for m in METHODS:
                a,f,l = zip(*(stats[key][m] for key in keys),strict=True)
                summary.append(dict(setting=setting,source=source,directions=len(keys),method=m,
                                    accuracy=float(mean(a)),macro_f1=float(mean(f)),
                                    nll=None if l[0] is None else fmean(l)))
            for m in CONTROLS:
                p,a,f,l,n = zip(*(deltas[key][m] for key in keys),strict=True)
                totals = {k:sum(r[k] for r in p) for k in ('net','corrected','harmed','changed')}
                rates = {k+'_pp':float(100*mean(Fraction(v[k],size) for v,size in zip(p,n,strict=True)))
                         for k in ('corrected','harmed','changed')}
                contrast.append(dict(setting=setting,source=source,directions=len(keys),left=MAIN,right=m,**totals,**rates,
                                     accuracy_exact=str(mean(a)),accuracy_pp=float(100*mean(a)),
                                     macro_f1_pp=float(100*mean(f)),nll_delta=None if l[0] is None else fmean(l),
                                     pos=sum(v>0 for v in a),tied=sum(v==0 for v in a),neg=sum(v<0 for v in a)))
    return summary,contrast


def gate(comparisons):
    recorded = {}
    for setting in sorted({r['setting'] for r in comparisons}):
        blocks = [r for r in comparisons if r['setting']==setting]
        overall = {r['right']:Fraction(r['accuracy_exact']) for r in blocks if r['source'] is None}
        groups = {str(r['source']):Fraction(r['accuracy_exact']) for r in blocks
                  if r['source'] is not None and r['right']==BASE}
        positive = sum(v>0 for v in groups.values())
        checks = dict(base_gain_at_least_half_pp=overall[BASE]>=MINIMUM,
                      at_least_three_positive_source_groups=positive>=3)
        checks.update({f'strictly_above_{m}':overall[m]>0 for m in CONTROLS if m!=BASE})
        recorded[setting] = dict(passed=all(checks.values()),clauses=checks,positive_source_groups=positive,
                                 group_gain_vs_C1_exact={k:str(v) for k,v in groups.items()},
                                 accuracy_exact={m:str(v) for m,v in overall.items()},
                                 accuracy_pp={m:float(100*v) for m,v in overall.items()})
    advance = recorded['proteins_edge']['passed'] and any(
        recorded[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=recorded,continue_real_development=advance)


def score(out,cfg,prep,obs_sha,*,execution):
    arrays,models = observed(out,cfg,prep,obs_sha)
    write_json(out/'score-execution.json',execution)
    roles = prep.reference.roles
    endpoints,pairs,stats,deltas,geometry = [],[],{},{},[]
    with data.ScopedLabels(prep.archive.by_setting,roles,'replay_E') as labels:
        for key,role in roles.items():
            with labels.direction(key):
                y = labels.load(key,'E',role['roles']['E'])
            ep,pa,stats[key],deltas[key] = block_scores(role,arrays[key],y)
            endpoints.extend(ep)
            pairs.extend(pa)
            model = models[key]
            for name,diag in model['geometry'].items():
                geometry.append(dict(**weighted_residuals.id_fields(role),method=name,m=len(model['target_refs']),**diag,
                    TargetPair_SourcePair_disagreement_E=int((arrays[key]['TargetPair']!=arrays[key]['SourcePair']).sum())))
    data.validate_receipt(roles,labels.actual,'replay_E')
    summary,paired = summarize(roles,stats,deltas)
    tables = {'endpoints':endpoints,'paired-blocks':pairs,'summary':summary,'paired-summary':paired,'geometry':geometry}
    for name,rows in tables.items():
        write_table(out/f'{name}.csv',rows)
    write_json(out/'e-label-access.json',labels.actual)
    write_json(out/'gate.json',gate(paired))
    files = ['OBS.json','score-execution.json','e-label-access.json','gate.json',*[f'{name}.csv' for name in tables]]
    return prior_run.seal(out,'FINAL',files,dict(obs_sha256=obs_sha))
