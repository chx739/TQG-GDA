"""Run the registered 24 source-size directions with D/C and E-only label ports."""
from dataclasses import dataclass
from fractions import Fraction
import json
from statistics import fmean

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from . import geometry_models as geom, residual_inputs as data, true_rate_replay, weighted_residuals

PAIRS = tuple(('TargetFull', name) for name in geom.METHODS if name!='TargetFull')
META = ('executed-config.json','roles.json','fit-label-access.json','source-topology.json','source-provenance.json')
MINIMUM_GAIN = Fraction(1,200)


@dataclass(frozen=True)
class Reference:
    plan: dict
    roles: dict
    topology: dict
    provenance: dict


def authenticate(here, cfg):
    """Reuse only the original OBS roles and input authorities."""
    root = here/cfg['source_attempt']
    seal = roc_transfer.stage(root, 'OBS', cfg['source_obs_sha256'])
    files = seal['files']
    prior = {}
    for name in ('executed-config','roles','source-topology','source-provenance'):
        prior[name] = json.loads(prior_run.read_member(root,files,f'{name}.json'))
    plan = prior['executed-config']['plan']
    expected = data.grid(plan)
    if (seal['predecessors']!=weighted_residuals.parents(plan)
            or set(files)!=set(weighted_residuals.members(expected))
            or set(prior['roles'])!=set(expected)
            or (plan['settings'],plan['domains'],plan['seed'])!=(
                ['mutagenicity_edge','mutagenicity_node','proteins_edge'],[0,1,2,3],3405)):
        raise PermissionError('Original source-size OBS differs from the registered 24 directions')
    for key,row in prior['roles'].items():
        if weighted_residuals.id_fields(row)!=expected[key] or row['status']!='ok':
            raise PermissionError(f'Original direction differs: {key}')
    return Reference(plan,prior['roles'],prior['source-topology'],prior['source-provenance'])


def prepare(repo, reference):
    archive = data.Archive(repo,reference.plan)
    true_rate_replay.validate_e_roles(archive,reference)
    roles,features,topology = archive.prepare()
    if roles!=reference.roles or topology!=reference.topology:
        raise PermissionError('Reconstructed source roles or topology changed from attempt-002')
    return archive, features


def members(roles):
    return [*META, *prior_run.inventory(roles,'predictions'),*(f'models/{key}.json' for key in roles)]


def freeze(output, cfg, reference, archive, features):
    for name in ('models','predictions'):
        (output/name).mkdir()
    with data.ScopedLabels(archive.by_setting,reference.roles,'fit') as port:
        for key,row in reference.roles.items():
            ids = row['roles']
            with port.direction(key):
                yd = port.load(key,'D',ids['D'])
                yc = port.load(key,'C',ids['C'])
            feature = features[row['setting'],row['source']]
            fit_features = {i:feature[i] for pool in ('D','C','U') for i in ids[pool]}
            model = geom.fit(fit_features,row,yd,yc,archive.plan['solver'])
            arrays = geom.decisions(model,feature,ids['E'])
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ArithmeticError(f'Malformed predictions: {key}')
            write_json(output/f'models/{key}.json',model)
            prior_run.save_predictions(output, f'predictions/{key}.npz', arrays)
            print(f'R9 geometry {key} C={model["selected_base_C"]} references={model["m"]}',flush=True)
    data.validate_receipt(reference.roles,port.actual,'fit')
    for name,obj in (('roles',reference.roles),('fit-label-access',port.actual),
                     ('source-topology',reference.topology),('source-provenance',reference.provenance)):
        write_json(output/f'{name}.json',obj)
    return prior_run.seal(output,'OBS',members(reference.roles),dict(source_obs_sha256=cfg['source_obs_sha256']))


def observed(output, cfg, reference, features, obs_sha):
    prior = prior_run.verified_members(output,'OBS',obs_sha,members(reference.roles),
                                       dict(source_obs_sha256=cfg['source_obs_sha256']))
    for name,expected in (('roles',reference.roles),('source-topology',reference.topology),
                          ('source-provenance',reference.provenance)):
        if json.loads(prior[f'{name}.json'])!=expected:
            raise PermissionError(f'Sealed {name} differs from the historical authority')
    data.validate_receipt(reference.roles,json.loads(prior['fit-label-access.json']),'fit')
    predictions, models = {}, {}
    for key,row in reference.roles.items():
        ids = row['roles']
        x = inputs.npz_member(prior[f'predictions/{key}.npz'])
        model = json.loads(prior[f'models/{key}.json'])
        m = min(len(ids['C']),len(ids['U']))
        t,s = (geom.select_references(ids[pool],m,row['setting'],row['source']) for pool in ('U','C'))
        if (m,t,s)!=(model['m'],model['target'],model['source']):
            raise PermissionError(f'Sealed reference IDs differ: {key}')
        if model['status']=='ok':
            candidates,base_c = model['candidates'],model['selected_base_C']
            if (base_c!=candidates['Base']['selected_C'] or
                    any(fit['selected_C']!=base_c for name,fit in candidates.items() if name in geom.AUGMENTED)):
                raise PermissionError(f'Augmented model changed the selected Base C: {key}')
        feature = features[row['setting'],row['source']]
        expected = geom.decisions(model,feature,ids['E'])
        if (set(x)!=set(expected) or any(x[k].dtype!=a.dtype or not np.isfinite(a).all()
                                       or not np.array_equal(x[k],a) for k,a in expected.items())):
            raise PermissionError(f'Sealed E arrays fail replay: {key}')
        predictions[key],models[key] = x,model
    return predictions,models


def block_scores(role, actions, y):
    ids = actions['e_ids']
    if y.shape!=ids.shape or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid directional E label vector')
    ident = weighted_residuals.id_fields(role)
    result, endpoints, pairs = {}, [], []
    for name in geom.METHODS:
        tn,fp,fn,tp = confusion = roc_score.confusion(actions[name],y)
        acc,f1 = Fraction(tn+tp,len(y)),roc_score.f1(confusion)
        z = actions.get(f'margin_{name}')
        nll = None if z is None else float(np.logaddexp(0.,(1-2*y)*z).mean())
        result[name] = acc,f1,nll
        endpoints.append(dict(**ident,method=name,n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,
                              positive=tp+fp,accuracy=float(acc),macro_f1=float(f1),nll=nll))
    diffs = {}
    for l,r in PAIRS:
        count = true_rate_replay.paired_counts(actions[l],actions[r],y)
        a,f = result[l][0]-result[r][0],result[l][1]-result[r][1]
        if a!=Fraction(count['net'],len(y)) or count['changed']!=count['corrected']+count['harmed']:
            raise ArithmeticError('Paired gains differ from the binary action decomposition')
        diffs[l,r] = count, a, f, len(y)
        pairs.append(dict(**ident,left=l,right=r,n=len(y),**count,
                          accuracy_exact=str(a),accuracy_pp=float(100*a),macro_f1_pp=float(100*f),
                          disagreement_pp=100*count['changed']/len(y)))
    return result,diffs,endpoints,pairs


def _mean(values):
    values = list(values)
    return sum(values)/len(values)


def summarize(reference, endpoints, pairs):
    summary, contrasts = [], []
    for setting in reference.plan['settings']:
        for group in (None,*reference.plan['domains']):
            keys = [k for k,r in reference.roles.items()
                    if r['setting']==setting and (group is None or r['source']==group)]
            if len(keys)!=(8 if group is None else 2):
                raise ValueError('Missing paired direction in the source group')
            for name in geom.METHODS:
                acc,f1,nll = zip(*(endpoints[k][name] for k in keys),strict=True)
                accuracy = _mean(acc)
                summary.append(dict(setting=setting,source=group,method=name,directions=len(keys),
                                    accuracy_exact=str(accuracy),accuracy=float(accuracy),macro_f1=float(_mean(f1)),
                                    nll=None if nll[0] is None else fmean(nll)))
            for l,r in PAIRS:
                counts,acc,f1,n = zip(*(pairs[k][l,r] for k in keys),strict=True)
                accuracy = _mean(acc)
                totals = {m:sum(c[m] for c in counts) for m in ('net','corrected','harmed','changed')}
                rates = {m+'_pp':float(100*_mean(Fraction(c[m],d) for c,d in zip(counts,n,strict=True)))
                         for m in ('corrected','harmed','changed')}
                contrasts.append(dict(setting=setting,source=group,left=l,right=r,directions=len(keys),n=sum(n),
                                      accuracy_exact=str(accuracy),accuracy_pp=float(100*accuracy),
                                      macro_f1_pp=float(100*_mean(f1)),**totals,**rates))
    return summary,contrasts


def gate(reference, contrasts):
    settings = {}
    for setting in reference.plan['settings']:
        rows = [r for r in contrasts if r['setting']==setting and r['left']=='TargetFull']
        full = {r['right']:Fraction(r['accuracy_exact']) for r in rows if r['source'] is None}
        positive_groups = sum(Fraction(r['accuracy_exact'])>0 for r in rows
                              if r['source'] is not None and r['right']=='Base')
        passed = (full['Base']>=MINIMUM_GAIN and positive_groups>=3 and
                  all(full[m]>0 for m in ('SourceFull','TargetDiag','WL','Size')))
        settings[setting] = dict(passed=passed,positive_source_groups=positive_groups,
                                 accuracy_difference_exact={k:str(v) for k,v in full.items()},
                                 accuracy_difference_pp={k:float(100*v) for k,v in full.items()})
    proceed = settings['proteins_edge']['passed'] and any(
        settings[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=settings,continue_real_four_domain_pilot=proceed)


def score(output, cfg, reference, archive, features, obs_sha, *, execution=None):
    predictions,models = observed(output,cfg,reference,features,obs_sha)
    if execution is not None:
        write_json(output/'score-execution.json', execution)
    endpoints, pairs, endpoints_by_key, pairs_by_key, geometry = [], [], {}, {}, []
    with data.ScopedLabels(archive.by_setting,reference.roles,'replay_E') as port:
        for key,row in reference.roles.items():
            with port.direction(key):
                y = port.load(key,'E',row['roles']['E'])
            ep,pa,e_rows,p_rows = block_scores(row,predictions[key],y)
            endpoints_by_key[key],pairs_by_key[key] = ep,pa
            endpoints.extend(e_rows)
            pairs.extend(p_rows)
            model = models[key]
            for arm in geom.AUGMENTED:
                stats = model.get('geometry',{}).get(arm,{})
                geometry.append(dict(**weighted_residuals.id_fields(row),arm=arm,m=model['m'],
                                     status=model['status'],selected_base_C=model['selected_base_C'],
                                     D_wl_variance=model.get('D_wl_variance'),scale=stats.get('scale'),
                                     D_extra_variance=stats.get('D_extra_variance'),
                                     D_matched_variance=stats.get('D_matched_variance'),
                                     fallback=stats.get('fallback')))
    data.validate_receipt(reference.roles,port.actual,'replay_E')
    ep,pa = summarize(reference,endpoints_by_key,pairs_by_key)
    tables = {'endpoints':endpoints,'paired-blocks':pairs,'endpoint-summary':ep,'paired-summary':pa,'geometry':geometry}
    for name,rows in tables.items():
        write_table(output/f'{name}.csv',rows)
    write_json(output/'gate.json',gate(reference,pa))
    write_json(output/'e-label-access.json',port.actual)
    return prior_run.seal(output,'FINAL',['OBS.json','gate.json','e-label-access.json','score-execution.json',
                           *[f'{n}.csv' for n in tables]],
                          dict(obs_sha256=obs_sha))
