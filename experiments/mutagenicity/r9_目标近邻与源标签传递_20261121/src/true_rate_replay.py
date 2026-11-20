"""Replay exact published U/E rates and separately score the sealed E actions."""
from dataclasses import dataclass
from fractions import Fraction
import csv
import io
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_rate, roc_score, roc_transfer, thresholds
from . import residual_inputs as data, residual_models as models, weighted_residuals


RULES = {'A':('U','U'),'B':('E','U'),'C':('E','E')}
METHODS = ('WL',*RULES)
PAIRS = (('A','WL'),('B','WL'),('C','WL'),('B','A'),('C','B'))
METADATA = ('executed-config.json','roles.json','rates.json')


@dataclass
class Reference:
    plan: dict
    roles: dict
    topology: dict
    counts: dict
    predictions: dict


def parents(cfg):
    return {k:cfg[k] for k in ('source_obs_sha256','source_final_sha256')}


def csv_rows(payload):
    return csv.DictReader(io.StringIO(payload.decode('utf-8')))


def published_counts(payload, roles):
    votes = {}
    for row in csv_rows(payload):
        key = data.key(row['setting'],int(row['source']),row['direction'],int(row['seed']))
        pool,z,method = row['pool'],int(row['size_action']),row['method']
        n,p = int(row['n']),int(row['positive'])
        if key not in roles or pool not in ('U','E') or z not in (0,1) or method not in models.ESTIMATORS or not 0<=p<=n or n<=0:
            raise PermissionError('Unexpected published U/E stratum or count')
        count, seen = votes.setdefault((key,pool,z),((n,p),set()))
        if count!=(n,p) or method in seen:
            raise PermissionError(f'Inconsistent published U/E counts: {key}/{pool}/{z}')
        seen.add(method)
    if any(methods!=set(models.ESTIMATORS) for _,methods in votes.values()):
        raise PermissionError('Published U/E strata do not have matching estimator counts')
    return {k:count for k,(count,_) in votes.items()}


def authenticate(here, cfg):
    root = here/cfg['source_attempt']
    obs = roc_transfer.stage(root,'OBS',cfg['source_obs_sha256'])
    final = roc_transfer.stage(root,'FINAL',cfg['source_final_sha256'])
    obs_files, score_files = obs['files'], final['files']
    if final['predecessors']!={'obs_sha256':cfg['source_obs_sha256']} or score_files['OBS.json']!=cfg['source_obs_sha256']:
        raise PermissionError('Original FINAL is not bound to the registered OBS')

    def old(relative):
        return prior_run.read_member(root,obs_files,relative)

    plan = json.loads(old('executed-config.json'))['plan']
    roles = json.loads(old('roles.json'))
    topology = json.loads(old('source-topology.json'))
    keys = data.grid(plan)
    if len(keys)!=24 or set(keys)!=set(roles) or set(obs_files)!=set(weighted_residuals.members(keys)):
        raise PermissionError('Original experiment does not cover all 24 registered directions')
    access = json.loads(prior_run.read_member(root,score_files,'score-label-access.json'))
    data.validate_receipt(roles,access,'score')
    counts = published_counts(prior_run.read_member(root,score_files,'proportion-strata.csv'),roles)
    predictions = {}
    for key, identity in keys.items():
        role = roles[key]
        if weighted_residuals.id_fields(role)!=identity or role['status']!='ok':
            raise PermissionError(f'Unexpected original direction: {key}')
        needed = ['u_ids','e_ids','wl_u','wl_e','size_u','size_e','WL']
        x = inputs.npz_member(old(f'predictions/{key}.npz'),needed)
        for pool in ('U','E'):
            col = pool.lower()
            n = len(role['roles'][pool])
            if (x[f'{col}_ids'].dtype!=np.int64 or x[f'{col}_ids'].tolist()!=role['roles'][pool]
                    or n==0):
                raise PermissionError(f'Original {pool} order changed: {key}')
            for method in ('wl','size'):
                z = x[f'{method}_{col}']
                if z.dtype!=np.float64 or z.shape!=(n,) or not np.isfinite(z).all():
                    raise PermissionError(f'Malformed original {pool} margin: {key}/{method}')
            layers = x[f'size_{col}']>0.
            for z in (0,1):
                count = counts.get((key,pool,z),(0,0))
                if count[0]!=int((layers==z).sum()):
                    raise PermissionError(f'Published stratum size differs from OBS: {key}/{pool}/{z}')
        if x['WL'].dtype!=np.int64 or not np.array_equal(x['WL'],x['wl_e']>0.):
            raise PermissionError(f'Original WL differs from its strict zero threshold: {key}')
        predictions[key] = x
    return Reference(plan,roles,topology,counts,predictions)


def action_rules(key, reference):
    x = reference.predictions[key]
    layers = {p:x[f'size_{p.lower()}']>0. for p in ('U','E')}
    actions = {m:x['WL'].copy() for m in RULES}
    strata = []
    for z in (0,1):
        e = layers['E']==z
        counts = {p:reference.counts.get((key,p,z),(0,0)) for p in ('U','E')}
        traces = {}
        for m,(rate_pool,threshold_pool) in RULES.items():
            n,positive = counts[rate_pool]
            rate = Fraction(positive,n) if n else None
            reason = 'empty_E' if not counts['E'][0] else 'fallback_WL_empty_U' if not counts['U'][0] else 'threshold'
            cut,k = None,None
            if reason=='threshold':
                scores = x[f'wl_{threshold_pool.lower()}'][layers[threshold_pool]==z]
                cut,k = roc_rate.quantile(scores,rate)
                actions[m][e] = roc_rate.classify(x['wl_e'][e],cut)
            traces[m] = dict(rate_pool=rate_pool,threshold_pool=threshold_pool,
                             rate_exact=None if rate is None else str(rate),status=reason,
                             threshold=None if cut is None else thresholds.encode_threshold(cut),
                             threshold_pool_positives=k)
        strata.append(dict(size_action=z,counts={p:dict(n=n,positive=y) for p,(n,y) in counts.items()},rules=traces))
    return dict(e_ids=x['e_ids'],WL=x['WL'],**actions),strata


def members(keys):
    return [*METADATA,*prior_run.inventory(keys,'predictions')]


def freeze(out, cfg, reference):
    (out/'predictions').mkdir()
    rates = {}
    for key in reference.roles:
        actions,rates[key] = action_rules(key,reference)
        prior_run.save_predictions(out,f'predictions/{key}.npz',actions)
    write_json(out/'roles.json',reference.roles)
    write_json(out/'rates.json',rates)
    return prior_run.seal(out,'OBS',members(reference.roles),parents(cfg))


def verify_actions(out, cfg, reference, obs_sha):
    record = prior_run.verified_members(out,'OBS',obs_sha,members(reference.roles),parents(cfg))
    certificates = json.loads(record['rates.json'])
    if json.loads(record['roles.json'])!=reference.roles or set(certificates)!=set(reference.roles):
        raise PermissionError('Replay OBS does not contain all original E roles')
    actions = {}
    for key in reference.roles:
        measured = inputs.npz_member(record[f'predictions/{key}.npz'])
        expected, trace = action_rules(key,reference)
        if set(measured)!=set(expected) or certificates[key]!=trace:
            raise PermissionError(f'Invalid replay certificate: {key}')
        for m, a in expected.items():
            if measured[m].dtype!=np.int64 or not np.array_equal(measured[m],a):
                raise PermissionError(f'Frozen E actions differ: {key}/{m}')
        actions[key] = measured
    return actions


def validate_e_roles(archive, reference):
    for key, row in reference.roles.items():
        group = f"{row['setting']}-M{row['source']}-seed-{row['seed']}"
        fitted_roles = archive.fit_access[group]
        record = reference.topology[group]
        s,v = record['source_ids'],record['validation_ids']
        if s!=fitted_roles['source_ids'] or v!=fitted_roles['validation_ids']:
            raise PermissionError('R9 S/V differs from the authenticated R10 FIT')
        cut, roles = data.split(s,v,dict(zip(s+v,record['sizes'],strict=True)))
        if cut!=record['cut'] or cut!=row['cut'] or roles[row['direction']]!=row['roles']:
            raise PermissionError(f'R9 fitting/score split differs: {key}')


def paired_counts(left, right, y):
    l,r = left==y,right==y
    gain,loss = int((l&~r).sum()),int((~l&r).sum())
    return dict(corrected=gain,harmed=loss,net=gain-loss,changed=int((left!=right).sum()))


def summaries(score_by_key, pairs_by_key, reference, cfg):
    rows, contrasts, checks = [],[],{}
    for setting in reference.plan['settings']:
        keys = [k for k,r in reference.roles.items() if r['setting']==setting]
        if len(keys)!=8:
            raise PermissionError(f'{setting} must have eight paired directions')
        totals = {}
        for m in METHODS:
            means = [sum(score_by_key[k,m][i] for k in keys)/len(keys) for i in (0,1)]
            totals[m] = means
            rows.append(dict(setting=setting,method=m,directions=len(keys),
                             accuracy=float(means[0]),macro_f1=float(means[1])))
        value = 100*totals['A'][0]
        lo,hi = map(Fraction,cfg['published_a_accuracy_ranges_pp'][setting])
        # Compare to the interval at the precision in which the endpoints were published.
        in_range = lo-Fraction(1,200)<=value<=hi+Fraction(1,200)
        if not in_range:
            raise ArithmeticError(f'{setting} A Accuracy is outside the published feasible interval')
        checks[setting] = dict(accuracy_pp=float(value),published_low_pp=float(lo),published_high_pp=float(hi),within=True)
        for l,r in PAIRS:
            scores = [pairs_by_key[k,l,r] for k in keys]
            contrasts.append(dict(setting=setting,left=l,right=r,directions=len(keys),
                **{f:sum(row[f] for row in scores) for f in ('corrected','harmed','net','changed')},
                accuracy_pp=float(100*(totals[l][0]-totals[r][0])),
                macro_f1_pp=float(100*(totals[l][1]-totals[r][1]))))
    return rows,contrasts,checks


def score(out, repo, cfg, reference, obs_sha):
    actions = verify_actions(out,cfg,reference,obs_sha)
    archive = data.Archive(repo,reference.plan,load_graphs=False)
    validate_e_roles(archive,reference)
    endpoints,pairs,result,contrasts = [],[],{},{}
    with data.ScopedLabels(archive.by_setting,reference.roles,'replay_E') as port:
        for key, role in reference.roles.items():
            ids = role['roles']['E']
            with port.direction(key):
                y = port.load(key,'E',ids)
            identity = weighted_residuals.id_fields(role)
            layer = reference.predictions[key]['size_e']>0.
            for z in (0,1):
                if int(y[layer==z].sum())!=reference.counts.get((key,'E',z),(0,0))[1]:
                    raise PermissionError(f'E labels differ from the published stratum: {key}/{z}')
            for m in METHODS:
                c = roc_score.confusion(actions[key][m],y)
                tn,fp,fn,tp = c
                acc,f1 = Fraction(tn+tp,len(y)),roc_score.f1(c)
                result[key,m] = acc,f1
                endpoints.append(dict(**identity,method=m,n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,
                                      accuracy=float(acc),macro_f1=float(f1)))
            for l,r in PAIRS:
                counts = paired_counts(actions[key][l],actions[key][r],y)
                contrasts[key,l,r] = counts
                accuracy = Fraction(counts['net'],len(y))
                macro_f1 = result[key,l][1]-result[key,r][1]
                pairs.append(dict(**identity,left=l,right=r,n=len(y),**counts,
                                  accuracy_pp=float(100*accuracy),macro_f1_pp=float(100*macro_f1)))
    data.validate_receipt(reference.roles,port.actual,'replay_E')
    summary,pa_summary,bounds = summaries(result,contrasts,reference,cfg)
    csvs = {'endpoints':endpoints,'paired-blocks':pairs,'endpoint-summary':summary,'paired-summary':pa_summary}
    for name,rows in csvs.items():
        write_table(out/f'{name}.csv',rows)
    write_json(out/'a-range-checks.json',bounds)
    write_json(out/'e-label-access.json',port.actual)
    finals = ['OBS.json','a-range-checks.json','e-label-access.json',*[f'{n}.csv' for n in csvs]]
    return prior_run.seal(out,'FINAL',finals,dict(obs_sha256=obs_sha))
