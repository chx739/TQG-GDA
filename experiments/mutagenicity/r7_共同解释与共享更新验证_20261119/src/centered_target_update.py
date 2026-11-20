"""Replay the frozen differences between the R7 Full and calibrated-teacher heads."""
from fractions import Fraction
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation
from . import head_crossover as cross, head_crossover_scoring as holdout_score, strong_anchor_refit as refit
from . import strong_anchor_scoring as scoring

META = ('executed-config.json','roles.json','weight-differences.json')
COMPARISONS = ('Anchor','WL','Size','Full','CalTeacher','Reversed','Other1','Other2','OtherMean','EqualCount')


def parents(plan):
    return {'anchor_obs_sha256':plan['anchor_obs_sha256']}


def members(roles):
    return [*META,*prior_run.inventory(roles,'predictions')]


def teacher_difference(fits,anchor):
    d_full,d_cal = (fits[refit.arm(anchor,m)] for m in ('Full','CalTeacher'))
    if (d_full.ndim!=1 or d_full.shape!=d_cal.shape
            or not np.isfinite(d_full).all() or not np.isfinite(d_cal).all()):
        raise PermissionError('Frozen teacher updates have inconsistent coordinates')
    return d_full-d_cal


def predict(role,original,h,group):
    ids = original['ids']
    target = role['target']
    margins = {a:original[f'margin_{a}'] for a in refit.ANCHORS}
    decisions = {}
    for a in refit.ANCHORS:
        c = margins[a]
        for t,delta in group.items():
            margins[refit.arm(a,'Centered',t)] = c+h@delta[a]
        shift = h@group[target][a]
        margins[refit.arm(a,'Reversed')] = c-shift
        margins[refit.arm(a,'Full')] = original[f'margin_{refit.arm(a,"Full",target)}']
        margins[refit.arm(a,'CalTeacher')] = original[f'margin_{refit.arm(a,"CalTeacher")}']
        count = int((margins[refit.arm(a,'Centered',target)]>0.).sum())
        decisions[refit.arm(a,'EqualCount')],_ = cross.equal_count(ids,c,count)
    if any(m.shape!=ids.shape or not np.isfinite(m).all() for m in margins.values()):
        raise ArithmeticError('Invalid H replay margin')
    return dict(ids=ids,**decisions,**{name:(z>0.).astype(np.int64) for name,z in margins.items()},
                **{f'margin_{m}':z for m,z in margins.items()})


def parent_actions(here,plan,cfg,c):
    path = here/plan['anchor_attempt']
    sha = plan['anchor_obs_sha256']
    roles,x,notes = scoring.observed(path,cfg,c,sha)
    registry = roc_transfer.stage(path,'OBS',sha)
    old_cfg = json.loads(prior_run.read_member(path,registry['files'],'executed-config.json'))['plan']
    if old_cfg!=cfg:
        raise PermissionError('Centered replay refers to a different strong-anchor configuration')
    by_group, weights = {}, {}
    for key,r in roles.items():
        group = (r['setting'],r['source'],r['seed'])
        relative = f'fits/{key}.npz'
        raw = {relative:prior_run.read_member(path,registry['files'],relative)}
        fits = scoring.read_models(raw,key,r,notes[key])
        delta = {a:teacher_difference(fits,a) for a in refit.ANCHORS}
        by_group.setdefault(group,{})[r['target']] = delta
        weights[key] = {a:d.tolist() for a,d in delta.items()}
    predicted = {}
    for setting,source,seed,tasks in observation.group_tasks(cfg,[cfg['phase1_seed']]):
        group = by_group[setting,source,seed]
        for task in tasks:
            key = inputs.key(**task)
            raw = inputs.npz_member(c.read('R7_OBS',f'blocks/{key}/holdout_standardization.npz'),
                                    ['holdout_u','holdout_ids'])
            ids,h = raw['holdout_ids'],raw['holdout_u']
            np.testing.assert_array_equal(ids, x[key]['ids'])
            if (h.dtype!=np.float64 or h.shape!=(len(ids),len(next(iter(group.values()))['WL']))
                    or not np.isfinite(h).all() or not np.all(h[:,-1]==1.)):
                raise PermissionError('Frozen source coordinates do not match the centered update')
            predicted[key] = predict(roles[key],x[key],h,group)
    return roles,weights,predicted


def freeze(output,here,plan,cfg,c):
    roles,weights,predicted = parent_actions(here,plan,cfg,c)
    (output/'predictions').mkdir()
    for key,x in predicted.items():
        prior_run.save_predictions(output,f'predictions/{key}.npz',x)
    write_json(output/'roles.json',roles)
    write_json(output/'weight-differences.json',weights)
    return prior_run.seal(output,'OBS',members(roles),parents(plan))


def observed(output,here,plan,cfg,c,obs):
    _,keys = observation.observations_expected(cfg,[cfg['phase1_seed']])
    raw = prior_run.verified_members(output,'OBS',obs,members(keys),parents(plan))
    roles,weights,predicted = parent_actions(here,plan,cfg,c)
    if (json.loads(raw['roles.json'])!=roles or json.loads(raw['weight-differences.json'])!=weights
            or set(predicted)!=set(keys)):
        raise PermissionError('Centered OBS roles or weights fail replay')
    for key,expected in predicted.items():
        x = inputs.npz_member(raw[f'predictions/{key}.npz'])
        if set(x)!=set(expected) or any(x[k].dtype!=v.dtype or not np.array_equal(x[k],v) for k,v in expected.items()):
            raise PermissionError(f'Centered OBS failed prediction replay: {key}')
    return roles,predicted


def aliases(role,anchor):
    a = refit.arm
    t = role['target']
    others = sorted(set(role['reference_targets'])-{t})
    return {
        **{m:(m,) for m in refit.ANCHORS}, 'Anchor':(anchor,),
        'Centered':(a(anchor,'Centered',t),),
        **{m:(a(anchor,m),) for m in ('Full','CalTeacher','EqualCount','Reversed')},
        'Other1':(a(anchor,'Centered',others[0]),), 'Other2':(a(anchor,'Centered',others[1]),),
        'OtherMean':tuple(a(anchor,'Centered',r) for r in others),
    }


def gate(comparisons,cfg):
    settings = {}
    for setting,anchor in cfg['primary_anchors'].items():
        rows = {r['right']:r for r in comparisons if r['setting']==setting and r['anchor']==anchor}
        f = {m:Fraction(r['accuracy_exact']) for m,r in rows.items()}
        positive = rows['Anchor']['positive']
        clauses = dict(proper_improvement=f['Anchor']>=Fraction(*cfg['minimum_accuracy_fraction']),
                       positive_directions=positive>=cfg['minimum_positive_directions'])
        clauses.update({f'strictly_above_{m}':f[m]>0 for m in ('WL','Size','Reversed','OtherMean')})
        settings[setting] = dict(anchor=anchor,passed_accuracy_gate=all(clauses.values()),
                                  above_equal_count=f['EqualCount']>0,clauses=clauses,
                                  positive=positive,tied=rows['Anchor']['tied'],negative=rows['Anchor']['negative'],
                                  differences_exact={m:str(x) for m,x in f.items()},
                                  differences_pp={m:float(100*x) for m,x in f.items()})
    def qualifies(v, require_equal_count):
        return v['passed_accuracy_gate'] and (not require_equal_count or v['above_equal_count'])

    def valid(require_equal_count):
        return qualifies(settings['proteins_edge'],require_equal_count) and any(
            qualifies(v,require_equal_count) for k,v in settings.items() if k.startswith('mutagenicity_'))

    return dict(settings=settings,passes_cross_setting_accuracy=valid(False),continue_remaining_seeds=valid(True))


def score(output,repo,here,plan,cfg,c,obs,*,execution):
    roles,x = observed(output,here,plan,cfg,c,obs)
    write_json(output/'score-execution.json',execution)
    cohorts,access,_ = holdout_score.read_h_cohorts(repo,c,roles)
    endpoints,paired,stats,delta = [],[],{},{}
    for key,r in roles.items():
        labels,_,_ = cohorts[r['setting'],r['target']]
        ep,pa,stats[key],delta[key] = scoring.score_block(r,x[key],labels,active='Centered',
                                                         make_views=aliases,comparisons=COMPARISONS)
        endpoints.extend(ep)
        paired.extend(pa)
    ep,pa = scoring.aggregate(roles,stats,delta,active='Centered')
    tables = {'endpoints':endpoints,'paired-blocks':paired,'endpoint-summary':ep,'paired-summary':pa}
    for name,rows in tables.items():
        write_table(output/f'{name}.csv',rows)
    write_json(output/'gate.json',gate(pa,cfg))
    write_json(output/'h-label-access.json',dict(obs_sha256=obs,actual_reads=access))
    files = ['OBS.json','score-execution.json','gate.json','h-label-access.json',*[f'{m}.csv' for m in tables]]
    return prior_run.seal(output,'FINAL',files,dict(obs_sha256=obs))
