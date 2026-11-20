"""Verify frozen anchor refits and score only the original H graph roles."""
from fractions import Fraction
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation
from . import head_crossover_scoring as scoring, strong_anchor_refit as refit

OTHER_CONTROLS = ('B0Teacher','CalTeacher','ScoreOnly','OriginalE','Other1','Other2','OtherMean','EqualCount')


def read_models(raw,key,role,notes):
    fits = inputs.npz_member(raw[f'fits/{key}.npz'])
    names = ['v0','Q','original_moment','teacher_moment','calibrated_moment','original_B0_A','refit_E_update','original_E_A']
    names.extend(f'{a}_A' for a in refit.ANCHORS)
    names.extend(refit.arm(a,m) for a in refit.ANCHORS for m in refit.UPDATES)
    if (set(fits)!={'A_ids',*names} or any(fits[k].dtype!=np.float64 or not np.isfinite(fits[k]).all() for k in names)
            or fits['A_ids'].dtype!=np.int64 or fits['A_ids'].tolist()!=role['adapt_ids']
            or notes['A_n']!=len(role['adapt_ids'])):
        raise PermissionError('Sealed update inputs differ from roles')
    return fits


def observed(output, cfg, c, obs):
    _,keys = observation.observations_expected(cfg,[cfg['phase1_seed']])
    raw = prior_run.verified_members(output,'OBS',obs,refit.members(keys),refit.parents(cfg))
    roles = json.loads(raw['roles.json'])
    notes = json.loads(raw['fit-notes.json'])
    if set(roles)!=set(keys) or set(notes)!=set(keys):
        raise PermissionError('Missing required anchor refit block')
    fits,directions,predictions = {},{},{}
    for setting,source,seed,tasks in observation.group_tasks(cfg,[cfg['phase1_seed']]):
        subset = {t['target']:inputs.key(**t) for t in tasks}
        for task in tasks:
            key = inputs.key(**task)
            r = roles[key]
            if (any(r[k]!=v for k,v in task.items()) or r['reference_targets']!=sorted(subset)
                    or r['adapt_ids']==r['holdout_ids'] or set(r['adapt_ids'])&set(r['holdout_ids'])):
                raise PermissionError('Sealed H role is different from the preregistered direction')
            fits[key] = read_models(raw,key,r,notes[key])
            directions[key] = subset
    root = Path(cfg['structural_archive'])
    archive = roc_transfer.stage(root,'OBS',cfg['structural_obs_sha256'])
    if archive['predecessors']!={'fit_sha256':cfg['structural_fit_sha256']}:
        raise PermissionError('Historical R10 H points to another FIT')
    for key,role in roles.items():
        task = {k:role[k] for k in ('setting','source','target','seed')}
        item = refit.read_reference(c,task,cfg)
        if (item['manifest_sha']!=role['manifest_sha256'] or item['ids']['holdout']!=role['holdout_ids']
                or item['ids']['adapt']!=role['adapt_ids']):
            raise PermissionError('New archive did not preserve immutable R7 role identity')
        rel = f'predictions/{key}.npz'
        if archive['files'][rel]!=role['R10_H_prediction_sha256']:
            raise PermissionError('New archive does not reuse the expected R10 held predictions')
        old = inputs.npz_member(prior_run.read_member(root,archive['files'],rel),
                                ['ids',*(f'logits_{v}' for v in refit.ANCHORS.values())])
        np.testing.assert_array_equal(old['ids'],role['holdout_ids'])
        anchors_h = {a:inputs.margin(old[f'logits_{m}']) for a,m in refit.ANCHORS.items()}
        group = {t:fits[k] for t,k in directions[key].items()}
        expected = refit.predict_h(item,group,anchors_h,role['target'])
        x = inputs.npz_member(raw[rel])
        if set(x)!=set(expected) or any(x[k].dtype!=v.dtype or not np.array_equal(x[k],v) for k,v in expected.items()):
            raise PermissionError(f'Frozen H arrays cannot be replayed: {key}')
        predictions[key] = x
    return roles,predictions,notes


def aliases(role,anchor):
    target = role['target']
    a = refit.arm
    others = sorted(set(role['reference_targets'])-{target})
    names = {name:(name,) for name in ('WL','Size','B0','E')}
    names.update({name:(a(anchor,name),) for name in OTHER_CONTROLS
                  if name not in ('Other1','Other2','OtherMean')})
    names.update({'Anchor':(anchor,), 'Full':(a(anchor,'Full',target),),
                  'Other1':(a(anchor,'Full',others[0]),), 'Other2':(a(anchor,'Full',others[1]),),
                  'OtherMean':tuple(a(anchor,'Full',t) for t in others)})
    return names


def mean(indices):
    values = tuple(indices)
    return sum(values)/len(values)


def score_block(role,x,y,*,active='Full',make_views=aliases,
                comparisons=('Anchor','WL','Size',*OTHER_CONTROLS)):
    if y.shape!=x['ids'].shape or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid binary H role')
    n = len(y)
    ids_fields = {k:role[k] for k in ('setting','source','target','seed')}
    raw = {m:val for m,val in x.items() if m!='ids' and not m.startswith('margin_')}
    flags = {m:a==y for m,a in raw.items()}
    cells = {m:roc_score.confusion(a,y) for m,a in raw.items()}
    nll = {m:float(np.logaddexp(0.,(1-2*y)*x[f'margin_{m}']).mean()) for m in raw if f'margin_{m}' in x}
    endpoints,contrasts,stats,delta = [],[],{},{}
    for anchor in refit.ANCHORS:
        views = make_views(role,anchor)
        for name,members in views.items():
            point = (mean(Fraction(int(flags[m].sum()),n) for m in members),
                     mean(roc_score.f1(cells[m]) for m in members),
                     None if name=='EqualCount' else fmean(nll[m] for m in members))
            tn,fp,fn,tp = (mean(Fraction(v) for v in col) for col in zip(*(cells[m] for m in members),strict=True))
            stats[anchor,name] = point
            endpoints.append(dict(**ids_fields,anchor=anchor,method=name,n=n,accuracy=float(point[0]),
                                  macro_f1=float(point[1]),nll=point[2],
                                  tn=str(tn),fp=str(fp),fn=str(fn),tp=str(tp),positive_count=str(tp+fp)))
        for name in comparisons:
            l,r = stats[anchor,active],stats[anchor,name]
            corrected,harmed,changed = scoring.paired_counts(views[active],views[name],raw,flags)
            a,f = l[0]-r[0],l[1]-r[1]
            if a!=(corrected-harmed)/n or changed!=corrected+harmed:
                raise ArithmeticError('Strong-anchor paired classifications do not reconcile')
            loss = None if r[2] is None else l[2]-r[2]
            delta[anchor,name] = dict(acc=a,f1=f,nll=loss,corrected=corrected,harmed=harmed,changed=changed,n=n)
            contrasts.append(dict(**ids_fields,anchor=anchor,left=active,right=name,n=n,
                                  corrected=str(corrected),harmed=str(harmed),changed=str(changed),
                                  accuracy_exact=str(a),accuracy_pp=float(100*a),macro_f1_pp=float(100*f),nll_delta=loss))
    return endpoints,contrasts,stats,delta


def aggregate(roles,scores,contrasts,*,active='Full'):
    endpoints,comparisons = [],[]
    for setting in sorted({r['setting'] for r in roles.values()}):
        keys = [k for k,r in roles.items() if r['setting']==setting]
        if len(keys)!=12:
            raise PermissionError('Each setting requires twelve directions')
        for anchor,name in scores[keys[0]]:
            a,f,nll = zip(*(scores[k][anchor,name] for k in keys),strict=True)
            endpoints.append(dict(setting=setting,anchor=anchor,method=name,directions=len(keys),
                                  accuracy=float(mean(a)),macro_f1=float(mean(f)),
                                  nll=None if nll[0] is None else fmean(nll)))
        for anchor,name in contrasts[keys[0]]:
            rows = [contrasts[k][anchor,name] for k in keys]
            a = [r['acc'] for r in rows]
            overall = mean(a)
            totals = {field:str(sum(r[field] for r in rows)) for field in ('corrected','harmed','changed')}
            comparisons.append(dict(setting=setting,anchor=anchor,left=active,right=name,directions=len(keys),**totals,
                positive=sum(v>0 for v in a),tied=sum(v==0 for v in a),negative=sum(v<0 for v in a),
                accuracy_exact=str(overall),accuracy_pp=float(100*overall),
                macro_f1_pp=float(100*mean(r['f1'] for r in rows)),
                nll_delta=None if rows[0]['nll'] is None else fmean(r['nll'] for r in rows)))
    return endpoints,comparisons


def gate(comparisons,cfg):
    settings = {}
    for setting,anchor in cfg['primary_anchors'].items():
        rows = {r['right']:r for r in comparisons if r['setting']==setting and r['anchor']==anchor}
        fractions = {m:Fraction(r['accuracy_exact']) for m,r in rows.items()}
        positives = rows['Anchor']['positive']
        clauses = dict(proper_improvement=fractions['Anchor']>=Fraction(*cfg['minimum_accuracy_fraction']),
                       positive_directions=positives>=cfg['minimum_positive_directions'])
        clauses.update({f'strictly_above_{m}':fractions[m]>0 for m in ('B0Teacher','ScoreOnly','OtherMean','WL','Size')})
        settings[setting] = dict(anchor=anchor,passed=all(clauses.values()),clauses=clauses,
                                  positive=positives,tied=rows['Anchor']['tied'],negative=rows['Anchor']['negative'],
                                  differences_exact={m:str(x) for m,x in fractions.items()},
                                  differences_pp={m:float(100*x) for m,x in fractions.items()})
    advance = settings['proteins_edge']['passed'] and any(
        v['passed'] for k,v in settings.items() if k.startswith('mutagenicity_'))
    return dict(settings=settings,continue_remaining_seeds=advance)


def score(output,repo,cfg,c,obs,*,execution):
    roles,actions,_ = observed(output,cfg,c,obs)
    write_json(output/'score-execution.json',execution)
    cohorts,access,_ = scoring.read_h_cohorts(repo,c,roles)
    endpoints,comparisons,results,deltas = [],[],{},{}
    for key,role in roles.items():
        y,_,_ = cohorts[role['setting'],role['target']]
        ep,pa,results[key],deltas[key] = score_block(role,actions[key],y)
        endpoints.extend(ep)
        comparisons.extend(pa)
    summary,paired = aggregate(roles,results,deltas)
    tables = {'endpoints':endpoints,'paired-blocks':comparisons,'endpoint-summary':summary,'paired-summary':paired}
    for name,rows in tables.items():
        write_table(output/f'{name}.csv',rows)
    write_json(output/'gate.json',gate(paired,cfg))
    write_json(output/'h-label-access.json',dict(obs_sha256=obs,actual_reads=access))
    files = ['OBS.json','score-execution.json','gate.json','h-label-access.json',*[f'{m}.csv' for m in tables]]
    return prior_run.seal(output,'FINAL',files,dict(obs_sha256=obs))
