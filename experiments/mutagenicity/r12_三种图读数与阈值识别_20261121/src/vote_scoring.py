"""Recheck the complete vote pilot, then count paired H corrections."""
from fractions import Fraction
from itertools import permutations
import json
from statistics import fmean

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from . import conditional_votes as experiment, vote_mixture


PAIRS = tuple(('Own',method) for method in (*experiment.METHODS,'OtherMean') if method!='Own')
METRICS = ('accuracy','macro_f1','recall_0','recall_1')


def verify_observation(out, plan, cfg, obs_sha):
    keys = prior_run.registered_keys(plan)
    obs = prior_run.verified_members(out,'OBS',obs_sha,experiment.members(keys),experiment.parents(cfg))
    roles, sources, rules = (json.loads(obs[f'{name}.json']) for name in ('roles','source-counts','rules'))
    if set(roles)!=set(keys) or set(rules)!=set(keys) or set(sources)!={r['group'] for r in roles.values()}:
        raise PermissionError('Wrong frozen source or task coverage')
    rows = {}
    for key in keys:
        role = roles[key]
        if key != inputs.key(**{k:role[k] for k in experiment.IDENTITY}):
            raise PermissionError(f'Wrong frozen task identity: {key}')
        x = inputs.npz_member(obs[f'predictions/{key}.npz'])
        if set(x)!={'ids','a_ids','a_bins','h_bins',*experiment.METHODS}:
            raise PermissionError(f'Wrong frozen prediction columns: {key}')
        if any(v.dtype!=np.int64 or v.ndim!=1 for v in x.values()):
            raise PermissionError(f'Non-integer or malformed frozen arrays: {key}')
        np.testing.assert_array_equal(x['ids'],role['holdout_ids'])
        np.testing.assert_array_equal(x['a_ids'],role['selected_a_ids'])
        np.testing.assert_equal(x['a_bins'].shape,x['a_ids'].shape)
        if not np.isin(x['a_bins'],np.arange(8)).all():
            raise PermissionError(f'Invalid joint A vote codes: {key}')
        for m in experiment.METHODS:
            if not np.isin(x[m],[0,1]).all() or x[m].shape!=x['ids'].shape:
                raise PermissionError(f'Invalid hard decision for {key}/{m}')
        np.testing.assert_array_equal(x['h_bins'],vote_mixture.encode(x['B0'],x['WL'],x['Size']))
        source = np.asarray(sources[role['group']]['counts'],dtype=np.int64)
        if source.sum()!=len(sources[role['group']]['v_ids']):
            raise PermissionError(f'V counts do not cover the source V role: {key}')
        counts = np.bincount(x['a_bins'],minlength=8).reshape(2,4)
        if vote_mixture.rules(source,counts)!=rules[key]:
            raise PermissionError(f'Wrong fitted table in sealed roles: {key}')
        swaps = [inputs.key(role['setting'],role['source'],t,role['seed']) for t in role['other_targets']]
        for name, expected in vote_mixture.predict(rules[key],[rules[arm] for arm in swaps],x['h_bins']).items():
            np.testing.assert_array_equal(x[name],expected)
        rows[key] = x
    return roles, rows


def compare(flags, weights):
    own = flags['Own']
    for _, right in PAIRS:
        other = ('Other1','Other2') if right=='OtherMean' else (right,)
        changed = np.mean([flags[arm] for arm in other],axis=0)
        gain = Fraction(sum(int((own & ~flags[arm]).sum()) for arm in other),len(other))
        harm = Fraction(sum(int((~own & flags[arm]).sum()) for arm in other),len(other))
        samples = None if weights is None else weights @ (own.astype(float)-changed)
        yield right, gain, harm, samples


def by_setting(endpoints, plan):
    for setting in plan['settings']:
        for target in (None,*plan['domains']):
            for method in (*experiment.METHODS,'OtherMean'):
                arms = ('Other1','Other2') if method=='OtherMean' else (method,)
                rows = [r for r in endpoints if r['setting']==setting and r['method'] in arms
                        and (target is None or target==r['target'])]
                yield dict(setting=setting,scope='setting' if target is None else 'target',target=target,method=method,
                           blocks=len(rows)//len(arms),
                           constant_blocks=sum(r['constant'] for r in rows)/len(arms),
                           **{k:fmean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
                              for k in METRICS})


def gate(plan, cfg, paired_values):
    directions = list(permutations(plan['domains'],2))
    cutoff = Fraction(*cfg['minimum_accuracy_fraction'])
    compared = ('Static','Global',*experiment.FIXED,'OtherMean')
    seeds = plan['seeds']
    by_setting = {}
    for setting in plan['settings']:
        changes = {arm:[paired_values[setting,s,t,seed,'Own',arm]
                        for s,t in directions for seed in seeds] for arm in compared}
        diffs = {arm:sum(rows)/len(rows) for arm,rows in changes.items()}
        best = min(experiment.FIXED,key=diffs.__getitem__)
        won = sum(sum(paired_values[setting,s,t,seed,'Own','Static'] for seed in seeds)>0 for s,t in directions)
        gaps_failed = [arm for arm in ('Static','Global',best) if diffs[arm]<cutoff]
        passed = not gaps_failed and diffs['OtherMean']>0 and won>=cfg['minimum_positive_directions']
        by_setting[setting] = dict(passed=passed,highest_fixed=best,positive_directions_vs_static=won,
                                   failed_minimum=gaps_failed,beats_other_mean=diffs['OtherMean']>0,
                                   gains_pp={arm:float(100*v) for arm,v in diffs.items()})
    extend = by_setting['proteins_edge']['passed'] and any(
        by_setting[name]['passed'] for name in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=by_setting,extend_to_reserve_seeds=extend)


def lock_rows(actions, flags):
    agree = actions['B0']==actions['WL']
    size_agrees = actions['B0']==actions['Size']
    change = actions['Own']!=actions['OwnLock']
    if np.any(change & ~agree):
        raise ValueError('OwnLock differs on a B0/WL disagreement')
    corrected = flags['Own'] & ~flags['OwnLock']
    harmed = flags['OwnLock'] & ~flags['Own']
    for name, scope in (('all_agree',agree),('size_agrees',agree & size_agrees),
                        ('size_opposes',agree & ~size_agrees)):
        yield dict(stratum=name,graphs=int(scope.sum()),both_wrong=int((scope & ~flags['B0']).sum()),
                   changed=int((scope & change).sum()),corrected=int((scope & corrected).sum()),
                   harmed=int((scope & harmed).sum()),net=int((scope & corrected).sum()-(scope & harmed).sum()))


def scored_cells(bins, y, actions):
    for code in range(8):
        mask = bins==code
        corrected = (actions['Own']==y) & (actions['OwnLock']!=y)
        harmed = (actions['Own']!=y) & (actions['OwnLock']==y)
        yield dict(size=code//4,b0=(code//2)%2,wl=code%2,graphs=int(mask.sum()),
                   true_0=int((mask & (y==0)).sum()),true_1=int((mask & (y==1)).sum()),
                   own_positive=int((mask & (actions['Own']==1)).sum()),
                   lock_positive=int((mask & (actions['OwnLock']==1)).sum()),
                   corrected=int((mask & corrected).sum()),harmed=int((mask & harmed).sum()))


def score(out, repo, cfg, history, obs_sha):
    plan = history.plan
    roles, predictions = verify_observation(out,plan,cfg,obs_sha)
    cohorts, grants, access = {},{},{}
    for key,role in roles.items():
        task = {k:role[k] for k in experiment.IDENTITY}
        manifest,checksum,context = historical.context(history.c,task,repo)
        h = manifest['ids']['holdout']
        if (checksum!=role['manifest_sha256'] or h!=role['holdout_ids'] or manifest['ids']!=role['ids']
                or h!=predictions[key]['ids'].tolist()):
            raise PermissionError(f'Wrong H label grant for {key}')
        cohort = task['setting'],task['target']
        grant = inputs.raw_grant(context,manifest,'holdout')
        if grants.setdefault(cohort,grant)!=grant:
            raise PermissionError('Shared H cohort has different role grants')
    reps, draw_seed, minimum, _ = paired._settings(history.c.plan)
    with inputs.GraphLabelSession('holdout') as labels:
        for (setting,target),grant in grants.items():
            ids = np.asarray(grant.ids)
            y = labels.load(grant,grant.ids)
            if y.shape!=ids.shape or not np.isin(y,[0,1]).all():
                raise ValueError('Unexpected binary H labels')
            order = np.argsort(ids,kind='stable')
            cohorts[setting,target] = y,order,paired._cohort(setting,target,ids[order],y[order],reps,draw_seed,minimum)
            access[f'{setting}-M{target}'] = ids.tolist()
    ends, comparisons, lock, cells = [],[],[],[]
    points, curves = {},{}
    for key,role in roles.items():
        identity = {k:role[k] for k in experiment.IDENTITY}
        x = predictions[key]
        y,order,cohort = cohorts[role['setting'],role['target']]
        actions = {arm:x[arm] for arm in experiment.METHODS}
        flags = {arm:a==y for arm,a in actions.items()}
        for arm,pred in actions.items():
            tn,fp,fn,tp = roc_score.confusion(pred,y)
            ends.append(dict(**identity,method=arm,n=len(y),correct=tp+tn,accuracy=(tp+tn)/len(y),
                             macro_f1=float(roc_score.f1((tn,fp,fn,tp))),recall_0=tn/(tn+fp) if tn+fp else None,
                             recall_1=tp/(tp+fn) if tp+fn else None,positive_count=int(pred.sum()),
                             constant=bool(pred.min()==pred.max()),tn=tn,fp=fp,fn=fn,tp=tp))
        for other, correct, harm, samples in compare({m:f[order] for m,f in flags.items()},cohort['weights']):
            key_pair = (*identity.values(),'Own',other)
            change = (correct-harm)/len(y)
            points[key_pair] = change
            if samples is not None:
                curves[key_pair] = samples
            comparisons.append(dict(**identity,left='Own',right=other,n=len(y),corrected=str(correct),
                                    harmed=str(harm),net=str(correct-harm),accuracy_pp=float(100*change)))
        lock.extend(dict(**identity,**row) for row in lock_rows(actions,flags))
        cells.extend(dict(**identity,**row) for row in scored_cells(x['h_bins'],y,actions))

    tables = {'endpoints':ends,'endpoint-summary':list(by_setting(ends,plan)),
              'paired-blocks':comparisons,'paired-summary':run.summarize_pairs(plan,points,curves,PAIRS),
              'own-lock':lock,'h-cells':cells}
    for name, rows in tables.items():
        write_table(out/f'{name}.csv',rows)
    write_json(out/'gate.json',gate(plan,cfg,points))
    write_json(out/'h-label-access.json',dict(obs_sha256=obs_sha,actual_reads=access,adaptation_labels_read=False))
    write_json(out/'cohort-audit.json',[entry[2]['audit'] for entry in cohorts.values()])
    return prior_run.seal(out,'FINAL',['OBS.json','gate.json','h-label-access.json','cohort-audit.json',
                                       *[f'{name}.csv' for name in tables]],dict(obs_sha256=obs_sha))
