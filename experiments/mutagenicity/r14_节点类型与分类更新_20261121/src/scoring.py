"""Open only authenticated H labels, after all fitted directions and predictions are sealed."""
from fractions import Fraction
import json

import numpy as np
from scipy.special import expit

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run as storage
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src.run import summarize_pairs
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src.scoring import auc
from . import activations, experiment

GATE_ARMS = ('Base', 'B0', 'SourceCal', 'True', 'Hist', 'V_Binding', 'Other_Binding', 'Base_equal_positive')
MAIN_PAIRS = tuple(('Binding', arm) for arm in (*GATE_ARMS, 'Other_Binding_1', 'Other_Binding_2'))
MAIN_PAIRS += tuple((arm, 'Base') for arm in experiment.METHODS if arm not in ('Binding','Base'))
OPTIONAL = tuple(('Binding', arm) for arm in experiment.HISTORICAL)
FINAL_MEMBERS = ('OBS.json', 'endpoints.csv', 'endpoint-summary.csv', 'paired-blocks.csv', 'paired-summary.csv',
                 'local-risk.csv', 'local-summary.csv', 'gate.json', 'h-label-access.json', 'cohort-audit.json')


def methods_for(role):
    return experiment.METHODS + tuple(arm for arm in experiment.HISTORICAL if arm in role['historical_members'])


def checked_predictions(role, x):
    ids = role['held_ids']
    np.testing.assert_array_equal(x['ids'], ids)
    methods = methods_for(role)
    arrays = {f'{stem}_{m}' for m in methods for stem in ('logits','decision')}
    arrays |= {f'{stem}_{m}' for m in experiment.ACTIVE_ARMS for stem in ('velocity','residual_velocity')}
    if set(x) != {'ids','base_margin'} | arrays:
        raise PermissionError('Prediction arrays differ from the sealed method registry')
    np.testing.assert_array_equal(x['base_margin'], activations.margin(x['logits_Base']))
    for arm in methods:
        z, a = x[f'logits_{arm}'], x[f'decision_{arm}']
        if z.shape != (len(ids),2) or not np.isfinite(z).all() or                 a.shape != (len(ids),) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('Invalid frozen R14 inference arrays')
        ref = (observation.equal_positive(x['ids'], z, int(x['decision_Binding'].sum()))
               if arm == 'Base_equal_positive' else z.argmax(1))
        np.testing.assert_array_equal(a, ref)
    np.testing.assert_array_equal(x['logits_Base_equal_positive'], x['logits_Base'])
    for arm in experiment.ACTIVE_ARMS:
        v, res = x[f'velocity_{arm}'], x[f'residual_velocity_{arm}']
        if v.shape != (len(ids),) or res.shape != v.shape or not np.isfinite([v,res]).all():
            raise ValueError('Invalid frozen R14 direction velocities')
        expected, _, _ = activations.residualize(x['base_margin'], v)
        np.testing.assert_allclose(res, expected, atol=1e-12, rtol=0.)
    return methods


def gate(plan, cfg, points):
    directions = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
    fixed = plan['phase1_seed']
    threshold = Fraction(*cfg['minimum_binding_gain_fraction'])
    settings = {}
    for setting in plan['settings']:
        paired_by_arm = {arm: [points.get((setting,s,t,fixed,'Binding',arm)) for s,t in directions] for arm in GATE_ARMS}
        average = {arm: None if None in values else sum(values)/len(values) for arm, values in paired_by_arm.items()}
        positive = sum(x is not None and x > 0 for x in paired_by_arm['Base'])
        passed = (average['Base'] is not None and average['Base'] >= threshold
                  and all(average[arm] is not None and average[arm] > 0 for arm in GATE_ARMS if arm != 'Base')
                  and positive >= cfg['minimum_positive_directions'])
        settings[setting] = dict(passed=passed, positive_directions_vs_base=positive,
                                 mean_pp={k: None if v is None else float(100*v) for k,v in average.items()})
    ready = settings['proteins_edge']['passed'] and any(
        row['passed'] for s,row in settings.items() if s.startswith('mutagenicity_'))
    return dict(extend=ready, settings=settings)


def contrast(a, b, flags, actions, draws):
    refs = ('Other_Binding_1','Other_Binding_2') if b=='Other_Binding' else (b,)
    if any(r not in flags for r in refs):
        return None
    x = flags[a]
    c, h, d = (sum(int(fn(r).sum()) for r in refs) for fn in (
        lambda r: x & ~flags[r], lambda r: ~x & flags[r], lambda r: actions[a] != actions[r]))
    corrected, harmed, changed = (Fraction(i,len(refs)) for i in (c,h,d))
    net = corrected-harmed
    if net != int(x.sum()) - sum(Fraction(int(flags[r].sum()), len(refs)) for r in refs):
        raise AssertionError('Paired arithmetic disagrees')
    samples = draws[a] - np.mean([draws[r] for r in refs],axis=0) if draws else None
    return net/len(x), samples, dict(corrected_exact=str(corrected), harmed_exact=str(harmed),
                                      changed_exact=str(changed), net_exact=str(net))


def mean_rows(plan, seeds, rows, names, metrics):
    expected = len(seeds) * len(plan['domains']) * (len(plan['domains'])-1)
    for setting in plan['settings']:
        for arm in names:
            matches = [x for x in rows if x['setting']==setting and x['method']==arm]
            if matches:
                metrics_mean = {key: float(np.mean([r[key] for r in matches]))
                                if all(r[key] is not None for r in matches) else None for key in metrics}
                yield dict(setting=setting,method=arm,coverage=len(matches),grid=expected,**metrics_mean)


def score(output, repo, plan, cfg, seeds, fit_sha, obs_sha, *, pilot):
    files = storage.verified_members(output,'OBS',obs_sha,experiment.obs_members(plan,seeds),{'fit_sha256':fit_sha})
    frozen = experiment.frozen_fit(output,plan,seeds,fit_sha)
    fit_roles = json.loads(frozen['fit-roles.json'])
    roles = json.loads(files['prediction-roles.json'])
    groups, tasks = observation.observations_expected(plan,seeds)
    if set(roles)!=set(tasks) or set(fit_roles)!=set(groups):
        raise PermissionError('The sealed prediction/fitting grid is incomplete')
    archive = historical.CArchive(plan)
    original = json.loads(frozen['historical-plan.json'])
    if archive.plan!=original:
        raise PermissionError('Historical plan no longer authentic')
    authorized, predictions = {}, {}
    for key, role in roles.items():
        task = {field:role[field] for field in experiment.IDENTITY}
        manifest, checksum, context = historical.context(archive,task,repo)
        recorded = next(r for r in fit_roles[role['group']]['tasks'] if r11.key(
            **{k:r[k] for k in experiment.IDENTITY}) == key)
        if checksum!=role['manifest_sha256'] or checksum!=recorded['manifest_sha256'] or                 manifest['ids']['holdout']!=role['held_ids'] or r11.key(**task)!=key:
            raise PermissionError('Holdout label authorization does not match sealed predictions')
        x = r11.npz_member(files[f'predictions/{key}.npz'])
        methods = checked_predictions(role,x)
        grant = r11.raw_grant(context,manifest,'holdout')
        name = task['setting'], task['target']
        if authorized.setdefault(name,grant)!=grant:
            raise PermissionError('Shared H label authorization differs across source/models')
        predictions[key] = methods,x,task

    reps, seed, minimum, _ = paired._settings(original)
    cohorts, access, points, intervals = {}, {}, {}, {}
    ends, comparisons, local = [], [], []
    # Include all modeling, prediction and source-role checks before this port.
    with r11.GraphLabelSession('holdout') as labels:
        for key, (methods,x,task) in predictions.items():
            s, source, target, model_seed = (task[k] for k in experiment.IDENTITY)
            name = (s,target)
            ids = x['ids']
            if name not in cohorts:
                y = labels.load(authorized[name],roles[key]['held_ids'])
                order = np.argsort(ids, kind='stable')
                cohorts[name] = y,order,paired._cohort(s,target,ids[order],y[order],reps,seed,minimum)
                access[key] = roles[key]['held_ids']
            y, order, cohort = cohorts[name]
            np.testing.assert_array_equal(ids[order],cohort['ids'])
            actions = {arm:x[f'decision_{arm}'] for arm in methods}
            flags = {arm:action==y for arm,action in actions.items()}
            values = cohort['weights'] @ np.column_stack([flags[m][order] for m in methods]) if cohort['weights'] is not None else None
            draws = {} if values is None else dict(zip(methods,values.T))
            z = {arm:x[f'logits_{arm}'] for arm in methods}
            score_by_arm = {}
            for arm in methods:
                metrics = evaluate.classification(z[arm],y,actions['Base'],decisions=actions[arm])
                if arm=='Base_equal_positive':
                    metrics['sampled_nll']=None
                row = dict(**task,method=arm,auroc=auc(z[arm],y),positive_count=int(actions[arm].sum()),**metrics)
                score_by_arm[arm] = row
                ends.append(row)
            for lhs,rhs in (*MAIN_PAIRS,*OPTIONAL):
                pair = contrast(lhs,rhs,flags,actions,draws)
                if pair is None:
                    continue
                value, bootstrap, counts = pair
                pair_key = (s,source,target,model_seed,lhs,rhs)
                points[pair_key] = value
                if bootstrap is not None:
                    intervals[pair_key] = bootstrap
                comparisons.append(dict(**task,left=lhs,right=rhs,n=len(y),accuracy_pp=float(100*value),**counts))
            residual_y = y - expit(x['base_margin'])
            for arm in experiment.ACTIVE_ARMS:
                v, res = x[f'velocity_{arm}'], x[f'residual_velocity_{arm}']
                local.append(dict(**task,method=arm,local_nll_improvement=float(np.mean(residual_y*v)),
                                  residual_local_nll_improvement=float(np.mean(residual_y*res)),
                                  velocity_rms=float(np.sqrt(np.mean(v*v))),residual_rms=float(np.sqrt(np.mean(res*res))),
                                  finite_nll_improvement=score_by_arm['Base']['sampled_nll']-score_by_arm[arm]['sampled_nll'],
                                  finite_accuracy_gain_pp=100*(flags[arm].sum()-flags['Base'].sum())/len(y),
                                  auroc_gain=score_by_arm[arm]['auroc']-score_by_arm['Base']['auroc']
                                  if score_by_arm[arm]['auroc'] is not None else None))
    storage.verified_members(output,'OBS',obs_sha,experiment.obs_members(plan,seeds),{'fit_sha256':fit_sha})
    write_table(output/'endpoints.csv',ends)
    write_table(output/'endpoint-summary.csv', list(mean_rows(plan,seeds,ends,(*experiment.METHODS,*experiment.HISTORICAL),
                ('accuracy','balanced_accuracy','recall_0','recall_1','sampled_nll','auroc','predicted_class_1'))))
    write_table(output/'paired-blocks.csv',comparisons)
    write_table(output/'paired-summary.csv',summarize_pairs(plan|{'seeds':seeds},points,intervals,MAIN_PAIRS))
    write_table(output/'local-risk.csv',local)
    write_table(output/'local-summary.csv', list(mean_rows(plan,seeds,local,experiment.ACTIVE_ARMS,
                ('local_nll_improvement','residual_local_nll_improvement','velocity_rms','residual_rms',
                 'finite_nll_improvement','finite_accuracy_gain_pp','auroc_gain'))))
    decision = gate(plan,cfg,points) if pilot else None
    write_json(output/'gate.json',dict(pilot=pilot, decision=decision))
    write_json(output/'h-label-access.json',dict(obs_sha256=obs_sha, actual_h_reads=access))
    write_json(output/'cohort-audit.json',[row[2]['audit'] for row in cohorts.values()])
    return storage.seal(output,'FINAL',FINAL_MEMBERS, {'obs_sha256':obs_sha}),decision
