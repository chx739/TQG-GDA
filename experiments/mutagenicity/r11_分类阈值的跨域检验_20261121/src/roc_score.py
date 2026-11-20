"""Score the frozen R11 ROC decisions with shared target-class graph draws."""
from fractions import Fraction
import csv
from itertools import permutations
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import paired, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from . import inputs, prior_run, roc_transfer as transfer, run


METHODS = (*transfer.UNLABELED, *transfer.SUPERVISED)
GATE_CONTROLS = ('NativeV', 'DirectP', 'SourcePrior', 'OtherPriorMean', 'OtherQuantileMean',
                 'SamePriorCal', 'PoolMajority', 'MLLS')
PAIRS = tuple(('RateROC', arm) for arm in (*METHODS, *transfer.MEAN_ARMS) if arm != 'RateROC')
PAIRS += (('KnownRateROC', 'KnownNativeV'), ('KnownRateROC', 'PriorA'),
          ('KnownRateROC', 'MajorityA'), ('KnownRateROC', 'OwnA'), ('KnownRateROC', 'B0'))


def confusion(pred, y):
    return tuple(int(np.count_nonzero((pred == c) & (y == k))) for k in (0, 1) for c in (0, 1))


def f1(cells):
    tn, fp, fn, tp = cells
    return (Fraction(tn, max(2*tn+fp+fn, 1)) + Fraction(tp, max(2*tp+fp+fn, 1)))


def f1_draws(pred, y, weights):
    flags = np.column_stack([(pred == c) & (y == k) for k in (0, 1) for c in (0, 1)])
    tn, fp, fn, tp = (weights @ flags).T
    dn, dp = 2*tn+fp+fn, 2*tp+fp+fn
    return np.divide(tn, dn, out=np.zeros_like(tn), where=dn!=0.) + np.divide(tp, dp, out=np.zeros_like(tp), where=dp!=0.)


def endpoint_summary(rows, plan):
    for setting in plan['settings']:
        for target in [None, *plan['domains']]:
            for method in METHODS:
                selected = [r for r in rows if (r['setting'], r['method']) == (setting, method)
                            and (target is None or r['target'] == target)]
                yield dict(setting=setting, target=target, scope='setting' if target is None else 'target',
                           method=method, blocks=len(selected), accuracy=fmean(r['accuracy'] for r in selected),
                           macro_f1=fmean(r['macro_f1'] for r in selected),
                           constant_blocks=sum(r['constant'] for r in selected))


def gate(plan, cfg, paired_values):
    directions = list(permutations(plan['domains'], 2))
    required = (*transfer.STRONG, *GATE_CONTROLS)
    out = {}
    for setting in plan['settings']:
        points = {arm:[paired_values[setting, src, tgt, seed, 'RateROC', arm]
                       for src,tgt in directions for seed in plan['seeds']]
                  for arm in required}
        averages = {arm:sum(vals)/len(vals) for arm,vals in points.items()}
        winners = sum(sum(paired_values[setting,s,t,seed,'RateROC','B0'] for seed in plan['seeds']) > 0
                      for s,t in directions)
        strongest = min(transfer.STRONG, key=averages.__getitem__)
        pass_gap = averages[strongest] >= Fraction(*cfg['minimum_accuracy_fraction'])
        pass_controls = [name for name in GATE_CONTROLS if averages[name] <= 0]
        passed = pass_gap and not pass_controls and winners >= cfg['positive_directions']
        out[setting] = dict(passed=passed, strongest_source=strongest, gain_over_strongest_pp=float(100*averages[strongest]),
                            positive_directions_vs_B0=winners, accuracy_gate_met=pass_gap,
                            controls_not_beaten=pass_controls, comparisons_pp={k:float(100*v) for k,v in averages.items()})
    expand = out['proteins_edge']['passed'] and (out['mutagenicity_edge']['passed'] or out['mutagenicity_node']['passed'])
    return dict(settings=out, extend_to_reserve_seeds=expand)


def paired_block(actions, y, weights):
    flags = {arm: action == y for arm, action in actions.items()}
    macro = {arm: f1(confusion(action,y)) for arm,action in actions.items()}
    macro_samples = {arm:f1_draws(action,y,weights) for arm,action in actions.items()} if weights is not None else {}
    for left, right in PAIRS:
        controls = transfer.MEAN_ARMS.get(right, (right,))
        l = flags[left]
        fixed = Fraction(sum(int((l & ~flags[arm]).sum()) for arm in controls), len(controls))
        harmed = Fraction(sum(int((~l & flags[arm]).sum()) for arm in controls), len(controls))
        delta_a = (fixed-harmed)/len(y)
        delta_f1 = macro[left] - sum(macro[arm] for arm in controls)/len(controls)
        draws_a = draws_f1 = None
        if weights is not None:
            delta = l.astype(float)-np.mean([flags[arm] for arm in controls],axis=0)
            draws_a = weights @ delta
            draws_f1 = macro_samples[left] - np.mean([macro_samples[arm] for arm in controls],axis=0)
        yield (left,right), fixed, harmed, delta_a, delta_f1, draws_a, draws_f1


def prior_error_summary(errors, plan):
    for setting in plan['settings']:
        rows = [r for r in errors if r['setting']==setting]
        for estimator in ('size', 'source'):
            vals = [r[f'{estimator}_error'] for r in rows]
            yield dict(setting=setting, estimator=estimator, blocks=len(rows),
                       signed_bias=fmean(vals), mean_absolute_error=fmean(map(abs, vals)),
                       majority_matches=sum(r[f'{estimator}_majority_matches'] for r in rows))


def audit_old_endpoints(cfg, bound, endpoints):
    prior, root = bound[3], bound[5]
    checked = {}
    for path, anchor, obs_sha, methods in (
            (prior, cfg['prior_final_sha256'], cfg['prior_obs_sha256'],
             ('B0', 'SourceV', 'SourceCal', 'MLLS', 'OwnA', 'PriorA', 'MajorityA')),
            (root, cfg['structural_final_sha256'], cfg['structural_obs_sha256'],
             ('B0', 'SourceCal', 'Size', 'WL_selected'))):
        seal = transfer.stage(path,'FINAL',anchor)
        if seal['predecessors'] != dict(obs_sha256=obs_sha):
            raise PermissionError('Historical score audit has the wrong prediction parent')
        table = prior_run.read_member(path, seal['files'], 'endpoints.csv').decode()
        index = {(r['setting'],int(r['source']),int(r['target']),int(r['seed']),r['method']):r
                 for r in csv.DictReader(table.splitlines()) if r['method'] in methods}
        count = 0
        for row in endpoints:
            name = row['method']
            name = 'WL_selected' if name == 'WL' else name
            if name not in methods:
                continue
            identity = tuple(row[k] for k in prior_run.ID_FIELDS)
            old = index[(*identity, name)]
            if (int(old['n']) != row['n'] or abs(float(old['accuracy'])-row['accuracy'])>1e-14
                    or ('macro_f1' in old and abs(float(old['macro_f1'])-row['macro_f1'])>1e-14)):
                raise ValueError(f'Historical H scoring differs for {identity}/{name}')
            count += 1
        checked[path.name] = dict(endpoints=count, endpoints_sha256=seal['files']['endpoints.csv'])
    return checked


def evaluate(output, repo, cfg, bound, parents, unlabeled_sha, obs_sha):
    plan = bound[0]
    keys = prior_run.registered_keys(plan)
    raw, roles, frozen = transfer.verify_deployment(output, plan, parents, unlabeled_sha)
    diag = prior_run.verified_members(output, 'OBS', obs_sha, transfer.labeled_members(keys),
                                     dict(unlabeled_sha256=unlabeled_sha, prior_obs_sha256=cfg['prior_obs_sha256']))
    counts = json.loads(diag['a-audit.json'])
    if set(counts)!=set(keys):
        raise PermissionError('Incomplete A-count diagnostics')
    for key in keys:
        known = inputs.npz_member(diag[f'a-diagnostic/{key}.npz'])
        if set(known) != {'ids', *transfer.SUPERVISED}:
            raise PermissionError(f'Invalid diagnostic decision set: {key}')
        prior_run.verified_predictions(known, roles[key]['holdout_ids'], transfer.SUPERVISED, transfer.SUPERVISED)
        frozen[key].update(known)
    r7 = json.loads(raw['historical-plan.json'])
    reps, boots, min_class, _ = paired._settings(r7)
    endpoints, pairs, errors = [], [], []
    points_a, points_f, curves_a, curves_f = {}, {}, {}, {}
    cohorts, accesses = {}, {}
    archive = Path(plan['archive'])
    with seals.verified_seal_chain(archive, 'R7_FINAL') as scope:
        grants = {}
        for key in keys:
            task = roles[key]
            manifest = prior_run.label_manifest(scope, key, task, 'holdout')
            cohort_id = task['setting'], task['target']
            grant = prior_run.label_port(manifest, task, 'holdout', repo)
            previous = grants.setdefault(cohort_id, grant)
            if previous != grant or tuple(task['holdout_ids']) != grant.ids:
                raise PermissionError(f'Shared H role or raw label grant differs: {key}')
        with inputs.GraphLabelSession('holdout') as labels:
            for (setting,target),grant in grants.items():
                y = labels.load(grant, grant.ids)
                ids = np.array(grant.ids)
                order = np.argsort(ids, kind='stable')
                cohort = paired._cohort(setting, target, ids[order], y[order], reps, boots, min_class)
                cohorts[setting,target] = dict(y=y[order], order=order, **{k:cohort[k] for k in ('weights','audit')})
                accesses[f'{setting}-M{target}'] = list(grant.ids)
    for key in keys:
        role = roles[key]
        identity = {k:role[k] for k in prior_run.ID_FIELDS}
        cohort = cohorts[role['setting'],role['target']]
        y, order = cohort['y'], cohort['order']
        n = len(y)
        actions = {arm:frozen[key][arm][order] for arm in METHODS}
        for arm,pred in actions.items():
            tn,fp,fn,tp = confusion(pred,y)
            endpoints.append(dict(**identity, method=arm, n=n, correct=tp+tn, accuracy=(tp+tn)/n,
                                  macro_f1=float(f1((tn,fp,fn,tp))), predicted_positive=int(pred.sum()),
                                  constant=not pred.any() or pred.all(), tn=tn,fp=fp,fn=fn,tp=tp))
        for (left,right),fixed,harmed,acc,macro,da,df in paired_block(actions,y,cohort['weights']):
            addr = (*identity.values(),left,right)
            points_a[addr],points_f[addr] = acc,macro
            if da is not None:
                curves_a[addr],curves_f[addr] = da,df
            pairs.append(dict(**identity,left=left,right=right,n=n,corrected=str(fixed),harmed=str(harmed),
                              net=str(fixed-harmed),accuracy_pp=float(100*acc),macro_f1_pp=float(100*macro)))
        r = counts[key]
        observed, size, source = (Fraction(r[k]) for k in ('observed_prior','size_prior','source_prior'))
        majority = observed > Fraction(1,2)
        errors.append(dict(**identity,a_n=r['n'],a_positive=r['positive'],observed_prior=float(observed),
                           size_prior=float(size),source_prior=float(source),size_error=float(size-observed),
                           source_error=float(source-observed),size_majority_matches=(size>Fraction(1,2))==majority,
                           source_majority_matches=(source>Fraction(1,2))==majority))
    score_audit = audit_old_endpoints(cfg, bound, endpoints)
    acc_summary = run.summarize_pairs(plan, points_a, curves_a, PAIRS)
    f1_summary = run.summarize_pairs(plan, points_f, curves_f, PAIRS)
    result = gate(plan,cfg,points_a)
    tables = {'endpoints':endpoints, 'endpoint-summary':list(endpoint_summary(endpoints,plan)),
              'paired-blocks':pairs, 'paired-accuracy-summary':acc_summary, 'paired-macrof1-summary':f1_summary,
              'prior-errors':errors, 'prior-error-summary':list(prior_error_summary(errors,plan))}
    for name,rows in tables.items():
        write_table(output/f'{name}.csv',rows)
    write_json(output/'gate.json',result)
    write_json(output/'historical-scoring-parity.json',score_audit)
    write_json(output/'h-label-access.json',dict(obs_sha256=obs_sha,actual_reads=accesses))
    write_json(output/'cohort-audit.json',[entry['audit'] for entry in cohorts.values()])
    members = ['OBS.json','gate.json','historical-scoring-parity.json','h-label-access.json','cohort-audit.json',
               *[f'{name}.csv' for name in tables]]
    return prior_run.seal(output,'FINAL',members,dict(obs_sha256=obs_sha))
