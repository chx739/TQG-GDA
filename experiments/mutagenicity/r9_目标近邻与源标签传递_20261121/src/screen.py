"""Run two isolated pseudo-directions; gate evaluation on sealed H actions."""
from fractions import Fraction
import json

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, fit, paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table

from . import bridge, inputs, roles, wl

CRITICAL = ('B', 'Direct', 'MatchedWeights', 'SourceBridge')
COMPARATORS = CRITICAL + ('PooledBridge',)
METHODS = ('TargetBridge', *COMPARATORS)


def task_key(direction, seed):
    return f"ps{direction['source']}_to_{direction['target']}-seed-{seed}"


def tasks(role_record, plan):
    for d in role_record['directions']:
        for seed in plan['adaptation_seeds']:
            yield task_key(d, seed), d, seed


def prediction_path(key):
    return f'predictions/{key}.npz'


def fit_direction(direction, label_reader, rows, fingerprint_keys, output, plan):
    s, h = direction['source_ids'], direction['holdout_ids']
    overlap_src = set(fingerprint_keys[i] for i in s)
    overlap = [i for i in h if fingerprint_keys[i] in overlap_src]
    identity = f"ps{direction['source']}_to_{direction['target']}"
    result = {}
    summaries = []
    names = []
    task = direction['source'], direction['target']
    try:
        if 'unavailable_reason' in direction:
            raise bridge.NoCandidates(direction['unavailable_reason'])
        source_labels = label_reader.load(*task, s)
        base, model = bridge.fit_source(rows, s, h, source_labels, plan['source_model'])
    except (ValueError, FloatingPointError) as error:
        for seed in plan['adaptation_seeds']:
            result[task_key(direction, seed)] = dict(status='unavailable', reason=str(error))
        model = dict(status='unavailable', reason=str(error))
    else:
        for seed in plan['adaptation_seeds']:
            key = task_key(direction, seed)
            split = direction['perturbations'][str(seed)]
            try:
                arrays, checks = bridge.predict(rows, fingerprint_keys, s, h, split['adapt'],
                                                split['source_middle'], source_labels, base, plan)
            except bridge.NoCandidates as error:
                result[key] = dict(status='unavailable', reason=str(error))
                continue
            result[key] = dict(status='available', **checks)
            path = prediction_path(key)
            fit.write_npz(output/path, **arrays)
            names.append(path)
            for arm in COMPARATORS:
                row = dict(source=direction['source'], target=direction['target'], seed=seed, comparator=arm, n=len(h),
                           n_actions_differ=int((arrays['action_'+arm] != arrays['action_TargetBridge']).sum()),
                           weight_diagnostics_available=arm != 'B')
                if arm != 'B':
                    row.update(mean_overlap=float(arrays[f'overlap_vs_{arm}'].mean()),
                               mean_support_jaccard=float(arrays[f'support_jaccard_vs_{arm}'].mean()),
                               mean_total_variation=float(arrays[f'total_variation_vs_{arm}'].mean()),
                               top_weight_source_equal=int(arrays[f'top_weight_source_equal_vs_{arm}'].sum()),
                               mean_support=float(arrays[f'support_{arm}'].mean()),
                               target_mean_support=float(arrays['support_TargetBridge'].mean()),
                               mean_entropy=float(arrays[f'entropy_{arm}'].mean()),
                               target_mean_entropy=float(arrays['entropy_TargetBridge'].mean()),
                               mean_effective_neighbors=float(arrays[f'effective_neighbors_{arm}'].mean()),
                               target_mean_effective_neighbors=float(arrays['effective_neighbors_TargetBridge'].mean()))
                summaries.append(row)
    paths = {f'directions/{identity}-model.json': model,
             f'directions/{identity}-checks.json': dict(source_label_ids=label_reader.actual.get(task, []), requested_source_label_ids=s,
               source_label_scope='source_fit_and_source_endpoint_voting_only', target_h_same_fingerprint_source_ids=overlap,
               status=result)}
    for path, value in paths.items():
        write_json(output/path, value)
    return result, summaries, [*names, *paths]


def freeze(output, plan, repo):
    evidence = inputs.SourceEvidence(repo, plan)
    counts, edges, rows = wl.make_records(evidence.graphs(), plan)
    role_record = roles.build(edges, counts, plan)
    for name in ('predictions', 'directions'):
        (output/name).mkdir()
    write_json(output/'roles.json', role_record)
    write_json(output/'source-provenance.json', evidence.provenance)
    write_json(output/'fingerprints.json', {
        str(graph_id): dict(undirected_edges=edges[graph_id], group_sha256=inputs.sha(wl.count_key(counts[graph_id])),
                            raw_layer_counts=[{color.hex(): count for color, count in sorted(layer.items())}
                                              for layer in counts[graph_id]])
        for graph_id in evidence.ids})
    fingerprints = {i: wl.count_key(counts[i]) for i in evidence.ids}
    statuses, summaries, paths = {}, [], ['executed-config.json', 'roles.json', 'source-provenance.json', 'fingerprints.json']
    # Each direction owns a separate fitted model and local source-label array.
    # Repeating A only recomputes propagation weights, without refitting B.
    reader = inputs.RoleLabels(evidence, role_record, phase='fit')
    for direction in role_record['directions']:
        status, summary, names = fit_direction(direction, reader, rows, fingerprints, output, plan)
        statuses.update(status)
        summaries.extend(summary)
        paths.extend(names)
    write_json(output/'block-status.json', statuses)
    if summaries:
        write_table(output/'action-and-weight-checks.csv', summaries)
        paths.append('action-and-weight-checks.csv')
    paths.append('block-status.json')
    write_json(output/'OBS.json', dict(stage='R9_OBS', r8_parent_sha256=plan['r8_observation_sha256'],
             files={name: digest(output/name) for name in paths}))
    return digest(output/'OBS.json')


def read_frozen(output, obs_sha, plan):
    seal = (output/'OBS.json').read_bytes()
    if inputs.sha(seal) != obs_sha:
        raise PermissionError('R9 observation seal changed')
    obs = json.loads(seal)
    if obs['stage'] != 'R9_OBS' or obs['r8_parent_sha256'] != plan['r8_observation_sha256']:
        raise PermissionError('R9 OBS identity differs')
    files = {}
    for relative, expected_hash in obs['files'].items():
        payload = (output/relative).read_bytes()
        if inputs.sha(payload) != expected_hash:
            raise PermissionError(f'R9 observation differs: {relative}')
        files[relative] = payload
    record, status = [json.loads(files[name]) for name in ('roles.json', 'block-status.json')]
    if [[d['source'], d['target']] for d in record['directions']] != plan['directions']:
        raise PermissionError('Both registered directions must be sealed')
    observed_tasks = {key for key, _, _ in tasks(record, plan)}
    if status.keys() != observed_tasks:
        raise PermissionError('R9 perturbation registry incomplete')
    for key, info in status.items():
        if (info['status'] == 'available') != (prediction_path(key) in files):
            raise PermissionError('R9 prediction files inconsistent with block status')
    for d in record['directions']:
        name = f"directions/ps{d['source']}_to_{d['target']}"
        if not {name+'-checks.json', name+'-model.json'} <= files.keys():
            raise PermissionError('R9 source direction lacks sealed model or label lineage')
    return files, record, status


def equal_average(rows, plan, field):
    values = [Fraction(row[field], row['n']) for row in rows]
    total = len(plan['directions'])*len(plan['adaptation_seeds'])
    if len(values) != total:
        raise ValueError('Incomplete exact aggregate grid')
    # Each registered direction has the same number of A perturbations.
    return sum(values) / total


def action_gate(files, record, status, plan):
    unavailable = [key for key, value in status.items() if value['status'] != 'available']
    if unavailable:
        return dict(scoring_permitted=False, reason='unavailable_blocks', unavailable=unavailable)
    counts = {arm: [] for arm in COMPARATORS}
    for key, d, seed in tasks(record, plan):
        arrays = inputs.npz_member(files[prediction_path(key)])
        np.testing.assert_array_equal(arrays['h_ids'], d['holdout_ids'])
        main = arrays['action_TargetBridge']
        for arm in COMPARATORS:
            n = int(np.count_nonzero(main != arrays['action_'+arm]))
            counts[arm].append(dict(source=d['source'], target=d['target'], seed=seed, n=len(main), disagree=n))
    rates = {arm: equal_average(group, plan, 'disagree') for arm, group in counts.items()}
    cutoff = Fraction(*plan['minimum_accuracy_gain_fraction'])
    too_small = [arm for arm in CRITICAL if rates[arm] < cutoff]
    return dict(scoring_permitted=not too_small, reason='action_bound_below_threshold' if too_small else 'scoring_required',
                threshold=str(cutoff), blocked_comparators=too_small,
                rows=counts, avg_disagreement={arm: float(value) for arm, value in rates.items()},
                exact_disagreement={arm: str(value) for arm, value in rates.items()})


def bootstrap_plan(plan):
    return dict(real_model_seeds=plan['adaptation_seeds'],
                phases={'P': dict(setting='mutagenicity_m0_pseudo', directions=plan['directions'])},
                evaluation=dict(cohort_bootstrap_replicates=plan['cohort_bootstrap_replicates'],
                                cohort_bootstrap_seed=plan['cohort_bootstrap_seed'],
                                paired_interval=dict(strata=['target_domain', 'class'],
                                                     reuse_draws_across_methods_seeds_and_source_directions=True,
                                                     conditional_on_fitted_models=True,
                                                     minimum_graphs_per_class_for_interval=1,
                                                     percentiles=[2.5, 97.5], level=.95)))


def paired_verdict(rows, plan):
    cutoff = Fraction(*plan['minimum_accuracy_gain_fraction'])
    fair, all_averages = {}, {}
    for arm in COMPARATORS:
        portion = [row for row in rows if row['comparator'] == arm]
        pooled = equal_average(portion, plan, 'net')
        by_direction = {}
        for s, t in plan['directions']:
            items = [Fraction(row['net'], row['n']) for row in portion if (row['source'], row['target']) == (s, t)]
            by_direction[f'{s}→{t}'] = sum(items)/len(items)
        meets = pooled >= cutoff and all(value >= 0 for value in by_direction.values())
        if arm == 'MatchedWeights':
            meets &= all(value > 0 for value in by_direction.values())
        if arm in CRITICAL:
            fair[arm] = meets
        all_averages[arm] = dict(pooled_fraction=str(pooled), pooled_pp=float(100*pooled),
                                 directions={d: str(v) for d, v in by_direction.items()})
    return dict(passed=all(fair.values()), passed_each_critical=fair, scores=all_averages,
                reason='supervised_accuracy_compared', scoring_label_pools=['directional_holdout'])


def score_all(files, record, plan, evidence):
    cohorts, rows, endpoints, label_ids = [], [], [], {}
    p = bootstrap_plan(plan)
    reader = inputs.RoleLabels(evidence, record, phase='score')
    replicates, bootstrap_seed, minimum, quantiles = paired._settings(p)
    for d in record['directions']:
        s, t, holdout = d['source'], d['target'], d['holdout_ids']
        y = reader.load(s, t, holdout)
        label_ids[f'{s}→{t}'] = reader.actual[s, t]
        ids = np.asarray(holdout)
        # Role construction stores H in graph-ID order. Reuse exactly the
        # same cohort draws for every arm, bootstrap summary and perturbation.
        cohort = paired._cohort('mutagenicity_m0_pseudo', t, ids, y, replicates, bootstrap_seed, minimum)
        for seed in plan['adaptation_seeds']:
            key = task_key(d, seed)
            x = inputs.npz_member(files[prediction_path(key)])
            np.testing.assert_array_equal(x['h_ids'], ids)
            raw = {arm: x['logp_'+arm] for arm in METHODS}
            actions = {arm: x['action_'+arm] for arm in METHODS}
            task = dict(setting='mutagenicity_m0_pseudo', source=s, target=t, seed=seed)
            cohorts.append(dict(**task, ids=ids, y=y, logits=raw, decisions=actions))
            for arm in METHODS:
                values = evaluate.classification(raw[arm], y, actions['B'], decisions=actions[arm])
                endpoints.append(dict(**task, method=arm, **values))
            main = actions['TargetBridge'] == y
            for arm in COMPARATORS:
                ctrl = actions[arm] == y
                corrected, harmed = int((main & ~ctrl).sum()), int((~main & ctrl).sum())
                diff = main.astype(float) - ctrl.astype(float)
                lower, upper = [None, None]
                if cohort['eligible']:
                    lower, upper = np.percentile(100*cohort['weights'] @ diff, quantiles).tolist()
                rows.append(dict(**task, comparator=arm, n=len(y), corrected=corrected, harmed=harmed,
                                 net=corrected-harmed, accuracy_pp=100*(corrected-harmed)/len(y),
                                 accuracy_ci_low=lower, accuracy_ci_high=upper))
    result = paired.paired_bootstrap(p, cohorts, phase='P',
        comparisons=[('TargetBridge', ref) for ref in COMPARATORS])
    result['interval_rows'] = [item for item in result['interval_rows'] if item['metric'] != 'directional_risk_gap']
    for key in ('directional_gap', 'candidate_max_recomputed_each_replicate'):
        result['metadata'].pop(key, None)
    return endpoints, rows, result, label_ids


def finish(output, obs_sha, plan, repo):
    files, record, status = read_frozen(output, obs_sha, plan)
    gate = action_gate(files, record, status, plan)
    write_json(output/'action-gate.json', gate)
    paths = ['OBS.json', 'action-gate.json', 'gate.json']
    if gate['scoring_permitted']:
        evidence = inputs.SourceEvidence(repo, plan)
        endpoints, rows, intervals, labels = score_all(files, record, plan, evidence)
        read_frozen(output, obs_sha, plan)
        write_table(output/'endpoints.csv', endpoints)
        write_table(output/'paired-seeds.csv', rows)
        write_json(output/'paired-intervals.json', intervals)
        write_json(output/'scoring-label-ids.json', labels)
        verdict = paired_verdict(rows, plan)
        paths.extend(['endpoints.csv', 'paired-seeds.csv', 'paired-intervals.json', 'scoring-label-ids.json'])
    else:
        verdict = dict(passed=False, reason=gate['reason'], blocked_comparators=gate.get('blocked_comparators', []),
                       scoring_label_pools=[])
    write_json(output/'gate.json', verdict)
    write_json(output/'FINAL.json', dict(stage='R9_FINAL', observation_sha256=obs_sha,
        files={name: digest(output/name) for name in paths}))
    return verdict
