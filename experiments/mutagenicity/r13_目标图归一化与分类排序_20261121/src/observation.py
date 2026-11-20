"""Run equal-budget S/A calibrations and freeze all H predictions before scoring."""
import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11_inputs
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run
from . import historical, normalization

IDENTITY = ('setting', 'source', 'target', 'seed')
METHODS_NODE = ('B0', 'SourceCal', 'S_node', 'A_node', 'Other_node_1', 'Other_node_2')
METHODS_GRAPH = ('S_graph', 'A_graph', 'Other_graph_1', 'Other_graph_2', 'B0_equal_positive')
HISTORICAL = ('SourceOnly', 'SourceFusion', 'GRADE')


def group_tasks(plan, seeds):
    for setting in plan['settings']:
        for source in plan['domains']:
            for seed in seeds:
                yield setting, source, seed, [dict(setting=setting, source=source, target=target, seed=seed)
                                              for target in plan['domains'] if target != source]


def observations_expected(plan, seeds):
    groups, tasks = [], []
    for s, source, seed, entries in group_tasks(plan, seeds):
        groups.append(f'{s}-M{source}-seed-{seed}')
        tasks.extend(r11_inputs.key(**t) for t in entries)
    return groups, tasks


def required_members(plan, seeds):
    groups, tasks = observations_expected(plan, seeds)
    return ['executed-config.json', 'historical-plan.json', 'roles.json',
            'input-access.json', *[f'stats/{name}.json' for name in groups],
            *[f'predictions/{name}.npz' for name in tasks]]


def roles_for_group(setting, items):
    source_ids = items[0]['ids']['source']
    if any(row['ids']['source'] != source_ids or row['ids']['validation'] != items[0]['ids']['validation']
           for row in items):
        raise PermissionError('Same-source cohorts differ across directions')
    sources = {'S': source_ids}
    sources.update({f"A{row['target']}": row['ids']['adapt'] for row in items})
    all_pools = list(sources.values()) + [items[0]['ids']['validation']] + [row['ids']['holdout'] for row in items]
    flat = [i for pool in all_pools for i in pool]
    if len(flat) != len(set(flat)):
        raise PermissionError('S, V and target A/H roles overlap within a source group')
    k = min(map(len, sources.values()))
    return k, {role: historical.selected_ids(setting, ids, k) for role, ids in sources.items()}


def equal_positive(ids, b0_logits, positive):
    if (not 0 <= positive <= len(ids) or len(set(ids)) != len(ids)
            or b0_logits.shape != (len(ids), 2)):
        raise ValueError('Invalid equal-positive diagnostic')
    margin = b0_logits[:, 1] - b0_logits[:, 0]
    order = np.lexsort((ids, -margin))
    decision = np.zeros(len(ids), dtype=np.int64)
    decision[order[:positive]] = 1
    return decision


def bind(task, c, repo, data, lineage, checksum):
    key = r11_inputs.key(**task)
    mf, key_checksum, current = historical.context(c, task, repo)
    obj = mf['provenance']['r5_files']
    if (current['root'] != data['root'] or obj.get(lineage['path']) != lineage['sha256']
            or (checksum is not None and key_checksum != checksum)):
        raise PermissionError(f'{key} does not use the same original checkpoint or task manifest')
    return mf, key_checksum


def evaluate_pool(original, cohorts, store, batch_size, plan, weights):
    """Construct the complete pool×weight matrix before evaluating adapted H."""
    arms = {}
    receipt = {}
    for pool, ids in cohorts.items():
        graphs = store.take(ids)
        for weighting in weights:
            name = f'{pool}_{weighting}'
            native, stats, changed = normalization.recalibrate(original, graphs, weighting, batch_size)
            replay, alt, _ = normalization.recalibrate(original, graphs, weighting, plan['check_batch_size'])
            width = plan['rebatch_tolerance']
            max_mean, max_var = 0., 0.
            for layer, measured in stats.items():
                expected = alt[layer]
                if any(measured[k] != expected[k] for k in ('graphs', 'nodes', 'effective_weight', 'epsilon', 'buffer_dtype')):
                    raise ValueError('The BN calibration cohort changed on re-batching')
                for field in ('mean', 'variance'):
                    x, y = np.asarray(measured[field]), np.asarray(expected[field])
                    np.testing.assert_allclose(x, y, **width)
                    discrepancy = float(np.max(np.abs(x - y)))
                    if field == 'mean':
                        max_mean = max(max_mean, discrepancy)
                    else:
                        max_var = max(max_var, discrepancy)
            arms[name] = (native, replay)
            receipt[name] = dict(pool=pool, weighting=weighting, ids=ids, moments=stats,
                                 changed_buffers=changed, repartition_max_abs_mean_error=max_mean,
                                 repartition_max_abs_variance_error=max_var)
    return arms, receipt


def name_map(target, others, weights):
    result = {}
    for weight in weights:
        result[f'S_{weight}'] = f'S_{weight}'
        result[f'A_{weight}'] = f'A{target}_{weight}'
        result.update({f'Other_{weight}_{slot}': f'A{t}_{weight}' for slot, t in enumerate(others, 1)})
    return result


def forward(calibrated, source_map, h, batch_size, plan):
    arrays, discrepancies = {}, {}
    for name, real in source_map.items():
        native, replay = calibrated[real]
        output = normalization.logits(native, h, batch_size)
        repartition = normalization.logits(replay, h, plan['check_batch_size'])
        discrepancies[name] = normalization.compare_logits(output, repartition, plan['rebatch_tolerance'])
        arrays[name] = output
    return arrays, discrepancies


def freeze(output, repo, plan, seeds, weights, *, check_only=False):
    c = historical.CArchive(plan)
    ref = historical.ReferenceArms(plan, repo)
    graphs = {}
    scopes, replay_checks = {}, []
    if not check_only:
        for directory in ('stats', 'predictions'):
            (output / directory).mkdir()
    for setting, source, seed, entries in group_tasks(plan, seeds):
        manifest, checksum, data = historical.context(c, entries[0], repo)
        original, lineage = historical.checkpoint(manifest, data)
        store = graphs.get(setting)
        if store is None:
            store = graphs[setting] = historical.GraphStore(data)
        batch_size = lineage['batch_size']
        items, held = [], {}
        # Authenticate selection metadata and reproduce the original buffers on each H.
        for task in entries:
            m, auth = bind(task, c, repo, data, lineage, checksum if task == entries[0] else None)
            t, h_ids = task['target'], m['ids']['holdout']
            source_score = c.b0(task, h_ids)
            h_graphs = store.take(h_ids)
            replay = normalization.logits(original, h_graphs, batch_size)
            error = normalization.compare_logits(replay, source_score, plan['original_buffer_tolerance'])
            second = normalization.logits(original, h_graphs, plan['check_batch_size'])
            rebatch = normalization.compare_logits(replay, second, plan['rebatch_tolerance'])
            approved, ref_sha = ref.get(task, h_ids, source_score)
            replay_checks.append(dict(**task, b0_max_abs_error=error, b0_repartition_max_abs_error=rebatch,
                                      historical_references=list(ref_sha)))
            items.append(task | {'ids': m['ids'], 'manifest_sha256': auth})
            held[t] = (h_ids, h_graphs, source_score, replay, error, rebatch, approved, ref_sha)
        k, cohorts = roles_for_group(setting, items)
        if check_only:
            continue
        calibrated, statistics = evaluate_pool(original, cohorts, store, batch_size, plan, weights)
        group_key = f'{setting}-M{source}-seed-{seed}'
        write_json(output / f'stats/{group_key}.json',
                   dict(setting=setting, source=source, seed=seed, graph_container_sha256=store.graph_sha,
                        original=lineage, selected_graphs_per_pool=k, fit_arms=statistics))
        for item in items:
            t = item['target']
            h_ids, h_graphs, b0, replay, error, rebatch, approved, sha = held[t]
            others = sorted(target for target in held if target != t)
            mapping = name_map(t, others, weights)
            logits, errors = forward(calibrated, mapping, h_graphs, batch_size, plan)
            logits.update(approved)
            logits['B0'] = b0
            arrays = {'ids': np.asarray(h_ids), 'B0_raw_replay': replay}
            for arm, z in logits.items():
                arrays[f'logits_{arm}'] = z
                arrays[f'decision_{arm}'] = z.argmax(1).astype(np.int64)
            if 'graph' in weights:
                positive = int(arrays['decision_A_graph'].sum())
                arrays['logits_B0_equal_positive'] = b0
                arrays['decision_B0_equal_positive'] = equal_positive(arrays['ids'], b0, positive)
            key = r11_inputs.key(**{k: item[k] for k in IDENTITY})
            prior_run.save_predictions(output, f'predictions/{key}.npz', arrays)
            scopes[key] = {key: item[key] for key in IDENTITY} | dict(
                group_key=group_key, held_ids=h_ids, manifest_sha256=item['manifest_sha256'],
                statistics_arms=mapping, observation_ref_sha256=sha, b0_max_abs_error=error,
                b0_repartition_max_abs_error=rebatch, repartition_logit_max_abs_error=errors)
        print(f'R13 sealed-ready {group_key}: K={k}, targets={sorted(held)}, '
              f'largest B0 error={max(row[4] for row in held.values()):.3g}', flush=True)
    if check_only:
        return replay_checks
    write_json(output / 'roles.json', scopes)
    write_json(output / 'input-access.json', c.digests)
    write_json(output / 'historical-plan.json', c.plan)
    return prior_run.seal(output, 'OBS', required_members(plan, seeds), {'r7_final_digest': plan['r7_final_digest']})


def registered_predictions(arrays, role, graph):
    ids = role['held_ids']
    np.testing.assert_array_equal(arrays['ids'], ids)
    methods = METHODS_NODE + (METHODS_GRAPH if graph else ()) + tuple(
        name for name in HISTORICAL if name in role['observation_ref_sha256'])
    expected = {'ids', 'B0_raw_replay'} | {f'{stem}_{name}' for stem in ('decision', 'logits') for name in methods}
    if set(arrays) != expected:
        raise PermissionError('A frozen prediction is absent or unregistered')
    for arm in methods:
        z, action = arrays[f'logits_{arm}'], arrays[f'decision_{arm}']
        if (z.shape != (len(ids), 2) or not np.isfinite(z).all()
                or action.shape != (len(ids),) or not np.issubdtype(action.dtype, np.integer)
                or not np.isin(action, (0, 1)).all()):
            raise ValueError(f'{arm} has invalid frozen logits or classes')
        expected_action = (equal_positive(arrays['ids'], arrays['logits_B0'], int(arrays['decision_A_graph'].sum()))
                           if arm == 'B0_equal_positive' else z.argmax(1))
        np.testing.assert_array_equal(action, expected_action)
    return methods
