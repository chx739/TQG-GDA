"""Cross two frozen R10 GRADE models on the original H roles."""
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import stdev

import numpy as np
import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs as r8_inputs, pilot
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as label_inputs, prior_run
from . import data, jobs, scoring, training, validation


METHODS = ('A1', 'A3', 'SourceOnly')
METADATA = ('executed-config.json', 'roles.json', 'lineage.json', 'source-sequences.csv')
RESULTS = ('endpoints.csv', 'matrix.csv', 'paired-seeds.csv', 'paired-summary.csv',
           'cohort-audit.json', 'label-access.json')


def tasks(plan):
    return {(seed, h): jobs.target_key(seed, h) for seed in plan['seeds'] for h in plan['targets']}


def prediction_file(key):
    return f'predictions/{key}.npz'


def verify_roles(plan, roles):
    expected = tasks(plan)
    if (set(roles['tasks']) != set(expected.values()) or
            set(roles['source_fit']) != {str(seed) for seed in plan['seeds']}):
        raise PermissionError('Wrong crossover task or source roster')
    pools = roles['pools']
    members = [i for pool in pools.values() for i in pool]
    if (set(pools) != {'S', 'A-M1', 'A-M3', 'H-M1', 'H-M3'}
            or any(not pool for pool in pools.values()) or len(set(members)) != len(members)
            or set(pools['S']) != {i for src in roles['source_fit'].values() for i in src}):
        raise PermissionError('Source, adaptation and holdout pools must be disjoint and complete')
    for (seed, h), key in expected.items():
        task = roles['tasks'][key]
        s = roles['source_fit'][str(seed)]
        if ((task['setting'], task['source'], task['target'], task['seed']) !=
                (plan['setting'], plan['source'], h, seed) or
                not s or len(set(s)) != len(s) or task['source_ids'] != s or
                task['a_ids'] != pools[f'A-M{h}'] or task['h_ids'] != pools[f'H-M{h}']):
            raise PermissionError(f'Crossover roles differ: {key}')


def authenticate(repo, plan):
    root = Path(plan['archive'])
    if digest(root / 'FINAL.json') != plan['final_sha256']:
        raise PermissionError('Original R10 FINAL differs from registration')
    final = json.loads((root / 'FINAL.json').read_bytes())
    expected = {'OBS.json', 'evaluation/COMPLETE.json',
                *(f'evaluation/{name}' for name in scoring.EVALUATION_FILES)}
    if (final['stage'] != 'R10_FINAL' or final['blocks'] != 10 or final['score_label_pools'] != ['holdout']
            or set(final['files']) != expected or final['files'] != scoring._hash_members(root, expected)):
        raise PermissionError('Original R10 FINAL inventory differs')
    held_members = [f'data/H-M{h}.pt' for h in plan['targets']]
    roles, archived = data.read_inputs(root, requested=[
        *held_members, 'data/S.pt', *(f'data/A-M{h}.pt' for h in plan['targets']),
        *(f'data/S-labels-{seed}.pt' for seed in plan['seeds']),
    ])
    native = archived['plan']
    if (plan['setting'], plan['source'], plan['targets'], plan['seeds']) != (
            'mutagenicity_edge', 0, [1, 3], list(range(3405, 3410))) or (
            native['setting'], native['directions'], native['seeds'], native['model']['mode']) != (
            plan['setting'], [[plan['source'], h] for h in plan['targets']], plan['seeds'], 'graph'):
        raise PermissionError('Crossover scope differs from the registered native GRADE run')
    verify_roles(plan, roles)
    obs_sha, original = scoring.verify_observation(root, roles, repo)
    if obs_sha != final['observation_sha256']:
        raise PermissionError('Original R10 OBS differs from FINAL')
    receipt = json.loads((root / 'observation/training-verification.json').read_bytes())
    complete, _ = validation.validate_jobs(root, None, sealed_ledgers=receipt['supervision_ledgers'])
    obs = json.loads((root / 'OBS.json').read_bytes())
    parents = dict(FINAL=plan['final_sha256'], OBS=obs_sha, INPUT=obs['input_sha256'],
                   R7_FINAL=roles['r7_final_digest'])
    return root, roles, native, original, complete, parents


def source_history(trace, epochs):
    orders = [row['source_order_sha256'] for row in trace if row['event'] == 'epoch']
    if len(orders) != epochs or any(not isinstance(x, str) or not re.fullmatch('[0-9a-f]{64}', x) for x in orders):
        raise ValueError('Missing or invalid per-update source order receipt')
    return orders


def sequence_row(seed, orders, a1, a3):
    row = dict(seed=seed, a1_graphs=a1, a3_graphs=a3, updates=len(orders['A1']))
    for left, right in combinations(METHODS, 2):
        matches = [step for step, (l, r) in enumerate(zip(orders[left], orders[right], strict=True), 1) if l == r]
        row[f'{left}_{right}_same_steps'] = len(matches)
        row[f'{left}_{right}_same_step_numbers'] = ' '.join(map(str, matches))
    return row


def validate_prediction(row, ids, historic, holdout):
    required = {'ids', *(f'{field}_{method}' for field in ('logits', 'actions') for method in METHODS)}
    if (set(row) != required or row['ids'].dtype != np.int64 or not np.array_equal(row['ids'], ids)):
        raise PermissionError('Mismatched crossover prediction inventory or H graph order')
    for method in METHODS:
        z, actions = row[f'logits_{method}'], row[f'actions_{method}']
        if (z.shape != (len(ids), 2) or z.dtype != np.float32 or not np.isfinite(z).all()
                or actions.dtype != np.int64 or not np.array_equal(actions, z.argmax(1))):
            raise PermissionError(f'Invalid native prediction: {method}')
    errors = {}
    for method, old in ((f'A{holdout}', 'GRADE'), ('SourceOnly', 'SourceOnly')):
        logits = row[f'logits_{method}']
        archived = historic[f'logits_{old}']
        if (not np.array_equal(row[f'actions_{method}'], historic[f'actions_{old}'])
                or not np.allclose(logits, archived, atol=1e-7, rtol=1e-6)):
            raise PermissionError(f'Crossover replay differs from original R10 H{holdout}/{old}')
        errors[method] = float(np.max(np.abs(logits - archived)))
    return errors


def freeze(output, plan, authenticated):
    root, roles, native, original, complete, parents = authenticated
    task_keys = tasks(plan)
    held = {h: torch.load(root / f'data/H-M{h}.pt', map_location='cpu', weights_only=True)
            for h in plan['targets']}
    for h, graphs in held.items():
        if set(graphs) != set(roles['pools'][f'H-M{h}']):
            raise PermissionError(f'Wrong topology records for H{h}')
    predictions = {key: {} for key in task_keys.values()}
    models = []
    orders = {seed: {} for seed in plan['seeds']}
    for name, seed, target, attempt, summary in complete:
        method = 'SourceOnly' if target is None else f'A{target}'
        model = training.new_model(native['model'], roles['node_features'], roles['classes'], native['execution']['device'])
        model.load_state_dict(torch.load(attempt / 'model.pt', map_location='cpu', weights_only=True))
        before = training.state_hash(model)
        if before != summary['final_parameter_sha256']:
            raise PermissionError(f'Checkpoint state differs: {name}')
        for h in plan['targets']:
            key = task_keys[seed, h]
            ids = roles['tasks'][key]['h_ids']
            prediction = training.infer(model, held[h], ids, native)
            if prediction['ids'].tolist() != ids:
                raise PermissionError(f'Graph order changed during inference: {name}/H{h}')
            predictions[key].update(ids=prediction['ids'].numpy(),
                                    **{f'logits_{method}': prediction['logits'].numpy(),
                                       f'actions_{method}': prediction['decisions'].numpy()})
        if training.state_hash(model) != before:
            raise PermissionError(f'Eval inference changed frozen state: {name}')
        trace = attempt / 'epochs.jsonl'
        orders[seed][method] = source_history([json.loads(line) for line in trace.read_text().splitlines()],
                                               native['model']['epochs'])
        done = attempt.parent / 'DONE.json'
        files = ['model.pt', 'epochs.jsonl', 'summary.json',
                 *(f'prediction-M{h}.pt' for h in summary['prediction_targets'])]
        models.append(dict(job=name, seed=seed, method=method, done_sha256=digest(done),
                           init_sha256=summary['initial_parameter_sha256'], final_state_sha256=before,
                           files={str((attempt / file).relative_to(root)): digest(attempt / file) for file in files}))
    (output / 'predictions').mkdir()
    for (_, h), key in task_keys.items():
        validate_prediction(predictions[key], roles['tasks'][key]['h_ids'], original[key], h)
        prior_run.save_predictions(output, prediction_file(key), predictions[key])
    write_json(output / 'roles.json', {key: roles[key] for key in ('pools', 'source_fit', 'tasks')})
    write_json(output / 'lineage.json', dict(archive=str(root), parents=parents, models=models,
               original_execution_sha256=digest(root / 'executed-config.json'),
               topology_sha256={name: digest(root / f'data/{name}.pt') for name in roles['pools']},
               training=native, prediction_cells=len(predictions) * len(METHODS)))
    pilot.write_table(output / 'source-sequences.csv', [
        sequence_row(seed, orders[seed], len(roles['pools']['A-M1']), len(roles['pools']['A-M3']))
        for seed in plan['seeds']
    ])
    return prior_run.seal(output, 'OBS', [*METADATA, *(prediction_file(k) for k in task_keys.values())], parents)


def verify_observation(output, plan, authenticated, checksum):
    _, roles, _, original, _, parents = authenticated
    keys = tasks(plan)
    frozen = prior_run.verified_members(output, 'OBS', checksum,
                                         [*METADATA, *(prediction_file(k) for k in keys.values())], parents)
    if json.loads(frozen['roles.json']) != {k: roles[k] for k in ('pools', 'source_fit', 'tasks')}:
        raise PermissionError('Crossover seal contains a different split')
    predictions = {}
    replay = []
    for (seed, h), key in keys.items():
        row = r8_inputs.npz_member(frozen[prediction_file(key)])
        errors = validate_prediction(row, roles['tasks'][key]['h_ids'], original[key], h)
        predictions[seed, h] = row
        replay.extend(dict(seed=seed, holdout=h, method=method, action_differences=0, max_abs_logit_error=error)
                      for method, error in errors.items())
    return predictions, replay


def holdout_labels(repo, plan, roles):
    grants = {}
    archive = Path(roles['r7_archive'])
    with seals.verified_seal_chain(archive, 'R7_FINAL') as scope:
        if json.loads((archive / 'seals/R7_FINAL.json').read_bytes())['digest'] != roles['r7_final_digest']:
            raise PermissionError('R7 FINAL identity differs from the original R10')
        for (seed, h), key in tasks(plan).items():
            task = roles['tasks'][key]
            blob = scope.read_bytes(task['r7_manifest'], 'R7_INPUT')
            if r8_inputs.sha(blob) != task['r7_manifest_sha256']:
                raise PermissionError(f'Original manifest differs: {key}')
            manifest = json.loads(blob)
            context = r5_inputs._context(roles['r7_plan'], task['setting'], task['source'], h, seed, repo)
            for stored, field in (('source_ids', 'source'), ('a_ids', 'adapt'), ('h_ids', 'holdout')):
                if task[stored] != manifest['ids'][field] or task[stored] != context['ids'][field]:
                    raise PermissionError(f'R5/R7/R10 role mismatch: {key}/{field}')
            grant = label_inputs.raw_grant(context, manifest, 'holdout')
            if h in grants and grants[h] != grant:
                raise PermissionError(f'H{h} grant differs by seed')
            grants[h] = grant
        with label_inputs.GraphLabelSession('holdout') as port:
            labels = {h: port.load(grant, grant.ids) for h, grant in grants.items()}
    return labels, {str(h): list(grant.ids) for h, grant in grants.items()}


def compare(left, right, y):
    correct_left, correct_right = left == y, right == y
    change = correct_left.astype(np.int8) - correct_right.astype(np.int8)
    corrected, harmed = int((change == 1).sum()), int((change == -1).sum())
    disagreement = int((left != right).sum())
    return Fraction(100 * (corrected - harmed), len(y)), change, dict(
        n=len(y), corrected=corrected, harmed=harmed, net=corrected - harmed, disagreements=disagreement,
        disagreement_pp=100 * disagreement / len(y))


def estimate(plan, original, predictions, labels):
    reps, seed_key, minimum, quantiles = paired._settings(original)
    endpoints, pairs, summary, audited_draws = [], [], [], []
    estimates = {}
    h1, h3 = plan['targets']
    own_pairs = {h1: ('A1', 'A3'), h3: ('A3', 'A1')}
    for h, (own, other) in own_pairs.items():
        y = labels[h]
        ids = predictions[plan['seeds'][0], h]['ids']
        order = np.argsort(ids, kind='stable')
        cohort = paired._cohort(plan['setting'], h, ids[order], y[order], reps, seed_key, minimum)
        audited_draws.append(cohort['audit'])
        for seed in plan['seeds']:
            x = predictions[seed, h]
            if not np.array_equal(x['ids'], ids) or y.shape != ids.shape or not np.isin(y, [0, 1]).all():
                raise ValueError('Crossover requires shared H graph order and binary labels across seeds')
            actions = {method: x[f'actions_{method}'] for method in METHODS}
            for method in METHODS:
                metrics = evaluate.classification(x[f'logits_{method}'], y, actions['SourceOnly'],
                                                  decisions=actions[method])
                endpoints.append(dict(holdout=h, seed=seed, method=method, **metrics))
            for left, right in ((own, other), ('A1', 'SourceOnly'), ('A3', 'SourceOnly')):
                point, change, counts = compare(actions[left], actions[right], y)
                draws = 100 * (cohort['weights'] @ change[order]) if cohort['eligible'] else None
                estimates.setdefault((h, left, right), {})[seed] = (point, draws)
                pairs.append(dict(holdout=h, seed=seed, left=left, right=right, accuracy_pp=float(point), **counts))

    first, second = (estimates[(h, *own_pairs[h])] for h in plan['targets'])
    interaction = {}
    for seed in plan['seeds']:
        p1, d1 = first[seed]
        p3, d3 = second[seed]
        point = (p1 + p3) / 2
        draws = (d1 + d3) / 2 if d1 is not None and d3 is not None else None
        interaction[seed] = point, draws
        pairs.append(dict(holdout='equal_mean', seed=seed, left='Own', right='Other', accuracy_pp=float(point)))
    estimates['equal_mean', 'Own', 'Other'] = interaction
    for (h, left, right), per_seed in estimates.items():
        points, draws = zip(*per_seed.values())
        point = float(sum(points) / len(points))
        low, high = (None, None) if any(d is None for d in draws) else np.percentile(np.mean(draws, axis=0), quantiles)
        summary.append(dict(holdout=h, left=left, right=right, seeds=len(points), accuracy_pp=point,
                            seed_sample_sd_pp=stdev(map(float, points)), ci95_low_pp=low, ci95_high_pp=high))
    matrix = []
    for h in plan['targets']:
        for method in METHODS:
            rows = [row for row in endpoints if row['holdout'] == h and row['method'] == method]
            row = dict(holdout=h, method=method, seeds=len(rows), n=rows[0]['n'])
            for metric in ('accuracy', 'macro_f1', 'sampled_nll', 'predicted_class_1'):
                values = [r[metric] for r in rows]
                row[metric] = float(np.mean(values))
                row[f'{metric}_seed_sample_sd'] = stdev(values)
            matrix.append(row)
    return endpoints, matrix, pairs, summary, dict(replicates=reps, seed=seed_key,
        percentiles=quantiles, cohorts=audited_draws, same_draws_across_models_and_seeds=True,
        interaction_combines_domains_per_replicate=True, conditional_on_frozen_models=True)


def score(output, repo, plan, authenticated, obs_sha):
    predictions, replay = verify_observation(output, plan, authenticated, obs_sha)
    roles = authenticated[1]
    labels, ids = holdout_labels(repo, plan, roles)
    endpoints, matrix, pairs, summary, audit = estimate(plan, roles['r7_plan'], predictions, labels)
    verify_observation(output, plan, authenticated, obs_sha)
    for name, table in zip(RESULTS[:4], (endpoints, matrix, pairs, summary), strict=True):
        pilot.write_table(output / name, table)
    write_json(output / 'cohort-audit.json', audit)
    write_json(output / 'label-access.json', dict(obs_sha256=obs_sha, pool='holdout', ids=ids))
    pilot.write_table(output / 'replay.csv', replay)
    return prior_run.seal(output, 'FINAL', ['OBS.json', 'replay.csv', *RESULTS], dict(OBS=obs_sha))
