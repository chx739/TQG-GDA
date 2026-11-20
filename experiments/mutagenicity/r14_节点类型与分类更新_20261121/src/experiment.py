"""Fit exclusively on S/V/A, then read H topology for frozen forward evaluation."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import torch

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run as storage
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, normalization, observation
from . import activations, recovery

IDENTITY = observation.IDENTITY
ACTIVE_ARMS = ('Binding', 'True', 'Hist', 'V_Binding', 'Other_Binding_1', 'Other_Binding_2')
METHODS = ('B0', 'SourceCal', 'Base', *ACTIVE_ARMS, 'Base_equal_positive')
HISTORICAL = ('S_node', 'S_graph', 'SourceFusion', 'GRADE')


def select(setting, ids, n):
    return sorted(ids, key=lambda x: (sha256(f'r14-pool-v1|{setting}|{x}'.encode('utf-8')).digest(), x))[:n]


def match_roles(setting, tasks):
    source, v = (tasks[0]['ids'][name] for name in ('source', 'validation'))
    if any(t['ids']['source'] != source or t['ids']['validation'] != v for t in tasks):
        raise PermissionError('Directions do not use the same original S and V')
    target = {f"A{t['target']}": t['ids']['adapt'] for t in tasks}
    scope = [source, v, *target.values(), *(t['ids']['holdout'] for t in tasks)]
    flat = [i for ids in scope for i in ids]
    if len(set(flat)) != len(flat) or any(not ids for ids in scope):
        raise PermissionError('Source or target graph roles overlap or are empty')
    k = min(len(v), *(len(ids) for ids in target.values()))
    return source, v, {name: select(setting, ids, k) for name, ids in {'V': v, **target}.items()}


def group_scope(c, repo, setting, original_tasks):
    first_id = original_tasks[0]
    first, identity, data = historical.context(c, first_id, repo)
    model, lineage = historical.checkpoint(first, data)
    tasks = []
    for task in original_tasks:
        if task == first_id:
            mf, checksum = first, identity
        else:
            mf, checksum = observation.bind(task, c, repo, data, lineage, None)
        tasks.append(dict(**task, ids=mf['ids'], manifest_sha256=checksum))
    s, v, pools = match_roles(setting, tasks)
    return model, lineage, data, tasks, s, v, pools


def fit_members(plan, seeds):
    groups, _ = observation.observations_expected(plan, seeds)
    return ['executed-config.json', 'fit-roles.json', 'historical-plan.json',
            *[f'fit/{name}.json' for name in groups], *[f'fit/{name}.npz' for name in groups]]


def obs_members(plan, seeds):
    _, tasks = observation.observations_expected(plan, seeds)
    return ['FIT.json', 'prediction-roles.json', *[f'predictions/{name}.npz' for name in tasks]]


def pack(arrays, prefix, masked, gradients=None):
    arrays.update({f'{prefix}_{name}': getattr(masked, name) for name in ('ids', 'owner', 'nodes', 'types')})
    if gradients is not None:
        arrays.update({f'{prefix}_grad_{arm}': value for arm, value in gradients.items()})


def fit(output, repo, plan, cfg, seeds, *, inspect=False):
    c = historical.CArchive(plan)
    stores, roles, preflight = {}, {}, []
    if not inspect:
        (output/'fit').mkdir()
    for setting, source, seed, entries in observation.group_tasks(plan, seeds):
        original, lineage, data, tasks, s, v, pools = group_scope(c, repo, setting, entries)
        if setting not in stores:
            stores[setting] = historical.GraphStore(data)
        store = stores[setting]
        group = f'{setting}-M{source}-seed-{seed}'
        if inspect:
            store.take(s)
            for ids in pools.values():
                store.take(ids)
            preflight.append(dict(group=group, source_graphs=len(s), v_graphs=len(v), equal_budget=len(pools['V']),
                                  graph_container_sha256=store.graph_sha, checkpoint=lineage['sha256']))
            continue
        batch = lineage['batch_size']
        base, moments, buffers = normalization.recalibrate(original, store.take(s), 'node', batch)
        dataset = data['spec']['tu_dataset']

        def collect(ids):
            return recovery.masked_activations(base, store.take(ids), ids, dataset, cfg['max_masks_per_graph'], batch)

        s_masks = collect(s)
        head = recovery.fit_type_head(s_masks, cfg['type_optimizer'])
        arrays = {}
        pack(arrays, 'S', s_masks)
        fitted = {}
        v_diagnostics = None
        for name, ids in pools.items():
            masked = collect(ids)
            graphs, metrics = recovery.gradients(masked, head)
            fitted[name] = dict(summary=metrics, directions=recovery.directions(
                graphs, metrics, len(head['seen']), cfg['zero_gradient_floor']))
            pack(arrays, name, masked, graphs)
            if name == 'V':
                full_v = masked if len(ids) == len(v) else collect(v)
                v_diagnostics = recovery.recovery_diagnostic(full_v, head)
        base_theta = activations.theta(base)
        step = cfg['move_fraction'] * float(np.linalg.norm(base_theta))
        source_stats = dict(graphs=len(s), mean_and_variance=moments, changed_buffers=buffers)
        write_json(output/f'fit/{group}.json', dict(setting=setting, source=source, seed=seed, original=lineage,
                   base=source_stats, graph_sha256=store.graph_sha, type_head=head, v_diagnostic=v_diagnostics,
                   base_theta=base_theta.tolist(), nominal_step=step, pools=fitted))
        storage.save_predictions(output, f'fit/{group}.npz', arrays)
        roles[group] = dict(setting=setting, source=source, seed=seed, source_fit=s, validation=v, pools=pools, tasks=tasks)
        print(f'R14 fit-ready {group}: S={len(s)} K={len(pools["V"])} '
              f'V type recovery={v_diagnostics["recovery_accuracy"]:.4f} '
              f'majority={v_diagnostics["majority_accuracy"]:.4f}', flush=True)
    if inspect:
        return preflight
    write_json(output/'fit-roles.json', roles)
    write_json(output/'historical-plan.json', c.plan)
    return storage.seal(output, 'FIT', fit_members(plan, seeds), {'r7_final_digest': plan['r7_final_digest']})


def restore_base(original, fitted):
    # Copy source moments into the original native buffer dtype.
    with torch.inference_mode():
        fitted_model = deepcopy(original).eval()
        layers = dict(fitted_model.encoder.named_modules())
        moments = fitted['base']['mean_and_variance']
        for name, rec in moments.items():
            layer = layers[name]
            if layer.eps != rec['epsilon'] or str(layer.running_mean.dtype) != rec['buffer_dtype']:
                raise ValueError('Original BN interface changed since fit')
            layer.running_mean.copy_(torch.tensor(rec['mean'], dtype=layer.running_mean.dtype))
            layer.running_var.copy_(torch.tensor(rec['variance'], dtype=layer.running_var.dtype))
        if normalization.verify_frozen(original, fitted_model, moments) != fitted['base']['changed_buffers']:
            raise ValueError('Restored Base changed unexpected buffers')
        np.testing.assert_array_equal(activations.theta(fitted_model), fitted['base_theta'])
    return fitted_model


def sources(target, other):
    return {'Binding': (f'A{target}', 'Binding'), 'True': (f'A{target}', 'True'),
            'Hist': (f'A{target}', 'Hist'), 'V_Binding': ('V', 'Binding'),
            'Other_Binding_1': (f'A{other[0]}', 'Binding'), 'Other_Binding_2': (f'A{other[1]}', 'Binding')}


class References:
    def __init__(self, plan, cfg, repo):
        self.standard = historical.ReferenceArms(plan, repo)
        self.r13 = repo / cfg['r13_attempt']
        self.obs = self.standard._observe(self.r13, 'OBS.json', 'OBS', cfg['r13_obs_sha256'])

    def for_block(self, task, ids, b0):
        references, receipt = self.standard.get(task, ids, b0)
        references.pop('SourceOnly', None)
        receipt.pop('SourceOnly', None)
        key = r11.key(**task)
        path = f'predictions/{key}.npz'
        if path in self.obs:
            names = ('B0', 'SourceCal', 'S_node', 'S_graph')
            arrays = r11.npz_member(historical.member(self.r13, self.obs, path),
                                   ['ids', *(f'{stem}_{arm}' for arm in names for stem in ('logits', 'decision'))])
            np.testing.assert_array_equal(arrays['ids'], ids)
            np.testing.assert_array_equal(arrays['logits_B0'], b0)
            np.testing.assert_array_equal(arrays['logits_SourceCal'], references['SourceCal'])
            for arm in ('S_node','S_graph'):
                arm_z = arrays[f'logits_{arm}']
                np.testing.assert_array_equal(arm_z.argmax(1), arrays[f'decision_{arm}'])
                references[arm], receipt[arm] = arm_z, self.obs[path]
        return references, receipt


def frozen_fit(output, plan, seeds, fit_sha):
    return storage.verified_members(output, 'FIT', fit_sha, fit_members(plan, seeds),
                                    {'r7_final_digest': plan['r7_final_digest']})


def predict_limited(h, theta, direction, step, fd_step, tolerance):
    theta_new = theta + step * direction
    logits = np.concatenate([batch.finite(theta_new) for batch in h])
    local = [batch.local(direction, fd_step) for batch in h]
    speed = np.concatenate([z[0] for z in local])
    fd = np.concatenate([z[1] for z in local])
    np.testing.assert_allclose(speed, fd, **tolerance)
    return logits, speed, float(np.max(np.abs(speed-fd))), min(z[2] for z in local)


def predict(output, repo, plan, cfg, seeds, fit_sha):
    files = frozen_fit(output, plan, seeds, fit_sha)
    sealed = json.loads(files['fit-roles.json'])
    c = historical.CArchive(plan)
    ref = References(plan, cfg, repo)
    stores, observed = {}, {}
    (output/'predictions').mkdir()
    for setting, source, seed, entries in observation.group_tasks(plan, seeds):
        group = f'{setting}-M{source}-seed-{seed}'
        fitted = json.loads(files[f'fit/{group}.json'])
        original, lineage, data, tasks, s, v, pools = group_scope(c, repo, setting, entries)
        expected = dict(setting=setting, source=source, seed=seed, source_fit=s, validation=v, pools=pools, tasks=tasks)
        if sealed[group] != expected or fitted['original'] != lineage:
            raise PermissionError('Frozen roles/checkpoint differ before H inference')
        base_model = restore_base(original, fitted)
        if setting not in stores:
            stores[setting] = historical.GraphStore(data)
        store = stores[setting]
        if store.graph_sha != fitted['graph_sha256']:
            raise PermissionError('Frozen graph inventory differs before H inference')
        base_theta, step = np.asarray(fitted['base_theta']), fitted['nominal_step']
        for row in tasks:
            task = {k: row[k] for k in IDENTITY}
            key = r11.key(**task)
            h_ids = row['ids']['holdout']
            h = activations.held_batches(base_model, store.take(h_ids), lineage['batch_size'])
            b0 = c.b0(task, h_ids)
            logits, receipt = ref.for_block(task, h_ids, b0)
            logits['B0'] = b0
            z = logits['Base'] = np.concatenate([batch.base_logits for batch in h])
            z_margin = activations.margin(z)
            alternatives = sorted(t['target'] for t in tasks if t['target'] != task['target'])
            mapping = sources(task['target'], alternatives)
            arrays = dict(ids=np.asarray(h_ids), base_margin=z_margin)
            check = {}
            for arm, (pool, target) in mapping.items():
                fitted_direction = fitted['pools'][pool]['directions'][target]
                direction = np.asarray(fitted_direction['unit'])
                if fitted_direction['fallback']:
                    np.testing.assert_array_equal(direction, np.zeros_like(direction))
                else:
                    np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-13, rtol=0.)
                logits[arm], speed, err, delta = predict_limited(
                    h, base_theta, direction, step, cfg['maximum_derivative_check_step'], cfg['derivative_check'])
                residual, coef, rank = activations.residualize(z_margin, speed)
                arrays[f'velocity_{arm}'] = speed
                arrays[f'residual_velocity_{arm}'] = residual
                native_move = (base_theta + step*direction).astype(np.float32).astype(np.float64) - base_theta
                check[arm] = dict(max_abs_finite_difference_error=err, smallest_checked_step=delta,
                                  velocity_affine_coefficients=coef.tolist(), affine_rank=rank,
                                  nominal_norm=float(step*np.linalg.norm(direction)),
                                  native_displacement=float(np.linalg.norm(native_move)))
            for arm, scores in logits.items():
                arrays[f'logits_{arm}'] = scores
                arrays[f'decision_{arm}'] = scores.argmax(axis=1).astype(np.int64)
            positives = int(arrays['decision_Binding'].sum())
            arrays['decision_Base_equal_positive'] = observation.equal_positive(arrays['ids'], z, positives)
            arrays['logits_Base_equal_positive'] = z
            storage.save_predictions(output, f'predictions/{key}.npz', arrays)
            observed[key] = dict(**task, group=group, held_ids=h_ids, manifest_sha256=row['manifest_sha256'],
                                 historical_members=receipt, directions={k: list(v) for k,v in mapping.items()},
                                 numerical_checks=check)
        print(f'R14 H predictions frozen-ready {group}', flush=True)
    write_json(output/'prediction-roles.json', observed)
    return storage.seal(output, 'OBS', obs_members(plan, seeds), {'fit_sha256':fit_sha})
