"""Fit only on allowed features; prediction files are assembled after fitting."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch

from experiments.common.io import write_json
from tsba.shared_update.config import SharedUpdateConfig
from tsba.shared_update.explanations import ExplanationSet
from tsba.shared_update.predict import corrected_logits, update_features
from tsba.shared_update.solver import solve_master, solve_shared_update

from .controls import solve_pointwise, solve_restricted


def config(plan, seed, tau):
    section = plan['shared_update']
    values = {field.name: section[field.name] for field in fields(SharedUpdateConfig)
              if field.name not in ('seed', 'tau')}
    return SharedUpdateConfig(seed=seed, tau=tau, **values)


def features(h, mean, scale):
    return update_features(*(torch.as_tensor(x, dtype=torch.float64) for x in (h, mean, scale))).numpy()


def logits(z, u, w, multiplier=1.0):
    return corrected_logits(*(torch.as_tensor(x, dtype=torch.float64) for x in (multiplier * z, u, w))).numpy()


def moment(q, u):
    return np.asarray(q).T @ u / len(u)


def master(z, u, stats, cfg):
    w, detail = solve_master(z, u, stats, regularization=cfg.regularization,
                             feasibility_tol=cfg.feasibility_tol, optimality_tol=cfg.optimality_tol,
                             master_maxiter=cfg.master_maxiter)
    detail.update(status='completed' if detail['valid'] else 'unverified',
                  stop_reason='master_solved' if detail['valid'] else 'invalid_master',
                  found_worst_risk=detail.get('found_worst_risk'))
    if not detail['valid']:
        detail['failed_trial_w'] = w.tolist()
        w = np.zeros_like(w)
    return w, detail


def write_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def run_width(plan, cfg, base, z, u, v0, ids):
    family = deepcopy(base)
    family.config.tau = cfg.tau
    family.threshold = family.reference_loss + cfg.tau
    family.fit_report['threshold'] = family.threshold
    zeros = np.zeros_like(v0)
    reference = family.fit()
    wj, j = solve_shared_update(z, u, family, cfg)
    j['valid'] = j['status'] == 'completed' and j['stop_reason'] not in (
        'local_search_incomplete', 'search_returned_no_candidate')
    if not j['candidates']:
        raise RuntimeError(f'No retained explanations: {j["stop_reason"]}')
    parameters = np.array([c['parameters'] for c in j['candidates']])
    stats = np.array([c['statistic'] for c in j['candidates']])
    qbank = np.stack([family._target_terms(theta)[-1] for theta in parameters])
    for q, s in zip(qbank, stats):
        if not np.allclose(moment(q, u), s, rtol=0, atol=1e-12):
            raise ValueError('Replayed posterior moment differs from retained statistic')
    we, e = master(z, u, [reference.statistic], cfg)
    options = plan['controls']
    wp, p = solve_pointwise(z, u, qbank, cfg, options['pointwise']['exact_separation_limit'], ids=ids)
    corrections = dict(B0=zeros, E=we, J=wj, P=wp, C05=zeros, C075=zeros)
    records = {name: dict(valid=True, status='fixed', stop_reason='fixed') for name in ('B0', 'C05', 'C075')}
    records.update(E=e, J=j, P=p)
    if cfg.tau == plan['shared_update']['primary_tau']:
        corrections['P_mass'], records['P_mass'] = solve_pointwise(
            z, u, qbank, cfg, options['pointwise_mass']['exact_separation_limit'], mass=True, ids=ids)
        for name in ('J_bias', 'J_cal'):
            corrections[name], records[name] = solve_restricted(z, u, stats, v0, cfg, mode=name)
    arrays = dict(theta=parameters, stats=stats, ref_theta=reference.parameters,
                  **{f'w_{name}': w for name, w in corrections.items()})
    for name, w in corrections.items():
        arrays[f'adapt_logits_{name}'] = predict(name, z, u, w)
        arrays[f'adapt_probability_{name}'] = softmax(arrays[f'adapt_logits_{name}'], axis=1)
    return arrays, records


def predict(name, z, u, w):
    multiplier = {'C05': .5, 'C075': .75}.get(name, 1.)
    return logits(z, u, w, multiplier)


def width_key(tau):
    return f'tau-{tau:.3f}'


def explanation_family(plan, source_x, source_y, target_x, cfg, v0=None):
    variant = plan.get('separation_variant', 'stock')
    if variant == 'stock':
        return ExplanationSet(source_x, source_y, target_x, cfg)
    if variant == 'restore_trust_v1':
        from .restored_separation import RestoredExplanationSet
        return RestoredExplanationSet(source_x, source_y, target_x, cfg)
    if variant in ('projected_restore_v1', 'anchored_projected_v1', 'source_mixture_v1'):
        if variant == 'source_mixture_v1':
            from .mixture_explanations import MixtureExplanationSet as family_class
        elif variant == 'anchored_projected_v1':
            from .anchored_explanations import AnchoredExplanationSet as family_class
        else:
            from .projected_explanations import ProjectedExplanationSet as family_class
        if v0 is None or v0.shape != (2, source_x.shape[1] + 1):
            raise ValueError('Projected estimation requires the frozen binary source head in fit coordinates')
        options = plan.get('source_mixture_options', {}) if variant == 'source_mixture_v1' else {}
        return family_class(source_x, source_y, target_x, cfg,
                            source_head_normal=v0[1, :-1] - v0[0, :-1], **options)
    raise ValueError(f'Unregistered separation algorithm: {variant}')


def fitter_source_files(plan):
    """Modules that must remain identical across gated adaptation phases."""
    variant = plan.get('separation_variant', 'stock')
    files = ['fit.py', 'controls.py']
    if variant != 'stock':
        files.append('restored_separation.py')
    if variant in ('projected_restore_v1', 'anchored_projected_v1', 'source_mixture_v1'):
        files.append('projected_explanations.py')
    if variant in ('anchored_projected_v1', 'source_mixture_v1'):
        files.append('anchored_explanations.py')
    if variant == 'source_mixture_v1':
        files.append('mixture_explanations.py')
    return files


def fit_block(plan, seed, block, *, phase='S'):
    """This worker receives only its own A features and source-fit rows."""
    torch.set_num_threads(1)
    block = Path(block)
    data = read_npz(block / 'fit_input.npz')
    requested = plan['phases'][phase]['width_grid']
    primary = config(plan, seed, plan['shared_update']['primary_tau'])
    train = data['source_h']
    mean = train.mean(axis=0)
    scale = np.maximum(np.sqrt(np.mean((train - mean) ** 2, axis=0)), primary.coordinate_floor)
    source_u = features(train, mean, scale)
    u = features(data['adapt_h'], mean, scale)
    v0 = np.column_stack((data['source_head_weight'] * scale,
                          data['source_head_weight'] @ mean + data['source_head_bias']))
    v0 -= v0.mean(axis=0)
    z = data['adapt_z0']
    if not np.allclose(u @ v0.T, z - z.mean(axis=1, keepdims=True), rtol=1e-10, atol=1e-12):
        raise ValueError('Source head does not reconstruct adaptation logits')
    base = explanation_family(plan, source_u[:, :-1], data['source_y'], u[:, :-1], primary, v0)
    base.fit()
    if hasattr(base, 'projection_report'):
        base.fit_report['projection'] = base.projection_report
    write_npz(block / 'standardization.npz', mean=mean, scale=scale, source_u=source_u, adapt_u=u, adapt_z0=z, v0=v0)
    for tau in requested:
        cfg = config(plan, seed, tau)
        arrays, records = run_width(plan, cfg, base, z, u, v0, data['adapt_ids'])
        if phase == 'P' and tau == plan['phases']['P']['independent_audit']['tau']:
            from .real_audit import audit

            supplemental, audit_records = audit(plan, cfg, base, z, u, data['adapt_ids'], arrays, records)
            arrays.update(supplemental)
            records.update(audit_records)
        key = width_key(tau)
        write_npz(block / f'{key}-observed.npz', **arrays)
        write_json(block / f'{key}-observed.json', records)
    return block.name
