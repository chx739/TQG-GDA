"""Frozen-prediction replay, held-back oracles, and synthetic endpoints."""
import numpy as np
from scipy.special import log_softmax, softmax

from experiments.common.io import digest, write_json

from .fit import config, features, master, moment, predict, read_npz, width_key, write_npz
from .seals import verify_seal


def methods(arrays):
    return [key[2:] for key in arrays if key.startswith('w_')]


def exact_methods(plan, tau, *, oracle=False):
    main = tau == plan['shared_update']['primary_tau']
    names = set(plan['controls']['primary_groups'])
    if not main:
        names -= {'P_mass', 'J_bias', 'J_cal'}
    if not oracle:
        return names - {'O'}
    return {'O', 'O_analytic'}


def replay_heldout(plan, block, *, phase='S'):
    data = read_npz(block / 'holdout_input.npz')
    standard = read_npz(block / 'standardization.npz')
    u = features(data['holdout_h'], standard['mean'], standard['scale'])
    z = data['holdout_z0']
    expected_z = z - z.mean(axis=1, keepdims=True)
    if not np.allclose(u @ standard['v0'].T, expected_z, rtol=1e-10, atol=1e-12):
        raise ValueError('Source head does not reconstruct holdout logits')
    write_npz(block / 'holdout_standardization.npz', holdout_u=u, holdout_z0=z, holdout_ids=data['holdout_ids'])
    for tau in plan['phases'][phase]['width_grid']:
        key = width_key(tau)
        observed = read_npz(block / f'{key}-observed.npz')
        expected = exact_methods(plan, tau)
        if phase == 'P' and tau == plan['phases']['P']['independent_audit']['tau']:
            expected |= {'J_plus', 'P_plus', 'P_mass_plus'}
        if set(methods(observed)) != expected:
            raise ValueError(f'{block.name}/{key}: incomplete unsupervised arms')
        if methods(observed) == []:
            raise ValueError('No frozen methods')
        output = replay_predictions(standard, dict(holdout_u=u, holdout_z0=z), observed)
        write_npz(block / f'{key}-heldout.npz', **output)


def replay_predictions(standard, held, arrays):
    output = {}
    for name in methods(arrays):
        old = arrays[f'adapt_logits_{name}']
        regenerated = predict(name, standard['adapt_z0'], standard['adapt_u'], arrays[f'w_{name}'])
        new = predict(name, held['holdout_z0'], held['holdout_u'], arrays[f'w_{name}'])
        if (not np.array_equal(old, regenerated) or not np.array_equal(old.argmax(1), regenerated.argmax(1))
                or not np.array_equal(arrays[f'adapt_probability_{name}'], softmax(regenerated, axis=1))):
            raise ValueError(f'{name}: frozen adaptation logits or probabilities fail to replay')
        if not np.isfinite(new).all():
            raise ValueError(f'{name}: nonfinite holdout logits')
        output[f'holdout_logits_{name}'] = new
        output[f'holdout_probability_{name}'] = softmax(new, axis=1)
    return output


def _committed_truth(block, pool):
    import json

    paths = json.loads((block / 'commitments.json').read_text())
    name = f'secret_{pool}.npz'
    if digest(block / name) != paths[name]:
        raise ValueError(f'Truth commitment has changed: {block.name}/{name}')
    return read_npz(block / name)


def vault(root, block, pool):
    stage = {'adapt': 'R7_OBS', 'holdout': 'R7_ORACLE'}[pool]
    verify_seal(root, stage)
    return _committed_truth(block, pool)


def fit_oracles(plan, root, specs):
    verify_seal(root, 'R7_OBS')
    for block, _, seed in specs:
        truth = _committed_truth(block, 'adapt')
        standard = read_npz(block / 'standardization.npz')
        held = read_npz(block / 'holdout_standardization.npz')
        z, u = standard['adapt_z0'], standard['adapt_u']
        classes = z.shape[1]
        for tau in plan['phases']['S']['width_grid']:
            key = width_key(tau)
            cfg = config(plan, seed, tau)
            arrays, records = {}, {}
            for name, q in (('O', np.eye(classes)[truth['adapt_y']]), ('O_analytic', truth['adapt_q'])):
                arrays[f'stat_{name}'] = moment(q, u)
                w, records[name] = master(z, u, [arrays[f'stat_{name}']], cfg)
                arrays[f'w_{name}'] = w
                arrays[f'adapt_logits_{name}'] = predict(name, z, u, w)
                arrays[f'adapt_probability_{name}'] = softmax(arrays[f'adapt_logits_{name}'], axis=1)
            arrays.update(replay_predictions(standard, held, arrays))
            write_npz(block / f'{key}-oracle.npz', **arrays)
            write_json(block / f'{key}-oracle.json', records)


def loss(logits, q):
    return -np.sum(q * log_softmax(logits, axis=1), axis=1)


def classification(logits, y, b0):
    pred = logits.argmax(axis=1)
    confusion = np.bincount(2 * y + pred, minlength=4).reshape(2, 2)
    row_count, col_count = confusion.sum(axis=1), confusion.sum(axis=0)
    recall = [float(correct / count) if count else None for correct, count in zip(confusion.diagonal(), row_count)]
    corrected = int(np.sum((b0 != y) & (pred == y)))
    harmed = int(np.sum((b0 == y) & (pred != y)))
    f1 = 2 * confusion.diagonal() / np.maximum(row_count + col_count, 1)
    return dict(sampled_nll=float(loss(logits, np.eye(2)[y]).mean()),
                accuracy=float(np.mean(pred == y)), balanced_accuracy=np.mean(recall).item() if all(row_count) else None,
                macro_f1=float(f1.mean()), recall_0=recall[0], recall_1=recall[1],
                corrected=corrected, harmed=harmed, net_corrections=corrected - harmed,
                n=len(y), class_1_count=int(row_count[1]), predicted_class_1=float(pred.mean()))


def endpoint(result, a, h, b0a, b0h, truth, worst, **key):
    change = h - b0h
    change -= change.mean(axis=1, keepdims=True)
    arisk = float(np.mean(loss(a, truth['adapt_q']) - loss(b0a, truth['adapt_q'])))
    hrisk = float(np.mean(loss(h, truth['holdout_q']) - loss(b0h, truth['holdout_q'])))
    applied = bool(result['valid'])
    return dict(**key, valid=applied, status=result['status'], stop_reason=result['stop_reason'],
                holdout_expected_nll=float(loss(h, truth['holdout_q']).mean()) if applied else None,
                holdout_expected_nll_delta=hrisk if applied else None,
                centered_rms=float(np.sqrt(np.mean(change ** 2))) if applied else None,
                adapt_true_risk_delta=arisk if applied else None,
                found_worst_risk=worst,
                true_directional_risk_gap=(arisk - worst if applied and worst is not None else None),
                **(classification(h, truth['holdout_y'], b0h.argmax(axis=1)) if applied else {}))
