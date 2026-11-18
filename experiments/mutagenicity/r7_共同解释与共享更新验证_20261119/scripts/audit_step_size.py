"""Replay fixed logit paths on an already sealed R7 pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax

from experiments.common.io import write_json

from ..src.evaluate import classification, loss
from ..src.fit import read_npz, width_key
from ..src.r5_inputs import load_labels
from ..src.seals import verify_seal

STEPS = (0., .0625, .125, .25, .5, .75, 1.)
METHODS = ('J', 'E', 'J_cal', 'O')
SOURCE_CE_CAP = .005


def _replay_block(root, block, width):
    y = load_labels(root, str((block / 'manifest.json').relative_to(root)), 'holdout')
    data = read_npz(block / 'fit_input.npz')
    standard = read_npz(block / 'standardization.npz')
    observed = read_npz(block / f'{width}-observed.npz')
    heldout = read_npz(block / f'{width}-heldout.npz')
    oracle = read_npz(block / f'{width}-oracle.npz')
    diagnostics = json.loads((block / f'{width}-observed.json').read_text())
    diagnostics['O'] = json.loads((block / f'{width}-oracle.json').read_text())
    z0 = heldout['holdout_logits_B0']
    z_source = data['source_z0']
    q, q_source = np.eye(2)[y], np.eye(2)[data['source_y']]
    base = classification(z0, y, z0.argmax(1))
    source_baseline = float(loss(z_source, q_source).mean())
    rows = []
    for name in METHODS:
        key = dict(block=block.name, method=name, valid=diagnostics[name]['valid'])
        if not key['valid']:
            rows.append(key | {'reason': diagnostics[name]['stop_reason']})
            continue
        predictions, corrections = (oracle, oracle) if name == 'O' else (heldout, observed)
        shift = predictions[f'holdout_logits_{name}'] - z0
        source_shift = standard['source_u'] @ corrections[f'w_{name}'].T
        slope = float(np.mean(np.sum((softmax(z0, axis=1) - q) * shift, axis=1)))

        def source_change(alpha):
            return float(loss(z_source + alpha * source_shift, q_source).mean()) - source_baseline

        guarded = 1.
        if source_change(1.) > SOURCE_CE_CAP:
            guarded = brentq(lambda alpha: source_change(alpha) - SOURCE_CE_CAP, 0., 1., xtol=1e-14)
        for step, alpha in [*((x, x) for x in STEPS), ('source_ce_cap', guarded)]:
            sample = classification(z0 + alpha * shift, y, z0.argmax(1))
            rows.append(key | dict(step=step, alpha=alpha, nll_slope=slope,
                                   source_fit_nll_delta=source_change(alpha),
                                   nll_delta=sample['sampled_nll'] - base['sampled_nll'],
                                   ba_pp=100 * (sample['balanced_accuracy'] - base['balanced_accuracy']),
                                   net=sample['net_corrections']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.pilot.resolve()
    final = verify_seal(root, 'R7_FINAL')
    plan = json.loads((root / 'executed-plan.json').read_text())
    width = width_key(plan['shared_update']['primary_tau'])
    rows = [row for block in sorted((root / 'blocks').iterdir()) for row in _replay_block(root, block, width)]
    write_json(args.output, dict(scope='posthoc diagnostic on sealed development graphs; no model fit',
                                 final=final['digest'], budget=SOURCE_CE_CAP, results=rows))


if __name__ == '__main__':
    main()
