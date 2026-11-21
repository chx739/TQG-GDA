"""Compare a sealed R19 OBS with the separately frozen, unscored predictions."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer


ARMS = dict(WL='Static', Intercept='Bias', SourceBg='Source', TargetBg='Target', Raw='Raw', Mixture='Mixture',
            Count='Count')
TOL = 1e-12


def verify(run, obs_sha, blind, manifest_sha):
    obs = roc_transfer.stage(run, 'OBS', obs_sha)
    raw = {name: prior_run.read_member(run, obs['files'], name) for name in obs['files']}
    if digest(blind / 'manifest.json') != manifest_sha:
        raise PermissionError('Independent prediction manifest differs')
    manifest = json.loads((blind / 'manifest.json').read_bytes())
    if set(manifest) != {'blind.py', 'decisions.csv', 'fits.json', 'summary.json'}:
        raise PermissionError('Independent material list differs')
    for name, checksum in manifest.items():
        if digest(blind / name) != checksum:
            raise PermissionError(f'Independent member differs: {name}')

    blind_rows = {}
    with (blind / 'decisions.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            k = (row['key'], int(row['id']))
            if k in blind_rows:
                raise ValueError(f'Duplicate independent graph decision: {k}')
            blind_rows[k] = row
    roles, fits = (json.loads(raw[f'{name}.json']) for name in ('roles', 'fits'))
    blind_fits = json.loads((blind / 'fits.json').read_bytes())
    if set(roles) != set(blind_fits) or set(fits) != set(roles):
        raise PermissionError('Independent task set differs')

    largest_excess = 0.
    error_by_arm = dict.fromkeys((name for name in ARMS if name != 'Count'), 0.)
    total = 0
    for key, role in roles.items():
        values = inputs.npz_member(raw[f'predictions/{key}.npz'])
        np.testing.assert_array_equal(values['ids'], role['roles']['E'])
        records = [blind_rows.pop((key, gid)) for gid in values['ids']]
        fit, other = fits[key], blind_fits[key]
        if (fit['references']['D'], fit['references']['U']) != (other['source_ref'], other['target_ref']):
            raise PermissionError(f'Independent D/U reference IDs differ: {key}')
        candidate = fit['mixture_prior']
        prior_milli = 1000 * Fraction(candidate) if candidate is not None else None
        if prior_milli != other['pi_milli']:
            raise PermissionError(f'Independent mixture grid selection differs: {key}')
        checks = [(fit[name], other[alias]) for name, alias in (('pS', 'ps'), ('pU', 'pu'), ('pMix', 'pm'))]
        for variant, coef_name, scale_name in (('background', 'w', 'scale'), ('raw', 'raw_w', 'raw_scale')):
            model = fit[variant]
            rms = np.array(model['rms'])
            scale = np.divide(1., rms*np.sqrt(len(rms)), out=np.zeros_like(rms), where=rms != 0)
            checks.extend(((model['coef'], other[coef_name]), (scale, other[scale_name])))
        for a, b in checks:
            nums = np.array([float(Fraction(v)) for v in a])
            np.testing.assert_allclose(nums, b, atol=TOL, rtol=TOL)
            largest_excess = max(largest_excess, float(np.max(np.abs(nums - b))))
        total += len(records)
        for name, alias in ARMS.items():
            np.testing.assert_array_equal(values[name], [int(r[f'a_{alias}']) for r in records])
            if name != 'Count':
                ref = np.array([float(r[f'z_{alias}']) for r in records])
                actual = values[f'margin_{name}']
                np.testing.assert_allclose(actual, ref, atol=TOL, rtol=TOL)
                error_by_arm[name] = max(error_by_arm[name], float(np.max(np.abs(actual-ref))))
    if blind_rows or total != 869:
        raise ValueError('Independent prediction coverage differs')
    return dict(obs_sha256=obs_sha, obs_members=len(raw), independent_manifest_sha256=manifest_sha,
                independent_decisions_sha256=manifest['decisions.csv'], directions=len(roles), graph_trials=total,
                exact_actions=total*len(ARMS), continuous_margins=total*len(error_by_arm), max_abs_margin_error=error_by_arm,
                max_abs_fit_error=largest_excess, exact_reference_ids_and_mixture_grid=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('obs_sha256')
    parser.add_argument('blind', type=Path)
    parser.add_argument('manifest_sha256')
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.obs_sha256, args.blind, args.manifest_sha256)
    write_json(args.receipt, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
