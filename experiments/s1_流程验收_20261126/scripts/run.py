"""End-to-end smoke; real TU smoke stops before opening evaluation labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from cela.artifacts import digest, read_json, write_json
from .synthetic import make_fixture


def run_smoke(task, output, raw_dir=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiment = Path(__file__).resolve().parents[1]
    config = (make_fixture(output / 'fixture') if task == 'synthetic' else
              experiment / 'configs' / f'{task}.json')
    source = output / 'source.pt'
    adaptation = output / 'adaptation.json'
    predictions = output / 'predictions.json'
    target_raw = ['--raw-dir', str(raw_dir)] if raw_dir else []
    base = [sys.executable, '-m', 'scripts.run_task']
    def execute(phase, args):
        subprocess.run([*base, phase, *map(str, args)], check=True)

    execute('train', ['--config', config, '--output', source, *target_raw])
    frozen_source = digest(source)
    execute('adapt', ['--config', config, '--source', source, '--output', adaptation, *target_raw])
    predict_args = ['--config', config, '--source', source, '--adaptation', adaptation, *target_raw]
    execute('predict', [*predict_args, '--output', predictions])
    replay = output / 'predictions-reloaded.json'
    execute('predict', [*predict_args, '--output', replay])
    if digest(replay) != digest(predictions) or digest(source) != frozen_source:
        raise ValueError('reload parity or source immutability check failed')
    if task == 'synthetic':
        execute('score', ['--predictions', predictions, '--labels', config.parent/'H-labels.json',
                          '--output', output/'scores.json'])
    state, holdout = read_json(adaptation), read_json(predictions)
    import torch
    source_record = torch.load(source, weights_only=True, map_location='cpu')
    modes = set(state['modes'])
    expected = {'zero','structural_only','prior_only','prior_calibrated','v2_global'}
    if modes != expected:
        raise ValueError(f'incomplete smoke modes: {modes}')
    deviations = []
    for row in holdout['records']:
        if set(row['modes']) != modes:
            raise ValueError('incomplete predictions')
        for m in row['modes'].values():
            for field in ('probability', 'likelihood_weighted_probability'):
                prob = torch.tensor(m[field], dtype=torch.float64)
                if not (torch.isfinite(prob).all() and (prob >= 0).all() and abs(float(prob.sum()) - 1) < 1e-9):
                    raise ValueError(f'invalid {field}')
        zero = row['modes']['zero']['log_probability']
        struct = row['modes']['structural_only']['log_probability']
        deviations.append(abs((struct[1]-struct[0]) - (zero[1]-zero[0])))
    summary = dict(task=task, adaptation_graphs=len(state['ids']), holdout_graphs=len(holdout['records']),
                   source_graphs_including_validation=len(source_record['source_ids']),
                   source_train_graphs=len(source_record['preprocessing']['fit_ids']),
                   epochs=len(source_record['validation_ce']), best_epoch=source_record['best_epoch'],
                   validation_ce=source_record['validation_ce'],
                   source_checkpoint_unchanged=True, reload_predictions_identical=True, modes=sorted(modes),
                   component_ids=state['modes']['structural_only']['components'],
                   structural_eta=state['modes']['structural_only']['eta'],
                   calibrated_nu=state['modes']['prior_calibrated']['nu'],
                   prior_only_nu=state['modes']['prior_only']['nu'],
                   max_abs_holdout_structural_log_odds_change=max(deviations), scored=(task == 'synthetic'),
                   diagnostics=state['diagnostics'], prediction_sha256=digest(predictions),
                   prediction_modes_normalized=True)
    if task == 'synthetic':
        summary['scores'] = read_json(output / 'scores.json')
    write_json(output / 'smoke-summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['synthetic','Mutagenicity','PROTEINS'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path)
    a = p.parse_args()
    s = run_smoke(a.task, a.output, a.raw_dir)
    print(json.dumps({k:s[k] for k in ('task','holdout_graphs','scored','reload_predictions_identical')}))


if __name__ == '__main__':
    main()
