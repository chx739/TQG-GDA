"""Synthetic CLI integration test, including all shipped ablations and evaluation.

All generated graphs, checkpoints and logs must be outside the package.
Synthetic accuracies are engineering checks, not research results.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import torch
from helpers import fixture_files, small_config
from tqg_gda.utils import load_config, merge, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT):
        p.error('Smoke output must be outside the package')
    output.mkdir(parents=True, exist_ok=False)
    cache, split, target = fixture_files(output / 'clean_inputs', target_label=None)
    poison_cache, poison_split, poison_target = fixture_files(output / 'poison_inputs', target_label=999)
    config = small_config()
    config_path = output / 'config.json'; write_json(config_path, config)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')

    def run(name, stage, cfg=config_path, inputs=(cache, split, target), extra=()):
        cp, sp, tp = inputs
        command = [sys.executable, str(ROOT / 'scripts/run_experiment.py'), stage,
            '--data-path', str(cp), '--source-split', str(sp), '--target-index', str(tp),
            '--dataset', 'fixture', '--idx-dir', str(cp.parent), '--config', str(cfg),
            '--target-policy', 'transductive-unlabeled', '--output', str(output / name), *map(str, extra)]
        with (output / (name + '.log')).open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    run('fresh', 'all')
    run('resumed', 'adapt', extra=('--source-checkpoint', output/'fresh/source/best.pt',
                                 '--resume-adapt', output/'fresh/adaptation/step_1.pt'))
    run('poisoned', 'all', inputs=(poison_cache, poison_split, poison_target))
    a = torch.load(output/'fresh/adaptation/last.pt', weights_only=False)
    b = torch.load(output/'resumed/adaptation/last.pt', weights_only=False)
    c = torch.load(output/'poisoned/adaptation/last.pt', weights_only=False)
    for name, value in a['model'].items():
        assert torch.equal(value, b['model'][name]), ('adapt resume', name)
        assert torch.equal(value, c['model'][name]), ('target label isolation', name)
    assert a['selection']['step'] == b['selection']['step']
    short = merge(config, {'warmup': {'epochs': 1}})
    short_path = output/'short.json'; write_json(short_path, short)
    run('source_first', 'source', cfg=short_path)
    run('source_resumed', 'source', extra=('--resume-source', output/'source_first/source/last.pt'))
    first = torch.load(output/'fresh/source/last.pt', weights_only=False)
    second = torch.load(output/'source_resumed/source/last.pt', weights_only=False)
    assert all(torch.equal(v, second['model'][n]) for n, v in first['model'].items())
    variants = ['source_only_same_fit', 'source_only_all_labels', 'random_probes', 'source_probes', 'odd_only', 'linear_jvp']
    for name in variants:
        variant = load_config(ROOT/'configs'/f'{name}.json')
        # Small fixture budgets while preserving the shipped method/ablation switches.
        variant = merge(config, {key: variant[key] for key in ('method', 'probes')})
        if name == 'source_only_all_labels':
            variant = merge(variant, {'warmup': {'source_pool': 'all_source'},
                                     'adaptation': {'source_pool': 'all_source'}})
        path = output/(name+'.json'); write_json(path, variant)
        run(name, 'all', cfg=path)
    # Exercise the FP32 path with the real graph model as well as the FP64 default.
    fp32_path = output/'float32.json'; write_json(fp32_path, merge(config, {'dtype': 'float32'}))
    run('float32', 'all', cfg=fp32_path)
    with (output/'evaluate.log').open('x') as log:
        subprocess.run([sys.executable, str(ROOT/'scripts/evaluate.py'),
            '--checkpoint', str(output/'fresh/adaptation/selected.pt'), '--data-path', str(cache),
            '--source-split', str(split), '--target-index', str(target),
            '--output', str(output/'target_evaluation.json'), '--predictions', str(output/'predictions.csv')],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(status='PASSED', from_zero_source_and_adaptation=True, adaptation_steps=3,
        exact_source_resume=True, exact_adaptation_resume=True, target_label_poisoning_bitwise_invariant=True,
        cli_ablations=variants, float32_graph_run=True, separate_target_evaluation=True,
        formal_paper_result=False, outputs=str(output))
    write_json(output/'smoke_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
