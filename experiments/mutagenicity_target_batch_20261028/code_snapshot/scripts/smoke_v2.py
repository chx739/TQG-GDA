"""Small synthetic CLI checks for v2, its controls, and exact recovery.

Outputs are engineering checks, not dataset effectiveness experiments.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import torch
from helpers import fixture_files, small_config
from tqg_gda.utils import load_config, merge, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT):
        parser.error('Smoke outputs must be outside the repository')
    output.mkdir(parents=True, exist_ok=False)
    clean = fixture_files(output/'clean_inputs', target_label=None)
    poisoned = fixture_files(output/'poisoned_inputs', target_label=999)
    base = merge(load_config(ROOT/'configs/v2/tqg_v2.json'),
        {key: value for key, value in small_config().items()
         if key in ('model', 'split', 'use_node_attributes', 'warmup', 'adaptation')})
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')

    def run(name, config, *, inputs=clean, checkpoint=None, resume=None):
        config_path = output/(name+'.json')
        write_json(config_path, config)
        cache, split, target = inputs
        command = [sys.executable, str(ROOT/'scripts/run_experiment.py'), 'adapt' if checkpoint else 'all',
            '--data-path', str(cache), '--source-split', str(split), '--target-index', str(target),
            '--target-policy', 'transductive-unlabeled', '--dataset', 'fixture',
            '--idx-dir', str(cache.parent), '--config', str(config_path), '--output', str(output/name)]
        if checkpoint:
            command += ['--source-checkpoint', str(checkpoint)]
        if resume:
            command += ['--resume-adapt', str(resume)]
        with (output/(name+'.log')).open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)

    run('joint', base)
    source = output/'joint/source/best.pt'
    run('resumed', base, checkpoint=source, resume=output/'joint/adaptation/step_1.pt')
    run('poisoned', base, inputs=poisoned)
    reference = torch.load(output/'joint/adaptation/last.pt', weights_only=False)
    for name in ('resumed', 'poisoned'):
        candidate = torch.load(output/name/'adaptation/last.pt', weights_only=False)
        assert all(torch.equal(value, candidate['model'][key]) for key, value in reference['model'].items()), name
    variants = {
        'mean_only_krr': dict(v2=dict(calibration=dict(mode='mean_only'))),
        'representation_only': dict(v2=dict(signature='representation_only')),
        'source_only_sgd': dict(method='source_only'),
        'pseudo_labels_sgd': dict(method='tqg', adaptation=dict(feedback_predictor='pseudo_labels', confidence_threshold=.9)),
        'permuted_labels': dict(adaptation=dict(query_label_mode='permute_within_batch', query_permutation_seed=113)),
    }
    for name, changes in variants.items():
        run(name, merge(base, changes), checkpoint=source)
    run('source_all_sgd', merge(base, dict(method='source_only',
        warmup=dict(source_pool='all_source'), adaptation=dict(source_pool='all_source'))))
    history = [json.loads(line) for line in (output/'joint/adaptation/history.jsonl').read_text().splitlines()]
    assert all(row['query']['status'] == 'gradient_feedback' for row in history)
    assert all(row['execution']['feedback_gradient_identity_error'] < 1e-12 for row in history)
    assert all(row['execution']['sgd_step_identity_error'] < 1e-12 for row in history)
    result = dict(status='PASSED', exact_resume=True, target_label_poisoning_bitwise_invariant=True,
        variants=[*variants, 'source_all_sgd'], joint_steps=len(history), formal_paper_result=False,
        target_labels_evaluated=False, output=str(output))
    write_json(output/'smoke_result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
