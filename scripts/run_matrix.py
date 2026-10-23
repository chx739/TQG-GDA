"""Run a fixed ablation/seed matrix; target labels are never evaluated here."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True); p.add_argument('--output', required=True)
    p.add_argument('--source-split', default=str(ROOT / 'data/splits/PROTEINS_idx_0_source.json'))
    p.add_argument('--dataset', default='PROTEINS'); p.add_argument('--idx-dir', default=str(ROOT / 'idx'))
    p.add_argument('--index-family', choices=('idx', 'node'), default='idx')
    p.add_argument('--source-domain', type=int, default=0); p.add_argument('--target-domain', type=int, default=1)
    p.add_argument('--device', default='cpu'); p.add_argument('--seeds', nargs='+', type=int, default=[3405, 3406, 3407])
    p.add_argument('--configs', nargs='+', default=['source_only_same_fit', 'source_only_all_labels',
        'tqg_development', 'random_probes', 'source_probes', 'odd_only', 'linear_jvp'])
    args = p.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    for name in args.configs:
        for seed in args.seeds:
            run = output / f'{name}_seed{seed}'
            command = [sys.executable, str(ROOT / 'scripts/run_experiment.py'), 'all',
                '--data-path', args.data_path, '--source-split', args.source_split,
                '--dataset', args.dataset, '--idx-dir', args.idx_dir, '--index-family', args.index_family,
                '--source-domain', str(args.source_domain), '--target-domain', str(args.target_domain),
                '--target-policy', 'transductive-unlabeled', '--config', str(ROOT / 'configs' / f'{name}.json'),
                '--seed', str(seed), '--device', args.device, '--output', str(run)]
            print('Running', name, seed, flush=True)
            with (output / f'{name}_seed{seed}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
