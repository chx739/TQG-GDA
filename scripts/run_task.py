"""Run one stage of a CELA experiment from the repository or an installed wheel."""
import argparse
import json
from pathlib import Path

# Load the numerical backend before PyTorch when MKL inherits a threading layer.
from cela import pipeline
from cela.artifacts import read_json
import torch


def config_from(path, raw_dir=None):
    cfg = read_json(path)
    if not isinstance(cfg, dict) or set(cfg) - {'data', 'settings'} or 'data' not in cfg:
        raise ValueError('config must contain data and optionally settings')
    data = cfg['data']
    for key in ('manifest', 'raw_dir'):
        if key in data:
            data[key] = str((path.parent / data[key]).resolve())
    if raw_dir is not None:
        data['raw_dir'] = str(raw_dir.resolve())
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='phase', required=True)
    for phase in ('train', 'adapt', 'predict', 'score'):
        command = phases.add_parser(phase)
        command.add_argument('--output', required=True, type=Path, help='new artifact path')
        command.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads')
        if phase != 'score':
            command.add_argument('--config', type=Path, required=True)
            command.add_argument('--raw-dir', type=Path, help='TU raw directory override')
        if phase in ('adapt', 'predict'):
            command.add_argument('--source', type=Path, required=True)
        if phase == 'predict':
            command.add_argument('--adaptation', type=Path, required=True)
        if phase == 'score':
            command.add_argument('--predictions', type=Path, required=True)
            command.add_argument('--labels', type=Path, required=True, help='JSON map of H ID to raw class')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    torch.set_num_threads(args.threads)
    if args.phase == 'score':
        report = pipeline.score(args.predictions, args.labels, args.output)
    else:
        cfg = config_from(args.config, args.raw_dir)
        if args.phase == 'train':
            report = pipeline.train(cfg, args.output)
        elif args.phase == 'adapt':
            report = pipeline.adapt(cfg, args.source, args.output)
        else:
            report = pipeline.predict(cfg, args.source, args.adaptation, args.output)
    print(json.dumps(dict(phase=args.phase, output=str(args.output), **report), ensure_ascii=False))


if __name__ == '__main__':
    main()
