"""Run warmup, adaptation, or both on an explicitly identified TU/PyG task."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tqg_gda.data import GraphDomainTask
from tqg_gda.training import run_warmup, run_adaptation
from tqg_gda.utils import load_config, validate_config, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('source', 'adapt', 'all'))
    p.add_argument('--data-path', required=True)
    p.add_argument('--source-split', default=str(ROOT / 'data/splits/PROTEINS_idx_0_source.json'))
    p.add_argument('--dataset', default='PROTEINS')
    p.add_argument('--idx-dir', default=str(ROOT / 'idx'))
    p.add_argument('--index-family', choices=('idx', 'node'), default='idx')
    p.add_argument('--source-domain', type=int, default=0)
    p.add_argument('--target-domain', type=int, default=1)
    p.add_argument('--target-index')
    p.add_argument('--target-policy', choices=('transductive-unlabeled',))
    p.add_argument('--config', default=str(ROOT / 'configs/tqg_development.json'))
    p.add_argument('--source-checkpoint')
    p.add_argument('--resume-source')
    p.add_argument('--resume-adapt')
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    config = validate_config(load_config(args.config))
    if args.seed is not None:
        config['seed'] = args.seed
    if args.stage == 'adapt' and not args.source_checkpoint:
        p.error('adapt requires --source-checkpoint from this TQG source trainer')
    if args.source_checkpoint and args.stage != 'adapt':
        p.error('--source-checkpoint applies to stage adapt')
    if args.resume_adapt and args.stage != 'adapt':
        p.error('--resume-adapt requires stage adapt')
    if args.resume_source and args.stage == 'adapt':
        p.error('--resume-source requires source or all')
    target_required = args.stage != 'source' and config['method'] in ('tqg', 'tqg_v2')
    if target_required and not args.target_policy:
        p.error('TQG adaptation requires --target-policy transductive-unlabeled')
    common = dict(cache_path=args.data_path, source_split=args.source_split,
        fit_fraction=config['split']['fit_fraction'], split_seed=config['split']['seed'],
        use_node_attributes=config['use_node_attributes'], preprocess_pool=config['warmup']['source_pool'])
    task = GraphDomainTask(**common)
    if task.manifest['dataset'] != args.dataset:
        raise ValueError('Dataset differs from source manifest')
    if task.manifest.get('family', args.index_family) != args.index_family or task.manifest.get('source_domain', args.source_domain) != args.source_domain:
        raise ValueError('Source domain/family differs from the fixed source manifest')
    source_index = Path(args.idx_dir) / f'{args.index_family}_{args.dataset}_{args.source_domain}.npy'
    if source_index.exists():
        ids = np.load(source_index, allow_pickle=False)
        if set(ids.tolist()) != set(task.train_ids + task.validation_ids):
            raise ValueError('Source index differs from fixed source train/validation union')
        if task.manifest.get('domain_index_sha256') and sha256(source_index) != task.manifest['domain_index_sha256']:
            raise ValueError('Source index bytes differ from manifest')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'resolved_config.json', config)
    source_checkpoint = args.source_checkpoint
    if args.stage in ('source', 'all'):
        run_warmup(task, config, output / 'source', args.device, args.resume_source)
        source_checkpoint = output / 'source/best.pt'
    if args.stage in ('adapt', 'all'):
        if target_required:
            target_index = args.target_index or Path(args.idx_dir) / f'{args.index_family}_{args.dataset}_{args.target_domain}.npy'
            task = GraphDomainTask(**common, target_index=target_index, target_policy=args.target_policy)
        run_adaptation(task, config, source_checkpoint, output / 'adaptation', args.device, args.resume_adapt)
    print(json.dumps({'output': str(output), 'target_labels_used': False}), flush=True)


if __name__ == '__main__':
    main()
