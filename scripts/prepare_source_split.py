"""Create an explicitly NEW source protocol for a native cache and source-domain index.

This does not rewrite the bundled, byte-locked original PROTEINS protocol.
"""
import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch_geometric.data.separate import separate
from tqg_gda.data import load_cache, validate_ids
from tqg_gda.utils import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True); p.add_argument('--source-index', required=True)
    p.add_argument('--dataset', required=True); p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--seed', type=int, default=431); p.add_argument('--output', required=True)
    args = p.parse_args()
    if not 0 < args.validation_fraction < 1:
        p.error('validation-fraction must be between zero and one')
    if Path(args.output).exists():
        raise FileExistsError('Use a new manifest path')
    raw, slices, _ = load_cache(args.data_path)
    index = np.load(args.source_index, allow_pickle=False)
    if index.ndim != 1 or index.dtype.kind not in 'iu':
        raise ValueError('Source index must be a one-dimensional integer array')
    ids = list(map(int, index.tolist())); validate_ids(ids, len(slices['x']) - 1, 'source')
    groups = {}
    for i in ids:
        g = separate(raw.__class__, raw, i, slices, decrement=False)
        if (g.y is None or g.y.numel() != 1 or not torch.isfinite(g.y).all()
                or not torch.equal(g.y, g.y.long())):
            raise ValueError('Expected one finite integer source label per graph')
        groups.setdefault(int(g.y.item()), []).append(i)
    train, validation = [], []
    generator = random.Random(args.seed)
    for label in sorted(groups):
        group = sorted(groups[label]); generator.shuffle(group)
        if len(group) < 3:
            raise ValueError('Each class needs >=3 graphs for fit/query plus validation')
        n = max(1, min(len(group) - 2, round(len(group) * args.validation_fraction)))
        validation.extend(group[:n]); train.extend(group[n:])
    write_json(args.output, dict(dataset=args.dataset, source_train_indices=sorted(train),
        source_validation_indices=sorted(validation), dataset_cache_sha256=sha256(args.data_path),
        domain_index_sha256=sha256(args.source_index), split_regenerated=True, seed=args.seed,
        protocol='Explicit new source protocol; not the original bundled PROTEINS split'))
    print(args.output)


if __name__ == '__main__':
    main()
