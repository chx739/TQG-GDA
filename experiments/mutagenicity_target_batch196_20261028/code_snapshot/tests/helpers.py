import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from tqg_gda.data import GraphDomainTask
from tqg_gda.utils import sha256, load_config, merge

ROOT = Path(__file__).resolve().parents[1]


def fixture_files(root, target_label=999):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    graphs = []
    for i in range(24):
        n = 3 + i % 3
        label = i % 2
        x = torch.tensor([[.2 + .07 * i + .3 * label, .4 + .1 * j + .15 * label]
                          for j in range(n)], dtype=torch.float64)
        edges = [(j, (j + 1) % n) for j in range(n)] + [((j + 1) % n, j) for j in range(n)]
        y = label if i < 16 or target_label is None else target_label
        graphs.append(Data(x=x, edge_index=torch.tensor(edges).t().contiguous(), y=torch.tensor([y])))
    data, slices = InMemoryDataset.collate(graphs)
    cache = root / 'data.pt'
    torch.save((data.to_dict(), slices, dict(num_node_attributes=2, num_node_labels=0,
                                           num_edge_attributes=0, num_edge_labels=0), Data), cache)
    source = root / 'idx_fixture_0.npy'; target = root / 'idx_fixture_1.npy'
    np.save(source, np.arange(16)); np.save(target, np.arange(16, 24))
    split = root / 'source.json'
    split.write_text(json.dumps(dict(dataset='fixture', source_train_indices=list(range(12)),
        source_validation_indices=list(range(12, 16)), dataset_cache_sha256=sha256(cache),
        domain_index_sha256=sha256(source))))
    return cache, split, target


def small_config(**override):
    config = load_config(ROOT / 'configs/tqg_development.json')
    config = merge(config, dict(use_node_attributes=True,
        model=dict(hidden_dim=8, num_layers=2), split=dict(fit_fraction=.75),
        warmup=dict(epochs=2, batch_size=4),
        adaptation=dict(steps=3, fit_batch_size=4, query_batch_size=4, target_batch_size=4,
                        validation_interval=1, checkpoint_interval=1), probes=dict(count=2)))
    return merge(config, override)


def task_from(files, config, target=True):
    cache, split, index = files
    return GraphDomainTask(cache, split, target_index=index if target else None,
        target_policy='transductive-unlabeled' if target else None,
        fit_fraction=config['split']['fit_fraction'], split_seed=config['split']['seed'],
        use_node_attributes=config['use_node_attributes'], preprocess_pool=config['warmup']['source_pool'])
