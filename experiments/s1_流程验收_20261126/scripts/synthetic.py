"""Write a deterministic TU-format engineering fixture with separate label capabilities."""
import json
from pathlib import Path
import random

from cela.data import build_tu_index


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def make_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    dataset = 'CELAToy'
    raw = root / 'raw'
    raw.mkdir()
    raw_lines = {name: [] for name in ('graph_indicator', 'A', 'edge_labels', 'node_labels')}
    labels = {}
    offset = 1
    rng = random.Random(3405)
    ids = [str(i) for i in range(20)]
    for i, key in enumerate(ids):
        y = i % 2
        labels[key] = y
        n = 6 + i % 3
        perm = list(range(n))
        rng.shuffle(perm)
        # Cycles and paths give nontrivial candidate sets. Colors are exogenous.
        edges = [(a, a + 1) for a in range(n - 1)]
        if y:
            edges.append((n - 1, 0))
        if i in (12, 15, 17):
            edges.append((0, 2))
        for node in range(n):
            raw_lines['graph_indicator'].append(str(i + 1))
            raw_lines['node_labels'].append(str((node + i) % 3))
        for a, b in edges:
            u, v = offset + perm[a], offset + perm[b]
            for pair in ((u, v), (v, u)):
                raw_lines['A'].append(f'{pair[0]}, {pair[1]}')
                raw_lines['edge_labels'].append(str((a + b) % 2))
        offset += n
    for field, lines in raw_lines.items():
        (raw / f'{dataset}_{field}.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
    roles = dict(train=list(range(8)), val=list(range(8, 12)), A=list(range(12, 16)), H=list(range(16, 20)))
    for role in ('train', 'val', 'H'):
        write_json(root / f'{role}-labels.json', {str(i): labels[str(i)] for i in roles[role]})
    write_json(root / 'manifest.json', dict(dataset=dataset, roles=roles, typed_edges=True, tu_index='tu_index.json',
               node_type_provenance='Synthetic node colors assigned by graph and node index.',
               source_labels={'train': 'train-labels.json', 'val': 'val-labels.json'}))
    build_tu_index(dict(dataset=dataset, raw_dir=raw, manifest=root / 'manifest.json'), root / 'tu_index.json')
    write_json(root / 'config.json', dict(data=dict(dataset=dataset, raw_dir='raw', manifest='manifest.json'),
               settings=dict(epochs=2, bootstrap_samples=4)))
    return root / 'config.json'
