"""Freeze matched R8 graph IDs and role-isolated R5 graph/label stores."""
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs, pilot


def parent_dir(repo):
    return (repo/'experiments/mutagenicity/r8_图结构观察量与类别约束_20261121'/'results/attempt-001')


def prepare(root, repo, plan):
    parent = parent_dir(repo)
    r8 = json.loads((parent/'executed-config.json').read_bytes())['plan']
    for key in ('setting', 'directions', 'seeds'):
        if r8[key] != plan[key]:
            raise ValueError(f'R10 scope differs from R8 at {key}')
    frozen = pilot.verified_observations(parent, plan['parent_r8_obs_sha256'], r8)
    sources = json.loads(frozen['source-statistics.json'])
    statuses = json.loads(frozen['block-status.json'])
    decoded = root/'data'
    decoded.mkdir()
    roles = {}
    with inputs.registered_inputs(r8, repo) as (_, scope, _, r7, contexts):
        first = next(iter(contexts.values()))[3]
        read = first['read']
        raw_graphs = torch.load(io.BytesIO(read('inputs/graphs.pt', 'INPUT')), weights_only=True, map_location='cpu')
        inventory = json.loads(read('inputs/graph_hashes.json', 'INPUT'))
        task_manifest = {}
        for key, (relative, checksum, manifest, context) in contexts.items():
            task = statuses[key]['task']
            seed, target = task['seed'], task['target']
            ids = manifest['ids']
            if checksum != statuses[key]['manifest_sha256'] or relative != statuses[key]['manifest']:
                raise ValueError(f'{key}: source R7 manifest and R8 recorded identity differ')
            if sources[f'M0-seed-{seed}']['source_fit_ids'] != ids['source']:
                raise ValueError(f'{key}: R8 does not authorize these source labels')
            for pool, letter in (('adapt', 'A'), ('holdout', 'H')):
                inherited = inputs.npz_member(frozen[f'predictions/{key}-{letter}.npz'], ['ids'])
                np.testing.assert_array_equal(inherited['ids'], ids[pool])
            task_manifest[key] = dict(**task, r7_manifest=relative, r7_manifest_sha256=checksum,
                                      source_ids=ids['source'], a_ids=ids['adapt'], h_ids=ids['holdout'])
        for seed in plan['seeds']:
            entries = [value for value in task_manifest.values() if value['seed'] == seed]
            s = entries[0]['source_ids']
            if any(row['source_ids'] != s for row in entries):
                raise ValueError('A source role differs between R10 matched directions')
            y_rel = f'source_labels/M0/seed{seed}_fit.pt'
            y = torch.load(io.BytesIO(read(y_rel, 'INPUT')), weights_only=True, map_location='cpu')
            if y.shape != (len(s),) or y.dtype != torch.int64 or y.unique().tolist() != [0, 1]:
                raise ValueError('Registered source role does not define the binary class space')
            counts = next(context[3]['roles']['class_counts']['fit'] for key, context in contexts.items()
                          if context[2]['seed']==seed)
            if torch.bincount(y, minlength=2).tolist() != counts:
                raise ValueError('Source label counts disagree with the authenticated R5 roles')
            torch.save(y, decoded/f'S-labels-{seed}.pt')
            roles[str(seed)] = s
        pools = {'S': sorted({i for s in roles.values() for i in s})}
        for _, target in plan['directions']:
            entries = [value for value in task_manifest.values() if value['target']==target]
            for key, label in (('a_ids', 'A'), ('h_ids', 'H')):
                base = entries[0][key]
                if any(row[key] != base for row in entries):
                    raise ValueError('Target pool differs by source model seed')
                pools[f'{label}-M{target}'] = base
        pooled = [i for ids in pools.values() for i in ids]
        if len(set(pooled)) != len(pooled):
            raise ValueError('R10 fit/adapt/holdout topology pools overlap')
        node_dims = set()
        for pool, ids in pools.items():
            values = {}
            for i in ids:
                graph = raw_graphs[i]
                if set(graph) != {'x', 'edge_index', 'edge_attr'}:
                    raise PermissionError('R5 graph has a field outside the approved topology schema')
                measured = {name: inputs.sha(x.numpy().tobytes()) for name, x in graph.items()}
                if inventory[i]['id'] != i or inventory[i]['input_sha256'] != measured:
                    raise ValueError(f'R5 graph cache identity differs for {i}')
                x, edges = graph['x'], graph['edge_index']
                if (x.ndim != 2 or not x.is_floating_point() or not bool(torch.isfinite(x).all())
                        or edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2):
                    raise ValueError('The frozen graph cannot feed the native GCN')
                node_dims.add(x.shape[1])
                values[i] = {'x': x.float().contiguous(), 'edge_index': edges.contiguous()}
            torch.save(values, decoded/f'{pool}.pt')
        if len(node_dims) != 1:
            raise ValueError('Inconsistent native node feature columns')
    write_json(root/'roles.json', dict(tasks=task_manifest, source_fit=roles, pools=pools,
               node_features=node_dims.pop(), classes=2, r7_plan=r7, r7_archive=r8['r7_archive'],
               r7_final_digest=r8['r7_final_digest'], r8_obs_sha256=plan['parent_r8_obs_sha256'],
               r5_graph_cache_sha256=first['provenance']['r5_files']['inputs/graphs.pt']))


def read_inputs(root, *, requested=()):
    seal = json.loads((root/'INPUT.json').read_bytes())
    required = list(requested) + ['roles.json','executed-config.json']
    for relative in required:
        if digest(root/relative) != seal['files'].get(relative):
            raise PermissionError(f'Changed role-scoped R10 input: {relative}')
    return json.loads((root/'roles.json').read_bytes()), json.loads((root/'executed-config.json').read_bytes())
