"""Authenticate exact R5 topology and R7 pilot cohorts without reading labels."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from .composition import graph_counts


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def npz_member(blob, keys=None):
    with np.load(io.BytesIO(blob), allow_pickle=False) as member:
        return {key: member[key] for key in (member.files if keys is None else keys)}


def task_grid(plan):
    for source, target in plan['directions']:
        for seed in plan['seeds']:
            name = f"{plan['setting']}-M{source}_to_M{target}-seed-{seed}"
            yield name, dict(setting=plan['setting'], source=source, target=target, seed=seed)


def prior_donor(repo):
    return (repo / 'experiments/mutagenicity/r7_共同解释与共享更新验证_20261119'
            / 'results/iteration-14/accuracy-prior-control')


def donor_read(root, final_hash):
    final = (root/'FINAL.json').read_bytes()
    if sha(final) != final_hash:
        raise ValueError('R7 prior-control FINAL differs from registered identity')
    latest = json.loads(final)
    obs = (root/'OBS.json').read_bytes()
    if sha(obs) != latest['observation_sha256']:
        raise ValueError('R7 prior-control OBS chain differs')
    files = json.loads(obs)['files'] | latest['files']

    def read(name):
        blob = (root/name).read_bytes()
        if files.get(name) != sha(blob):
            raise ValueError(f'R7 prior-control artifact differs: {name}')
        return blob

    for name in files:
        read(name)
    return read


@contextmanager
def registered_inputs(plan, repo):
    root = Path(plan['r7_archive']).resolve()
    donor = donor_read(prior_donor(repo), plan['r7_prior_control_final_sha256'])
    if json.loads(donor('OBS.json'))['parent'] != plan['r7_final_digest']:
        raise ValueError('R7 prior-control evidence points to another P run')
    with seals.verified_seal_chain(root, 'R7_FINAL') as scope:
        if json.loads((root/'seals/R7_FINAL.json').read_bytes())['digest'] != plan['r7_final_digest']:
            raise ValueError('R7 P FINAL differs from registered identity')
        original = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        phase = original['phases']['P']
        if (phase['setting'] != plan['setting'] or phase['directions'] != plan['directions']
                or original['real_model_seeds'] != plan['seeds']
                or original['shared_update']['primary_tau'] != plan['r7_tau']):
            raise ValueError('R8 cohort does not match the R7 registration')
        contexts = {}
        for key, task in task_grid(plan):
            path = f'blocks/{key}/manifest.json'
            blob = scope.read_bytes(path, 'R7_INPUT')
            manifest = json.loads(blob)
            if any(manifest[field] != value for field, value in task.items()) or manifest['r7_plan'] != original:
                raise ValueError(f'{key}: manifest task or executed plan differs')
            context = r5_inputs._context(original, **task, repository=repo)
            if manifest['ids'] != context['ids']:
                raise ValueError(f'{key}: R7 IDs differ from authenticated R5 roles and folds')
            contexts[key] = (path, sha(blob), manifest, context)
        yield root, scope, donor, original, contexts


def raw_topology(plan, expected):
    """Read only four allowlisted raw topology members and verify their semantics."""
    spec = plan['input']
    root = Path(spec['raw_directory'])
    dataset = plan['dataset']
    suffixes = ('A', 'graph_indicator', 'node_labels', 'edge_labels')
    arrays, hashes = {}, {}
    for suffix in suffixes:
        name = f'{dataset}_{suffix}.txt'
        blob = (root/name).read_bytes()
        if sha(blob) != spec['files'][name]['sha256']:
            raise ValueError(f'R5 raw topology bytes differ: {name}')
        arrays[suffix] = np.loadtxt(io.BytesIO(blob), dtype=np.int64, delimiter=',',
                                    ndmin=2 if suffix == 'A' else 1)
        hashes[name] = sha(blob)
    raw_edges, memberships = arrays['A']-1, arrays['graph_indicator']-1
    nodes, edge_types = arrays['node_labels'], arrays['edge_labels']
    if (raw_edges.ndim != 2 or raw_edges.shape[1] != 2 or memberships.ndim != 1
            or nodes.ndim != 1 or edge_types.ndim != 1 or len(nodes) != len(memberships)
            or len(edge_types) != len(raw_edges) or np.any(np.diff(memberships) < 0)
            or np.unique(memberships).tolist() != list(range(expected))
            or raw_edges.min() < 0 or raw_edges.max() >= len(memberships)
            or not np.array_equal(memberships[raw_edges[:, 0]], memberships[raw_edges[:, 1]])):
        raise ValueError('Unsupported raw graph topology')
    for values, width, field in ((nodes, spec['node_feature_dim'], 'node'), (edge_types, spec['edge_feature_dim'], 'edge')):
        if not np.array_equal(np.unique(values)-values.min(), np.arange(width)):
            raise ValueError(f'Unsupported raw discrete {field} encoding')
    nonself = raw_edges[:, 0] != raw_edges[:, 1]
    record = dict(raw_file_sha256=hashes, raw_directed_rows=len(raw_edges), raw_self_loops=int((~nonself).sum()),
                  raw_duplicate_nonself_directed_rows=int(nonself.sum()-len(np.unique(raw_edges[nonself], axis=0))),
                  raw_node_type_values=np.unique(nodes).tolist(), raw_edge_type_values=np.unique(edge_types).tolist())
    return arrays, record


def graph_preflight(context, plan):
    read = context['read']
    blobs = {name: read(f'inputs/{name}', 'INPUT') for name in ('graphs.pt', 'graph_hashes.json')}
    graphs = torch.load(io.BytesIO(blobs['graphs.pt']), weights_only=True, map_location='cpu')
    inventory = json.loads(blobs['graph_hashes.json'])
    if type(graphs) is not list or len(graphs) != len(inventory) or len(graphs) != context['total']:
        raise ValueError('Wrong R5 graph archive structure')
    comp = plan['composition']
    raw, checks = raw_topology(context['old_plan'], len(graphs))
    if checks['raw_duplicate_nonself_directed_rows']:
        raise ValueError('Unsupported duplicate raw directed edges')
    group = raw['graph_indicator']-1
    offsets = np.r_[0, np.cumsum(np.bincount(group))]
    edges = raw['A']-1
    edge_groups = group[edges[:, 0]]
    keep = edges[:, 0] != edges[:, 1]
    grouped = np.argsort(edge_groups, kind='stable')
    edge_offsets = np.r_[0, np.cumsum(np.bincount(edge_groups, minlength=len(graphs)))]
    node_types = raw['node_labels']-raw['node_labels'].min()
    edge_types = raw['edge_labels']-raw['edge_labels'].min()
    counts = []
    for number, (graph, old) in enumerate(zip(graphs, inventory)):
        entry = graph_counts(graph, comp['node_types'], comp['edge_types'])
        counts.append(entry)
        sha_graph = {key: sha(value.numpy().tobytes()) for key, value in graph.items()}
        if old != dict(id=number, nodes=entry[0].total(), undirected_edges=entry[1].total(), input_sha256=sha_graph):
            raise ValueError(f'R5 per-graph bytes or counts differ for ID {number}')
        start, stop = offsets[number:number+2]
        selected = grouped[edge_offsets[number]:edge_offsets[number+1]]
        selected = selected[keep[selected]]
        from_raw = {(int(a-start), int(b-start)): int(c)
                    for (a, b), c in zip(edges[selected], edge_types[selected])}
        from_seal = {tuple(pair): kind for pair, kind in zip(
            graph['edge_index'].T.tolist(), graph['edge_attr'].argmax(1).tolist())}
        if (from_raw != from_seal or
                not np.array_equal(node_types[start:stop], graph['x'].argmax(1).numpy())):
            raise ValueError(f'Raw types or loop-stripped topology differ for graph {number}')
    checks.update(graph_count=len(counts), undirected_edges=sum(e.total() for _, e in counts),
                  no_edge_graphs=sum(not e for _, e in counts), reversed_edges_type_matched=True,
                  permitted_graph_fields=['x', 'edge_index', 'edge_attr'], graph_hashes_verified=len(counts),
                  r5_input_digest=context['provenance']['r5_seal_digests']['INPUT'],
                  graph_archive_hashes={name: sha(blob) for name, blob in blobs.items()})
    return counts, checks


def preflight(plan, repo):
    with registered_inputs(plan, repo) as (_, scope, donor, original, contexts):
        counts, checks = graph_preflight(next(iter(contexts.values()))[3], plan)
        blocks = {}
        for key, (path, manifest_hash, manifest, _) in contexts.items():
            role_sizes = {}
            for file, roles in (('fit_input.npz', ('source', 'adapt')),
                                ('validation_input.npz', ('validation',)), ('holdout_input.npz', ('holdout',))):
                values = npz_member(scope.read_bytes(f'blocks/{key}/{file}', 'R7_INPUT'),
                                    [role+'_ids' for role in roles])
                for role in roles:
                    np.testing.assert_array_equal(values[role+'_ids'], manifest['ids'][role])
                    role_sizes[role] = len(manifest['ids'][role])
            previous = npz_member(donor(f'predictions/{key}.npz'), ['ids'])
            np.testing.assert_array_equal(previous['ids'], manifest['ids']['holdout'])
            blocks[key] = dict(manifest_sha256=manifest_hash, roles=role_sizes)
    return dict(label_values_loaded=False, blocks=blocks, topology=checks,
                r7_final_digest=plan['r7_final_digest'], r7_prior_final_sha256=plan['r7_prior_control_final_sha256'])
