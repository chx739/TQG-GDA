"""Graph and frozen-action conditions that every registered ordered pair must pass."""
from collections import defaultdict
from fractions import Fraction
import json
from statistics import mean

import numpy as np
import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import discrete_types
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation, readout_scale

WIDTHS = {'mutagenicity_edge': (14, 3), 'mutagenicity_node': (14, 3), 'proteins_edge': (3, 0)}
COUNT_FIELDS = ('structural', 'opposite', 'can_flip')


def signature(graph, node_width, edge_width):
    """Keep parallel arcs, isolates and a separate bin for each loop type."""
    n = graph.x.shape[0]
    if n < 1:
        raise ValueError('Empty graphs are outside the registered topology input')
    nodes = discrete_types(graph.x, node_width, n, 'node')
    edges = graph.edge_index
    if (edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2 or
            (edges.numel() and not bool(((0 <= edges) & (edges < n)).all()))):
        raise ValueError('Invalid archived directed endpoints')
    if edge_width:
        types = np.asarray(discrete_types(graph.edge_attr, edge_width, edges.shape[1], 'edge'), dtype=np.intp)
    else:
        if graph.edge_attr is not None:
            raise ValueError('Unexpected edge attributes in the untyped setting')
        types = np.zeros(edges.shape[1], dtype=np.intp)
    width = max(edge_width, 1)
    incoming_outgoing_loop = np.zeros((n, 3 * width), dtype=np.int64)
    src, dst = edges.numpy()
    loops = src == dst
    ordinary = ~loops
    np.add.at(incoming_outgoing_loop, (dst[ordinary], types[ordinary]), 1)
    np.add.at(incoming_outgoing_loop, (src[ordinary], width + types[ordinary]), 1)
    np.add.at(incoming_outgoing_loop, (src[loops], 2 * width + types[loops]), 1)
    topology = n, tuple(sorted(map(tuple, incoming_outgoing_loop.tolist())))
    histogram = np.bincount(nodes, minlength=node_width)
    return topology, histogram


def count_queries(queries, references, qmargin, rmargin):
    if (len(queries) != len(qmargin) or len(references) != len(rmargin) or
            any(z.dtype != np.float64 or z.ndim != 1 or not np.isfinite(z).all() for z in (qmargin, rmargin))):
        raise ValueError('Expected finite original float64 margins in Q/R order')
    buckets = defaultdict(list)
    for (topology, histogram), z in zip(references, rmargin, strict=True):
        buckets[topology].append((histogram, z))
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    for (topology, hist), z in zip(queries, qmargin, strict=True):
        support = False
        opposite = False
        can_flip = False
        for other, a in buckets.get(topology, ()):
            if np.abs(hist - other).sum() != 2:
                continue
            support = True
            if (z > 0.) == (a > 0.):
                continue
            opposite = True
            # sign(z + a), evaluated as a comparison to retain strict ties.
            if (a > -z) != (z > 0.):
                can_flip = True
                break
        counts['structural'] += support
        counts['opposite'] += opposite
        counts['can_flip'] += can_flip
    return counts


def summarize(rows, cfg):
    summary = []
    passed = {}
    threshold = Fraction(*cfg['minimum_flip_fraction'])
    for setting in cfg['settings']:
        subset = [row for row in rows if row['setting'] == setting]
        if len(subset) != 12 or len({(r['source'], r['target']) for r in subset}) != 12:
            raise ValueError('Each setting needs the full twelve-direction grid')
        rates = {k: mean(Fraction(r[k], r['q_graphs']) for r in subset) for k in COUNT_FIELDS}
        passed[setting] = rates['can_flip'] >= threshold
        summary.append(dict(setting=setting, directions=len(subset),
                            q_graphs=sum(r['q_graphs'] for r in subset), r_graphs=sum(r['r_graphs'] for r in subset),
                            **{k: sum(r[k] for r in subset) for k in COUNT_FIELDS},
                            **{f'{k}_equal_direction_percent': float(100 * rates[k]) for k in COUNT_FIELDS},
                            reaches_minimum=passed[setting]))
    proceed = passed['proteins_edge'] and (passed['mutagenicity_edge'] or passed['mutagenicity_node'])
    return summary, dict(minimum_pp=float(100 * threshold), setting_passes=passed, proceed_to_exact_pairing=proceed)


def check_qr(row, task, query, ref, manifest_sha):
    target = str(task['target'])
    if (any(row[k] != v for k, v in task.items()) or row['manifest_sha256'] != manifest_sha or
            row['query_ids'] != query or row['reference_ids'][target] != ref or set(query) & set(ref) or
            row['role_read_hashes'][f'Q{target}'] != readout_scale.identity(query) or
            row['role_read_hashes'][f'R{target}'] != readout_scale.identity(ref)):
        raise PermissionError('Authenticated R13 Q/R, task or manifest identity differs')


def run(out, repo, cfg):
    r13 = repo / cfg['r13_attempt']
    final_path = r13 / 'FINAL.json'
    if digest(final_path) != cfg['r13_final_sha256']:
        raise PermissionError('R13 readout FINAL differs from the registered artifact')
    receipt = json.loads(final_path.read_bytes())
    if receipt['stage'] != 'FINAL':
        raise PermissionError('Invalid R13 stage')
    files = receipt['files']
    execution = json.loads(prior_run.read_member(r13, files, 'executed-config.json'))
    old_roles = json.loads(prior_run.read_member(r13, files, 'roles.json'))
    native = execution['historical_plan']
    if (cfg['settings'], cfg['seed'], native['domains']) != (native['settings'], native['phase1_seed'], list(range(4))):
        raise PermissionError('R9 audit grid differs from R13')
    archive = historical.CArchive(native)
    checks = repo / execution['plan']['r9_checks']
    checks_sha = execution['plan']['r9_checks_sha256']
    parent = dict(r7_final_digest=native['r7_final_digest'], r9_checks_sha256=checks_sha)
    if receipt['predecessors'] != parent or digest(checks) != checks_sha:
        raise PermissionError('R13 R7/R9 authentication does not match the readout receipt')
    old_checks = json.loads(checks.read_bytes())
    if old_checks['r7_c_final'] != native['r7_final_digest']:
        raise PermissionError('R9 Q/R hash receipts reference a different R7')
    groups = {(row['setting'], row['source']): row for row in old_checks['groups']}
    _, blocks = observation.observations_expected(native, [cfg['seed']])
    if set(old_roles) != set(blocks):
        raise PermissionError('R13 role grid is incomplete')
    stores, roots, cache, rows, access = {}, {}, {}, [], {}
    for setting, source, seed, tasks in observation.group_tasks(native, [cfg['seed']]):
        manifests = {}
        for task in tasks:
            key = inputs.key(**task)
            manifest, checksum, ctx = historical.context(archive, task, repo)
            manifests[key] = task | dict(ids=manifest['ids'], manifest_sha256=checksum)
            if setting not in stores:
                stores[setting] = historical.GraphStore(ctx)
                roots[setting] = ctx['root']
                cache[setting] = {}
            if (stores[setting].graph_sha != old_roles[key]['graph_archive_sha256'] or
                    ctx['root'] != roots[setting]):
                raise PermissionError(f'R13 topology identity differs for {key}')
        query, refs, _ = readout_scale.matched_roles(setting, source, list(manifests.values()), groups[setting, source])
        store = stores[setting]
        for task in tasks:
            key = inputs.key(**task)
            t = str(task['target'])
            q, ref = query[t], refs[t]
            role = manifests[key]
            check_qr(old_roles[key], task, q, ref, role['manifest_sha256'])
            # The only graph rows materialized here are the approved same-target Q and R.
            ids = q + ref
            graphs = store.take(ids)
            traits = cache[setting]
            for gid, graph in zip(ids, graphs, strict=True):
                if gid not in traits:
                    traits[gid] = signature(graph, *WIDTHS[setting])
            logits = readout_scale.b0_adapt(archive, task, role['ids']['adapt'], ids)
            z = logits[:, 1] - logits[:, 0]
            rel = f'predictions/{key}.npz'
            old = inputs.npz_member(prior_run.read_member(r13, files, rel), ['ids', 'B0', 'archived_B0'])
            if (not np.array_equal(old['ids'], q) or not np.array_equal(old['archived_B0'], logits[:len(q)])
                    or not np.array_equal(old['B0'] > 0., z[:len(q)] > 0.)):
                raise PermissionError(f'R7 B0 differs from the original R13 row for {key}')
            counts = count_queries([traits[i] for i in q], [traits[i] for i in ref], z[:len(q)], z[len(q):])
            rows.append(dict(**task, q_graphs=len(q), r_graphs=len(ref), **counts,
                             **{f'{k}_percent': 100 * counts[k] / len(q) for k in COUNT_FIELDS}))
            access[key] = dict(manifest_sha256=role['manifest_sha256'], r13_row_sha256=files[rel],
                               graph_archive_sha256=store.graph_sha, q_ids_sha256=readout_scale.identity(q),
                               r_ids_sha256=readout_scale.identity(ref), q_actions_matched=len(q))
        print(f'R9 paired-action {setting} M{source} seed {seed} completed', flush=True)
    aggregate, gate = summarize(rows, cfg)
    write_table(out / 'directions.csv', rows)
    write_table(out / 'settings.csv', aggregate)
    write_json(out / 'gate.json', gate)
    write_json(out / 'access.json', dict(reused_qr_roles=access, r7_files=archive.digests, graph_pools=['Q', 'R'],
                   used_holdout_inputs=False, used_graph_labels=False, predictions_rechecked=sum(r['q_graphs'] for r in rows)))
    outputs = ('executed-config.json', 'directions.csv', 'settings.csv', 'gate.json', 'access.json')
    return prior_run.seal(out, 'FINAL', outputs, dict(r13_final_sha256=cfg['r13_final_sha256'], **parent))
