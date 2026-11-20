"""Replay docs/local-vocabulary.md from 2d97b61. All rates use [0,1]."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from statistics import fmean
import argparse
import csv
import json
import mmap
import struct
import sys

sys.dont_write_bytecode = True
SETTINGS = ('mutagenicity_edge', 'mutagenicity_node', 'proteins_edge')
DOMAINS = tuple(range(4))
R7_FINAL = 'a97d6884ee74c456fce8b7c7e3ae9a5a5d6a72e22ac4b7b8835a06bf130a420c'
R9 = Path('experiments/mutagenicity/r9_目标近邻与源标签传递_20261121')
R9_FINGERPRINTS = 'e96060ec8045a2eb5303e34d3582de34e440ba2c7d656a5d9b62575186cd52ac'
POOLS = ('local', 'validation', 'foreign1', 'foreign2')
PAIRS = list(combinations(POOLS, 2))


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def digest(blob):
    return sha256(blob).hexdigest()


def signature(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def member(root, relative, identity):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'member escaped root')
    blob = path.read_bytes()
    require(digest(blob) == identity, f'checksum differs for {path}')
    return json.loads(blob)


class Archive:
    def __init__(self, root):
        self.root = root
        parent = None
        for stage in ('R7_INPUT', 'R7_OBS', 'R7_ORACLE', 'R7_FINAL'):
            seal = json.loads((root / 'seals' / f'{stage}.json').read_bytes())
            sig = seal.pop('digest')
            require(seal['stage'] == stage and seal['parent'] == parent and sig == signature(seal), stage)
            if stage == 'R7_INPUT':
                self.files = seal['files']
            parent = sig
        require(parent == R7_FINAL, 'C final identity differs')
        self.plan = self.read('executed-plan.json')
        phase = self.plan['phases']['C']
        require(phase['settings'] == list(SETTINGS) and phase['domains'] == list(DOMAINS)
                and 3405 in self.plan['real_model_seeds'], 'C grid differs')

    def read(self, name):
        return member(self.root, name, self.files[name])

    def manifest(self, setting, source, target):
        name = f'blocks/{setting}-M{source}_to_M{target}-seed-3405/manifest.json'
        m = self.read(name)
        require((m['setting'], m['source'], m['target'], m['seed']) == (setting, source, target, 3405)
                and m['r7_plan'] == self.plan, 'C block identity differs')
        for role in ('source', 'validation', 'adapt', 'holdout'):
            require(signature(m['ids'][role]) == m['provenance']['id_hashes'][role], 'C role hash differs')
        return m


class Inventory:
    def __init__(self, setting, prov):
        self.root = Path(prov['archive_root'])
        self.seal_sha = prov['r5_files']['seals/R5_INPUT.json']
        seal = member(self.root, 'seals/R5_INPUT.json', self.seal_sha)
        require(seal['stage'] == 'R5_INPUT' and seal['digest'] == prov['r5_seal_digests']['INPUT']
                and signature({k: seal[k] for k in ('stage', 'files')}) == seal['digest'], 'R5 seal')
        self.files = seal['files']
        self.domains = self.read('inputs/domains.json')
        self.hashes = self.read('inputs/graph_hashes.json')
        graph_path = self.root / 'inputs/graphs.pt'
        self.file_sha = self.files['inputs/graphs.pt']
        # Whole-container identity only. All decoded graph tensor reads go through Group.take.
        with graph_path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as blob:
            require(digest(blob) == self.file_sha, 'graph container checksum differs')
        self.mapped = torch.load(graph_path, mmap=True, weights_only=True, map_location='cpu')
        require(len(self.mapped) == len(self.hashes) and sorted(sum(self.domains.values(), []))
                == list(range(len(self.hashes))), 'R5 inventory or domain partition')
        self.setting = setting

    def read(self, relative):
        return member(self.root, relative, self.files[relative])


class Group:
    def __init__(self, archive, inventory, source):
        self.inventory = inventory
        self.source = source
        self.targets = [t for t in DOMAINS if t != source]
        self.roles = {}
        self.touched = set()
        forbidden = set()
        source_role = inventory.read(f'roles/M{source}/seed3405.json')
        self.roles['S'] = source_role['fit']
        self.roles['V'] = source_role['validation']
        require(sorted(self.roles['S'] + self.roles['V']) == sorted(inventory.domains[str(source)]),
                'source partition differs')
        setting = inventory.setting
        spec, = [s for s in archive.plan['real_settings'] if s['id'] == setting]
        for t in self.targets:
            m = archive.manifest(setting, source, t)
            p = m['provenance']
            require(Path(p['archive_root']) == inventory.root and
                    p['r5_files']['seals/R5_INPUT.json'] == inventory.seal_sha, 'R5 provenance')
            a, h = split_target_ids(archive.plan, spec['tu_dataset'], inventory.domains[str(t)])
            require(m['ids'] == dict(source=self.roles['S'], validation=self.roles['V'], adapt=a, holdout=h),
                    'R5 and R7 roles differ')
            self.roles[t] = a
            forbidden.update(h)
        granted = sum(self.roles.values(), [])
        require(len(granted) == len(set(granted)) and not (set(granted) & forbidden), 'role overlap')
        self.grants = {role: frozenset(ids) for role, ids in self.roles.items()}

    def take(self, role, i):
        require(i in self.grants[role] and i not in self.touched, 'ungranted or repeated row')
        self.touched.add(i)
        g = self.inventory.mapped[i]
        require(set(g) == {'x', 'edge_index', 'edge_attr'}, 'unexpected graph fields')
        actual = {k: None if v is None else digest(v.numpy().tobytes()) for k, v in g.items()}
        row = self.inventory.hashes[i]
        require(row['id'] == i and row['input_sha256'] == actual, 'individual graph hash')
        if 'nodes' in row:
            require(row['nodes'] == len(g['x']) and row['undirected_edges'] * 2 == g['edge_index'].shape[1],
                    'registered graph dimensions differ')
        return g


@dataclass
class Record:
    layers: list
    key: bytes


def encode(g, setting, old):
    node_width, edge_width = (3, 1) if setting == 'proteins_edge' else (14, 3)
    if setting == 'proteins_edge':
        require(g['edge_attr'] is None, 'expected untyped PROTEINS edges')
        g = dict(g, edge_attr=torch.ones((g['edge_index'].shape[1], 1), dtype=torch.int8))
    # The original R9 implementation checks one-hot values and symmetric, simple, typed arcs.
    baseline, n_edges = wl.fingerprints(g, dict(node_types=node_width, edge_types=edge_width))
    nodes = g['x'].argmax(dim=1).tolist()
    labels = g['edge_attr'].argmax(dim=1).tolist()
    adjacency = [[] for _ in nodes]
    for (s, t), label in zip(g['edge_index'].T.tolist(), labels):
        adjacency[s].append((label, t))
    pack = lambda x: struct.pack('>I', x)
    colors = [sha256(b'R9-WL0\0' + pack(n)).digest() for n in nodes]
    layers = [colors]
    for radius in (1, 2):
        next_colors = []
        for pos, neighbors in enumerate(adjacency):
            terms = sorted((edge, colors[target]) for edge, target in neighbors)
            payload = b''.join(pack(edge) + c for edge, c in terms)
            preimage = b'R9-WL\0' + pack(radius) + colors[pos] + pack(len(terms)) + payload
            next_colors.append(sha256(preimage).digest())
        colors = next_colors
        layers.append(colors)
    require([Counter(layer) for layer in layers] == baseline, 'R9 color counts differ')
    key = sha256(wl.count_key(baseline)).digest()
    if old is not None:
        require(n_edges == old['undirected_edges'] and key.hex() == old['group_sha256']
                and [{c.hex(): n for c, n in layer.items()} for layer in baseline] == old['raw_layer_counts'],
                'archived R9 edges/layer counts/group differ')
    return Record(layers, key)


def order(setting, ids):
    def key(i):
        require(type(i) is int, 'noninteger graph ID')
        return sha256(f'r9-local-v1|{setting}|{i}'.encode('utf-8')).digest(), i
    return sorted(ids, key=key)


class Index:
    def __init__(self, records, ids, r):
        # Store every distinct supplying fingerprint. Exclusion is relative to each query.
        self.keys = defaultdict(set)
        self.fingerprints = Counter(records[i].key for i in ids)
        for i in ids:
            record = records[i]
            for c in set(record.layers[r]):
                self.keys[c].add(record.key)

    def hits(self, eligible, fingerprint, dedup):
        return {c for c in eligible if c in self.keys
                and (not dedup or len(self.keys[c]) > 1 or fingerprint not in self.keys[c])}


def record_rates(rows):
    return {k: fmean(r[k] for r in rows) for k in rows[0]}


def flags(hits):
    out = {f'graph_any_{p}': bool(hits[p]) for p in POOLS}
    for p, q in PAIRS:
        out[f'graph_changed_{p}_{q}'] = bool(hits[p] ^ hits[q])
    return out


def analyze(group, archive_old):
    roles, ts = group.roles, group.targets
    setting, source = group.inventory.setting, group.source
    ordered = {t: order(setting, roles[t]) for t in ts}
    queries = {t: ids[::2] for t, ids in ordered.items()}
    references = {t: ids[1::2] for t, ids in ordered.items()}
    k = min(len(roles['V']), *(len(ids) for ids in references.values()))
    require(k > 0 and all(set(queries[t]).isdisjoint(references[t]) for t in ts), 'invalid Q/R split')
    chosen = {t: ref[:k] for t, ref in references.items()}
    chosen['V'] = order(setting, roles['V'])[:k]
    require(all(len(ids) == k for ids in chosen.values()), 'unequal reference budget')
    picks = {'S': roles['S'], 'V': chosen['V'], **{t: queries[t] + chosen[t] for t in ts}}
    old = archive_old if setting == 'mutagenicity_edge' and source == 0 else {}
    if old:
        require(set(map(int, old)) == set(roles['S']), 'R9 archived source scope differs')
    records = {i: encode(group.take(role, i), setting, old.get(str(i)) if role == 'S' else None)
               for role, ids in picks.items() for i in ids}
    require(group.touched == set(records) and len(records) == sum(map(len, picks.values())),
            'scope accounting differs')
    vocab = [{c for i in roles['S'] for c in records[i].layers[r]} for r in (0, 1, 2)]
    indexes = {r: {pool: Index(records, ids, r) for pool, ids in chosen.items()} for r in (1, 2)}
    rows, unions = [], []
    for t in ts:
        other = [f for f in ts if f != t]
        sources = dict(zip(POOLS, (t, 'V', *other)))
        base = dict(setting=setting, source=source, target=t, seed=3405, foreign1_domain=other[0],
                    foreign2_domain=other[1], n_source=len(roles['S']), n_validation_available=len(roles['V']),
                    n_target_a=len(roles[t]), n_queries=len(queries[t]), n_reference_available=len(references[t]),
                    n_foreign1_reference_available=len(references[other[0]]),
                    n_foreign2_reference_available=len(references[other[1]]), n_reference_each=k)
        combined = {i: {p: set() for p in POOLS} for i in queries[t]}
        for r in (1, 2):
            rates = []
            for i in queries[t]:
                rec = records[i]
                current, previous = rec.layers[r], rec.layers[r-1]
                unknown = [c not in vocab[r] for c in current]
                eligible = Counter(c for c, par, u in zip(current, previous, unknown) if u and par in vocab[r-1])
                n = len(current)
                sample = dict(oov=sum(unknown)/n, eligible=eligible.total()/n)
                for dedup in (False, True):
                    hits = {}
                    for p in POOLS:
                        box = indexes[r][sources[p]]
                        hits[p] = box.hits(eligible, rec.key, dedup)
                        field = f'hit_{p}' if dedup else f'raw_hit_{p}'
                        sample[field] = sum(eligible[c] for c in hits[p])/n
                        if dedup:
                            require(sample[field] <= sample[f'raw_hit_{p}'] <= sample['eligible']
                                    <= sample['oov'] <= 1, 'hit order')
                            combined[i][p].update((r, c) for c in hits[p])
                        # Directly replay the exclusion over candidate graph IDs on every affected query.
                        if dedup and rec.key in box.fingerprints:
                            direct = {c for j in chosen[sources[p]] if records[j].key != rec.key
                                      for c in records[j].layers[r] if c in eligible}
                            require(hits[p] == direct, 'query fingerprint exclusion differs')
                    if dedup:
                        sample.update(flags(hits))
                rates.append(sample)
            values = record_rates(rates)
            values['delta_local_validation'] = values['hit_local'] - values['hit_validation']
            values['delta_local_foreign1'] = values['hit_local'] - values['hit_foreign1']
            values['delta_local_foreign2'] = values['hit_local'] - values['hit_foreign2']
            values['delta_local_foreign_mean'] = (values['delta_local_foreign1'] + values['delta_local_foreign2']) / 2
            rows.append(dict(**base, radius=r, **values))
        unions.append(dict(**base, **record_rates([flags(hits) for hits in combined.values()])))
    receipt = dict(setting=setting, source=source, raw_graphs_read=len(records), n_s=len(roles['S']),
                   n_v_read=k, n_q_read=sum(map(len, queries.values())), n_r_read=len(ts)*k,
                   n_r9_archived_fingerprints=len(old),
                   n_authorized_total=sum(map(len, roles.values())), graph_archive_sha256=group.inventory.file_sha,
                   sha256_granted_role_ids={str(r): signature(ids) for r, ids in roles.items()},
                   sha256_graph_ids_read={str(r): signature(ids) for r, ids in picks.items()},
                   n_q_by_target={str(t):len(queries[t]) for t in ts}, k=k)
    return rows, unions, receipt


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
    from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.r5_inputs import split_target_ids
    torch.set_num_threads(1)
    archive = Archive(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    full, joint, receipts = [], [], []
    for setting in SETTINGS:
        inventory = Inventory(setting, archive.manifest(setting, 0, 1)['provenance'])
        for s in DOMAINS:
            group = Group(archive, inventory, s)
            historical = {}
            if setting == 'mutagenicity_edge' and s == 0:
                historical = member(args.repo / R9, 'results/attempt-001/fingerprints.json', R9_FINGERPRINTS)
                require(len(historical) == 868, 'R9 archived cohort size differs')
            rows, unions, checks = analyze(group, historical)
            full.extend(rows)
            joint.extend(unions)
            receipts.append(checks)
            print(setting, s, checks['raw_graphs_read'], checks['k'], flush=True)
    require(len(full) == 72 and len(joint) == 36, 'incomplete results')
    write_csv(args.output / 'directions-by-radius.csv', full)
    write_csv(args.output / 'directions-union.csv', joint)
    (args.output / 'checks.json').write_text(json.dumps(dict(r7_c_final=R7_FINAL,
        r9_archived_fingerprints_sha256=R9_FINGERPRINTS, groups=receipts), indent=2) + '\n')
