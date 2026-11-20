"""Independent per-node bitmask count with group-scoped authenticated graph reads."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
import argparse
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import audit


def check(expected, observed):
    for key, value in observed.items():
        if abs(float(expected[key]) - value) > 2e-14:
            raise ValueError((key, value, expected[key]))


def graph_rates(flags):
    result = {f'graph_any_{p}': bool(flags[idx]) for idx, p in enumerate(audit.POOLS)}
    for (a, p), (b, q) in combinations(enumerate(audit.POOLS), 2):
        result[f'graph_changed_{p}_{q}'] = bool(flags[a] ^ flags[b])
    return result


def bitmaps(records, pool, depth):
    occurrences, identical = defaultdict(int), defaultdict(int)
    for j, i in enumerate(pool):
        bit = 1 << j
        rec = records[i]
        identical[rec.key] |= bit
        for c in set(rec.layers[depth]):
            occurrences[c] |= bit
    return occurrences, identical


def check_group(grant, expected, joint):
    setting, source = grant.inventory.setting, grant.source
    def order(ids):
        return sorted(ids, key=lambda i: (hashlib.sha256(
            ('r9-local-v1|' + setting + '|' + str(i)).encode('utf-8')).hexdigest(), i))
    queries = {}
    references = {}
    for t in grant.targets:
        ordered = order(grant.roles[t])
        queries[t] = [i for j,i in enumerate(ordered) if j % 2 == 0]
        references[t] = [i for j,i in enumerate(ordered) if j % 2]
    k = min(len(grant.roles['V']), min(map(len, references.values())))
    selected = {t: p[:k] for t,p in references.items()}
    selected['V'] = order(grant.roles['V'])[:k]
    records = {}
    for role in grant.roles:
        if role == 'S':
            ids = grant.roles['S']
        elif role == 'V':
            ids = selected['V']
        else:
            ids = queries[role] + selected[role]
        for i in ids:
            records[i] = audit.encode(grant.take(role, i), setting, None)
    vocabulary = []
    for r in range(3):
        colors = set()
        for i in grant.roles['S']:
            colors.update(records[i].layers[r])
        vocabulary.append(colors)
    for target in grant.targets:
        c1, c2 = sorted(set(grant.targets) - {target})
        sources = [target, 'V', c1, c2]
        union = defaultdict(list)
        for r in (1,2):
            hits = [bitmaps(records, selected[src], r) for src in sources]
            by_metric = defaultdict(list)
            for i in queries[target]:
                rec = records[i]
                unknown, eligible = 0, 0
                raw = [0] * 4
                flags = [0] * 4
                for pos, cur in enumerate(rec.layers[r]):
                    if cur in vocabulary[r]:
                        continue
                    unknown += 1
                    if rec.layers[r-1][pos] not in vocabulary[r-1]:
                        continue
                    eligible += 1
                    for j,(found, matched) in enumerate(hits):
                        owners = found.get(cur, 0)
                        raw[j] += bool(owners)
                        if owners & ~matched.get(rec.key, 0):
                            flags[j] |= 1 << pos
                n = len(rec.layers[r])
                row = dict(oov=unknown/n, eligible=eligible/n, **graph_rates(flags))
                for j,p in enumerate(audit.POOLS):
                    row[f'hit_{p}'] = flags[j].bit_count()/n
                    row[f'raw_hit_{p}'] = raw[j]/n
                for name, value in row.items():
                    by_metric[name].append(value)
                union[i].append((flags, n))
            means = {name:fmean(values) for name, values in by_metric.items()}
            for name in audit.POOLS[1:]:
                means['delta_local_'+name]=means['hit_local'] - means['hit_'+name]
            means['delta_local_foreign_mean']=means['hit_local'] - .5*(means['hit_foreign1']+means['hit_foreign2'])
            e=expected[setting, source, target, r]
            check(e, dict(**means, n_reference_each=k, n_queries=len(queries[target]),
                          foreign1_domain=c1, foreign2_domain=c2))
        all_flags=defaultdict(list)
        for i, ((first,n), (second,m)) in union.items():
            assert n == m
            merged = [u | (v << n) for u,v in zip(first,second)]
            for name, value in graph_rates(merged).items():
                all_flags[name].append(value)
        check(joint[setting,source,target], {name:fmean(vals) for name,vals in all_flags.items()})
    return set(i for ids in queries.values() for i in ids), set(records)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    sys.path.insert(0,str(a.repo.resolve()))
    import torch
    from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
    from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.r5_inputs import split_target_ids
    audit.torch, audit.wl, audit.split_target_ids = torch, wl, split_target_ids
    torch.set_num_threads(1)
    c=audit.Archive(a.archive)
    with (a.output/'directions-by-radius.csv').open() as f:
        expected={(r['setting'], int(r['source']),int(r['target']),int(r['radius'])):r for r in csv.DictReader(f)}
    with (a.output/'directions-union.csv').open() as f:
        joint={(r['setting'],int(r['source']),int(r['target'])):r for r in csv.DictReader(f)}
    assert len(expected)==72 and len(joint)==36
    counts={}
    for setting in audit.SETTINGS:
        store=audit.Inventory(setting,c.manifest(setting,0,1)['provenance'])
        qs, selected=set(),set()
        for src in audit.DOMAINS:
            g=audit.Group(c,store,src)
            q, ids=check_group(g,expected,joint)
            qs.update(q)
            selected.update(ids)
        counts[setting]=dict(distinct_queries_within_setting=len(qs),distinct_graphs_read_within_setting=len(selected))
        print(setting, counts[setting], flush=True)
    files={name:audit.digest((a.output/name).read_bytes()) for name in
           ('directions-by-radius.csv','directions-union.csv','summary.csv','checks.json')}
    (a.output/'verification.json').write_text(json.dumps(dict(directions=36, radius_rows=72,
           tolerance=2e-14, alternate_count='node-position bitmaps and candidate row bitmaps',
           graph_counts=counts, files=files), indent=2)+'\n')
