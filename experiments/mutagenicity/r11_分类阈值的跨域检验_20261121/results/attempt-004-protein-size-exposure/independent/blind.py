"""Independent R11 audit. All data value reads are granted graph-only, B0 or S training values."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import torch

REPO = Path('/home/hxchen/research/GDA/TSBA/repository')
EXP = REPO / 'experiments/mutagenicity/r11_分类阈值的跨域检验_20261121'
R7 = Path('/home/hxchen/research/GDA/TSBA/experiments/r7/attempt-016/C')
R5 = Path('/home/hxchen/research/GDA/TSBA/experiments/r5/proteins_edge/attempt-001')
RAW_INDICATOR = Path('/home/hxchen/research/GDA/TQG/data/PROTEINS_official_20261023/PROTEINS/raw/PROTEINS_graph_indicator.txt')
OUT = Path(__file__).parent
SETTING = 'proteins_edge'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def checked(path, digest):
    blob = path.read_bytes()
    assert sha(blob) == digest, str(path)
    return blob

def fit(s_nodes, y):
    pos = sum(int(n) for n, label in zip(s_nodes, y, strict=True) if label == 1)
    neg = Counter(int(n) for n, label in zip(s_nodes, y, strict=True) if label == 0)
    if not pos:
        return math.inf
    if not neg:
        return 0.0
    def derivative(a):
        return math.fsum(int(count)*n/math.expm1(a*n) if a*n < 700 else 0.0 for n, count in neg.items())-pos
    upper = 1.0/float(max(s_nodes))
    while derivative(upper) > 0:
        upper *= 2
    return brentq(derivative, 1e-20, upper, xtol=5e-324, rtol=4*np.finfo(float).eps)

def b(nodes, lam):
    return np.exp(-lam*np.asarray(nodes, dtype=float))

def query(a_values, h_values, q):
    """Exhaust threshold options in exact rational distance on sorted tied scores."""
    counts = Counter(a_values.tolist())
    levels = sorted(counts)
    options = [(0, math.inf), (len(a_values), -math.inf)]
    remaining = len(a_values)
    for cut in levels[:-1]:
        remaining -= counts[cut]
        options.append((remaining, cut))
    k, cut = min(options, key=lambda item: (abs(Fraction(item[0],len(a_values))-q), item[0]))
    assert np.count_nonzero(a_values>cut) == k
    return h_values>cut, cut, k

def bits(v):
    return ''.join('1' if b else '0' for b in v)

def seq(ids):
    return '|'.join(str(x) for x in ids)

def save(name, records):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

prior_root = EXP / 'results/attempt-002-prior'
prior_cfg = json.loads((EXP/'configs/roc-rate-transfer.json').read_bytes())
unlabeled = json.loads(checked(prior_root/'UNLABELED.json', prior_cfg['prior_unlabeled_sha256']))
roles = json.loads(checked(prior_root/'roles.json', unlabeled['files']['roles.json']))
m7 = {stage:json.loads((R7/'seals'/f'R7_{stage}.json').read_bytes()) for stage in ['INPUT','OBS']}
for stage, record in m7.items():
    core = {k:record[k] for k in ('stage','parent','files')}
    assert sha(json.dumps(core,sort_keys=True,separators=(',',':'),allow_nan=False).encode()) == record['digest']
assert m7['OBS']['parent'] == m7['INPUT']['digest']
m5raw = (R5/'seals/R5_INPUT.json').read_bytes()
m5 = json.loads(m5raw)
graph_file = 'inputs/graphs.pt'
graphs = torch.load(io.BytesIO(checked(R5/graph_file,m5['files'][graph_file])), weights_only=True, map_location='cpu')
assert all(set(graph) == {'x','edge_index','edge_attr'} for graph in graphs)
nodes = np.array([graph['x'].shape[0] for graph in graphs], dtype=np.int64)
raw = np.loadtxt(RAW_INDICATOR, dtype=np.int64)
np.testing.assert_array_equal(nodes, np.bincount(raw)[1:])
assert np.all(nodes > 0)
frows, brows = [], []
for source in range(4):
    for seed in range(3405,3410):
        groups = {}
        train_ids = None
        labels = None
        for target in range(4):
            if target == source:
                continue
            key = f'{SETTING}-M{source}_to_M{target}-seed-{seed}'
            role = roles[key]
            prefix = 'blocks/'+key+'/'
            def read7(name, stage='INPUT'):
                path = prefix+name
                return checked(R7/path,m7[stage]['files'][path])
            manifest_data = read7('manifest.json')
            assert sha(manifest_data) == role['manifest_sha256']
            manifest = json.loads(manifest_data)
            ids = manifest['ids']
            assert ids['holdout'] == role['holdout_ids']
            prov = manifest['provenance']
            assert Path(prov['archive_root']) == R5 and prov['r5_files']['seals/R5_INPUT.json'] == sha(m5raw)
            a = role['selected_a_ids']
            assert len(a) == len(set(a)) and set(a) <= set(ids['adapt']) and set(a).isdisjoint(ids['holdout'])
            source_file = f'source_labels/M{source}/seed{seed}_fit.pt'
            assert prov['r5_files'][source_file] == m5['files'][source_file]
            if labels is None:
                labels = torch.load(io.BytesIO(checked(R5/source_file,prov['r5_files'][source_file])), weights_only=True, map_location='cpu').numpy()
                train_ids = ids['source']
                assert len(train_ids) == len(labels) and len(set(train_ids)) == len(labels) and np.isin(labels,[0,1]).all()
            assert train_ids == ids['source']
            assert set(train_ids).isdisjoint(a+ids['holdout'])
            def margins(group, orig, saved):
                with np.load(io.BytesIO(read7(orig)), allow_pickle=False) as original:
                    assert original[group+'_ids'].tolist() == ids[group]
                    z0 = original[group+'_z0']
                    if group == 'adapt':
                        assert original['source_ids'].tolist() == train_ids
                        np.testing.assert_array_equal(labels,original['source_y'])
                with np.load(io.BytesIO(read7(saved,'OBS')), allow_pickle=False) as frozen:
                    logits = frozen[group+'_logits_B0']
                assert logits.dtype == np.float64 and logits.shape == (len(ids[group]),2)
                values = logits[:,1]-logits[:,0]
                np.testing.assert_allclose(values,z0[:,1]-z0[:,0],rtol=0,atol=1e-12)
                return values
            full_a = margins('adapt','fit_input.npz','tau-0.005-observed.npz')
            h = margins('holdout','holdout_input.npz','tau-0.005-heldout.npz')
            groups[target] = {'a_ids':a,'h_ids':ids['holdout'],'h':h,'full_a_ids':ids['adapt'],
                'a':np.asarray([full_a[ids['adapt'].index(gid)] for gid in a]),'key':key,'other':role['other_targets']}
        budget = min(len(group['full_a_ids']) for group in groups.values())
        source_nodes = nodes[train_ids]
        lam = fit(source_nodes,labels)
        n1 = int(labels.sum())
        frows.append(dict(source=source,seed=seed,n=len(labels),n0=len(labels)-n1,n1=n1,
                          nodes_min=min(source_nodes),nodes_max=max(source_nodes),lambda_mle=lam,
                          source_ids_sha256=sha(seq(train_ids).encode())))
        for group in groups.values():
            assert group['a_ids'] == sorted(group['full_a_ids'], key=lambda n: (hashlib.sha256(f'r11-v1|{SETTING}|{n}'.encode()).digest(),n))[:budget]
            p = float(b(nodes[group['a_ids']],lam).mean())
            group['q'] = Fraction.from_float(p)
        for target, group in groups.items():
            source_majority = 2*n1 > len(labels)
            h_ids = group['h_ids']
            row = dict(key=group['key'],source=source,target=target,seed=seed,lambda_mle=lam,a_ids=seq(group['a_ids']),
                       h_ids=seq(h_ids),h_n=len(h_ids),b0=bits(group['h']>0),
                       source_majority=bits([source_majority]*len(h_ids)),single=bits(b(nodes[h_ids],lam)>0.5))
            assert group['other'] == sorted(set(groups)-{target})
            for idx, donor in enumerate([target,*group['other']]):
                prefix = ['own','other1','other2'][idx]
                supplied = groups[donor]
                row[prefix+'_target'] = donor
                row[prefix+'_p'] = float(supplied['q'])
                row[prefix+'_p_fraction'] = str(supplied['q'])
                row[prefix+'_pool'] = bits([supplied['q'] > Fraction(1,2)]*len(h_ids))
                action, cut, count = query(group['a'],group['h'],supplied['q'])
                row[prefix+'_prior_direct'] = bits(action)
                row[prefix+'_prior_cut'] = cut
                row[prefix+'_prior_a_count'] = count
                action, cut, count = query(supplied['a'],group['h'],group['q'])
                row[prefix+'_quantile_direct'] = bits(action)
                row[prefix+'_quantile_cut'] = cut
                row[prefix+'_quantile_a_count'] = count
            assert row['own_prior_direct'] == row['own_quantile_direct']
            brows.append(row)
assert len(frows) == 20 and len(brows) == 60
save('blind-fits.csv',frows)
save('blind-actions.csv',brows)
for name in ['blind.py','blind-fits.csv','blind-actions.csv']:
    print(sha((OUT/name).read_bytes()),name)
print('60 blocks; 20 fits;',sum(row['h_n'] for row in brows),'graph appearances')
