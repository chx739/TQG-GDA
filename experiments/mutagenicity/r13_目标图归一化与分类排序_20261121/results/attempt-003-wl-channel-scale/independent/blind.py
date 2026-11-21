"""Blind action-capacity screen. No data category files or evaluations are read."""
import hashlib
import io
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

repo = Path('/home/hxchen/research/GDA/TSBA/repository')
out = Path(__file__).parent
sys.path.insert(0, str(repo))
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl

def read_json(path):
    return json.loads(path.read_bytes())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(name, value):
    (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def hashed(pool, m, key):
    return sorted(pool, key=lambda gid: (hashlib.sha256(f'channel-v1|{key}|{gid}'.encode()).digest(), gid))[:m]

def adjust(h, base, donor_sd, reference):
    sd = reference.std(axis=0)
    ratio = np.divide(donor_sd, sd, out=np.ones_like(sd), where=(donor_sd > 1e-12) & (sd > 1e-12))
    delta = (reference - reference.mean(axis=0)) @ (ratio-1)
    assert abs(delta.mean()) <= 1e-12
    return base + (h-reference.mean(axis=0)) @ (ratio-1), ratio

def matched(ids, static, positive):
    v = np.zeros(len(ids), dtype=bool)
    v[np.lexsort((ids,-static))[:int(positive)]] = True
    return v

def counts(x):
    return int(np.count_nonzero(x))

def main():
    root = next(repo.glob('experiments/mutagenicity/r9_*'))
    locked = root/'results/feature-dropout-transfer-001'
    files = read_json(locked/'OBS.json')['files']
    for name in ['roles.json','executed-config.json']:
        assert digest(locked/name) == files[name]
    roles = read_json(locked/'roles.json')
    cfg = read_json(locked/'executed-config.json')['plan']
    original = root/'results/feature-dropout-screen/frozen-run.zip'
    assert digest(original) == cfg['zip_sha256']
    assert len(roles)==24
    outrows = []
    hard = {}
    torch.set_num_threads(1)
    with ZipFile(original) as archive:
        for setting in ('mutagenicity_edge','mutagenicity_node','proteins_edge'):
            r5 = repo.parent/'experiments/r5'/setting/'attempt-001'
            graph_file = r5/'inputs/graphs.pt'
            assert digest(graph_file) == read_json(r5/'seals/R5_INPUT.json')['files']['inputs/graphs.pt']
            graphs = torch.load(graph_file, mmap=True, weights_only=True, map_location='cpu')
            scoped = {key:row for key,row in roles.items() if row['setting']==setting}
            ids = {i for row in scoped.values() for name in ('E','D') for i in row['roles'][name]}
            ids.update(i for row in scoped.values() for name in ('target_reference','source_reference') for i in row[name])
            features = {}
            for i in sorted(ids):
                g = dict(graphs[i])
                assert set(g)=={'x','edge_index','edge_attr'}
                if setting=='proteins_edge':
                    assert g['edge_attr'] is None
                    g['edge_attr'] = torch.ones((g['edge_index'].shape[1],1),dtype=g['x'].dtype)
                sizes = (3,1) if setting=='proteins_edge' else (14,3)
                color,_ = wl.fingerprints(g, dict(node_types=sizes[0],edge_types=sizes[1]))
                features[i] = wl.normalize(color)
            for key,row in sorted(scoped.items()):
                model_name = f'models/{key}.npz'
                assert digest(locked/model_name)==files[model_name]
                with np.load(locked/model_name,allow_pickle=False) as model:
                    w = model['Static']
                    # ORIGINAL = ('Target', 'Static', 'Source', 'Perm').
                    bias = float(model['intercepts'][1])
                d,c,u,e = (row['roles'][i] for i in ('D','C','U','E'))
                assert len(set(d)|set(c)|set(u)|set(e))==sum(map(len,(d,c,u,e)))
                m = row['m']
                dr = hashed(d,m,key)
                tr,sr = row['target_reference'],row['source_reference']
                assert len(dr)==len(tr)==len(sr)==m and set(tr)<=set(u) and set(sr)<=set(c)
                lex = wl.vocabulary(features,d)
                assert len(lex)==len(w)
                radii = np.array([rad for rad,word in lex])
                pool = dr+tr+sr+e
                x = wl.sparse_rows(features,pool,lex)
                h = np.stack([x[:,radii==rad]@w[radii==rad] for rad in range(3)],axis=1)
                e_ids = np.asarray(e)
                h_d,h_u,h_c,h_e = np.split(h,[m,2*m,3*m])
                check = h_e.sum(axis=1)+bias
                with np.load(io.BytesIO(archive.read(f'{key}.npz')), allow_pickle=False) as old:
                    np.testing.assert_array_equal(e_ids,old['e'])
                    base = old['Static']
                err = float(np.max(np.abs(check-base)))
                assert err<=cfg['absolute_replay_tolerance'] and np.array_equal(check>0,base>0)
                z,ratio = adjust(h_e, base, h_d.std(axis=0), h_u)
                p,source_ratio = adjust(h_e, base, h_d.std(axis=0), h_c)
                static,action,placebo = base>0,z>0,p>0
                eq = matched(e_ids,base,action.sum())
                assert np.isfinite(z).all() and np.isfinite(p).all() and eq.sum()==action.sum()
                hard[key] = np.column_stack([e_ids,base, z,p,eq.astype(int)])
                outrows.append(dict(key=key,setting=setting,source=row['source'],direction=row['direction'],
                                    m=m,n=len(e),original_error=err,ratio=ratio.tolist(),source_ratio=source_ratio.tolist(),
                                    D_reference=dr,target_n_pos=counts(action),static_n_pos=counts(static),
                                    placebo_n_pos=counts(placebo),vs_static=counts(static!=action),
                                    vs_matched=counts(eq!=action),vs_placebo=counts(placebo!=action)))
    np.savez_compressed(out/'decisions.npz', **hard)
    dump('directions.json',outrows)
    summary = {}
    for setting in sorted({r['setting'] for r in outrows}):
        rows = [r for r in outrows if r['setting']==setting]
        assert len(rows)==8
        report = {}
        for field in ('vs_static','vs_matched','vs_placebo'):
            by_source={str(i):sum(r[field] for r in rows if r['source']==i) for i in range(4)}
            f = sum(Fraction(r[field],r['n']) for r in rows)/len(rows)
            report[field]=dict(count=sum(by_source.values()), source_counts=by_source, source_coverage=sum(v>0 for v in by_source.values()), 
                               directions=sum(r[field]>0 for r in rows), direction_equal_pct=100*float(f), exact_fraction=str(f))
        coverage = report['vs_matched']['source_coverage']
        report.update(n=sum(r['n'] for r in rows), max_replay_error=max(r['original_error'] for r in rows),
                      eligible_for_later_evaluation=(Fraction(report['vs_matched']['exact_fraction'])>=Fraction(1,200) and coverage>=3))
        summary[setting]=report
    dump('summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
