from pathlib import Path
import csv
from fractions import Fraction
import hashlib
import io
import json
from zipfile import ZipFile

import numpy as np

out = Path(__file__).absolute().parent
repo = Path('/home/hxchen/research/GDA/TSBA/repository')
results = repo/'experiments/mutagenicity'
original = next(results.glob('r9_*/results/feature-dropout-transfer-001'))
counts_zip = next(results.glob('r19_*/results/structure-screen/structure-run.zip'))
frozen = Path('/tmp/r19-independent-2c31050-E8WsOh')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_checked(path, expected):
    payload = path.read_bytes()
    assert digest(payload)==expected, path
    return payload

cert = json.loads(read_checked(frozen/'manifest.json','8a3d44caa222d9078a5ca834dcd683d3c896d475951898f58c7e8d16e1a26df2'))
fit = json.loads(read_checked(frozen/'fits.json',cert['fits.json']))
registry = json.loads(read_checked(original/'OBS.json','b1bb43490443d5af0ed3d060dbf9844eeece7e5de564be18d8f189d849441862'))
roles = json.loads(read_checked(original/'roles.json',registry['files']['roles.json']))
with ZipFile(counts_zip) as z:
    raw = z.read('per_graph_counts.csv')
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
counts_sha=digest(raw)
o = {int(row['graph']):[int(row[f'O{j}'])/int(row['nodes']) for j in range(4)] for row in rows}
assert len(o)==len(rows)
all_rows,detail,seen,role_audit=[],[],set(),{}
for key,model in fit.items():
    row = roles[key]
    stage = row['roles']
    assert row['setting']=='mutagenicity_edge'
    role_sets = {name:set(gids) for name,gids in stage.items()}
    assert sum(map(len,stage.values()))==len(set.union(*role_sets.values()))
    assert set(row['supplement'])==role_sets['U']-set(row['target_reference'])
    r19ref = model['target_ref']
    assert len(r19ref)==len(set(r19ref)) and set(r19ref)<=role_sets['U']
    assert set(model['source_ref'])<=role_sets['D']
    def sample(name):
        order=sorted(stage[name],key=lambda i:hashlib.sha256(f'ring-r19|{key}|{name}|{i}'.encode()).digest())
        return order[:min(len(stage[p]) for p in ('U','D','C'))]
    assert r19ref==sample('U') and model['source_ref']==sample('D')
    member=f'predictions/{key}.npz'
    with np.load(io.BytesIO(read_checked(original/member,registry['files'][member]))) as z:
        gids,z0=z['ids'],z['margin_Static']
        assert np.array_equal(gids,np.array(row['supplement'])) and np.array_equal(z0>0,z['Static'])
        mask = ~np.isin(gids,r19ref)
        ids,base=gids[mask],z0[mask]
    excluded=set.union(set(row['target_reference']),set(r19ref))
    assert set(ids)==role_sets['U']-excluded and not set(ids)&set.union(role_sets['E'],role_sets['C'],role_sets['D'])
    assert not seen.intersection(ids)
    seen.update(ids)
    slope=np.asarray(model['scale'])*np.asarray(model['w'])
    delta_coeff=(np.asarray(model['ps'])-np.asarray(model['pu']))*slope
    delta=np.array([o[i] for i in ids])@delta_coeff
    bias=float((np.array([o[i] for i in r19ref])@delta_coeff).mean())
    main=(base+delta)>0
    count=np.zeros(len(base),dtype=bool)
    count[np.lexsort((ids,-base))[:main.sum()]]=True
    offset=(base+bias)>0
    static=base>0
    actions={'static':static,'count':count,'offset':offset}
    nr={name:int((a!=main).sum()) for name,a in actions.items()}
    assert nr['count']%2==0
    role_audit[key]={'r19_ref':r19ref,'R9_supplement':row['supplement'],'evaluation':ids.tolist()}
    all_rows.append(dict(key=key,source=row['source'],direction=row['direction'],U=len(stage['U']),
                         R9_supp=len(gids),R19_ref=len(r19ref),removed=int((~mask).sum()),n=len(ids),
                         static_positive=int(static.sum()),delta_positive=int(main.sum()),intercept=bias,
                         offset_positive=int(offset.sum()),**{f'diff_{k}':v for k,v in nr.items()}))
    for i,g in enumerate(ids):
        detail.append(dict(key=key,id=int(g),static_z=float(base[i]),delta=float(delta[i]),offset_value=bias,
                           main=int(main[i]),**{k:int(a[i]) for k,a in actions.items()}))

def table(path,data):
    with (out/path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=data[0])
        w.writeheader()
        w.writerows(data)

table('directions.csv',all_rows)
table('unlabeled-decisions.csv',detail)
(out/'roles.json').write_text(json.dumps(role_audit,separators=(',',':')))
agg={}
for k in actions:
    deltas=[r[f'diff_{k}'] for r in all_rows]
    equal=sum(Fraction(v,r['n']) for v,r in zip(deltas,all_rows))/len(all_rows)
    sources=sorted({r['source'] for r,v in zip(all_rows,deltas) if v})
    agg[k]=dict(n=sum(deltas),directional_bound_exact=str(equal),pp=float(100*equal),
                n_directions=sum(v>0 for v in deltas),sources=sources)
pass_gate=(Fraction(agg['count']['directional_bound_exact'])>=Fraction(5,1000) and len(agg['count']['sources'])>=3)
(out/'summary.json').write_text(json.dumps(dict(total=len(seen),results=agg,gate_pass=pass_gate),indent=2))
files=['capacity.py','directions.csv','unlabeled-decisions.csv','roles.json','summary.json']
(out/'manifest.json').write_text(json.dumps(dict(files={name:digest((out/name).read_bytes()) for name in files},
     ring_counts=counts_sha,original_r9_obs=digest((original/'OBS.json').read_bytes()),frozen_r19=cert['fits.json']),indent=2))
print((out/'directions.csv').read_text(),(out/'summary.json').read_text(),digest((out/'manifest.json').read_bytes()))
