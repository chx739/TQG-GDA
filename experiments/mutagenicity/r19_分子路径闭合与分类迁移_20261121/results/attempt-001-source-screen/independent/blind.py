import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scipy.linalg import solve

repo = Path('/home/hxchen/research/GDA/TSBA/repository')
sys.path.insert(0, str(repo))
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl

out = Path(__file__).resolve().parent
base = repo/'experiments/mutagenicity'
old = next(base.glob('r9_*/results/feature-dropout-screen'))
structure = next(base.glob('r19_*/results/structure-screen/structure-run.zip'))
role_file = next(base.glob('r9_*/results/reference-geometry-001/roles.json'))
root = repo.parent/'experiments/r5/mutagenicity_edge/attempt-001/inputs'
rawmeta = json.loads((root/'label_records.json').read_bytes())
def sha(x):
    return hashlib.sha256(x).hexdigest()

with ZipFile(structure) as z:
    rows = list(csv.DictReader(io.StringIO(z.read('per_graph_counts.csv').decode())))
    structure_meta = json.loads(z.read('inputs.json'))
assert structure_meta['R9_roles'] == sha(role_file.read_bytes())
byid = {int(row['graph']): row for row in rows}
assert len(byid) == len(rows)
counts = {
    what: {g: np.array([int(row[f'{what}{j}'])/int(row['nodes']) for j in range(4)]) for g,row in byid.items()}
    for what in ('C','O')
}
def ring(ids):
    return [np.stack([counts[what][g] for g in ids]) for what in ('C','O')]

def ratio(C, O, fallback):
    return np.divide(C,O,out=fallback.copy(),where=O>0)

def rate(ids, fallback):
    c,o = ring(ids)
    return ratio(c.sum(0),o.sum(0),fallback)

def ref(key, group, ids, m):
    return sorted(ids,key=lambda g: hashlib.sha256(f'ring-r19|{key}|{group}|{g}'.encode()).digest())[:m]

def fit_ridge(feature, response):
    rms = np.sqrt(np.mean(feature**2,axis=0))
    scale = np.divide(1, rms*np.sqrt(feature.shape[1]), out=np.zeros_like(rms), where=rms>0)
    A = feature * scale
    w = np.linalg.lstsq(np.vstack([A, np.eye(A.shape[1])]),np.r_[response,np.zeros(A.shape[1])],rcond=None)[0]
    return w*scale, scale, w

def mix_bg(ps,pu,D,yd,uref):
    dC, dO = ring(D)
    if len(np.unique(yd)) == 1:
        return ps, None
    means = np.stack([np.stack([arr[yd==j].mean(0) for j in (0,1)]) for arr in (dC,dO)])
    full = rate(D,np.zeros(4))
    def h(p):
        cm,om = means[:,0,:]*(1-p) + means[:,1,:]*p
        return ratio(cm,om,full)
    hp = h(yd.mean())
    opp = ring(uref)[1].sum(0)
    scores = []
    for pi in range(1001):
        p = np.clip(ps+(h(pi/1000)-hp),0,1)
        scores.append((float(opp@((p-pu)**2)),pi,p))
    _,pi,bg=min(scores,key=lambda x:x[:2])
    return bg, pi

def src(role, name):
    assert name in ('D','C') and len(set(i for ids in role.values() for i in ids))==sum(map(len,role.values()))
    result = []
    with Path(rawmeta['path']).open('rb',buffering=0) as f:
        for gid in role[name]:
            pos,n = rawmeta['offsets'][gid]
            f.seek(pos)
            result.append(rawmeta['mapping'][str(int(f.read(n)))])
    return np.array(result,dtype=np.int64)

def base_ridge(XD, XC, XE, yd, yc):
    # Free intercept via centering the Gram matrices on the training set.
    K = (XD @ XD.T).toarray()
    J = K - K.mean(0)[None,:] - K.mean(1)[:,None] + K.mean()
    def cent(Q):
        Kq = (Q @ XD.T).toarray()
        return Kq - K.mean(0)[None,:] - Kq.mean(1)[:,None] + K.mean()
    C,E = cent(XC),cent(XE)
    states = []
    for lam in (0.01,0.1,1.,10.):
        coef = solve(J+lam*np.eye(len(K)),yd-yd.mean(),assume_a='pos')
        z = C @ coef + yd.mean()
        crit = (-np.count_nonzero((z>0)==(yc>0)),np.mean((z-yc)**2),-lam)
        states.append((crit,z,coef))
    crit,zc,coef = min(states,key=lambda state:state[0])
    return -crit[2],zc,E@coef + yd.mean()

roles = {k:r for k,r in json.loads(role_file.read_bytes()).items() if r['setting']=='mutagenicity_edge'}
archive = torch.load(root/'graphs.pt',map_location='cpu',weights_only=True)
assert all(set(g)=={'x','edge_attr','edge_index'} for g in archive)
manifest = json.loads((old/'freeze.json').read_bytes())
summary = {}
csvrows=[]
fits={}
for key,row in roles.items():
    role=row['roles']
    d,c,u,e=[role[k] for k in ('D','C','U','E')]
    ids = [*d,*c,*e]
    feats = {g:wl.normalize(wl.fingerprints(archive[g],dict(node_types=14,edge_types=3))[0]) for g in ids}
    lex = wl.vocabulary(feats,d)
    xd,xc,xe = (wl.sparse_rows(feats,part,lex) for part in (d,c,e))
    yd,yc=src(role,'D'),src(role,'C')
    lam,fc,fe=base_ridge(xd,xc,xe,2*yd-1,2*yc-1)
    with ZipFile(old/'frozen-run.zip') as z:
        raw=z.read(f'{key}.npz')
    assert sha(raw)==manifest['rows'][key]['prediction_sha']
    with np.load(io.BytesIO(raw)) as z:
        assert np.array_equal(z['e'],e)
        replay_err=float(np.max(abs(fe-z['Static'])))
        assert replay_err<1.e-10 and np.array_equal(fe>0,z['Static']>0)
    assert lam == manifest['rows'][key]['params']['Static']
    m=min(len(d),len(u),len(c))
    sr,ur = ref(key,'D',d,m),ref(key,'U',u,m)
    ps=rate(sr,rate(d,np.zeros(4)))
    pu=rate(ur,ps)
    pm,mil=mix_bg(ps,pu,d,yd,ur)
    cc,oc=ring(c)
    ce,oe=ring(e)
    response=(2*yc-1)-fc
    slope,scale,w=fit_ridge(cc-ps*oc,response)
    raw_slope,raw_scale,raw_w=fit_ridge(np.column_stack([cc,oc]),response)
    z={
        'Static':fe,
        'Bias':fe+response.mean(),
        'Source':fe+(ce-ps*oe)@slope,
        'Target':fe+(ce-pu*oe)@slope,
        'Raw':fe+np.column_stack([ce,oe])@raw_slope,
        'Mixture':fe+(ce-pm*oe)@slope
    }
    np.testing.assert_allclose(z['Target']-z['Source'],(oe*(ps-pu))@slope,rtol=1e-12,atol=4e-16)
    assert np.array_equal(z['Target'][oe.sum(1)==0],fe[oe.sum(1)==0])
    action={method:(values>0).astype(int) for method,values in z.items()}
    count=np.zeros(len(e),int)
    count[np.lexsort((e,-z['Source']))[:action['Target'].sum()]]=1
    action['Count']=count
    for j, gid in enumerate(e):
        csvrows.append({'key':key,'id':gid, **{f'z_{k}':float(v[j]) for k,v in z.items()},
                        **{f'a_{k}':int(v[j]) for k,v in action.items()}})
    summary[key]={'E':len(e),'m':m,'lambda':lam,'replay_err':replay_err,
                  'main_pos':int(action['Target'].sum()),
                  'main_diff':{m:int((action['Target']!=v).sum()) for m,v in action.items()}}
    fits[key]=dict(source_ref=sr,target_ref=ur,ps=ps.tolist(),pu=pu.tolist(),pm=pm.tolist(),pi_milli=mil,
                   scale=scale.tolist(),w=w.tolist(),raw_scale=raw_scale.tolist(),raw_w=raw_w.tolist())
    print(key, summary[key],flush=True)

with (out/'decisions.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]))
    writer.writeheader()
    writer.writerows(csvrows)
(out/'fits.json').write_text(json.dumps(fits,indent=2))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'manifest.json').write_text(json.dumps({p:sha((out/p).read_bytes()) for p in ['blind.py','decisions.csv','fits.json','summary.json']},indent=2))
