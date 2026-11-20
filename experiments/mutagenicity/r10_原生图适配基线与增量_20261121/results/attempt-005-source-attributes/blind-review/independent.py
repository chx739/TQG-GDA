"""Independent blind reconstruction of the eight source-local PROTEINS tasks.

PYTHONDONTWRITEBYTECODE=1 and one BLAS thread are recommended. No E labels.
"""
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from scipy.linalg import solve
from scipy.special import expit, logit
from torch_geometric.data import Data

REPO = Path('/home/hxchen/research/GDA/TSBA/repository')
EXPERIMENTS = REPO / 'experiments/mutagenicity'
sys.path.insert(0, str(REPO))
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import residual_inputs, wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src.structural import features as structural_features

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
R9 = next(EXPERIMENTS.glob('r9_*'))
OLD = R9 / 'results/reference-geometry-001'
ORIGINAL = R9 / 'results/attempt-002-weighted-residuals'
ARMS = ('WL','Size','Base','Attribute')
STRENGTHS = (.1, 1., 10., 100.)

def sealed(blob, digest):
    if hashlib.sha256(blob).hexdigest() != digest:
        raise AssertionError('Input digest changed')
    return blob

def member(folder, manifest, rel):
    return sealed((folder/rel).read_bytes(), manifest['files'][rel])

def read_raw(plan):
    root = Path(plan['raw_directory'])
    blobs = {}
    for suffix in ('A','graph_indicator','node_labels','node_attributes'):
        name = f'PROTEINS_{suffix}.txt'
        blobs[suffix] = sealed((root/name).read_bytes(),plan['files'][name]['sha256'])
    graph_ids = np.fromstring(blobs['graph_indicator'].decode(),sep=' ',dtype=int)-1
    assert graph_ids.min()==0 and graph_ids.max()==1112
    assert not (np.diff(graph_ids)<0).any()
    types = blobs['node_labels'].splitlines()
    attrs = blobs['node_attributes'].splitlines()
    assert len(graph_ids)==len(types)==len(attrs)
    return graph_ids, types, attrs

def attributes(gid, encoded, table):
    graph_ids, node_types, attributes_ = table
    first, stop = np.searchsorted(graph_ids,[gid,gid+1])
    kinds = np.array([int(i) for i in node_types[first:stop]])
    values = np.array([float(i) for i in attributes_[first:stop]])
    assert np.isfinite(values).all() and set(kinds) <= set(range(3))
    assert np.array_equal(encoded.numpy(), np.eye(3)[kinds])
    def get_medians(k, v):
        return np.array([statistics.median(v[k==t]) if (k==t).any() else np.nan for t in range(3)])
    med = get_medians(kinds,values)
    assert np.array_equal(med,get_medians(kinds[::-1],values[::-1]),equal_nan=True)
    perm = values.copy()
    for t in range(3):
        perm[kinds==t] = perm[kinds==t][::-1]
    assert np.array_equal(med,get_medians(kinds,perm),equal_nan=True)
    return med

def design(archive, row, table):
    ids = sum((row['roles'][r] for r in ('D','C','E')),[])
    n = len(row['roles']['D'])
    rows, sizes, attrs = {}, [], []
    for g in ids:
        raw = archive.graphs[g]
        assert set(raw)=={'x','edge_index','edge_attr'}
        hashes={k:None if x is None else hashlib.sha256(x.numpy().tobytes()).hexdigest() for k,x in raw.items()}
        assert archive.graph_hashes[g]['id']==g and hashes == archive.graph_hashes[g]['input_sha256']
        rows[g], z = structural_features(Data(**raw),'proteins_edge')
        sizes.append(z)
        attrs.append(attributes(g,raw['x'],table))
    vocab = wl.vocabulary(rows,ids[:n])
    x = wl.sparse_rows(rows,ids,vocab)
    size = np.asarray(sizes)
    sd = size[:n].std(axis=0)
    sd[np.ptp(size[:n],axis=0)==0] = 1.
    size = (size-size[:n].mean(axis=0))/sd
    a = np.asarray(attrs)
    scaled = np.zeros_like(a)
    for j in range(a.shape[1]):
        train = a[:n,j][np.isfinite(a[:n,j])]
        supplied = np.isfinite(a[:,j])
        if train.size and np.ptp(train)>0:
            scaled[supplied,j] = (a[supplied,j] - train.mean())/train.std(ddof=0)
    k_wl = (x@x[:n].T).toarray()
    k_size = size @ size[:n].T
    base = k_wl + k_size
    return dict(WL=k_wl,Size=k_size,Base=base,Attribute=base+scaled@scaled[:n].T)

def optimum(k,y,c):
    # Dual IRLS in the row span of D. At the solution, alpha=C(y-p) and sum(alpha)=0.
    n = len(y)
    alpha = np.zeros(n)
    b = logit(y.mean())
    def objective(a,bias):
        proj = k@a
        return np.logaddexp(0,(1-2*y)*(proj+bias)).sum() + (a@proj)/(2*c)
    for _ in range(100):
        margin = k@alpha+b
        p=expit(margin)
        curvature=c*p*(1-p)
        matrix=np.eye(n)+curvature[:,None]*k
        vals=np.column_stack([c*(y-p)+curvature*margin,curvature])
        v,t = solve(matrix,vals,check_finite=True).T
        new_b = v.sum()/t.sum()
        new_alpha = v-t*new_b
        da,db = new_alpha-alpha,new_b-b
        if max(abs(k@da+db))<1e-10:
            alpha,b=new_alpha,new_b
            break
        step,previous = 1.,objective(alpha,b)
        while objective(alpha+step*da,b+step*db)>previous+5e-11:
            step*=0.5
            if step<1e-8:
                raise ArithmeticError('IRLS line search')
        alpha+=step*da
        b+=step*db
    else:
        raise ArithmeticError('IRLS failed')
    residual=c*(y-expit(k@alpha+b))
    assert max(abs(residual.sum()), max(abs(alpha-residual))) < 5e-8
    return alpha,b

def fit_predict(k,y,val):
    n,nc=len(y),len(val)
    if np.unique(y).size==1:
        return None,np.full(len(k)-n-nc, logit((y.sum()+.5)/(n+1)))
    candidates = {}
    for strength in STRENGTHS:
        a,b = optimum(k[:n],y,strength)
        cal = k[n:n+nc]@a+b
        errors = np.count_nonzero((cal>0)!=val)
        nll=np.logaddexp(0.,(1-2*val)*cal).mean()
        candidates[(errors,nll,strength)]=(a,b)
    choice=min(candidates)
    a,b=candidates[choice]
    return choice[-1], k[n+nc:]@a+b

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original=json.loads(sealed((ORIGINAL/'OBS.json').read_bytes(),'6298811e840521a07d4134131e02130240649e4ef6fdf71d5b204034ca039617'))
    old=json.loads(sealed((OLD/'OBS.json').read_bytes(),'823ff737f381767bf445c034e18983b2eed8e817e7095fe2bfbf82a2ea1dbee2'))
    roles=json.loads(member(ORIGINAL,original,'roles.json'))
    assert roles==json.loads(member(OLD,old,'roles.json'))
    roles={k:r for k,r in roles.items() if r['setting']=='proteins_edge'}
    assert len(roles)==8
    cfg=json.loads((R9/'configs/weighted-source-residuals.json').read_bytes())
    archive = residual_inputs.Archive(REPO,cfg,load_graphs=False).by_setting['proteins_edge']
    provenance=json.loads(member(ORIGINAL,original,'source-provenance.json'))['proteins_edge']
    graph_path=archive._member('inputs/graphs.pt','INPUT')
    with graph_path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==provenance['inputs/graphs.pt']
    archive.graphs=torch.load(graph_path,mmap=True,weights_only=True,map_location='cpu')
    archive.graph_hashes=json.loads(archive._read('inputs/graph_hashes.json'))
    assert archive.receipt['inputs/graph_hashes.json']==provenance['inputs/graph_hashes.json']
    table=read_raw(archive.plan['input'])
    decisions,diagnostics,all_margins=[],{},{}
    with residual_inputs.ScopedLabels({'proteins_edge':archive},roles,'fit') as port:
        for key in sorted(roles):
            row=roles[key]
            assert row['seed']==3405 and row['status']=='ok'
            e=row['roles']['E']
            kernels=design(archive,row,table)
            with port.direction(key):
                y,c=(port.load(key,r,row['roles'][r]) for r in ('D','C'))
                results={arm:fit_predict(kernels[arm],y,c) for arm in ARMS}
                del y,c
            selected={arm:v[0] for arm,v in results.items()}
            margins={arm:v[1] for arm,v in results.items()}
            acts={arm:(v>0).astype(np.int64) for arm,v in margins.items()}
            model=json.loads(member(OLD,old,f'models/{key}.json'))
            max_delta={}
            with np.load(io.BytesIO(member(OLD,old,f'predictions/{key}.npz')),allow_pickle=False) as history:
                assert np.array_equal(e,history['e_ids'])
                for arm in ARMS[:-1]:
                    expected=model['candidates'][arm]['selected_C'] if model['status']=='ok' else None
                    assert selected[arm]==expected and np.array_equal(acts[arm],history[arm])
                    max_delta[arm]=float(np.max(np.abs(margins[arm]-history[f'margin_{arm}'])))
            counts={arm:int(np.count_nonzero(acts[arm]!=acts['Attribute'])) for arm in ARMS[:-1]}
            diagnostics[key]=dict(n=len(e),selected_C=selected,disagreements=counts,baseline_margin_max_delta=max_delta)
            for i in np.argsort(e):
                decisions.append('\t'.join(map(str,[key,row['source'],e[i],*(acts[arm][i] for arm in ARMS)]))+'\n')
            all_margins[key]={'graph_id':e,'margins':{arm:val.tolist() for arm,val in margins.items()}}
            print(key,len(e),selected,counts,flush=True)
        residual_inputs.validate_receipt(roles,port.actual,'fit')
        (OUT/'fit-access.json').write_text(json.dumps(port.actual,separators=(',',':'))+'\n')
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (OUT/'margins.json').write_text(json.dumps(all_margins,separators=(',',':'))+'\n')
    actions=''.join(decisions).encode('ascii')
    (OUT/'actions.tsv').write_bytes(actions)
    print('SHA-256', hashlib.sha256(actions).hexdigest(), 'records',len(decisions),flush=True)
if __name__=='__main__':
    main()
