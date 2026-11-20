"""Stand-alone blind R17 checker. No experiment implementation is imported."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, hashlib, io, json, math, struct, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, LinearOperator
import torch

out=Path(__file__).resolve().parent
repo=Path('/home/hxchen/research/GDA/TSBA/repository')
r9=next(repo.glob('experiments/mutagenicity/r9*'))/'results'
old=r9/'attempt-002-weighted-residuals'; ref=r9/'reference-geometry-001'
filehash=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def load_json(p):
 return json.loads(p.read_bytes())
def checked_json(p, h):
 assert filehash(p)==h
 return load_json(p)
def checked_old(root, seal, name):
 return checked_json(root/name,seal['files'][name])
def record(p,obj):
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def instant():
 return datetime.now(timezone.utc).isoformat()
def table(p,rows,columns):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,columns,delimiter='\t',lineterminator='\n'); w.writeheader();w.writerows(rows)

def feature_rows(g, feature_ids):
 assert set(g)=={'x','edge_index','edge_attr'}
 types=g['x'].numpy(); assert np.all(types.sum(1)==1) and np.all((types==0)|(types==1))
 t=types.argmax(1)
 edge=g['edge_index'].numpy().T
 et=np.zeros(len(edge),int) if g['edge_attr'] is None else g['edge_attr'].numpy().argmax(1)
 adj=[[] for _ in range(len(t))]
 for (a,b),e in zip(edge,et):
  adj[a].append((int(e),int(b)))
 take=[i for i,nei in enumerate(adj) if len({v for _,v in nei if v!=i})>=2]
 if not take: take=list(range(len(t)))
 def intern(k):
  return feature_ids.setdefault(k,len(feature_ids))
 colors=[intern((0,int(a))) for a in t]
 rows=[{},{}]
 for depth in range(3):
  for row,which in zip(rows,(colors,[colors[i] for i in take])):
   count=Counter(which); common=math.gcd(*count.values())
   count={k:c//common for k,c in count.items()}
   norm=math.sqrt(3*sum(c*c for c in count.values()))
   row.update({k:c/norm for k,c in count.items()})
  if depth<2:
   colors=[intern((depth+1,colors[i],tuple(sorted((e,colors[j]) for e,j in nei)))) for i,nei in enumerate(adj)]
 return tuple(rows)

def csr(rows,ids,col,view):
 data=[]; idx=[]; ptr=[0]
 for i in ids:
  for word,val in rows[i][view].items():
   if word in col:
    data.append(val);idx.append(col[word])
  ptr.append(len(data))
 ret=csr_matrix((data,idx,ptr),shape=(len(ids),len(col)),dtype=float);ret.sort_indices()
 return ret

def fit(X,y,XE,V,T):
 n,p=X.shape
 if min(y)==max(y):
  b=math.log((y.sum()+.5)/(n-y.sum()+.5))
  return np.full(XE.shape[0],b),dict(constant=b)
 V=V if V is not None and T>1e-14 else None
 def loss_grad(beta):
  w=beta[:-1]; score=X@w+beta[-1]; r=expit(score)-y
  grad=(X.T@r+w)/n
  val=np.logaddexp(0.,(1-2*y)*score).mean()+np.dot(w,w)/(2*n)
  if V is not None:
   z=V@w
   val+=np.dot(z,z)/(2*T);grad+=(V.T@z)/T
  return float(val),np.r_[grad,r.mean()]
 beta=np.zeros(p+1);beta[-1]=math.log(y.mean()/(1-y.mean()))
 opt=minimize(loss_grad,beta,method='L-BFGS-B',jac=True,options={'gtol':5e-12,'ftol':1e-15,'maxiter':1200,'maxls':50})
 beta=opt.x
 # Independent Newton polishing stabilizes decisions at small margins.
 for it in range(3):
  val,grad=loss_grad(beta)
  if np.linalg.norm(grad,np.inf)<1e-11: break
  eta=X@beta[:-1]+beta[-1]; curvature=expit(eta)*expit(-eta)
  def Hv(delta):
   z=curvature*(X@delta[:-1]+delta[-1])
   a=(X.T@z+delta[:-1])/n
   if V is not None: a+=V.T@(V@delta[:-1])/T
   return np.r_[a,z.mean()]
  h=LinearOperator((p+1,p+1),matvec=Hv,dtype=float)
  step,flag=cg(h,grad,rtol=1e-10,atol=1e-16,maxiter=2000)
  assert flag==0
  beta-=step
 val,grad=loss_grad(beta);res=float(np.linalg.norm(grad,np.inf))
 assert res<1e-8
 return XE@beta[:-1]+beta[-1],dict(max_gradient=res,objective=val)
def hashed(ids,m,setting,source):
 return sorted(ids,key=lambda g:(hashlib.sha256(f'r17-pair|{setting}|{source}|{g}'.encode()).digest(),g))[:m]
def diff(F,L):
 v=(F-L).tocsr(); v.eliminate_zeros()
 nonzero=int(np.diff(v.indptr).astype(bool).sum()); T=float(v.multiply(v).sum())
 return v,T,nonzero
def get_y(path,meta,ids,authorized):
 assert tuple(ids)==tuple(authorized)
 yy=[]
 with path.open('rb',buffering=0) as f:
  for i in ids:
   pos,length=meta['offsets'][i];f.seek(pos); x=f.read(length);assert len(x)==length
   yy.append(meta['mapping'][str(int(x))])
 return np.asarray(yy,dtype=int)

def main():
 start=instant(); elapsed=time.monotonic(); t0=elapsed
 old_obs=checked_json(old/'OBS.json','6298811e840521a07d4134131e02130240649e4ef6fdf71d5b204034ca039617')
 ref_obs=load_json(ref/'OBS.json')
 roles=checked_old(old,old_obs,'roles.json')
 assert roles==checked_old(ref,ref_obs,'roles.json')
 prov=checked_old(old,old_obs,'source-provenance.json')
 names=['WL1','TargetPair','SourcePair','Shuffled','EqualPositive','WL','Size','Base']
 rows=[]; margins=[]; blocks=[]; receipt={}; fits={}; groups={}
 for setting in sorted(prov):
  data=repo.parent/'experiments/r5'/setting/'attempt-001/inputs'
  check=prov[setting]
  assert filehash(data/'graphs.pt')==check['inputs/graphs.pt']
  graphs=torch.load(data/'graphs.pt',weights_only=True,mmap=True,map_location='cpu')
  meta=checked_json(data/'label_records.json',check['inputs/label_records.json'])
  path=Path(meta['path']); assert filehash(path)==meta['sha256']
  for source in range(4):
   tasks=sorted((k,r) for k,r in roles.items() if r['setting']==setting and r['source']==source)
   assert len(tasks)==2
   gids=sorted(set(g for _,r in tasks for group in ('D','U','E') for g in r['roles'][group]))
   feature_ids={}; feats={g:feature_rows(graphs[g],feature_ids) for g in gids}
   for key,r in tasks:
    assert r['status']=='ok' and r['seed']==3405
    D,U,E=[r['roles'][p] for p in ('D','U','E')]
    assert len(set(D+U+E))==len(D)+len(U)+len(E)
    y=get_y(path,meta,D,roles[key]['roles']['D'])
    receipt[key]={'D':D}
    m=min(len(D),len(U)); dr=hashed(D,m,setting,source); ur=hashed(U,m,setting,source)
    col={j:i for i,j in enumerate(sorted(set(key for i in D for key in feats[i][0])))}
    X=csr(feats,D,col,0);XE=csr(feats,E,col,0)
    FF=csr(feats,ur,col,0); LL=csr(feats,ur,col,1)
    penalties={'WL1':(None,0,0),'TargetPair':diff(FF,LL),'SourcePair':diff(csr(feats,dr,col,0),csr(feats,dr,col,1)),'Shuffled':diff(FF,LL[np.roll(np.arange(m),-1)])}
    scores={}; fits[key]={'m':m,'D_reference_ids':dr,'U_reference_ids':ur,'fitted':{}}
    for name,(V,T,nz) in penalties.items():
     scores[name],opt=fit(X,y,XE,V,T); fits[key]['fitted'][name]=dict(T=T,nonzero_rows=nz,**opt)
    relative=f'predictions/{key}.npz'
    assert filehash(ref/relative)==ref_obs['files'][relative] and filehash(old/relative)==old_obs['files'][relative]
    with np.load(ref/relative,allow_pickle=False) as rr:
     np.testing.assert_array_equal(rr['e_ids'],E)
     for name in ('WL','Size','Base'):
      scores[name]=rr['margin_'+name]; np.testing.assert_array_equal(scores[name]>0,rr[name])
    with np.load(old/relative,allow_pickle=False) as oo:
     np.testing.assert_array_equal(oo['e_ids'],E);np.testing.assert_array_equal(scores['WL1']>0,oo['WL'])
    actions={k:(sc>0).astype(int) for k,sc in scores.items()}
    order=np.lexsort((E,-scores['WL1'])); equal=np.zeros(len(E),int);equal[order[:sum(actions['TargetPair'])]]=1;actions['EqualPositive']=equal
    focal=actions['TargetPair']
    blocks.append(dict(task=key,n=len(E),m=m,**{name:int(np.count_nonzero(focal!=actions[name])) for name in names if name!='TargetPair'}))
    for j,g in enumerate(E):
     idcols={'task':key,'graph_id':g}
     rows.append(dict(**idcols,**{name:int(actions[name][j]) for name in names}))
     margins.append(dict(**idcols,**{name:float(scores[name][j]) for name in scores}))
    print(instant(),key,'E',len(E),'action bounds',blocks[-1],flush=True)
    groups.setdefault(setting,[]).append(blocks[-1])
  del graphs
 sort=lambda r:(r['task'],r['graph_id'])
 cols=['task','graph_id']
 table(out/'actions.tsv',sorted(rows,key=sort),cols+names)
 table(out/'margins.tsv',sorted(margins,key=sort),cols+[name for name in names if name!='EqualPositive'])
 table(out/'bounds.tsv',sorted(blocks,key=lambda r:r['task']),list(blocks[0]))
 summary={}
 for setting,bs in groups.items():
  summary[setting]={name:dict(macro_upper_pp=sum(100*b[name]/b['n'] for b in bs)/len(bs),maximum_win_directions=sum(b[name]>0 for b in bs),maximum_positive_historical_sources=sum(any(b[name]>0 for b in bs if f'-M{source}-' in b['task']) for source in range(4))) for name in names if name!='TargetPair'}
 record(out/'summary.json',summary);record(out/'label-access.json',receipt);record(out/'checks.json',fits)
 files=['independent.py','actions.tsv','bounds.tsv','margins.tsv','summary.json','label-access.json','checks.json']
 record(out/'FROZEN.json',dict(start_utc=start,ended_utc=instant(),elapsed_seconds=time.monotonic()-t0,role_file_sha256=old_obs['files']['roles.json'],observations_old=filehash(old/'OBS.json'),observations_reference=filehash(ref/'OBS.json'),canonical_actions_schema=cols+names,sort_order='task (ASCII lexicographic), graph_id (integer ascending); TSV header included; LF UTF-8',scope='D role only for active task; no E labels decoded; two settings can share historical IDs',total_graph_predictions=len(rows),tasks=len(roles),files={f:filehash(out/f) for f in files}))
 print((out/'FROZEN.json').read_text(),flush=True)
if __name__=='__main__':
 main()
