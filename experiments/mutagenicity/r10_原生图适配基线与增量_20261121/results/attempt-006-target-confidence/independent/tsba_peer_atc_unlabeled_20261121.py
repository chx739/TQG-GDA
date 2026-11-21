from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from types import SimpleNamespace
import io, json, numpy as np, torch
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
D=Path('experiments/mutagenicity'); r12=next(D.glob('r12*/results/attempt-002-conditional-votes')); r11=next(D.glob('r11*/results/attempt-003-roc')); r16=next(D.glob('r16*/results/attempt-001')); ext=Path('/home/hxchen/research/GDA/TSBA/experiments')
def j(path): return json.loads(path.read_bytes())
def ref(path,sha):
 b=path.read_bytes(); assert sha256(b).hexdigest()==sha;return b
def read(root,key,m): return ref(root/key,m[key])
seal=j(r12/'OBS.json')['files'];roles=json.loads(read(r12,'roles.json',seal));count=json.loads(read(r12,'source-counts.json',seal)); access=j(ext/'r10/attempt-003-structural/fit-access.json')
@lru_cache
def graph_archive(setting):
 p=ext/'r5'/setting/'attempt-001/inputs/graphs.pt';sha=sha256(p.read_bytes()).hexdigest(); assert set(row['graph_container_sha256'] for key,row in access.items() if key.startswith(setting+'-'))=={sha}
 return torch.load(p,mmap=True,weights_only=True,map_location='cpu')
@lru_cache(None)
def feature(setting,gid):
 obj=graph_archive(setting)[gid]; assert set(obj)=={'x','edge_index','edge_attr'}
 return structural.features(SimpleNamespace(**obj),setting)
def estimated(v,u,c,original=False):
 n=len(v)
 if c in (0,n): return Fraction(int(c),n)
 if original:
  cut=np.sort(v)[n-c-1]; return Fraction(int((u>=cut).sum()),len(u))
 cut=np.sort(v)[-c]; partial=Fraction(int(c)-int((v>cut).sum()),int((v==cut).sum()))
 assert int((v>cut).sum())+partial*int((v==cut).sum())==c
 return (int((u>cut).sum())+partial*int((u==cut).sum()))/len(u)
bits=np.array([[(i>>1)&1,i&1,(i>>2)&1] for i in range(8)])
stats={}
for k,r in roles.items():
 group=r['group'];s=r['setting'];S=count[group];V=np.asarray(S['counts']).reshape(2,8);correct=V[0]@(1-bits)+V[1]@bits
 vf=json.loads(ref(r16/'fits'/f'{group}.json',S['r16_fit_member_sha256'])); assert vf['validation_ids']==S['v_ids']; v=np.array(vf['validation_margins'])
 assert np.array_equal(np.bincount(((v>0)@np.array([2,1,4])),minlength=8),V.sum(0))
 model=json.loads(ref(ext/'r10/attempt-003-structural/fits'/f'{group}.json',S['structure_sha256']))
 key=f'predictions/{k}.npz'; ad,hd,ab=np.load(io.BytesIO(read(r12,key,seal))),r['holdout_ids'],r['selected_a_ids']; np.testing.assert_array_equal(ab,ad['a_ids']); np.testing.assert_array_equal(hd,ad['ids']); assert set(ab).isdisjoint(hd)
 ro=np.load(io.BytesIO(ref(r11/'deployment'/f'{k}.npz',r['evidence']['r11_sha256']))); z=structural.predict(model,[feature(s,gid) for gid in ab]); np.testing.assert_array_equal(z['Size'],ro['a_size_margin']); u=np.stack([ro['a_margin'],z['WL_selected'],z['Size']],axis=1)
 np.testing.assert_array_equal((u>0)@np.array([2,1,4]),ad['a_bins'])
 src=max(range(3),key=lambda i:(correct[i],-i)); anch=2 if s=='proteins_edge' else 1
 pr=tuple(estimated(np.abs(v[:,i]),np.abs(u[:,i]),correct[i]) for i in range(3)); orig=tuple(estimated(np.abs(v[:,i]),np.abs(u[:,i]),correct[i],True) for i in range(3))
 sel=max(range(3),key=lambda i:(pr[i],correct[i],-i)); other=max(range(3),key=lambda i:(orig[i],correct[i],-i))
 stats.setdefault(s,[]).append((r['source'],r['target'],sel,src,anch,int((bits[ad['h_bins'],sel]!=bits[ad['h_bins'],anch]).sum()),len(hd),other,tuple(map(float,pr))))
for s,rows in stats.items():
 print(s,'orders by source:', [[r[2] for r in sorted(rows) if r[0]==i] for i in range(4)],'B0/WL/Size',np.bincount([r[2] for r in rows],minlength=3).tolist(),'changed vs anchor',sum(r[5]>0 for r in rows),'vs source select',sum(r[2]!=r[3] for r in rows),'diff orig code',sum(r[2]!=r[7] for r in rows),'anchor action frac',np.mean([r[5]/r[6] for r in rows]),flush=True)
 print('ATC by direction',[(r[0],r[1],np.round(r[-1],3).tolist()) for r in sorted(rows)])
