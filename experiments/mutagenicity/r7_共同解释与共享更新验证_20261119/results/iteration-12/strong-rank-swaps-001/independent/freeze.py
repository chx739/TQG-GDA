"""Freeze label-blind R7 swap membership with all references and anchor counts."""
from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

out = Path(__file__).resolve().parent
repo = Path('/home/hxchen/research/GDA/TSBA/repository')
root = repo / 'experiments/mutagenicity'
def one(pattern):
    item, = root.glob(pattern)
    return item

raw = {
    'cross':one('r7_*/results/iteration-12/reference-head-crossover-001'),
    'center':one('r7_*/results/iteration-12/centered-target-update-001'),
    'votes':one('r12_*/results/attempt-002-conditional-votes'),
    'r10':Path('/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-003-structural'),
    'r11':one('r11_*/results/attempt-001'),
}
def digest(blob):
    return sha256(blob).hexdigest()
def jd(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
seals = {k:json.loads((d/'OBS.json').read_bytes()) for k,d in raw.items()}
seal_hash = {k:digest((d/'OBS.json').read_bytes()) for k,d in raw.items()}
for k in ['r10','r11']:
    assert seal_hash[k] == seals['cross']['predecessors'][k+'_obs_sha256']
access = {k:{} for k in raw}
def member(k,name):
    b=(raw[k]/name).read_bytes()
    h=digest(b)
    assert h==seals[k]['files'][name],(k,name)
    access[k][name]=h
    return b
roles = {k:json.loads(member(k,'roles.json')) for k in raw}
groups = json.loads(member('cross','groups.json'))
settings = ('mutagenicity_edge','mutagenicity_node','proteins_edge')
def block_id(g,s,t,seed):
    return f'{g}-M{s}_to_M{t}-seed-{seed}'
grid=[(g,s,t,seed) for g in settings for s in range(4) for t in range(4) if s!=t for seed in range(3405,3410)]
for k in raw:
    keys={block_id(*x) for x in grid if k not in ('center','votes') or x[3]==3405}
    assert set(roles[k])==keys
    assert {f'predictions/{i}.npz' for i in keys}=={i for i in seals[k]['files'] if i.startswith('predictions/')}
def pred(k,name,members):
    with np.load(BytesIO(member(k,'predictions/'+name+'.npz')),allow_pickle=False) as z:
        return {n:z[n] for n in members}
def check_action(a,n):
    assert a.shape==(n,) and a.dtype.kind in 'biu' and np.isin(a,[0,1]).all()
def binary_arrays(k,name,fields,ids):
    p=pred(k,name,['ids',*fields])
    np.testing.assert_array_equal(p.pop('ids'),ids)
    for a in p.values():
        check_action(a,len(ids))
    return p
pool={}
complete_counts=Counter()
incomplete=[]
blks=[]
graph_fields=['setting','source','target','seed','reference','own','complete','valid_head','anchor','graph_id','e','count','anchor_action','stratum']
bins=['count_0_to_e_1_anchor_count','count_1_to_e_0_anchor_count','count_0_to_e_1_anchor_e','count_1_to_e_0_anchor_e']
with (out/'graphs.csv').open('w',newline='') as f:
    gw=csv.writer(f,lineterminator='\n')
    gw.writerow(graph_fields)
    for setting,s,t,seed in grid:
        k=block_id(setting,s,t,seed)
        rec=roles['cross'][k]
        inds=rec['holdout_ids']
        ids=np.array(inds)
        adapt=rec['adapt_ids']
        assert len(set(inds+adapt))==len(inds)+len(adapt)
        assert pool.setdefault((setting,t),(inds,adapt))==(inds,adapt)
        relevant=['cross','r10','r11']+(['center','votes'] if seed==3405 else [])
        for name in relevant:
            r=roles[name][k]
            assert (r['setting'],r['source'],r['target'],r['seed'])==(setting,s,t,seed)
            assert (r['manifest_sha256'],r['holdout_ids'])==(rec['manifest_sha256'],inds)
        for name in ('r10','r11'):
            assert rec[name+'_prediction_sha256']==seals[name]['files']['predictions/'+k+'.npz']
        grp=groups[rec['group']]
        complete=all(head['valid'] for target in grp['status'].values() for head in target.values())
        assert rec['complete']==grp['complete']==complete
        if complete:
            complete_counts[setting]+=1
        else:
            incomplete.append(k)
        refs=sorted(set(range(4))-{s})
        names=[tag for r in refs for tag in (f'E_A{r}',f'E_count_A{r}',f'margin_E_A{r}')]
        z=pred('cross',k,['ids','margin_B0','B0','WL','Size','SourceV',*names])
        np.testing.assert_array_equal(z['ids'],ids)
        n=len(ids)
        margin=z['margin_B0']
        assert margin.shape==(n,) and np.isfinite(margin).all()
        order=np.lexsort((ids,-margin))
        np.testing.assert_array_equal(z['B0'],margin>0)
        struct=binary_arrays('r10',k,['action_B0','action_WL_selected','action_Size'],ids)
        v=binary_arrays('r11',k,['B0','SourceV'],ids)
        for x,y in (('B0','action_B0'),('WL','action_WL_selected'),('Size','action_Size')):
            np.testing.assert_array_equal(z[x],struct[y])
        for x in ('B0','SourceV'):
            np.testing.assert_array_equal(z[x],v[x])
        strong='Size' if setting=='proteins_edge' else 'WL'
        anchors={name:z[name] for name in (strong,'SourceV')}
        if seed==3405:
            others=['Static','B2','DirectP','PoolMajority']
            vote=binary_arrays('votes',k,['B0','WL','Size',*others],ids)
            for x in ('B0','WL','Size'):
                np.testing.assert_array_equal(z[x],vote[x])
            c_name=f'{strong}_Centered_A{t}'
            cc=binary_arrays('center',k,['WL','Size',c_name],ids)
            for x in ('WL','Size'):
                np.testing.assert_array_equal(z[x],cc[x])
            anchors.update({x:vote[x] for x in others},Centered=cc[c_name])
        for r in refs:
            e=z[f'E_A{r}']
            q=z[f'E_count_A{r}']
            check_action(e,n)
            check_action(q,n)
            er=rec['head_records'][f'E_A{r}']
            valid=grp['status'][str(r)]['E']['valid']
            assert er['reference']==r and er['valid']==valid and er['positives']==int(e.sum()) and er['fallback']==(not valid)
            np.testing.assert_array_equal(e,z[f'margin_E_A{r}']>0)
            if not valid:
                np.testing.assert_array_equal(e,z['B0'])
            eq=np.zeros(n,dtype=int)
            eq[order[:int(e.sum())]]=1
            np.testing.assert_array_equal(q,eq)
            swap=np.flatnonzero(e!=q)
            for anchor,a in anchors.items():
                counts=[0,0,0,0]
                key=[setting,s,t,seed,r,int(r==t),int(complete),int(valid),anchor]
                for i in swap:
                    equal_count=bool(a[i]==q[i])
                    assert equal_count!=bool(a[i]==e[i])
                    counts[(0 if equal_count else 2)+int(q[i])] += 1
                    gw.writerow([*key,int(ids[i]),int(e[i]),int(q[i]),int(a[i]),'anchor_count' if equal_count else 'anchor_e'])
                assert sum(counts)==len(swap)
                assert counts[0]+counts[2]==counts[1]+counts[3]
                blks.append([*key,n,int(e.sum()),len(swap),*counts])
for g in settings:
    flat=[i for t in range(4) for p in pool[g,t] for i in p]
    assert len(set(flat))==len(flat)
assert complete_counts == dict(zip(settings,(60,60,57)))
with (out/'blocks.csv').open('w',newline='') as f:
    w=csv.writer(f,lineterminator='\n'); w.writerow([*graph_fields[:9],'n','e_positive','n_swap',*bins]);w.writerows(blks)
jd(out/'inputs.json',{k:{'root':str(raw[k]),'obs_sha256':seal_hash[k],'read_members':v} for k,v in access.items()})
jd(out/'design.json',dict(complete_by_setting=dict(complete_counts),incomplete_blocks=incomplete,all_blocks=len(grid),directions=36,seeds=list(range(3405,3410)),scope='historical H pools: 180 block tasks; retain full and original E/J complete flag; per eligible block consider each of the 3 non-source reference E heads; compare each E to its own positive-count B0 ranking.',primary='own=1 and complete=1; anchor WL for mutagenicity and Size for proteins. SourceV sensitivity; additional seed 3405 anchors use same heldout tasks and own-reference Centered.',membership='Per nonidentical graph/action E versus count, anchor_count iff anchor equals count; anchor_e iff anchor equals E. Direction names count_0_to_e_1 etc. Graph IDs repeated across seed/source/reference/anchor are separate occurrences. One block row per anchor per reference including zero swaps.',test='Recompute count via B0 margin descending and ID ascending. Validate every input digest against its historical OBS, current grid, task/role manifest, ordered graph IDs and action members including original R10/R11. Only historical OBS, roles/groups, and named frozen prediction members accessed.'))
artifacts=['freeze.py','inputs.json','design.json','blocks.csv','graphs.csv']
with (out/'SHA256SUMS').open('w') as f:
    for name in artifacts:
        f.write(digest((out/name).read_bytes())+'  '+name+'\n')
print('SHA256SUMS',digest((out/'SHA256SUMS').read_bytes()),'block_reference_anchor_rows',len(blks),'complete',complete_counts)
