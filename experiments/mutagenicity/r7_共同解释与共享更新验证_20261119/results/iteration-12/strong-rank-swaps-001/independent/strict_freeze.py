"""Strict positive-count preservation from frozen swap membership, without labels."""
from collections import defaultdict,Counter
import csv
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

out=Path(__file__).resolve().parent
def digest(name): return sha256((out/name).read_bytes()).hexdigest()
primary_hash='c5bba74caa786b93a9c4e19afd4c54eb7fe9f823e9412f64d4d8fe2e871f10e3'
assert digest('SHA256SUMS')==primary_hash
for line in (out/'SHA256SUMS').read_text().splitlines():
    h,name=line.split()
    assert digest(name)==h
def read(name):
    with (out/name).open() as f:
        return list(csv.DictReader(f))
def write(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
def group(r):
    return r['setting'],int(r['source']),int(r['target']),int(r['seed']),r['anchor']
def blockref(r):
    return (*group(r),int(r['reference']))
blocks=read('blocks.csv')
graphs=read('graphs.csv')
assert len({blockref(r) for r in blocks})==len(blocks)
glookup=defaultdict(dict)
for r in graphs:
    key=blockref(r)
    gid=int(r['graph_id'])
    assert gid not in glookup[key]
    glookup[key][gid]=r
triples=defaultdict(dict)
def bern(k,n):
    return F(k,n) if n else F(0)
for r in blocks:
    key=blockref(r)
    n_in=int(r['count_0_to_e_1_anchor_count'])
    n_out=int(r['count_1_to_e_0_anchor_count'])
    eligible=Counter(int(g['count']) for g in glookup[key].values() if int(g['count'])==int(g['anchor_action']))
    assert eligible[0]==n_in and eligible[1]==n_out
    m=min(n_in,n_out)
    triples[group(r)][key[-1]]=m
    r.update(n_in=n_in,n_out=n_out,m=m)
for r in blocks:
    b=group(r)
    assert set(triples[b])==set(range(4))-{b[1]}
    L=min(triples[b].values())
    r['L']=L
    for k in ('m','L'):
        for side in ('in','out'):
            r[f'p_{side}_{k}']=bern(r[k],r[f'n_{side}'])
        r[f'bound_{k}_pp']=float(F(200*r[k],int(r['n'])))
write('strict-blocks.csv',blocks)
lookup={blockref(r):r for r in blocks}
for r in graphs:
    s=lookup[blockref(r)]
    side='in' if int(r['count'])==0 else 'out'
    for k in ('m','L'):
        r[f'select_p_{k}']=s[f'p_{side}_{k}'] if r['stratum']=='anchor_count' else F(0)
write('strict-graphs.csv',graphs)

# Two-stage directional mean, including zero switches. "other" averages both
# off-diagonal donors inside the same H block. Signs of contrast coefficients
# follow the archived anchor action for every particular graph in that block.
dirvalues=defaultdict(list)
for b,donors in triples.items():
    setting,s,t,seed,anchor=b
    row=lookup[(*b,t)]
    oth=sorted(set(donors)-{t})
    n=int(row['n'])
    for budget in ('m','L'):
        probs={}
        for d in donors:
            rr=lookup[(*b,d)]
            probs[d]={gid:g[f'select_p_{budget}'] * (1-2*int(g['anchor_action'])) for gid,g in glookup[(*b,d)].items()}
        coeffs={
            'own':probs[t],
            'other_mean':{gid:mean(probs[d].get(gid,0) for d in oth) for gid in set().union(*(probs[d] for d in oth))}}
        coeffs['own_minus_other_mean']={gid:coeffs['own'].get(gid,0)-coeffs['other_mean'].get(gid,0)
                                        for gid in coeffs['own'].keys()|coeffs['other_mean'].keys()}
        for con,coef in coeffs.items():
            value=sum(map(abs,coef.values()),F(0))/n
            if con!='own_minus_other_mean':
                donorlist=[t] if con=='own' else oth
                assert value == mean(F(2*lookup[(*b,d)][budget],n) for d in donorlist)
            cohorts=['all']
            if int(row['complete']): cohorts.append('complete')
            if seed==3405: cohorts.append('seed3405')
            for coh in cohorts:
                dirvalues[coh,setting,s,t,anchor,budget,con].append((value,coef,n))
drows=[]
summaries=defaultdict(list)
for (coh,g,s,t,a,budget,con),vals in sorted(dirvalues.items()):
    coeff=defaultdict(F)
    for value,cs,n in vals:
        for gid,v in cs.items():
            coeff[gid]+=v/(n*len(vals))
    peak=sum(map(abs,coeff.values()),F(0))
    abs_mean=mean(v[0] for v in vals)
    assert peak<=abs_mean
    drows.append(dict(cohort=coh,setting=g,source=s,target=t,anchor=a,budget=budget,contrast=con,
        blocks=len(vals),nonzero_blocks=sum(v[0]>0 for v in vals),avg_block_bound_pp=float(100*abs_mean),
        direction_shared_graph_bound_pp=float(100*peak)))
    summaries[coh,g,a,budget,con].append((s,t,len(vals),sum(v[0]>0 for v in vals),abs_mean,coef_copy:=dict(coeff),peak))
write('strict-directions.csv',drows)
srows=[]
for (coh,g,a,budget,con),dirs in sorted(summaries.items()):
    assert len(dirs)==12
    coefficients=defaultdict(F)
    for s,t,n,nz,ab,co,peak in dirs:
        for gid,v in co.items():
            # In each setting a heldout graph has a single target domain.
            coefficients[t,gid]+=v/len(dirs)
    safe_bound=mean(d[4] for d in dirs)
    dir_bound=mean(d[6] for d in dirs)
    peak=sum(map(abs,coefficients.values()),F(0))
    assert peak<=dir_bound<=safe_bound
    srows.append(dict(cohort=coh,setting=g,anchor=a,budget=budget,contrast=con,blocks=sum(d[2] for d in dirs),
                      nonzero_blocks=sum(d[3] for d in dirs),directions=len(dirs),
                      nonzero_directions=sum(d[4]>0 for d in dirs),
                      avg_block_bound_pp=float(100*safe_bound),avg_direction_shared_graph_bound_pp=float(100*dir_bound),
                      setting_shared_graph_bound_pp=float(100*peak)))
write('strict-summary.csv',srows)
note={'input_sha256sums':primary_hash,'unit':'Expected Accuracy change in percentage points for a graph-uniform draw within its eligible sign stratum. Every draw flips exactly k base 0s to 1 and k base 1s to 0. All graphs outside the drawn subsets keep their archived anchor action. No random draws or labels used.',
      'direction':'count_0_to_e_1_anchor_count is incoming; count_1_to_e_0_anchor_count is outgoing.',
      'budgets':'m=min(n_in,n_out) per block, donor and anchor; L=min of the three m for that exact block and same anchor. Empty side receives selection probability 0. Both strict-graphs probabilities vanish when anchor agrees with E.',
      'aggregation':'all=180 scheduled H blocks; complete=177 C-complete blocks; seed3405=36 scheduled blocks. The five extra anchors only exist for seed 3405. All summaries average over donors, then seeds within directed transfer, then directions equally. nonzero counts mean strictly positive label-free expectation bound.',
      'bounds':'own and other_mean avg_block_bound_pp equal averages of 200m/N or 200L/N. For own minus off-diagonal mean, sum graphwise absolute signed selection-probability differences per block. Shared-graph bounds combine signed coefficients across seeds/blocks before taking absolute values. These are label-free worst-case bounds, not achieved scores.'}
(out/'strict-design.json').write_text(json.dumps(note,indent=2)+'\n')
with (out/'STRICT_SHA256SUMS').open('w') as f:
    for name in ['SHA256SUMS','strict_freeze.py','strict-design.json','strict-blocks.csv','strict-graphs.csv','strict-directions.csv','strict-summary.csv']:
        f.write(digest(name)+'  '+name+'\n')
print('STRICT_SHA256SUMS',digest('STRICT_SHA256SUMS'))
