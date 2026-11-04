"""Independent raw-JSONL audit of the frozen A2-b/A6 aggregate gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np


def read(path):return json.loads(Path(path).read_text())
def stream(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def main(root):
    started=time.process_time();summary=read(root/'results_summary.json');assert read(root/'audit.json')['passed']
    maxerror=0.;checks=0
    def equal(a,b):
        nonlocal maxerror,checks
        checks+=1
        if a is None or b is None:assert a is b;return
        error=abs(float(a)-float(b));maxerror=max(maxerror,error)
        assert error<1e-10,(a,b,error)
    def stats(a,expected):
        a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
        equal(len(a),expected['n']);equal(sum(a>0),expected['positive_seeds'])
        equal(float(a.mean()) if len(a) else None,expected['mean'])
        equal(float(a.std(ddof=1)) if len(a)>1 else None,expected['std'])
    perseed={};all_later={};alternate=[]
    for seed in range(3405,3410):
        gs=list(stream(root/str(seed)/'gradient_blocks.jsonl'))
        if root.name=='A6':
            flags={(r['token'],r['profile'],r['arm']):r['control_valid'] and r['valid_comparison'] for r in gs}
            for kind,rows,metric,blocks in [('gradient',gs,'c_A',8),('step',list(stream(root/str(seed)/'step_blocks.jsonl')),'R_Q',4)]:
                grouped=defaultdict(dict)
                for r in rows:
                    if r['profile']=='SOURCE' or r['arm']=='ORACLE_M':continue
                    key=(r['task'],r['profile'],r['snapshot_step'],r['block_id'])
                    assert r['arm'] not in grouped[key]
                    grouped[key][r['arm']]=r[metric] if flags[r['token'],r['profile'],r['arm']] else None
                values=defaultdict(lambda:defaultdict(list))
                for (task,profile,snapshot,block),arms in grouped.items():
                    for name in ['FULL','BASE','BASE_M']:
                        values[task,profile,name][snapshot].append(arms[name])
                    for name in ['BASE','BASE_M','CPL_AVG','CRL_AVG','CPL_M_AVG','CRL_M_AVG','FULL_PERM_M_AVG','FULL_RAND_M_AVG']:
                        control=arms[name] if not name.endswith('_AVG') else [arms[name[:-4]+'_'+str(i)] for i in [1,2,3]]
                        if isinstance(control,list):control=None if any(x is None for x in control) else sum(control)/3
                        delta=None if arms['FULL'] is None or control is None else arms['FULL']-control
                        values[task,profile,'DELTA_'+name][snapshot].append(delta)
                for (task,profile,metricname),snapshots in values.items():
                    assert set(snapshots)=={0,50,200} and all(len(x)==blocks for x in snapshots.values())
                    means={s:np.mean([x for x in xs if x is not None]) if any(x is not None for x in xs) else np.nan for s,xs in snapshots.items()}
                    coverage=sum(x is not None for xs in snapshots.values() for x in xs)/(3*blocks)
                    perseed[kind,task,profile,metricname,seed]=(float(np.mean(list(means.values()))),coverage)
                    if metricname=='FULL':
                        for snapshot,v in means.items():all_later[kind,task,profile,seed,snapshot]=v
        else:
            grouped=defaultdict(dict)
            for r in gs:
                if r['arm']=='FULL':grouped[r['task'],r['block_id']][r['profile']]=r
            vals=defaultdict(list)
            for (task,block),profiles in grouped.items():
                for name,r in profiles.items():
                    if '_NEAR_' not in name:continue
                    signature,n,_,selector,convention=name.split('_')
                    uniform=[profiles[f'{signature}_{n}_UNIFORM_{rep}_{convention}'] for rep in range(5)]
                    valid=all(x['control_valid'] and x['valid_comparison'] for x in [r]+uniform)
                    c=sum(x['c_A'] for x in uniform)/5;e=sum(x['e_A'] for x in uniform)/5
                    vals[task,name].append((r['c_A']-c,(e-r['e_A'])/e,e,r['e_A']) if valid else (np.nan,)*4)
            for (task,name),value in vals.items():
                assert len(value)==8
                a=np.asarray(value);perseed[task,name,seed]=(np.nanmean(a,axis=0),float(np.isfinite(a[:,0]).mean()))
    for result in summary['results']:
        task,profile=result['task'],result['profile']
        if root.name=='A6':
            passes={}
            for kind in ['gradient','step']:
                for metric,expected in result[kind].items():
                    data=[perseed[kind,task,profile,metric,seed] for seed in range(3405,3410)]
                    stats([x[0] for x in data],expected)
                    equal(min(x[1] for x in data),result[kind+'_min_coverage'][metric])
                names=['FULL','DELTA_FULL_PERM_M_AVG','DELTA_FULL_RAND_M_AVG']+(['DELTA_BASE'] if kind=='step' else [])
                verdict=True
                for metric in names:
                    pairs=[perseed[kind,task,profile,metric,seed] for seed in range(3405,3410)]
                    a=np.asarray([x[0] for x in pairs]);threshold=.05 if kind=='gradient' and metric!='FULL' else 1e-4 if kind=='step' and metric=='FULL' else 0
                    verdict &= bool(np.isfinite(a).all() and (a.mean()>=threshold if threshold else a.mean()>0) and sum(a>0)>=4 and min(x[1] for x in pairs)>=.8)
                passes[kind]=verdict
            guard=True
            for record in result['later']:
                for kind,key in [('gradient','nonpositive_direction'),('step','nonpositive_step')]:
                    a=np.array([all_later[kind,task,profile,seed,record['snapshot']] for seed in range(3405,3410)])
                    equal(sum(a<=0),record[key]);guard &= bool(np.isfinite(a).all() and sum(a<=0)<3)
            equal(passes['gradient'],result['direction_pass']);equal(passes['step'],result['step_pass'])
            equal(guard,result['later_guard_pass']);equal(passes['gradient'] and passes['step'] and guard,result['joint_pass'])
        else:
            data=[perseed[task,profile,seed] for seed in range(3405,3410)]
            a=np.array([x[0] for x in data]);coverage=min(x[1] for x in data)
            stats(a[:,0],result['c_delta']);stats(a[:,1],result['error_reduction']);equal(coverage,result['coverage'])
            gate=a[:,0].mean()>=.05 and a[:,1].mean()>=.1 and min(sum(a[:,0]>0),sum(a[:,1]>0))>=4 and coverage>=.8
            equal(gate,result['passed'])
            # Original prose does not fix mean-of-ratios vs ratio-of-means.
            # Preserve the published block-average and expose both conventions.
            ratio=(a[:,2]-a[:,3])/a[:,2]
            alternative=a[:,0].mean()>=.05 and ratio.mean()>=.1 and min(sum(a[:,0]>0),sum(ratio>0))>=4 and coverage>=.8
            alternate.append(dict(task=task,profile=profile,ratio_of_seed_mean_errors=float(ratio.mean()),
                positive_seeds=int(sum(ratio>0)),passed=bool(alternative),same_gate=bool(gate==alternative)))
    result=dict(passed=True,phase=root.name,checks=checks,max_absolute_error=maxerror,cpu_seconds=time.process_time()-started,
        scope='All summary means, SDs, positive-seed counts, coverage and per-task frozen gates independently recomputed from raw JSONL.',
        alternate_A2_error_aggregation=alternate)
    (root/'summary_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='alternate_A2_error_aggregation'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();main(Path(args.root))
