"""Seed-paired summaries of audited, preregistered feedback experiments.

Never selects candidates using development outcomes. LOCAL and KERNEL are
fixed source-CV algorithms; all other rows remain descriptive references.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load(root,name):
    return pd.concat([pd.read_json(root/str(seed)/name,lines=True) for seed in range(3405,3410)],ignore_index=True)


def statistics(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return dict(mean=float(a.mean()) if len(a) else None,std=float(a.std(ddof=1)) if len(a)>1 else None,
                positive_seeds=int((a>0).sum()),n=int(len(a)))


def paired_blocks(frame,metric):
    keys=['task','seed','snapshot_step','block_id','profile']
    assert not frame.duplicated(keys+['arm']).any()
    table=frame.pivot(index=keys,columns='arm',values=metric)
    valid=frame.pivot(index=keys,columns='arm',values='control_valid').fillna(False)
    if 'valid_comparison' in frame:
        valid &= frame.pivot(index=keys,columns='arm',values='valid_comparison').fillna(False)
    table=table.where(valid)
    # Require all three controls for a block-level null mean. Missing controls
    # remain missing and reduce coverage; they are never silently averaged out.
    for prefix in ['CPL','CRL','CPL_M','CRL_M','FULL_PERM_M','FULL_RAND_M']:
        columns=[f'{prefix}_{i}' for i in [1,2,3]]
        table[prefix+'_AVG']=table[columns].mean(axis=1).where(table[columns].notna().all(axis=1))
    for control in ['BASE','BASE_M','CPL_AVG','CRL_AVG','CPL_M_AVG','CRL_M_AVG','FULL_PERM_M_AVG','FULL_RAND_M_AVG']:
        table['DELTA_'+control]=table['FULL']-table[control]
    return table.reset_index()


def seed_summary(table,metrics):
    keys=['task','seed','profile']
    snapshot=table.groupby(keys+['snapshot_step'],dropna=False)[metrics].mean()
    seeds=snapshot.groupby(keys)[metrics].mean()
    coverage=table.groupby(keys)[metrics].count()/table.groupby(keys).size().to_numpy()[:,None]
    # Every scheduled snapshot must contribute; no dropping an entirely invalid
    # snapshot when averaging a seed. Coverage is separately reported per metric.
    n_snapshots=table.groupby(keys).snapshot_step.nunique()
    seeds=seeds.where(snapshot.groupby(keys).count().eq(n_snapshots,axis=0))
    return snapshot.reset_index(),seeds.reset_index(),coverage.reset_index()


def a6(root,g,s):
    gd=paired_blocks(g,'c_A');sd=paired_blocks(s[s.profile!='SOURCE'],'R_Q')
    metrics=['FULL','BASE','BASE_M']+['DELTA_'+x for x in ['BASE','BASE_M','CPL_AVG','CRL_AVG','CPL_M_AVG','CRL_M_AVG','FULL_PERM_M_AVG','FULL_RAND_M_AVG']]
    gs,gseed,gcoverage=seed_summary(gd,metrics);ss,sseed,scoverage=seed_summary(sd,metrics)
    for name,frame in [('gradient_snapshot',gs),('gradient_seed',gseed),('gradient_coverage',gcoverage),
                       ('step_snapshot',ss),('step_seed',sseed),('step_coverage',scoverage)]:frame.to_csv(root/(name+'.csv'),index=False)
    results=[]
    for (task,profile),gg in gseed.groupby(['task','profile']):
        st=sseed[(sseed.task==task)&(sseed.profile==profile)]
        gc=gcoverage[(gcoverage.task==task)&(gcoverage.profile==profile)]
        sc=scoverage[(scoverage.task==task)&(scoverage.profile==profile)]
        gm={m:statistics(gg[m]) for m in metrics};sm={m:statistics(st[m]) for m in metrics}
        gcriteria=['FULL','DELTA_FULL_PERM_M_AVG','DELTA_FULL_RAND_M_AVG']
        scriteria=['FULL','DELTA_BASE','DELTA_FULL_PERM_M_AVG','DELTA_FULL_RAND_M_AVG']
        direction=all(gm[m]['n']==5 and gm[m]['positive_seeds']>=4 and gm[m]['mean'] is not None and
            (gm[m]['mean']>0 if m=='FULL' else gm[m]['mean']>=.05) and gc[m].min()>=.8 for m in gcriteria)
        step=all(sm[m]['n']==5 and sm[m]['positive_seeds']>=4 and sm[m]['mean'] is not None and
            (sm[m]['mean']>=1e-4 if m=='FULL' else sm[m]['mean']>0) and sc[m].min()>=.8 for m in scriteria)
        later=[]
        for snapshot in [50,200]:
            cg=gs[(gs.task==task)&(gs.profile==profile)&(gs.snapshot_step==snapshot)].FULL
            cs=ss[(ss.task==task)&(ss.profile==profile)&(ss.snapshot_step==snapshot)].FULL
            later.append(dict(snapshot=snapshot,nonpositive_direction=int((cg<=0).sum()),nonpositive_step=int((cs<=0).sum()),
                              complete=bool(cg.notna().sum()==5 and cs.notna().sum()==5)))
        guard=all(x['complete'] and x['nonpositive_direction']<3 and x['nonpositive_step']<3 for x in later)
        results.append(dict(task=task,profile=profile,gradient=gm,step=sm,direction_pass=bool(direction),step_pass=bool(step),
            later_guard_pass=bool(guard),joint_pass=bool(direction and step and guard),later=later,
            gradient_min_coverage={m:float(gc[m].min()) for m in metrics},step_min_coverage={m:float(sc[m].min()) for m in metrics}))
    routes={name:dict(passed_tasks=sum(r['joint_pass'] for r in results if r['profile']==name)) for name in ['LOCAL','KERNEL']}
    for route in routes.values():route['passes_A6']=route['passed_tasks']>=2
    # Teacher / MEAN comparisons paired on exactly the same target block.
    choices={}
    for seed in range(3405,3410):
        for path in (root/str(seed)/'blocks').glob('*/selection.json'):
            choices[path.parent.name]=read(path)['kernel'][0]['signature']
    mechanism=[]
    for metric,table in [('c_A',gd),('R_Q',sd)]:
        full=table.pivot(index=['task','seed','snapshot_step','block_id'],columns='profile',values='FULL')
        for key,row in full.iterrows():
            task,seed,snapshot,block=key;sig=choices[f'{task.rsplit("_",1)[1]}_{seed}_{snapshot}_{block}']
            for candidate,teacher,mean in [('LOCAL','TEACHER_LOCAL','MEAN_H'),('KERNEL','TEACHER_'+sig,'MEAN_'+sig)]:
                mechanism.append(dict(task=task,seed=seed,snapshot_step=snapshot,block_id=block,metric=metric,candidate=candidate,
                    matched_teacher=teacher,teacher_delta=row[candidate]-row[teacher],mean_delta=row[candidate]-row[mean]))
    pd.DataFrame(mechanism).to_csv(root/'mechanism_paired_blocks.csv',index=False)
    return dict(results=results,routes=routes,continuous_training_triggered=any(x['passes_A6'] for x in routes.values()),
                combination_triggered=all(x['passes_A6'] for x in routes.values()))


def a2(root,g,s):
    full=g[g.arm=='FULL'].copy();parts=full.profile.str.split('_')
    # Names are H_64_UNIFORM_0_FIXED or H_64_Hnear_FIXED, and full196.
    metadata={}
    for p in (root/'3405/blocks').iterdir():
        for profile in read(p/'profiles.json')['profiles']:
            metadata[profile['profile']]={k:profile[k] for k in ['signature','n','selector','convention']}
    for col in ['signature','n','selector','convention']:full[col]=full.profile.map(lambda p:metadata[p][col])
    full['valid']=full.control_valid & full.valid_comparison
    full.loc[~full.valid,['c_A','e_A']]=np.nan
    keys=['task','seed','snapshot_step','block_id','signature','n','convention']
    uniform=full[full.selector=='UNIFORM'].groupby(keys)[['c_A','e_A']].agg(lambda v:v.mean() if v.notna().sum()==5 else np.nan)
    rows=[]
    for _,row in full[(full.n<196)&(full.selector!='UNIFORM')].iterrows():
        u=uniform.loc[tuple(row[k] for k in keys)]
        rows.append(dict(**{k:row[k] for k in keys},profile=row.profile,selector=row.selector,c_delta=row.c_A-u.c_A,
                         error_reduction=(u.e_A-row.e_A)/u.e_A if u.e_A>0 else np.nan))
    blocks=pd.DataFrame(rows);blocks.to_csv(root/'local_paired_blocks.csv',index=False)
    seeds=blocks.groupby(['task','seed','profile'])[['c_delta','error_reduction']].mean().reset_index()
    coverage=blocks.groupby(['task','seed','profile'])[['c_delta','error_reduction']].count()/8
    seeds.to_csv(root/'local_paired_seeds.csv',index=False);coverage.to_csv(root/'local_coverage.csv')
    results=[]
    for (task,profile),frame in seeds.groupby(['task','profile']):
        c,e=statistics(frame.c_delta),statistics(frame.error_reduction)
        cv=float(coverage.loc[(task,slice(None),profile),:].min().min())
        passed=c['n']==5 and e['n']==5 and c['mean']>=.05 and e['mean']>=.1 and min(c['positive_seeds'],e['positive_seeds'])>=4 and cv>=.8
        results.append(dict(task=task,profile=profile,c_delta=c,error_reduction=e,coverage=cv,passed=bool(passed)))
    return dict(results=results,passing_tasks_by_profile={p:sum(r['passed'] for r in results if r['profile']==p) for p in sorted(seeds.profile.unique())})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);audit=read(root/'audit.json');assert audit['passed'];started=time.process_time()
    g=load(root,'gradient_blocks.jsonl');s=load(root,'step_blocks.jsonl')
    assert len(g)==audit['counts']['gradient_rows'] and len(s)==audit['counts']['step_rows']
    # Step logs contain losses, while control validity is sealed in the matching
    # gradient log. Join by block/profile/arm rather than assuming a missing flag.
    join=['token','profile','arm']
    s=s.merge(g[join+['control_valid','valid_comparison']],on=join,how='left',validate='one_to_one')
    diagnostic=(s.profile=='SOURCE')|(s.arm=='ORACLE_M')
    assert s.loc[~diagnostic,'control_valid'].notna().all()
    s.loc[diagnostic,['control_valid','valid_comparison']]=True
    # Preserve every arm, failure and snapshot in an ordinary seed-level table.
    keys=['task','seed','profile','arm','snapshot_step','git_commit','protocol_sha256','config_name','hardware','gpu_hours','event_flags']
    ga=g.groupby(keys,dropna=False)[['c_A','c_B','e_A','gradient_norm','valid_comparison','control_valid']].mean().reset_index()
    sa=s.groupby(keys,dropna=False)[['R_Q','CE_B','BA_B','control_valid']].mean().reset_index()
    ga.merge(sa,on=keys,how='outer',suffixes=('_gradient','_step')).to_csv(root/'results.csv',index=False)
    summary=a6(root,g,s) if root.name=='A6' else a2(root,g,s)
    summary.update(phase=root.name,audited=True,final_labels_read=0,summary_cpu_seconds=time.process_time()-started,
        aggregation='Three nulls within block; blocks within snapshot; snapshots within model seed. Five paired model seeds; not five independent datasets.',
        coverage_policy='All scheduled blocks retained; invalid comparisons contribute to missing coverage. Gate requires minimum per-seed coverage >=0.8.')
    write(root/'results_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['results']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
