"""Aggregate paired P0 blocks within model seed; never reselect thresholds."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import time


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def deviation(values):
    values=[v for v in values if v is not None]
    return statistics.stdev(values) if len(values)>1 else None


def balanced_snapshot_mean(rows,key):
    times=sorted({r['snapshot_step'] for r in rows})
    return mean([mean([r[key] for r in rows if r['snapshot_step']==t]) for t in times])


def export_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def summarize(stage):
    stage=Path(stage).resolve()
    manifest=json.loads((stage/'experiment_manifest.json').read_text())
    plan=json.loads((stage.parent/'stage2_method/experiment_plan.yaml').read_text())
    seeds=plan['seeds']
    tasks=plan['tasks']
    gradients=[];steps=[];geometry=[]
    for seed in seeds:
        out=stage/'logs'/str(seed)
        done=json.loads((out/'done.json').read_text())
        if done['blocks']!=72 or done['git_commit']!=manifest['git_commit']:
            raise ValueError('Incomplete or differently versioned seed')
        gradients+=load_jsonl(out/'gradient_blocks.jsonl')
        steps+=load_jsonl(out/'one_step_blocks.jsonl')
        geometry+=load_jsonl(out/'geometry_stats.jsonl')
    if len(gradients)!=3600 or len(steps)!=6660 or len(geometry)!=360:
        raise ValueError(('Unexpected logical row count',len(gradients),len(steps),len(geometry)))
    export_csv(stage/'gradient_blocks.csv',gradients)
    export_csv(stage/'one_step_blocks.csv',steps)
    grouped=defaultdict(dict)
    for row in gradients:
        key=(row['task'],row['seed'],row['snapshot_step'],row['block_id'])
        grouped[key][row['arm']]=row
    block_metrics=[]
    for (task,seed,snapshot,block),arms in grouped.items():
        true=arms['TRUE']
        paired=(true['c_A'] is not None and all(arms[f'{kind}-{rep}']['c_A'] is not None
                for kind in ('PERM','RAND') for rep in (1,2,3)))
        p=mean([arms[f'PERM-{rep}']['c_A'] for rep in (1,2,3)])
        r=mean([arms[f'RAND-{rep}']['c_A'] for rep in (1,2,3)])
        block_metrics.append(dict(task=task,seed=seed,snapshot_step=snapshot,block_id=block,
            c_TRUE=true['c_A'],c_PERM=p,c_RAND=r,
            advantage_PERM=true['c_A']-p if paired else None,
            advantage_RAND=true['c_A']-r if paired else None,coverage=float(paired),
            e_TRUE=true['e_A'],c_B_TRUE=true['c_B'],norm_ratio=true['norm_ratio'],
            weighted_ratio=true['weighted_ratio'],B_space_outside_energy_fraction=true['B_space_outside_energy_fraction']))
    metric_keys=['c_TRUE','c_PERM','c_RAND','advantage_PERM','advantage_RAND','coverage',
                 'e_TRUE','c_B_TRUE','norm_ratio','weighted_ratio','B_space_outside_energy_fraction']
    d1_seed=[];snapshot_summary=[]
    for task in tasks:
        for seed in seeds:
            rows=[r for r in block_metrics if r['task']==task and r['seed']==seed]
            d1_seed.append(dict(task=task,seed=seed,**{k:balanced_snapshot_mean(rows,k) for k in metric_keys}))
            for t in plan['snapshots']['times']:
                these=[r for r in rows if r['snapshot_step']==t]
                snapshot_summary.append(dict(task=task,seed=seed,snapshot_step=t,**{k:mean([r[k] for r in these]) for k in metric_keys}))
    d1_tasks=[]
    for task in tasks:
        rows=[r for r in d1_seed if r['task']==task]
        result=dict(task=task,**{k:mean([r[k] for r in rows]) for k in metric_keys},
                    cosine_SD=deviation([r['c_TRUE'] for r in rows]),
                    PERM_seed_wins=sum(r['advantage_PERM'] is not None and r['advantage_PERM']>0 for r in rows),
                    RAND_seed_wins=sum(r['advantage_RAND'] is not None and r['advantage_RAND']>0 for r in rows))
        result['passed']=(result['c_TRUE'] is not None and result['c_TRUE']>0
            and result['advantage_PERM'] is not None and result['advantage_PERM']>=.05
            and result['advantage_RAND'] is not None and result['advantage_RAND']>=.05
            and result['PERM_seed_wins']>=3 and result['RAND_seed_wins']>=3 and result['coverage']>=.8)
        d1_tasks.append(result)
    d1_pass=sum(r['passed'] for r in d1_tasks)>=2
    export_csv(stage/'gradient_seed_summary.csv',d1_seed)
    export_csv(stage/'gradient_snapshot_seed_summary.csv',snapshot_summary)
    save(stage/'gradient_summary.json',dict(passed=d1_pass,tasks=d1_tasks,model_seeds=5,
         thresholds_frozen=True,blocks_are_independent_seeds=False))

    step_groups=defaultdict(dict)
    for row in steps:
        if row['arm']=='F': continue
        key=(row['task'],row['seed'],row['snapshot_step'],row['block_id'],row['eta'])
        step_groups[key][row['arm']]=row
    step_blocks=[]
    for (task,seed,snapshot,block,eta),arms in step_groups.items():
        source=arms['S'];query=arms['S+Q'];base=query['base_CE_B']
        valid=all(arms[f'S+{kind}-{rep}']['valid_comparison'] for kind in ('P','R') for rep in (1,2,3))
        d1_valid=next(r['coverage'] for r in block_metrics if (r['task'],r['seed'],r['snapshot_step'],r['block_id'])==(task,seed,snapshot,block))
        paired=bool(valid and d1_valid)
        p=mean([arms[f'S+P-{rep}']['CE_B'] for rep in (1,2,3)])
        r=mean([arms[f'S+R-{rep}']['CE_B'] for rep in (1,2,3)])
        step_blocks.append(dict(task=task,seed=seed,snapshot_step=snapshot,block_id=block,eta=eta,
            R_Q=(source['CE_B']-query['CE_B'])/base,
            advantage_PERM=(p-query['CE_B'])/base if paired else None,
            advantage_RAND=(r-query['CE_B'])/base if paired else None,
            coverage=float(paired),oracle_extra=(source['CE_B']-arms['S+O-M']['CE_B'])/base if arms['S+O-M']['valid_comparison'] else None,
            Q_alone_gain=(base-arms['Q']['CE_B'])/base,
            oracle_alone_gain=(base-arms['O-M']['CE_B'])/base if arms['O-M']['valid_comparison'] else None,
            oracle_raw_gain=(base-arms['O-RAW']['CE_B'])/base,
            source_gain=(base-source['CE_B'])/base,
            harmful=float(query['CE_B']>source['CE_B']+1e-8),
            A_query_gain=(query['base_CE_A']-query['CE_A'])/query['base_CE_A'],
            B_query_gain=(base-query['CE_B'])/base,
            BA_delta_vs_source=query['BA_B']-source['BA_B']))
    d2_keys=['R_Q','advantage_PERM','advantage_RAND','coverage','oracle_extra','Q_alone_gain',
             'oracle_alone_gain','oracle_raw_gain','source_gain','harmful','A_query_gain','B_query_gain','BA_delta_vs_source']
    d2_seed=[];d2_snap=[]
    for task in tasks:
        for seed in seeds:
            for eta in plan['D2']['etas']:
                rows=[r for r in step_blocks if r['task']==task and r['seed']==seed and r['eta']==eta]
                d2_seed.append(dict(task=task,seed=seed,eta=eta,**{k:balanced_snapshot_mean(rows,k) for k in d2_keys}))
                for t in plan['snapshots']['times']:
                    these=[r for r in rows if r['snapshot_step']==t]
                    d2_snap.append(dict(task=task,seed=seed,eta=eta,snapshot_step=t,**{k:mean([r[k] for r in these]) for k in d2_keys}))
    d2_tasks=[]
    for task in tasks:
        for eta in plan['D2']['etas']:
            rows=[r for r in d2_seed if r['task']==task and r['eta']==eta]
            result=dict(task=task,eta=eta,**{k:mean([r[k] for r in rows]) for k in d2_keys},
                        R_Q_SD=deviation([r['R_Q'] for r in rows]),seed_wins=sum(r['R_Q']>0 for r in rows))
            result['passed']=(result['R_Q']>=1e-4 and result['seed_wins']>=3 and result['coverage']>=.8
                and result['advantage_PERM'] is not None and result['advantage_PERM']>0
                and result['advantage_RAND'] is not None and result['advantage_RAND']>0)
            result['primary_eta']=eta==plan['D2']['primary_eta']
            d2_tasks.append(result)
    d2_pass=sum(r['passed'] for r in d2_tasks if r['primary_eta'])>=2
    export_csv(stage/'one_step_seed_summary.csv',d2_seed)
    export_csv(stage/'one_step_snapshot_seed_summary.csv',d2_snap)
    export_csv(stage/'one_step_paired_blocks.csv',step_blocks)
    save(stage/'one_step_summary.json',dict(passed=d2_pass,primary_eta=plan['D2']['primary_eta'],tasks=d2_tasks))

    results=[]
    for kind,rows,key,direction in [('D1',d1_seed,'c_TRUE','max'),('D2',d2_seed,'R_Q','max')]:
        for row in rows:
            results.append(dict(experiment_id=kind,config_name=manifest['settings']['config_name'],
                git_commit=manifest['git_commit'],gpu_hours=0.,hardware=manifest['hardware'],
                plan_sha256=manifest['plan_sha256'],event_flags='clean',metric_name=key,metric_direction=direction,
                metric_value=row[key],**row))
    export_csv(stage/'results.csv',results)
    solver_fail=sum(not x['statistics']['calibration']['converged'] for x in geometry)
    summary=dict(status='completed',D0_passed=True,D1_passed=d1_pass,D2_passed=d2_pass,
        eligible_for_D9=d1_pass and d2_pass,D1=d1_tasks,D2=[r for r in d2_tasks if r['primary_eta']],
        counts=dict(D1_blocks=360,D1_raw_rows=len(gradients),D2_blocks=180,D2_temporary_updates=6480,D2_frozen_references=180),
        nonconverged_calibration_blocks=solver_fail,numerical_fallbacks=0,final_labels_read=0,
        oracle_models_produced=0,continuous_adaptation_jobs=0,limitations=[
            'Development labels previously observed; this is exploratory diagnosis, not a blind test.',
            'Five model seeds; snapshots, blocks and permutations are not extra independent seeds.',
            'Empirical finite-batch gradients are not conditional expected gradients.',
            'One-step CE effects do not establish long-run balanced-accuracy improvement.'])
    save(stage/'results_summary.json',summary)
    export_csv(stage/'gradient_paired_blocks.csv',block_metrics)
    for filename,report in [('decision.md',summary),('hand_off.md',summary)]:
        (stage/filename).write_text(f'# P0诊断交接\n\nD0通过；D1：{d1_pass}；D2：{d2_pass}。'
            f'是否符合进入D9的预定门槛：{d1_pass and d2_pass}。'
            '详细数值见results_summary.json。未达到门槛时继续有目的的D3/4/8定位，不启动完整适配。'
            '本轮使用开发标签作独立Oracle诊断，最终标签未读取，临时参数不成为训练起点。\n')
    make_plots(stage,d1_seed,d2_seed,plan['D2']['primary_eta'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_plots(stage,d1,d2,primary_eta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    tasks=sorted({r['task'] for r in d1})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(len(tasks))
    for index,(key,label) in enumerate([('c_TRUE','TRUE'),('c_PERM','PERM'),('c_RAND','RAND')]):
        groups=[[r[key] for r in d1 if r['task']==task] for task in tasks]
        axes[0].bar(x+(index-1)*.23,[mean(g) or 0 for g in groups],width=.23,
                    yerr=[deviation(g) or 0 for g in groups],label=label,capsize=3)
    axes[0].set_xticks(x,[t.replace('_to_','→') for t in tasks]);axes[0].set_ylabel('Cosine with empirical target gradient')
    axes[0].axhline(0,color='black',lw=.7);axes[0].legend()
    for index,task in enumerate(tasks):
        values=[r['R_Q'] for r in d2 if r['task']==task and r['eta']==primary_eta]
        axes[1].scatter([index]*len(values),np.array(values)*100,s=25)
        axes[1].plot([index-.15,index+.15],[mean(values)*100]*2,color='black')
    axes[1].set_xticks(x,[t.replace('_to_','→') for t in tasks]);axes[1].axhline(0,color='black',lw=.7)
    axes[1].set_ylabel('Extra B-pool relative CE reduction (%)')
    fig.tight_layout();fig.savefig(stage/'figures/p0_primary.png',dpi=180);fig.savefig(stage/'figures/p0_primary.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    summarize(parser.parse_args().stage)
