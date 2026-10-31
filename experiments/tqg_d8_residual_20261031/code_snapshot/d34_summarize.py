"""Predeclared paired D3/D4 summaries; blocks and folds are not model seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from p0_summarize import export_csv,mean,deviation,balanced_snapshot_mean


def jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def aggregate_rows(rows,keys,metrics):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row.get(k) for k in keys)].append(row)
    result=[]
    for values,group in grouped.items():
        reference=group[0]
        record={k:v for k,v in zip(keys,values)}
        record.update({k:reference[k] for k in ('git_commit','plan_sha256','hardware','config_name','event_flags','gpu_hours') if k in reference})
        record.update({k:balanced_snapshot_mean(group,k) for k in metrics})
        result.append(record)
    return result


def summarize(stage):
    stage=Path(stage).resolve()
    manifest=json.loads((stage/'experiment_manifest.json').read_text())
    d3=[];d4=[];source=[];steps=[];kstats=[];spectra=[]
    for seed in range(3405,3410):
        folder=stage/'logs'/str(seed)
        done=json.loads((folder/'done.json').read_text())
        if done['blocks']!=96 or done['git_commit']!=manifest['git_commit']:raise ValueError('Incomplete/mixed execution')
        for name,dest in [('calibration_grid_blocks',d3),('signature_target_blocks',d4),('source_audit_blocks',source),
                          ('step_blocks',steps),('kernel_distance_stats',kstats),('omega_spectrum',spectra)]:
            dest.extend(jsonl(folder/(name+'.jsonl')))
    actual=(len(d3),len(d4),len(source),len(steps),len(kstats),len(spectra))
    if actual!=(5400,1800,600,8400,2400,5400):raise ValueError(('Row counts differ',actual))
    for name,rows in [('calibration_grid_blocks',d3),('signature_target_blocks',d4),('source_audit_blocks',source),('step_blocks',steps)]:
        export_csv(stage/(name+'.csv'),rows)
    all_grad=d3+d4+source
    metrics=['c_A','e_A','c_B','gradient_norm','norm_ratio','weighted_ratio','mean_error','max_error','weight_norm',
             'omega_deviation','feedback_cosine_vs_reference','feedback_norm_ratio_vs_reference','W_difference_vs_reference',
             'solve_seconds','physical_solve_seconds']
    keys=['experiment_id','scope','task','seed','arm','alpha']
    seed_grad=aggregate_rows(all_grad,keys,metrics)
    seed_steps=aggregate_rows(steps,keys+['scale'],['R_Q','CE_B','BA_B','AUROC_B'])
    snapshots=aggregate_rows(all_grad,keys+['snapshot_step'],metrics)
    export_csv(stage/'gradient_seed_summary.csv',seed_grad)
    export_csv(stage/'step_seed_summary.csv',seed_steps)
    export_csv(stage/'snapshot_seed_summary.csv',snapshots)
    export_csv(stage/'calibration_seed_summary.csv',[r for r in seed_grad if r['experiment_id']=='D3'])
    export_csv(stage/'signature_seed_summary.csv',[r for r in seed_grad if r['experiment_id']=='D4'])
    group=defaultdict(list)
    for row in seed_grad:group[(row['experiment_id'],row['scope'],row['task'],row['arm'],row['alpha'])].append(row)
    task_summary=[]
    for (experiment,scope,task,arm,alpha),rows in group.items():
        task_summary.append(dict(experiment_id=experiment,scope=scope,task=task,arm=arm,alpha=alpha,
            **{metric:mean([r[metric] for r in rows]) for metric in metrics},
            c_A_SD=deviation([r['c_A'] for r in rows]),e_A_SD=deviation([r['e_A'] for r in rows])))

    # Align every comparison by exact snapshot, fold, batch and model seed.
    block_keys=['scope','task','seed','snapshot_step','fold','block_id']
    paired=defaultdict(dict)
    for row in all_grad:
        if row['experiment_id']=='D3' and row['alpha']!=.005:continue
        paired[(row['experiment_id'],*(row[k] for k in block_keys))][row['arm']]=row
    step_pairs=defaultdict(dict)
    for row in steps:
        if row['scale']!='norm_matched' or (row['experiment_id']=='D3' and row['alpha']!=.005):continue
        step_pairs[(row['experiment_id'],*(row[k] for k in block_keys))][row['arm']]=row
    pairs=[]
    for key,arms in paired.items():
        experiment=key[0];candidate,reference=('JOINT','ISO') if experiment=='D3' else ('HA','H')
        c,r=arms[candidate],arms[reference]
        valid=c['c_A'] is not None and r['c_A'] is not None
        entry=dict(experiment_id=experiment,**{k:v for k,v in zip(block_keys,key[1:])},
            advantage_cosine=c['c_A']-r['c_A'] if valid else None,gradient_coverage=float(valid),
            candidate_cosine=c['c_A'],reference_cosine=r['c_A'],step_advantage=None,step_coverage=None)
        if key in step_pairs:
            sc,sr=step_pairs[key][candidate],step_pairs[key][reference]
            step_valid=sc['valid_comparison'] and sr['valid_comparison']
            entry.update(step_advantage=sc['R_Q']-sr['R_Q'] if step_valid else None,step_coverage=float(step_valid))
        pairs.append(entry)
    paired_seeds=aggregate_rows(pairs,['experiment_id','scope','task','seed'],
        ['advantage_cosine','gradient_coverage','candidate_cosine','reference_cosine','step_advantage','step_coverage'])
    export_csv(stage/'paired_blocks.csv',pairs)
    export_csv(stage/'paired_seed_summary.csv',paired_seeds)
    criteria=[]
    for experiment,scope,task in sorted({(r['experiment_id'],r['scope'],r['task']) for r in paired_seeds}):
        rows=[r for r in paired_seeds if (r['experiment_id'],r['scope'],r['task'])==(experiment,scope,task)]
        result=dict(experiment_id=experiment,scope=scope,task=task,
            cosine_advantage=mean([r['advantage_cosine'] for r in rows]),cosine_advantage_SD=deviation([r['advantage_cosine'] for r in rows]),
            cosine_seed_wins=sum(r['advantage_cosine'] is not None and r['advantage_cosine']>0 for r in rows),
            norm_matched_B_advantage=mean([r['step_advantage'] for r in rows]),
            gradient_coverage=mean([r['gradient_coverage'] for r in rows]),step_coverage=mean([r['step_coverage'] for r in rows]),
            source_only_descriptive=scope=='source')
        result['passed']=(result['cosine_advantage'] is not None and result['cosine_advantage']>=.03
            and result['cosine_seed_wins']>=3 and result['norm_matched_B_advantage'] is not None
            and result['norm_matched_B_advantage']>0 and result['gradient_coverage']>=.8 and result['step_coverage']>=.8)
        criteria.append(result)
    d3_pass=sum(r['passed'] for r in criteria if r['experiment_id']=='D3' and r['scope']=='target')>=2
    d4_pass=sum(r['passed'] for r in criteria if r['experiment_id']=='D4' and r['scope']=='target')>=2
    result_rows=[]
    for rows,metric,scale in [(seed_grad,'c_A','gradient'),(seed_steps,'R_Q',None)]:
        for row in rows:
            result_rows.append(dict(row,scale=scale or row['scale'],metric_name=metric,metric_value=row[metric],metric_direction='max'))
    export_csv(stage/'results.csv',result_rows)
    kernel_metrics=['K_SS_rank','K_SS_effective_rank','K_ST_mean','K_ST_max','offdiagonal_std',
        'source_nn_label_agreement','source_prior_agreement','source_nn_residual_MSE',
        'target_nn_label_agreement','target_nn_residual_MSE']
    export_csv(stage/'kernel_seed_summary.csv',aggregate_rows(kstats,['scope','task','seed','arm'],kernel_metrics))
    summary=dict(status='completed',D0_passed=True,D3_passed=d3_pass,D4_passed=d4_pass,
        criteria=criteria,task_settings=task_summary,
        counts=dict(D3_rows=5400,D4_target_rows=1800,D4_source_rows=600,step_rows=8400,model_seeds=5,
                    target_blocks=360,source_blocks=120,result_rows=len(result_rows)),
        D3_nonconverged=sum(not r['converged'] for r in d3),D4_nonconverged=sum(not r['converged'] for r in d4+source),
        ISO_MEAN_max_relative_error=max(r.get('ISO_MEAN_relative_error',0.) for r in d3),
        final_labels_read=0,continuous_adaptations=0,
        scope='D3/D4 incremental mechanism criteria only; no candidate D1/D2 revalidation or D9',
        limitations=['Observed development labels; no new blind test.',
            'Source bank98 and real bank196 differ; source-vs-target is descriptive, not a causal matched-count contrast.',
            'Empirical finite-batch label gradients can be noisy even within the source domain.',
            'Passing an incremental D3/D4 test does not establish overall correct feedback or long-run classification improvement.'])
    write(stage/'results_summary.json',summary)
    write(stage/'calibration_summary.json',dict(passed=d3_pass,criteria=[r for r in criteria if r['experiment_id']=='D3']))
    write(stage/'signature_summary.json',dict(passed=d4_pass,criteria=[r for r in criteria if r['experiment_id']=='D4']))
    make_plots(stage,task_summary,paired_seeds)
    (stage/'decision.md').write_text(f'# D3/D4判断\n\nD3预定增量门槛：{d3_pass}。D4预定增量门槛：{d4_pass}。'
        '比较按共同模型、逐块配对、模型种子内先平均；详细各任务标准见results_summary.json。'
        '即使某一增量通过，也必须另过候选D1/D2，才能进入D9。\n')
    (stage/'hand_off.md').write_text((stage/'decision.md').read_text())
    print(json.dumps(dict(D3_passed=d3_pass,D4_passed=d4_pass,criteria=criteria,counts=summary['counts']),ensure_ascii=False,indent=2))


def make_plots(stage,summary,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    tasks=['0_to_1','0_to_2','0_to_3']
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,task in zip(axes,tasks):
        for mode in ['MEAN','ISO','JOINT']:
            values=sorted([r for r in summary if r['experiment_id']=='D3' and r['task']==task and r['arm']==mode],key=lambda r:r['alpha'])
            ax.plot([r['alpha'] for r in values],[r['c_A'] for r in values],marker='o',label=mode)
        ax.set_xscale('log');ax.axhline(0,color='black',lw=.5);ax.set_title(task.replace('_to_','→'))
        ax.set_xlabel('alpha');ax.set_ylabel('Empirical target-gradient cosine')
    axes[0].legend();fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(stage/f'figures/regularization_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    kernels=['H','A','HA','CONST','HA_PERM'];x=np.arange(3)
    for i,kernel in enumerate(kernels):
        values=[next(r for r in summary if r['experiment_id']=='D4' and r['scope']=='target' and r['task']==task and r['arm']==kernel) for task in tasks]
        axes[0].bar(x+(i-2)*.15,[r['c_A'] for r in values],width=.15,label=kernel)
    axes[0].set_xticks(x,[t.replace('_to_','→') for t in tasks]);axes[0].axhline(0,color='black',lw=.5)
    axes[0].set_ylabel('Real-target empirical gradient cosine');axes[0].legend(fontsize=8)
    source=[next(r for r in summary if r['experiment_id']=='D4' and r['scope']=='source' and r['arm']==kernel) for kernel in kernels]
    axes[1].bar(np.arange(5),[r['c_A'] for r in source],yerr=[r['c_A_SD'] for r in source],capsize=3)
    axes[1].set_xticks(np.arange(5),kernels,rotation=25);axes[1].axhline(0,color='black',lw=.5)
    axes[1].set_ylabel('Within-source empirical gradient cosine');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(stage/f'figures/signature_comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    summarize(parser.parse_args().stage)
