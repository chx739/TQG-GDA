"""Verify a finished local archive, summarize all seeds, and export a public report.

This is post-run analysis, not a checkpoint or hyperparameter selector.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import torch
from experiments.common.io import digest, write_json


def load(path):
    return json.loads(path.read_text())


def stats(values):
    values=[v for v in values if v is not None]
    if not values:return dict(n=0,mean=None,sd=None,min=None,max=None)
    return dict(n=len(values),mean=statistics.mean(values),
                sd=statistics.stdev(values) if len(values)>1 else None,min=min(values),max=max(values))


def summarize(root):
    files=load(root/'manifest.json')
    for path,record in files.items():
        absolute=root/path
        if absolute.stat().st_size!=record['bytes'] or digest(absolute)!=record['sha256']:
            raise ValueError(f'Archive integrity failure: {path}')
    evaluations=load(root/'evaluation.json')
    grouped={};cell_rows=[];score_rows=[];trajectory_rows=[];decompositions=[];geometry_probes=[]
    for directory in sorted(root.glob('d*_seed*')):
        if not (directory/'cell_summary.json').exists():continue
        summary=load(directory/'cell_summary.json');dim=summary['dim'];seed=summary['seed']
        cell_rows.append(summary)
        for bank,metrics in load(directory/'score_queries.json').items():
            for pair,row in metrics['comparisons'].items():
                score_rows.append(dict(dim=dim,seed=seed,bank=bank,pair=pair,**row))
        nonlinear=load(directory/'E1_nonlinear.json')
        nonlinear_arrays=torch.load(directory/'E1_nonlinear.pt',weights_only=True)
        measurements=nonlinear_arrays.get('measurements',{})
        common=None
        for arrays in measurements.values():
            common=arrays['valid'].clone() if common is None else common & arrays['valid']
        for step in nonlinear['steps']:
            arrays=measurements[step['alpha']]
            error=(arrays['geometric']+arrays['v'])/summary['sigma']
            shared_error=float(error[common].square().mean().sqrt()) if common is not None and common.any() else None
            geometry_probes.append(dict(dim=dim,seed=seed,**step,
                                       common_valid_count=int(common.sum()) if common is not None else 0,
                                       E_geom_common_valid=shared_error))
        for arm in ('learned','empirical','population','zero'):
            trace=load(directory/arm/'adaptation.json')
            observations=load(directory/arm/'observations.json')
            count=sum(r.get('anchors',0) for r in observations)
            rounds=len(trace['history'])
            nonempty=[r for r in observations if r.get('anchors',0)]
            accepted=[r for r in nonempty if r['accepted']]
            measured=sum(r['geometry_first8']['measured'] for r in nonempty)
            attempted=sum(r['geometry_first8']['anchors'] for r in nonempty)
            rejects=Counter()
            for record in trace['history']:rejects.update(record.get('projection_rejections',{}))
            row=dict(dim=dim,seed=seed,arm=arm,accepted=trace['accepted'],rounds=rounds,
                     stop_reason=trace['stop_reason'],anchors_total=count,coverage=count/(128*rounds),
                     anchors_per_round=stats([r.get('anchors',0) for r in observations]),
                     clipping_round_mean=stats([r['clipped_fraction'] for r in nonempty])['mean'],
                     geometry_valid_fraction=measured/attempted if attempted else None,
                     geometry_attempted=attempted,geometry_measured=measured,
                     E_geom_round_mean=stats([r['geometry_first8']['E_geom'] for r in nonempty])['mean'],
                     E_geom_accepted_round_mean=stats([r['geometry_first8']['E_geom'] for r in accepted])['mean'],
                     E_lin_round_mean=stats([r['E_lin_all_anchors'] for r in nonempty])['mean'],
                     projection_rejections=dict(rejects),
                     reasons=dict(Counter(r['reason'] for r in observations)),
                     source_ce=summary['arms'][arm]['source_ce'])
            # Equal weight to rounds within each trajectory, then seeds/arms.
            for pair in ('learned_vs_population','empirical_vs_population','zero_vs_population'):
                row[pair+'_normal_rmse_round_mean']=stats([r['score']['comparisons'][pair]['normal_rmse'] for r in nonempty])['mean']
            trajectory_rows.append(row)
            path=directory/arm/'first_active_decomposition.json'
            if path.exists():decompositions.append(dict(dim=dim,seed=seed,arm=arm,**load(path)))
    for dim in (2,16):
        grouped[dim]={}
        for arm in ('learned','empirical','population','zero'):
            selected=[r for r in evaluations if r['dim']==dim and r['arm']==arm]
            successful=[r for r in selected if next(t for t in trajectory_rows if t['dim']==dim and t['seed']==r['seed'] and t['arm']==arm)['stop_reason']=='round_budget']
            delta=[r['delta_ba_pp'] for r in successful]
            paired=[r for r in trajectory_rows if r['dim']==dim and r['arm']==arm and r['stop_reason']=='round_budget']
            grouped[dim][arm]=dict(completed=len(successful),delta_ba_pp=stats(delta),
                                  delta_nll=stats([r['delta_nll'] for r in successful]),
                                  wins=sum(v>0 for v in delta),ties=sum(v==0 for v in delta),losses=sum(v<0 for v in delta),
                                  accepted=stats([r['accepted'] for r in paired]),coverage=stats([r['coverage'] for r in paired]))
    return dict(archive_path=str(root),integrity_files_verified=len(files),
                archive_bytes=sum(r['bytes'] for r in files.values()),
                execution_commit=load(root/'provenance.json')['git_commit'],
                failures=load(root/'failures.json'),analytic=load(root/'E1_analytic.json'),
                grouped=grouped,cells=cell_rows,evaluation=evaluations,score_queries=score_rows,
                trajectories=trajectory_rows,decompositions=decompositions,nonlinear_geometry=geometry_probes)


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in rows)


def plots(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    arms=['learned','empirical','population','zero']
    for axis,dim in zip(axes,[2,16]):
        for i,arm in enumerate(arms):
            rows=[r for r in summary['evaluation'] if r['dim']==dim and r['arm']==arm]
            vals=[r['delta_ba_pp'] for r in rows]
            axis.scatter([i+(j-2)*.045 for j in range(len(vals))],vals,s=24)
            if vals:axis.plot([i-.18,i+.18],[statistics.mean(vals)]*2,c='black')
        axis.axhline(0,c='gray',lw=.7);axis.set_xticks(range(4),arms,rotation=20)
        axis.set_title(f'd={dim}; all five seeds');axis.set_ylabel('Target BA change (percentage points)')
    fig.savefig(output/'E2-risk.png',dpi=180);fig.savefig(output/'E2-risk.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,dim in zip(axes,[2,16]):
        for seed in range(5):
            rows=sorted([r for r in summary['nonlinear_geometry'] if r['dim']==dim and r['seed']==seed],key=lambda r:r['alpha'])
            axis.plot([r['alpha'] for r in rows],[r['E_geom_common_valid'] for r in rows],marker='o',label=f'seed {seed}')
        axis.set_xscale('log',base=2);axis.set_yscale('log');axis.set_title(f'd={dim}, nonzero U2')
        axis.set_xlabel('Candidate step fraction');axis.set_ylabel('RMS geometric proxy error / sigma\n(common valid anchors across all steps)');axis.legend(fontsize=7)
    fig.savefig(output/'E1-geometry.png',dpi=180);fig.savefig(output/'E1-geometry.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,dim in zip(axes,[2,16]):
        for cell in [c for c in summary['cells'] if c['dim']==dim]:
            seed=cell['seed'];loss=cell['score_validation_loss']
            rows=[r for r in summary['score_queries'] if r['dim']==dim and r['seed']==seed and r['bank']=='initial_boundaries']
            learned=next((r['normal_rmse'] for r in rows if r['pair']=='learned_vs_population'),None)
            zero=next((r['normal_rmse'] for r in rows if r['pair']=='zero_vs_population'),None)
            if learned is not None:
                axis.scatter([loss],[learned],c='tab:blue',marker='o')
                axis.scatter([loss],[zero],c='gray',marker='x')
                axis.annotate(str(seed),(loss,learned),xytext=(4,4),textcoords='offset points')
        axis.set_title(f'd={dim}; blue=learned, gray x=zero')
        axis.set_xlabel('Selected fixed-noise validation loss')
        axis.set_ylabel('RMS[sigma * normal-score error]')
    fig.savefig(output/'E2-score-validation.png',dpi=180);fig.savefig(output/'E2-score-validation.svg');plt.close(fig)
