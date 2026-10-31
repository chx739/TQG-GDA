"""Auxiliary unlabeled timing/CONST audit and plots; primary results stay frozen."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from p0_summarize import export_csv


def constant_binary_readout(target_response,n,lambda_w):
    """Exact joint minimizer for the all-ones binary (q=1) kernel.

    u=mean A, E=(u-W1)(u-W1)^T. Symmetry gives W=c*u*1^T;
    minimizing (r+1)*(1-n*c)^2 + lambda*n*c^2 yields c below.
    This is an observer, never a replacement for the preregistered solver.
    """
    if target_response.shape[1]!=1:raise ValueError('Analytic CONST check is binary only')
    r=target_response.shape[2]
    u=target_response.mean(0).reshape(r)
    return ((r+1)/(n*(r+1)+lambda_w))*u[:,None].expand(r,n)


def run(stage,*,do_numerical=True):
    torch.set_num_threads(1)
    stage=Path(stage).resolve()
    manifest=json.loads((stage/'experiment_manifest.json').read_text())
    if do_numerical:
        start=time.perf_counter();rows=[]
        p0=Path(manifest['settings']['p0_stage'])
        for entry in json.loads((stage/'p0_cache_manifest.json').read_text())['files']:
            with np.load(p0/entry['cache'],allow_pickle=False) as payload:
                tensors={k:torch.from_numpy(payload[k].copy()) for k in ['source_h','target_h','source_A','target_A']}
            n=len(tensors['source_h'])
            kernels,_=kernel_family(tensors['source_h'],tensors['target_h'],tensors['source_A'],tensors['target_A'],torch.arange(n))
            symmetric=(kernels['HA'][0]+kernels['HA'][0].mT)/2
            started=time.perf_counter();torch.linalg.eigh(symmetric);factor_seconds=time.perf_counter()-started
            approximate,info=solve_observed(kernels['CONST'],tensors['target_A'],'JOINT',.01,
                {'mode':'joint','lambda_w':.01,'max_iterations':200,'gap_atol':1e-8,'gap_rtol':1e-6})
            exact=constant_binary_readout(tensors['target_A'],n,.01)
            denom=max(float(exact.norm()),1e-12)
            cosine=float((approximate*exact).sum()/(approximate.norm()*exact.norm())) if exact.norm()>1e-12 else None
            rows.append(dict(cache=entry['cache'],input_cache_sha256=entry['sha256'],source_graphs=n,
                source_eigh_seconds=factor_seconds,CONST_relative_weight_error=float((approximate-exact).norm())/denom,
                CONST_weight_cosine=cosine,CONST_gap=info['gap'],CONST_converged=info['converged']))
        export_csv(stage/'factorization_and_CONST_audit.csv',rows)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
        report=dict(scope='Auxiliary unlabeled measurement; no replacement of formal W, gradients, criteria or flags',
            code_commit=commit,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_eigh_timing='Separate single-worker rerun on same360 K_SS; formal solve_seconds includes its internal factorization; no efficiency comparison claim',
            CONST_check='Binary all-ones-kernel exact minimizer, post-hoc numerical audit of nonconverged control',
            blocks=len(rows),target_labels_read=0,elapsed_seconds=time.perf_counter()-start,
            source_eigh_median_seconds=statistics.median(r['source_eigh_seconds'] for r in rows),
            CONST_max_relative_weight_error=max(r['CONST_relative_weight_error'] for r in rows),
            CONST_min_weight_cosine=min(r['CONST_weight_cosine'] for r in rows if r['CONST_weight_cosine'] is not None),
            CONST_nonconverged=sum(not r['CONST_converged'] for r in rows))
        (stage/'numerical_supplement.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report),flush=True)
    matched_plots(stage)


def matched_plots(stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((stage/'step_seed_summary.csv').open()))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,task in enumerate(['0_to_1','0_to_2','0_to_3']):
        for row_index,experiment,arms in [(0,'D3',['MEAN','ISO','JOINT']),(1,'D4',['H','A','HA','CONST','HA_PERM'])]:
            ax=axes[row_index,col]
            for offset,scale in [(-.17,'raw'),(.17,'norm_matched')]:
                groups=[]
                for arm in arms:
                    values=[float(r['R_Q'])*100 for r in rows if r['experiment_id']==experiment and r['task']==task
                            and r['arm']==arm and r['scale']==scale and (experiment!='D3' or float(r['alpha'])==.005)]
                    groups.append(values)
                ax.bar(np.arange(len(arms))+offset,[statistics.mean(g) for g in groups],width=.34,
                    yerr=[statistics.stdev(g) for g in groups],capsize=2,label=scale)
            ax.set_xticks(np.arange(len(arms)),arms,rotation=25);ax.axhline(0,color='black',lw=.7)
            ax.set_title(experiment+' '+task.replace('_to_','→'));ax.set_ylabel('Extra relative B-pool CE reduction (%)')
    axes[0,0].legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(stage/f'figures/raw_vs_matched_steps.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args();run(args.stage,do_numerical=not args.plots_only)
