"""Frozen A4-a learning curves. Fits readers, never updates a GNN."""
from workflows._paths import REPO_ROOT
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from workflows.v2.p0.p0_experiment import read, write, ResearchExperiment, ExperimentConfig
from workflows.v2.assumptions.assumptions_experiment import cpu_seconds, append
from tqg_gda.assumption_diagnostics import ProbeLabelBoundary, ordered, split_folds, stable_seed, metrics
from tqg_gda.optimization_diagnostics import (bandwidth_kernel, effective_lambda, kernel_factor,
    ridge_grid, choose_kernel, mean_readout, check_mean_identity)
from tqg_gda.d34_diagnostics import solve_observed
from tqg_gda.utils import sha256, seed_everything


class OptimizationA4:
    def __init__(self, repo, stage, config):
        self.repo, self.stage = Path(repo).resolve(), Path(stage).resolve()
        self.code = REPO_ROOT
        self.config = str(Path(config).resolve())
        self.settings = read(self.config)
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.code, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.code):
            raise ValueError('Execution code must be clean')
        self.plan_sha = sha256(self.stage.parent/'stage2_method/method.md')
        freeze = read(self.stage.parent/'stage2_method/freeze_manifest.json')
        assert self.plan_sha == freeze['files']['method.md']
        assert sha256(self.config) == freeze['files']['experiment_plan.yaml']
        self.old = ResearchExperiment(ExperimentConfig(str(self.repo), self.settings['old_stage'],
                                    str(self.code/'configs/p0_diagnostics.json')))
        self.hardware = self.old.hardware
        if self.settings['final_labels_allowed'] or self.settings['device'] != 'cpu':
            raise ValueError('Frozen CPU / no-final protocol')

    def prepare(self):
        self.stage.mkdir(exist_ok=True)
        for sub in ['A4', 'D0', 'figures']:
            (self.stage/sub).mkdir(exist_ok=True)
        screen = []
        for rel in ['workflows/v2/optimization/optimization_a4.py', 'tqg_gda/optimization_diagnostics.py']:
            tree = ast.parse((self.code/rel).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['exec', 'eval', 'compile']:
                    screen.append(dict(file=rel, line=node.lineno))
                if isinstance(node, ast.Import) and any(x.name.split('.')[0] in ['requests', 'socket', 'urllib'] for x in node.names):
                    screen.append(dict(file=rel, line=node.lineno))
        assert not screen
        write(self.stage/'D0/ast_screen_a4.json', dict(passed=True, forbidden_findings=screen,
              local_subprocess_use='git identity and CPU worker orchestration; no network in experiment'))
        write(self.stage/'A4/manifest.json', dict(git_commit=self.commit, protocol_sha256=self.plan_sha,
            config=self.settings, config_sha256=sha256(self.config), hardware=self.hardware,
            expected_units=960, expected_inner_fits=276480, expected_refits=92160,
            expected_logical_fits=368640, final_labels_allowed=False, versions={k:importlib.metadata.version(k)
            for k in ['torch', 'torch-geometric', 'numpy', 'scipy', 'tensorboard']}))

    def features(self, seed, domain):
        path = Path(self.settings['feature_stage'])/f'features/{seed}_{domain}.npz'
        role = read(path.with_suffix('.json'))
        if sha256(path) != role['feature_sha256']:
            raise ValueError('Feature SHA mismatch')
        d = dict(np.load(path, allow_pickle=False));ids = d['ids'].tolist()
        assert ids == role['query']+role['A']+role['B'] and len(role['query']) == 196
        assert not set(role['query']) & set(role['fit'])
        h, a = torch.from_numpy(d['h']), torch.from_numpy(d['a'].reshape(len(ids), -1))
        hs = float(torch.quantile(torch.pdist(h[:196]).square(), .5).clamp_min(1e-16))
        aa = float(torch.quantile(torch.pdist(a[:196]).square(), .5).clamp_min(1e-16))
        dh = torch.cdist(h, h, compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/hs
        da = torch.cdist(a, a, compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/aa
        return d, role, {'H':dh, 'HA':dh+da}, dict(h_squared=hs, a_squared=aa, feature_sha256=sha256(path))

    def d0(self):
        start = cpu_seconds();seed_everything(9041, 1, True);cases=[]
        for q in [1, 2, 3]:
            x=torch.randn(11, 6, dtype=torch.float64)
            gram=torch.exp(-torch.cdist(x,x).square()/3)
            response=torch.randn(4,q,5,dtype=torch.float64)
            residual=torch.randn(7*q,dtype=torch.float64)
            cases.append(dict(q=q, **check_mean_identity(gram[:7,:7],gram[:7,7:],gram[7:,7:],response,
                                residual,.01,self.settings['calibration'])))
        for domain in self.settings['domains']:
            d, role, distances, scale=self.features(3405,domain)
            ids=d['ids'].tolist();target=[ids.index(i) for i in role['target_batch']]
            for signature in self.settings['signatures']:
                k=bandwidth_kernel(distances[signature],1)
                residual=torch.from_numpy(((d['source_y']-d['p'][:196,1])*(d['q'][1]-d['q'][0])))
                checks=check_mean_identity(torch.from_numpy(k[:196,:196]),torch.from_numpy(k[:196,target]),
                    torch.from_numpy(k[np.ix_(target,target)]),torch.from_numpy(d['a'][target]),residual,.005,self.settings['calibration'])
                cases.append(dict(domain=domain,signature=signature,**checks))
        passed=all(c['mean_vs_pointwise_max']<1e-8 and c['mean_vs_iso_max']<1e-8 for c in cases)
        write(self.stage/'D0/mean_identity.json',dict(passed=passed,cases=cases,cpu_seconds=cpu_seconds()-start,
                                                    final_labels_read=0,target_labels_read=0))
        if not passed:raise ValueError('MEAN/ISO/pointwise algebra disagreement')

    def worker(self, seed, pilot=False):
        seed_everything(seed,1,True)
        out=self.stage/'A4'/(f'pilot_{seed}' if pilot else str(seed));out.mkdir(exist_ok=False)
        (out/'models').mkdir();writer=SummaryWriter(str(out/'tensorboard'))
        started_cpu,started_wall=cpu_seconds(),time.time()
        base=dict(seed=seed,git_commit=self.commit,protocol_sha256=self.plan_sha,config_name=self.settings['config_name'],
                  gpu_hours=0.,hardware=self.hardware,event_flags='clean')
        write(out/'RUN_INFO.json',dict(base,started_at=started_wall,pilot=pilot))
        handles={name:(out/f'{name}.csv').open('w',newline='') for name in ['inner_cv','curves']}
        csv_writers={};units=fits=factors=0
        def row(name,record):
            if name not in csv_writers:
                csv_writers[name]=csv.DictWriter(handles[name],fieldnames=list(record));csv_writers[name].writeheader()
            csv_writers[name].writerow(record)
        try:
            for domain in ([1] if pilot else self.settings['domains']):
                d,role,distances,scales=self.features(seed,domain);ids=d['ids'].tolist();pos={i:j for j,i in enumerate(ids)}
                query,apool,bpool=role['query'],role['A'],role['B'];bp=[pos[i] for i in bpool]
                task=self.old.task(domain)
                boundary=ProbeLabelBoundary(task,apool,bpool,role['final'],out/'label_access.jsonl')
                source_y=dict(zip(query,d['source_y'].tolist()));outer=split_folds(query,4,'outer')
                target_batch=[pos[i] for i in role['target_batch']];target_a=d['a'][target_batch]
                all_kernels={sig:{c:bandwidth_kernel(distances[sig],c) for c in self.settings['bandwidth_multipliers']}
                             for sig in self.settings['signatures']}
                jobs=[];model_files=[]
                for fold in range(1 if pilot else 4):
                    source_audit=sorted(outer[fold]);source_train=sorted(set(query)-set(source_audit))
                    target_train=sorted(ordered(apool,'target_train',domain,fold)[:147])
                    target_y=dict(zip(target_train,boundary.read('A_train',target_train).tolist()))
                    for train_domain,pool in [('S',source_train),('T',target_train)]:
                        nested=ordered(pool,'A4-nested',domain,fold,train_domain)
                        for n in ([32,147] if pilot else self.settings['curve_sizes']):
                            train_ids=sorted(nested[:n]);tridx=[pos[i] for i in train_ids]
                            truth=source_y if train_domain=='S' else target_y
                            true=np.array([truth[i] for i in train_ids])
                            ys=np.array([true]+[true[np.random.default_rng(stable_seed('A4-null',seed,domain,fold,train_domain,n,rep)).permutation(n)]
                                              for rep in range(self.settings['null_repetitions'])])
                            inner=split_folds(train_ids,3,'A4-inner',fold)
                            extra=source_audit if train_domain=='S' else []
                            eval_ids=train_ids+extra+bpool;ev=[pos[i] for i in eval_ids]
                            for sig in self.settings['signatures']:
                                key=f'{domain}_{fold}_{train_domain}_{n}_{sig}'
                                meta=dict(base,task=f'0_to_{domain}',fold=fold,train_domain=train_domain,n=n,signature=sig)
                                cv=np.zeros((2,4,3,4));full=np.empty((2,4,3,4,len(ev)))
                                alpha=np.empty((2,4,3,4,n));spectra=[];dimensions=np.zeros((2,3,4));norms=np.zeros((2,4,3,4))
                                for ci,c in enumerate(self.settings['bandwidth_multipliers']):
                                    kernel=all_kernels[sig][c]
                                    for inner_id,va_ids in enumerate(inner):
                                        va=[train_ids.index(i) for i in va_ids];tr=sorted(set(range(n))-set(va))
                                        ix=np.array(tridx)[tr];vx=np.array(tridx)[va]
                                        kt=kernel[np.ix_(ix,ix)];ke=kernel[np.ix_(vx,ix)];factor=kernel_factor(kt);factors+=1
                                        for co,convention in enumerate(self.settings['lambda_conventions']):
                                            lam=[effective_lambda(v,len(tr),147,convention) for v in self.settings['kernel_lambdas']]
                                            fitted=ridge_grid(kt,ke,ys[:,tr],d['p'][ix,1],d['p'][vx,1],lam,factor)
                                            fits+=16
                                            for variant in range(4):
                                                for li,lbase in enumerate(self.settings['kernel_lambdas']):
                                                    score=metrics(ys[variant,va],fitted.probabilities[variant,li])['nll']
                                                    cv[co,variant,ci,li]+=score/3
                                                    row('inner_cv',dict(meta,convention=convention,variant=variant,bandwidth=c,
                                                        lambda_base=lbase,lambda_actual=lam[li],inner_fold=inner_id,n_train=len(tr),n_audit=len(va),nll=score))
                                    kt=kernel[np.ix_(tridx,tridx)];ke=kernel[np.ix_(ev,tridx)];factor=kernel_factor(kt);factors+=1
                                    spectra.append(factor[0])
                                    for co,convention in enumerate(self.settings['lambda_conventions']):
                                        lam=[effective_lambda(v,n,147,convention) for v in self.settings['kernel_lambdas']]
                                        fitted=ridge_grid(kt,ke,ys,d['p'][tridx,1],d['p'][ev,1],lam,factor)
                                        fits+=16;full[co,:,ci]=fitted.probabilities;alpha[co,:,ci]=fitted.alpha
                                        dimensions[co,ci]=fitted.effective_dimensions
                                        norms[co,:,ci]=fitted.rkhs_squared*float((d['q'][1]-d['q'][0])**2)
                                selection=[];readout_stats=[]
                                for co,convention in enumerate(self.settings['lambda_conventions']):
                                    for variant in range(4):
                                        ci,li=choose_kernel(cv[co,variant],self.settings['bandwidth_multipliers'],self.settings['kernel_lambdas'])
                                        c=self.settings['bandwidth_multipliers'][ci];lbase=self.settings['kernel_lambdas'][li]
                                        lam=effective_lambda(lbase,n,147,convention);kernel=all_kernels[sig][c]
                                        kt=kernel[np.ix_(tridx,tridx)];cross=kernel[np.ix_(tridx,target_batch)]
                                        wm=mean_readout(kt,cross,target_a,lam)
                                        wj,info=solve_observed((torch.from_numpy(kt),torch.from_numpy(cross),
                                            torch.from_numpy(kernel[np.ix_(target_batch,target_batch)])),torch.from_numpy(target_a),'JOINT',2*lam,self.settings['calibration'])
                                        selection.append(dict(convention=convention,variant=variant,bandwidth_index=ci,lambda_index=li,
                                            bandwidth=c,lambda_base=lbase,lambda_actual=lam,cv_nll=float(cv[co,variant,ci,li])))
                                        readout_stats.append(dict(convention=convention,variant=variant,mean_weight_norm=float(np.linalg.norm(wm)),
                                            joint_weight_norm=float(wj.norm()),joint_lambda=2*lam,joint_solver=info))
                                path=out/'models'/f'{key}.npz'
                                np.savez_compressed(path,predictions=full,alpha=alpha,train_ids=np.array(train_ids),eval_ids=np.array(eval_ids),
                                    training_labels=ys,cv=cv,kernel_eigenvalues=np.array(spectra),effective_dimension=dimensions,
                                    contrast_rkhs_squared=norms,query_contrast=d['q'])
                                write(path.with_suffix('.json'),dict(meta,inner_validation_ids=inner,source_audit=extra,B=bpool,
                                    selected=selection,readout_statistics=readout_stats,source_geometry=scales,model_sha256=sha256(path)))
                                model_files.extend([dict(path=str(p.relative_to(out)),sha256=sha256(p)) for p in [path,path.with_suffix('.json')]])
                                jobs.append(dict(meta=meta,path=path,source_audit=extra,train_ids=train_ids,selected=selection))
                                units+=1
                                elapsed=cpu_seconds()-started_cpu
                                append(out/'history.jsonl',dict(event='unit_complete',key=key,units=units,logical_fits=fits,cpu_seconds=elapsed))
                                writer.add_scalar('compute/cpu_seconds',elapsed,units)
                                write(out/'progress.json',dict(seed=seed,domain=domain,fold=fold,units=units,logical_fits=fits,
                                      cpu_seconds=elapsed,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                                if elapsed>self.settings['cpu_hour_cap']*3600/5 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>self.settings['memory_cap_gib']*1024**2/5:
                                    raise TimeoutError('Per-worker resource allocation exhausted')
                    print(json.dumps(dict(event='A4_fold',seed=seed,domain=domain,fold=fold,units=units,cpu_seconds=cpu_seconds()-started_cpu)),flush=True)
                handles['inner_cv'].flush()
                seal=out/f'{domain}_sealed_artifacts.json'
                write(seal,dict(domain=domain,models=model_files,feature_sha256=scales['feature_sha256'],
                    cv_prefix_bytes=(out/'inner_cv.csv').stat().st_size,cv_prefix_sha256=sha256(out/'inner_cv.csv')))
                boundary.seal(sha256(seal),len(jobs)*96)
                if pilot:continue
                by=boundary.read('B_evaluation',bpool)
                for job in jobs:
                    z=dict(np.load(job['path'],allow_pickle=False));meta=job['meta'];n=meta['n'];extra=job['source_audit']
                    sa=np.array([source_y[i] for i in extra]);true=z['training_labels'][0]
                    for co,convention in enumerate(self.settings['lambda_conventions']):
                        for variant in range(4):
                            chosen=job['selected'][co*4+variant];train_y=z['training_labels'][variant]
                            for ci,c in enumerate(self.settings['bandwidth_multipliers']):
                                for li,lbase in enumerate(self.settings['kernel_lambdas']):
                                    prediction=z['predictions'][co,variant,ci,li]
                                    pools=[('TRAIN',np.arange(n),train_y),('B',np.arange(n+len(extra),len(prediction)),by)]
                                    if len(extra):pools.append(('S_audit',np.arange(n,n+len(extra)),sa))
                                    for pool,ix,y in pools:
                                        oldp=d['p'][[pos[int(z['eval_ids'][i])] for i in ix],1]
                                        record=dict(meta,convention=convention,variant=variant,bandwidth=c,lambda_base=lbase,
                                            lambda_actual=effective_lambda(lbase,n,147,convention),pool=pool,
                                            selected=ci==chosen['bandwidth_index'] and li==chosen['lambda_index'],
                                            cv_nll=float(z['cv'][co,variant,ci,li]),effective_dimension=float(z['effective_dimension'][co,ci,li]),
                                            contrast_rkhs_norm=float(np.sqrt(max(0,z['contrast_rkhs_squared'][co,variant,ci,li]))),
                                            prior_nll=metrics(y,np.full(len(y),train_y.mean()))['nll'],model_nll=metrics(y,oldp)['nll'],
                                            **{('evaluation_n' if k=='n' else k):v for k,v in metrics(y,prediction[ix]).items()})
                                        row('curves',record)
                handles['curves'].flush()
            result=dict(seed=seed,pilot=pilot,completed=True,units=units,logical_fits=fits,physical_factorizations=factors,
                        cpu_seconds=cpu_seconds()-started_cpu,elapsed=time.time()-started_wall,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,git_commit=self.commit)
            write(out/'done.json',result);return result
        except Exception:
            write(out/'failed.json',dict(cpu_seconds=cpu_seconds()-started_cpu,traceback=traceback.format_exc(),units=units))
            raise
        finally:
            writer.close()
            for handle in handles.values():handle.close()

    def run(self):
        pilot=read(self.stage/'A4/pilot_3405/done.json');estimate=pilot['cpu_seconds']*960/pilot['units']*self.settings['compute_buffer']
        used=pilot['cpu_seconds']+read(self.stage/'D0/mean_identity.json')['cpu_seconds']
        if estimate+used>self.settings['cpu_hour_cap']*3600:raise TimeoutError('Pilot exceeds CPU budget')
        write(self.stage/'A4/compute_estimate.json',dict(pilot_cpu_seconds=pilot['cpu_seconds'],estimated_formal_cpu_seconds=estimate,
             cap_cpu_hours=self.settings['cpu_hour_cap'],units=960))
        start=time.time();results=[]
        with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(run_worker,str(self.repo),str(self.stage),self.config,seed):seed for seed in self.settings['seeds']}
            while pending:
                done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:results.append(f.result());del pending[f]
                states=[]
                for seed in self.settings['seeds']:
                    p=self.stage/f'A4/{seed}/progress.json'
                    if p.exists():states.append(read(p))
                print(json.dumps(dict(event='A4_progress',elapsed=time.time()-start,pending=list(pending.values()),states=states)),flush=True)
                current=used+sum(s['cpu_seconds'] for s in states)
                write(self.stage/'A4/compute_ledger.json',dict(status='running' if pending else 'completed',tracked_cpu_seconds=current,
                    completed_workers=results,elapsed_wall=time.time()-start,gpu_hours=0.,states=states))
                if current>self.settings['cpu_hour_cap']*3600:
                    for p in pool._processes.values():p.terminate()
                    raise TimeoutError('Global CPU hard cap')
        total=used+sum(r['cpu_seconds'] for r in results)
        write(self.stage/'A4/done.json',dict(completed=True,units=sum(r['units'] for r in results),
             logical_fits=sum(r['logical_fits'] for r in results),cpu_seconds_including_pilot_d0=total,workers=results,
             phase='A4 only; A2b/A6 not yet complete'))


def run_worker(repo,stage,config,seed):
    return OptimizationA4(repo,stage,config).worker(seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','d0','pilot','run']);p.add_argument('--repo',required=True)
    p.add_argument('--stage',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    torch.set_num_threads(1);e=OptimizationA4(a.repo,a.stage,a.config)
    if a.action=='prepare':e.prepare()
    elif a.action=='d0':e.d0()
    elif a.action=='pilot':e.worker(3405,True)
    else:e.run()


if __name__=='__main__':main()
