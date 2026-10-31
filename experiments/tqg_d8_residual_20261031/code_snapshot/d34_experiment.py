"""Frozen D3 calibration and D4 signature diagnostics; no adaptation trajectory."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import time
import traceback
import unittest

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from p0_experiment import ExperimentConfig,ResearchExperiment as P0Experiment,read,write
from tqg_gda.feedback_v2 import ParameterSnapshot
from tqg_gda.p0_diagnostics import (DevelopmentOracle,Geometry,append_json,cosine,evaluate_flat,
    gradient_metrics,gradient_of_ce,label_free,model_hash,normalized,sample_ids,tensor_hash,unlabeled_geometry)
from tqg_gda.d34_diagnostics import (SourceAuditOracle,kernel_family,kernel_statistics,
    solve_observed,source_residual)
from tqg_gda.utils import sha256,seed_everything


class ResearchExperiment(P0Experiment):
    def __init__(self,config):
        super().__init__(config)
        self.p0=Path(self.settings['p0_stage']).resolve()
        self.cache_index={r['path']:r for r in read(self.p0/'local_geometry_manifest.json')['files']}
        if self.settings['D4_reader']!='JOINT' or self.settings['D4_optional_MEAN_matrix']:
            raise ValueError('Frozen D4 matrix uses joint for every kernel')
        if self.settings['D3_step_blocks']!=self.plan['D2']['first_blocks']:
            raise ValueError('D3 one-step blocks must match the predeclared D2 subset')

    def ast_screen(self):
        findings=[]
        files=[self.code/'d34_experiment.py',self.code/'d34_summarize.py',self.code/'tqg_gda/d34_diagnostics.py']
        for p in files:
            if not p.exists():continue
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('eval','exec','compile'):
                    findings.append(dict(file=str(p),line=node.lineno))
        write(self.stage/'ast_screen.json',dict(passed=not findings,findings=findings,
              local_subprocess_justification='Frozen Git metadata and local CPU process pool; no experiment network'))
        if findings:raise ValueError('AST screen failed')

    def prepare(self):
        if not read(self.p0/'D0/engineering_checks.json')['passed']:raise ValueError('P0 D0 was not passed')
        super().prepare()
        checked=[]
        for seed in self.plan['seeds']:
            for domain in (1,2,3):
                for step in self.plan['snapshots']['times']:
                    for block in range(8):
                        path=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
                        if sha256(self.p0/path)!=self.cache_index[path]['sha256']:
                            raise ValueError('P0 cache differs from immutable manifest')
                        indices=self.p0/'logs'/str(seed)/f'{domain}_{seed}_{step}_{block}_indices.json'
                        checked.append(dict(cache=path,sha256=sha256(self.p0/path),indices=str(indices),indices_sha256=sha256(indices)))
        write(self.stage/'p0_cache_manifest.json',dict(source_run=str(self.p0),files=checked,
            source_execution_commit=read(self.p0/'experiment_manifest.json')['git_commit']))
        manifest=read(self.stage/'experiment_manifest.json')
        for key in ['expected_D1_blocks','expected_D2_blocks','expected_D2_temporary_updates']:manifest.pop(key,None)
        manifest.update(scope='D3/D4 only; frozen geometry and independent functional single steps',
            expected_D3_solves=5400,expected_D4_target_blocks=360,expected_D4_source_blocks=120,
            expected_D4_gradient_rows=2400,expected_step_rows=8400,
            D3_step_rows=5400,D4_step_rows=3000,
            D4_HA_target_reused_from_D3=360,
            source_geometry_builder_auxiliary_HA_solves=120,
            execution_resolution=dict(D3_steps='D2 first4 blocks, eta0.0003, beta0.1, all15 arms raw and equal norm',
                D3_equal_norm_reference='MEAN alpha0.005',D3_incremental_reference='ISO, matched alpha0.005',
                D4_equal_norm_reference='H',D4_incremental_reference='H at same joint lambda0.01',
                HA_PERM_seed_offset=self.settings['signature_permutation_offset'],
                source_fit_seed='source block seed +1000',source_bank_order='original IDs sorted',
                source_effect_comparison='Bank98 vs real196 is not a matched-count causal domain comparison',
                D4_optional_MEAN='Not run; all five kernels use joint0.01 regardless of D3 outcome'))
        write(self.stage/'experiment_manifest.json',manifest)
        print(json.dumps(dict(event='D34_prepared',caches_verified=len(checked))),flush=True)

    def cached_geometry(self,model,domain,seed,step,block):
        rel=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
        if sha256(self.p0/rel)!=self.cache_index[rel]['sha256']:raise ValueError('Cached geometry changed')
        with np.load(self.p0/rel,allow_pickle=False) as data:
            tensors={k:torch.from_numpy(data[k].copy()) for k in data.files}
        snapshot=ParameterSnapshot(model)
        if tensors['V'].shape[0]!=len(snapshot.flat):raise ValueError('Parameter layout mismatch')
        return Geometry(snapshot,tensors['V'],tensors['Q'],tensors['source_A'],tensors['target_A'],
            tensors['W'],tensors['source_p'],tensors['source_h'],tensors['target_h'],{}),rel

    def profiles(self,geo,block_seed,include_D3):
        generator=torch.Generator().manual_seed(block_seed+self.settings['signature_permutation_offset'])
        permutation=torch.randperm(len(geo.source_probabilities),generator=generator)
        started=time.perf_counter()
        kernels,distances=kernel_family(geo.source_representation,geo.target_representation,
            geo.source_response,geo.target_response,permutation,scale_epsilon=self.v2.scale_epsilon)
        kernel_seconds=time.perf_counter()-started
        profiles=[]
        if include_D3:
            for alpha in self.plan['D3']['alpha_grid']:
                for mode in ('MEAN','ISO','JOINT'):
                    lam=alpha if mode=='MEAN' else 2*alpha
                    w,info=solve_observed(kernels['HA'],geo.target_response,mode,lam,self.v2.calibration)
                    profiles.append(dict(experiment_id='D3',arm=mode,alpha=alpha,lambda_w=lam,
                                         kernel='HA',W=w,info=info,physical_solve_seconds=info['solve_seconds']))
        for name in self.plan['D4']['kernels']:
            reused=next((p for p in profiles if p['arm']=='JOINT' and p['alpha']==self.plan['D3']['primary_alpha']),None) if name=='HA' else None
            if reused is not None:
                w,info=reused['W'],dict(reused['info']);physical=0.
            else:
                w,info=solve_observed(kernels[name],geo.target_response,'JOINT',self.settings['D4_lambda_w'],self.v2.calibration)
                physical=info['solve_seconds']
            profiles.append(dict(experiment_id='D4',arm=name,alpha=None,lambda_w=self.settings['D4_lambda_w'],
                kernel=name,W=w,info=info,physical_solve_seconds=physical,solver_reused_from_D3=reused is not None))
        if include_D3:
            for alpha in self.plan['D3']['alpha_grid']:
                pair=[next(p for p in profiles if p['experiment_id']=='D3' and p['arm']==arm and p['alpha']==alpha)
                      for arm in ('MEAN','ISO')]
                error=float((pair[0]['W']-pair[1]['W']).norm())/max(float(pair[0]['W'].norm()),1e-12)
                if error>self.settings['relative_tolerance']:raise ValueError('ISO/MEAN equivalence failed')
                for p in pair:p['info']['ISO_MEAN_relative_error']=error
            reference=next(p['W'] for p in profiles if p['experiment_id']=='D3' and p['arm']=='JOINT' and p['alpha']==.005)
            relative=float((reference-geo.readout).norm())/max(float(geo.readout.norm()),1e-12)
            if relative>self.settings['relative_tolerance']:raise ValueError('D3 original arm does not replay P0 W')
        else:relative=None
        return profiles,kernels,dict(distances,kernel_build_seconds=kernel_seconds,
            source_signature_permutation=permutation.tolist(),P0_W_relative_replay_error=relative)

    def analyze(self,model,geo,profiles,kernels,distances,source_y,a_input,b_batches,fit,
                oracle,token,ids,base,out,writer,count,perform_steps,b_cache):
        started=time.perf_counter()
        residual=source_residual(geo.source_probabilities,source_y,geo.contrasts)
        for p in profiles:p['gradient']=-(geo.basis@(p['W']@residual))
        hashes={f'{p["experiment_id"]}/{p["arm"]}/{p["alpha"]}':tensor_hash(p['W']) for p in profiles}
        seal_digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        append_json(out/'readout_seals.jsonl',dict(token=token,profiles=hashes,time=time.time(),**base))
        oracle.seal(token,ids['A'],theta_hash=tensor_hash(geo.snapshot.flat),readout_hash=seal_digest)
        a_labels=oracle.labels(ids['A'],token=token,purpose='D3_D4_gradient_and_step_A')
        g_a=gradient_of_ce(geo.snapshot,[a_input],a_labels)
        if not b_cache:
            b_labels=oracle.labels(ids['B'],token=token,purpose='D3_D4_full_B_gradient_evaluation')
            b_cache.update(labels=b_labels,gradient=gradient_of_ce(geo.snapshot,b_batches,b_labels),
                           metrics=evaluate_flat(geo.snapshot,geo.snapshot.flat,b_batches,b_labels))
        b_labels,g_b,base_b=b_cache['labels'],b_cache['gradient'],b_cache['metrics']
        g_source=gradient_of_ce(geo.snapshot,[label_free(fit)],fit.y)
        d3_reference=next((p for p in profiles if p['experiment_id']=='D3' and p['arm']=='MEAN' and p['alpha']==.005),None)
        d4_reference=next(p for p in profiles if p['experiment_id']=='D4' and p['arm']=='H')
        if perform_steps:
            base_a=evaluate_flat(geo.snapshot,geo.snapshot.flat,[a_input],a_labels)
            source_a=evaluate_flat(geo.snapshot,geo.snapshot.flat-self.plan['D2']['primary_eta']*g_source,[a_input],a_labels)
            source_b=evaluate_flat(geo.snapshot,geo.snapshot.flat-self.plan['D2']['primary_eta']*g_source,b_batches,b_labels)
            append_json(out/'reference_blocks.jsonl',dict(base,base_CE_A=base_a['ce'],base_CE_B=base_b['ce'],
                source_CE_A=source_a['ce'],source_CE_B=source_b['ce'],source_BA_B=100*source_b['balanced_accuracy']))
        audit_residual=((torch.nn.functional.one_hot(a_labels,geo.source_probabilities.shape[1]).double()
                         -geo.snapshot.logits(a_input).detach().softmax(-1))@geo.contrasts).reshape(-1)
        for p in profiles:
            reference=d3_reference if p['experiment_id']=='D3' else d4_reference
            g=p['gradient'];ref=reference['gradient'];w=p['W'];refw=reference['W']
            scalars={k:v for k,v in p['info'].items() if k not in ('omega_eigenvalues','effective_lambda')}
            metrics=gradient_metrics(g,g_a,g_b,g_source,geo.basis,
                epsilon=self.settings['epsilon_gradient'],beta=self.plan['D2']['beta'])
            record=dict(base,experiment_id=p['experiment_id'],arm=p['arm'],alpha=p['alpha'],lambda_w=p['lambda_w'],
                kernel=p['kernel'],rank=geo.basis.shape[1],metric_name='c_A',metric_value=metrics['c_A'],metric_direction='max',
                **metrics,**scalars,
                physical_solve_seconds=p['physical_solve_seconds'],solver_reused_from_D3=p.get('solver_reused_from_D3',False),
                W_hash=tensor_hash(w),W_difference_vs_reference=float((w-refw).norm())/max(float(refw.norm()),1e-12),
                feedback_cosine_vs_reference=cosine(g,ref,self.settings['epsilon_gradient']),
                feedback_norm_ratio_vs_reference=float(g.norm()/ref.norm()) if ref.norm()>self.settings['epsilon_gradient'] else None,
                elapsed_seconds=time.perf_counter()-started)
            filename='calibration_grid_blocks.jsonl' if p['experiment_id']=='D3' else (
                'signature_target_blocks.jsonl' if base['label_scope']=='development_oracle' else 'source_audit_blocks.jsonl')
            append_json(out/filename,record)
            if p['experiment_id']=='D3':
                append_json(out/'omega_spectrum.jsonl',dict(base,alpha=p['alpha'],arm=p['arm'],lambda_w=p['lambda_w'],
                    omega_eigenvalues=p['info']['omega_eigenvalues'],effective_lambda=p['info']['effective_lambda'],
                    delta_min=p['info']['delta_min'],delta_max=p['info']['delta_max'],omega_deviation=p['info']['omega_deviation']))
            else:
                kstats=kernel_statistics(kernels[p['kernel']],source_y,residual,a_labels,audit_residual)
                append_json(out/'kernel_distance_stats.jsonl',dict(base,arm=p['arm'],**distances,**kstats))
            for metric in ('c_A','e_A'):
                if metrics[metric] is not None:writer.add_scalar(f'{p["experiment_id"]}/{base["task"]}/{p["arm"]}/{p["alpha"]}/{metric}',metrics[metric],count)
            if perform_steps:
                for scale in self.settings['step_scales']:
                    if scale=='norm_matched':vector,valid=normalized(g,float(ref.norm()),self.settings['epsilon_gradient'])
                    else:vector,valid=g,float(g.norm())>self.settings['epsilon_gradient']
                    eta,beta=self.plan['D2']['primary_eta'],self.plan['D2']['beta']
                    direction=g_source+beta*vector
                    flat=geo.snapshot.flat-eta*direction
                    ma=evaluate_flat(geo.snapshot,flat,[a_input],a_labels)
                    mb=evaluate_flat(geo.snapshot,flat,b_batches,b_labels)
                    append_json(out/'step_blocks.jsonl',dict(base,experiment_id=p['experiment_id'],arm=p['arm'],alpha=p['alpha'],
                        lambda_w=p['lambda_w'],scale=scale,eta=eta,beta=beta,normalization_reference=reference['arm'],
                        normalization_reference_norm=float(ref.norm()),applied_feedback_norm=float(vector.norm()),
                        valid_comparison=valid,invalid_reason='' if valid else 'zero_or_unmatchable_feedback',
                        CE_A=ma['ce'],CE_B=mb['ce'],base_CE_A=base_a['ce'],base_CE_B=base_b['ce'],
                        source_CE_B=source_b['ce'],R_Q=(source_b['ce']-mb['ce'])/base_b['ce'],
                        BA_B=100*mb['balanced_accuracy'],source_BA_B=100*source_b['balanced_accuracy'],AUROC_B=mb['auroc'],
                        first_order_CE_B=-eta*float(torch.dot(g_b,direction)),
                        metric_name='R_Q',metric_value=(source_b['ce']-mb['ce'])/base_b['ce'],metric_direction='max'))
        return len(profiles)

    def worker(self,seed,*,pilot=False):
        seed_everything(seed,self.settings['threads_per_worker'],True)
        out=self.stage/'logs'/(f'pilot_{seed}' if pilot else str(seed))
        out.mkdir(exist_ok=False)
        writer=SummaryWriter(str(out/'tensorboard'))
        started=time.perf_counter();count=0
        write(out/'RUN_INFO.json',dict(seed=seed,git_commit=self.commit,plan_sha256=self.plan_sha,config_name=self.settings['config_name'],
              hardware=self.hardware,started_at=time.time(),pilot=pilot))
        try:
            for domain in ([1] if pilot else [1,2,3]):
                task=self.task(domain);_,_,dev,final=self.pools(domain)
                oracle=DevelopmentOracle(task,dev,final,out/'label_access.jsonl',enabled=self.settings['allow_development_oracle'])
                for step in ([0] if pilot else self.plan['snapshots']['times']):
                    model,row=self.load(domain,seed,step);original=model_hash(model);b_cache={}
                    for block in range(1 if pilot else 8):
                        t=time.perf_counter();token=f'target_{domain}_{seed}_{step}_{block}'
                        input_index=self.p0/'logs'/str(seed)/f'{domain}_{seed}_{step}_{block}_indices.json'
                        ids=read(input_index)
                        geo,cache_path=self.cached_geometry(model,domain,seed,step,block)
                        profiles,kernels,distances=self.profiles(geo,ids['block_seed'],True)
                        copied=out/f'{token}_indices.json';write(copied,dict(ids,geometry_source=cache_path))
                        base=self.base(row,block,str(copied.relative_to(self.stage)))
                        base.update(scope='target',fold=-1,original_model_hash=original,P0_geometry=cache_path,
                                    P0_geometry_sha256=self.cache_index[cache_path]['sha256'])
                        self.analyze(model,geo,profiles,kernels,distances,task.batch(ids['query'],'query').y,
                            task.batch(ids['A'],'target'),self.batches(task,ids['B']),task.batch(ids['fit'],'fit'),
                            oracle,token,ids,base,out,writer,count,block in self.settings['D3_step_blocks'],b_cache)
                        if model_hash(model)!=original:raise ValueError('Target diagnostic changed original model')
                        append_json(out/'temporary_step_checks.jsonl',dict(token=token,passed=True,original_model_unchanged=True))
                        count+=1
                        write(out/'progress.json',dict(seed=seed,blocks_complete=count,scope='target',domain=domain,step=step,
                            block=block,elapsed_seconds=time.perf_counter()-started,last_block_seconds=time.perf_counter()-t))
                        if count%4==0 or pilot:print(json.dumps(dict(event='target_block',seed=seed,domain=domain,step=step,block=block)),flush=True)
            task=self.task(1)
            ordered=sorted(task.query_ids,key=lambda i:hashlib.sha256(f'D4|{i}'.encode()).hexdigest())
            halves=[ordered[:98],ordered[98:]]
            for step in ([0] if pilot else self.plan['snapshots']['times']):
                model,row=self.load(1,seed,step);original=model_hash(model)
                for fold in ([0] if pilot else [0,1]):
                    bank=tuple(sorted(halves[fold]));audit=tuple(sorted(halves[1-fold]))
                    source_oracle=SourceAuditOracle(task,bank,audit,out/'source_label_access.jsonl',enabled=self.settings['allow_source_audit_oracle'])
                    source=task.batch(bank,'query');source_y=source.y.clone()
                    for block in range(1 if pilot else 4):
                        t=time.perf_counter();token=f'source_{seed}_{step}_{fold}_{block}'
                        block_seed=64000000+(seed-3405)*10000+step*10+fold*1000+block
                        a_ids=sample_ids(audit,16,block_seed);b_ids=tuple(i for i in audit if i not in a_ids)
                        fit_ids=sample_ids(task.fit_ids,16,block_seed+self.settings['source_fit_seed_offset'])
                        a_input=label_free(task.batch(a_ids,'query'))
                        geo=unlabeled_geometry(model,label_free(source),a_input,self.v2)
                        profiles,kernels,distances=self.profiles(geo,block_seed,False)
                        ids=dict(A=list(a_ids),B=list(b_ids),fit=list(fit_ids),query=list(bank),audit=list(audit),block_seed=block_seed)
                        path=out/f'{token}_indices.json';write(path,ids)
                        base=self.base(row,block,str(path.relative_to(self.stage)))
                        base.update(task='source_audit',scope='source',label_scope='source_query_audit',fold=fold,original_model_hash=original)
                        self.analyze(model,geo,profiles,kernels,distances,source_y,a_input,
                            [label_free(task.batch(b_ids[i:i+self.settings['evaluation_microbatch']],'query'))
                             for i in range(0,len(b_ids),self.settings['evaluation_microbatch'])],
                            task.batch(fit_ids,'fit'),source_oracle,token,ids,base,out,writer,count,True,{})
                        if model_hash(model)!=original:raise ValueError('Source diagnostic changed original model')
                        append_json(out/'temporary_step_checks.jsonl',dict(token=token,passed=True,original_model_unchanged=True))
                        count+=1
                        write(out/'progress.json',dict(seed=seed,blocks_complete=count,scope='source',step=step,fold=fold,block=block,
                            elapsed_seconds=time.perf_counter()-started,last_block_seconds=time.perf_counter()-t))
                        if count%4==0 or pilot:print(json.dumps(dict(event='source_block',seed=seed,step=step,fold=fold,block=block)),flush=True)
            write(out/'done.json',dict(status='completed',seed=seed,blocks=count,elapsed_seconds=time.perf_counter()-started,
                  git_commit=self.commit,plan_sha256=self.plan_sha,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        except Exception:
            append_json(out/'monitor_events.jsonl',dict(event='failed',time=time.time(),traceback=traceback.format_exc()))
            raise
        finally:writer.close()
        return read(out/'done.json')

    def d0(self):
        seed_everything(3405,1,True)
        out=self.stage/'D0';out.mkdir(exist_ok=False)
        with (out/'unit_checks.log').open('w') as stream:
            suite=unittest.TestSuite()
            for pattern in ['test_d34_diagnostics.py','test_v2_math.py','test_p0_diagnostics.py']:
                suite.addTests(unittest.defaultTestLoader.discover(str(self.code/'tests'),pattern=pattern))
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not result.wasSuccessful():
            write(out/'engineering_checks.json',dict(passed=False,tests=result.testsRun))
            raise AssertionError('D3/D4 engineering suite failed')
        replays=[]
        for domain in (1,2,3):
            model,row=self.load(domain,3405,0)
            geo,rel=self.cached_geometry(model,domain,3405,0,0)
            ids=read(self.p0/'logs/3405'/f'{domain}_3405_0_0_indices.json')
            profiles,kernels,distances=self.profiles(geo,ids['block_seed'],True)
            task=self.task(domain)
            source=label_free(task.batch(ids['query'],'query'))
            a=task.batch(ids['A'],'target')
            predictions=geo.snapshot.logits(source).softmax(-1).detach()
            p_error=float((predictions-geo.source_probabilities).norm())
            jac=geo.snapshot.jacobian(a,geo.contrasts)
            expected=(jac@geo.basis).reshape_as(geo.target_response)
            a_error=float((expected-geo.target_response).norm())/max(float(expected.norm()),1e-12)
            if p_error>1e-10 or a_error>1e-8:raise AssertionError('P0 cache/model functional state mismatch')
            mean=next(p['W'] for p in profiles if p['experiment_id']=='D3' and p['arm']=='MEAN' and p['alpha']==.005)
            iso=next(p['W'] for p in profiles if p['experiment_id']=='D3' and p['arm']=='ISO' and p['alpha']==.005)
            error=float((mean-iso).norm())/max(float(mean.norm()),1e-12)
            if error>1e-8:raise AssertionError('Real-data ISO/MEAN equivalence failed')
            replays.append(dict(domain=domain,cache=rel,P0_W_error=distances['P0_W_relative_replay_error'],
                source_probability_error=p_error,target_response_relative_error=a_error,ISO_MEAN_relative_error=error))
        write(out/'engineering_checks.json',dict(passed=True,unit_tests=result.testsRun,real_replays=replays,
              git_commit=self.commit,research_efficacy_not_evaluated=True,final_labels_read=0))
        print(json.dumps(dict(event='D34_D0_passed',tests=result.testsRun,real_replays=replays)),flush=True)

    def run(self):
        if not read(self.stage/'D0/engineering_checks.json')['passed']:raise RuntimeError('D0 required')
        pilot=read(self.stage/'logs/pilot_3405/done.json')
        estimate=pilot['elapsed_seconds']/2*480/self.settings['workers']*self.settings['compute_buffer']*2
        if estimate>self.settings['package_wall_cap_seconds']:raise RuntimeError('Pilot exceeds fixed package budget')
        write(self.stage/'compute_estimate.json',dict(pilot_seconds=pilot['elapsed_seconds'],estimated_wall_seconds=estimate,
            buffer=self.settings['compute_buffer'],additional_contention_factor=2.,cap_seconds=self.settings['package_wall_cap_seconds']))
        started=time.time();finished=0;warned=set()
        print(json.dumps(dict(event='start_D3_D4',estimated_wall_seconds=estimate)),flush=True)
        with ProcessPoolExecutor(max_workers=self.settings['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(run_worker,self.config,seed):seed for seed in self.plan['seeds']}
            while pending:
                completed,_=wait(pending,timeout=self.settings['progress_interval_seconds'],return_when=FIRST_COMPLETED)
                for future in completed:
                    result=future.result();finished+=result['blocks'];del pending[future]
                    print(json.dumps(dict(event='seed_complete',**result)),flush=True)
                elapsed=time.time()-started;states=[]
                for seed in pending.values():
                    path=self.stage/'logs'/str(seed)/'progress.json'
                    if path.exists():states.append(read(path))
                record=dict(event='progress',elapsed_wall_seconds=elapsed,completed_blocks=finished+sum(r['blocks_complete'] for r in states),
                            remaining_seeds=list(pending.values()),gpu_hours=0.,continuous_training_jobs=0,status='running' if pending else 'completed')
                write(self.stage/'compute_ledger.json',record);print(json.dumps(record),flush=True)
                for fraction in (.5,.8):
                    if elapsed>=fraction*self.settings['package_wall_cap_seconds'] and fraction not in warned:
                        warned.add(fraction)
                        event=dict(event='budget_warning',fraction=fraction,time=time.time(),requires_extension_review=fraction==.8)
                        append_json(self.stage/'logs/monitor_events.jsonl',event);print(json.dumps(event),flush=True)
                if elapsed>self.settings['package_wall_cap_seconds']:
                    for process in pool._processes.values():process.terminate()
                    raise TimeoutError('D3/D4 hard wall cap')


def run_worker(config,seed):return ResearchExperiment(config).worker(seed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','d0','pilot','run'])
    parser.add_argument('--repo',required=True);parser.add_argument('--stage',required=True)
    parser.add_argument('--config',default='configs/d34_diagnostics.json')
    args=parser.parse_args()
    experiment=ResearchExperiment(ExperimentConfig(args.repo,args.stage,str(Path(args.config).resolve())))
    torch.set_num_threads(experiment.settings['threads_per_worker'])
    if args.action=='prepare':experiment.prepare()
    elif args.action=='d0':experiment.d0()
    elif args.action=='pilot':
        if not read(experiment.stage/'D0/engineering_checks.json')['passed']:raise RuntimeError('D0 failed')
        experiment.worker(3405,pilot=True)
    else:experiment.run()


if __name__=='__main__':main()
