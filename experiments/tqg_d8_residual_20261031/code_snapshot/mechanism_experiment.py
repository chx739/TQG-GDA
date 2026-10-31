"""Frozen D8 known-truth and real residual decomposition; no real adaptation."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from p0_experiment import ExperimentConfig,ResearchExperiment as P0,read,write,run_d0
from tqg_gda.models import GraphClassifier
from tqg_gda.feedback_v2 import ParameterSnapshot,query_matrices
from tqg_gda.p0_diagnostics import (DevelopmentOracle,Geometry,append_json,controls,cosine,evaluate_flat,
    gradient_metrics,gradient_of_ce,label_free,model_hash,normalized,sample_ids,tensor_hash,unlabeled_geometry)
from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from tqg_gda.mechanism_diagnostics import (graph_pool,linear_pool,LinearBinary,soft_gradient,soft_ce,
    soft_residual,decomposition_vectors,decomposition_steps)
from tqg_gda.utils import seed_everything,sha256
from tqg_gda.metrics import classification_metrics


class ResearchExperiment(P0):
    def __init__(self,config):
        super().__init__(config)
        self.p0=Path(self.settings['p0_stage']);self.h_stage=Path(self.settings['h_stage'])
        self.index={}
        for name,root in [('HA_JOINT',self.p0),('H_JOINT',self.h_stage)]:
            self.index[name]={r['path']:r for r in read(root/'local_geometry_manifest.json')['files']}
        self.protocol_sha=sha256(self.stage.parent/'execution_protocol.md')

    def ast_screen(self):
        super().ast_screen();findings=[]
        for relative in ['mechanism_experiment.py','mechanism_summarize.py','tqg_gda/mechanism_diagnostics.py']:
            for n in ast.walk(ast.parse((self.code/relative).read_text())):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['exec','eval','compile']:
                    findings.append(dict(path=relative,line=n.lineno))
                if isinstance(n,ast.Import) and any(x.name.split('.')[0] in ['requests','socket','urllib'] for x in n.names):
                    findings.append(dict(path=relative,line=n.lineno))
        write(self.stage/'mechanism_ast_screen.json',dict(passed=not findings,findings=findings))
        if findings:raise ValueError('AST screen failed')

    def prepare(self):
        super().prepare()
        inputs=[]
        for name,root in [('HA_JOINT',self.p0),('H_JOINT',self.h_stage)]:
            for seed in self.plan['seeds']:
                for domain in [1,2,3]:
                    for step in [0,50,200]:
                        for block in range(8):
                            rel=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
                            expected=self.index[name][rel]['sha256']
                            if sha256(root/rel)!=expected:raise ValueError('Frozen geometry changed')
                            inputs.append(dict(candidate=name,root=str(root),path=rel,sha256=expected))
        write(self.stage/'input_geometry_manifest.json',dict(files=inputs))
        m=read(self.stage/'experiment_manifest.json')
        for key in ['expected_D1_blocks','expected_D2_blocks','expected_D2_temporary_updates']:m.pop(key,None)
        m.update(scope='D8 plus real residual decomposition',execution_protocol_sha256=self.protocol_sha,
            expected=dict(linear_blocks=120,graph_blocks=480,graph_profiles=1440,real_blocks=360,real_profiles=720,
                real_step_updates=4680,synthetic_step_updates=2880),new_synthetic_source_models=5,new_real_pretraining=0)
        write(self.stage/'experiment_manifest.json',m)

    def provenance(self,seed):
        return dict(seed=seed,model_seed=seed,git_commit=self.commit,config_name=self.settings['config_name'],
            plan_sha256=self.plan_sha,execution_protocol_sha256=self.protocol_sha,hardware=self.hardware,
            gpu_hours=0.,event_flags='clean')

    def synthetic_profiles(self,model,source,a_input):
        geo=unlabeled_geometry(model,source,a_input,self.v2)
        kernels,kstats=kernel_family(geo.source_representation,geo.target_representation,
            geo.source_response,geo.target_response,torch.arange(len(geo.source_probabilities)))
        profiles=[]
        for name,kernel,mode,lam in self.settings['synthetic_methods']:
            w,info=solve_observed(kernels[kernel],geo.target_response,mode,lam,self.v2.calibration)
            profiles.append(dict(name=name,kernel=kernel,W=w,info=info))
        return geo,kernels,kstats,profiles

    def synthetic_analyze(self,geo,kernels,kstats,profiles,source_labels,source_eta,
            a_input,a_eta,a_labels,b_batches,b_eta,b_labels,fit,fit_labels,ids,base,out,writer,count,steps):
        # The caller seals ALL readouts before revealing synthetic truth.
        g_cond=soft_gradient(geo.snapshot,[a_input],a_eta)
        g_lab=gradient_of_ce(geo.snapshot,[a_input],a_labels)
        g_b=soft_gradient(geo.snapshot,b_batches,b_eta) if 'g_b' not in ids else ids['g_b']
        gs=gradient_of_ce(geo.snapshot,[fit],fit_labels)
        rcond=soft_residual(geo.source_probabilities,source_eta,geo.contrasts)
        if steps:
            baseline=soft_ce(geo.snapshot,geo.snapshot.flat,b_batches,b_eta)
            source_ce=soft_ce(geo.snapshot,geo.snapshot.flat-self.settings['synthetic_step_eta']*gs,b_batches,b_eta)
        for p in profiles:
            geo.readout=p['W']
            gradients,valid,_=controls(geo,source_labels,ids['block_seed'],self.plan['D1']['permutation_offsets'],
                self.plan['D1']['random_offsets'],self.settings['epsilon_gradient'])
            readcond=-(geo.basis@(p['W']@rcond))
            gradients.update(SOURCE_CONDITIONAL=readcond,ORACLE_CONDITIONAL=g_cond,ORACLE_LABEL=g_lab,ZERO=torch.zeros_like(g_cond))
            for arm,g in gradients.items():
                metrics=gradient_metrics(g,g_cond,g_b,gs,geo.basis,epsilon=self.settings['epsilon_gradient'])
                append_json(out/f"{base['layer']}_gradient_blocks.jsonl",dict(base,method=p['name'],arm=arm,
                    rank=geo.basis.shape[1],**metrics,c_empirical_A=cosine(g,g_lab,1e-12),
                    conditional_vs_empirical_cosine=cosine(g_cond,g_lab,1e-12),
                    W_hash=tensor_hash(p['W']),solver=p['info'],kernel_statistics=kstats,
                    K_ST_mean=float(kernels[p['kernel']][1].mean()),K_ST_max=float(kernels[p['kernel']][1].max()),
                    H_ST_norm=float(query_matrices(kernels[p['kernel']][1],kernels[p['kernel']][2],geo.target_response)[0].norm()),
                    target_conditional_gradient_norm=float(g_cond.norm()),
                    source_conditional_readout_norm=float(readcond.norm())))
            noise=gradients['TRUE']-readcond;bias=readcond-g_cond;total=gradients['TRUE']-g_cond
            denom=float(g_cond.square().sum())
            cross=2*float(torch.dot(noise,bias))
            identity=abs(float(total.square().sum())-float(noise.square().sum())-float(bias.square().sum())-cross)
            if identity>1e-8*(1+float(total.square().sum())):raise ValueError('Error decomposition identity failed')
            append_json(out/'error_decomposition.jsonl',dict(base,method=p['name'],
                source_label_noise_squared=float(noise.square().sum()),combined_mismatch_squared=float(bias.square().sum()),
                cross_term=cross,total_squared=float(total.square().sum()),conditional_gradient_squared=denom,
                normalized_noise_squared=float(noise.square().sum())/denom if denom>1e-24 else None,
                normalized_mismatch_squared=float(bias.square().sum())/denom if denom>1e-24 else None,
                normalized_cross_term=cross/denom if denom>1e-24 else None,
                identity_absolute_error=identity))
            if gradients['TRUE'].norm()>1e-12:
                writer.add_scalar(f"D8/{base['scenario']}/{p['name']}/c_conditional",cosine(gradients['TRUE'],g_cond,1e-12) or 0.,count)
            if steps:
                om,om_valid=normalized(g_cond,float(gradients['TRUE'].norm()),1e-12)
                beta,eta=self.settings['synthetic_step_beta'],self.settings['synthetic_step_eta']
                directions={'Q':beta*gradients['TRUE'],'S+TRUE':gs+beta*gradients['TRUE'],
                    'S+SOURCE_CONDITIONAL':gs+beta*readcond,'S+ORACLE-M':gs+beta*om}
                for arm,d in directions.items():
                    ce=soft_ce(geo.snapshot,geo.snapshot.flat-eta*d,b_batches,b_eta)
                    append_json(out/'synthetic_step_blocks.jsonl',dict(base,method=p['name'],arm=arm,eta=eta,beta=beta,
                        conditional_CE_B=ce,base_conditional_CE_B=baseline,source_conditional_CE_B=source_ce,
                        extra_gain_vs_source=(source_ce-ce)/baseline,absolute_gain=(baseline-ce)/baseline,
                        valid_comparison=om_valid if arm=='S+ORACLE-M' else True))

    def linear(self,seed,out,writer,pilot=False):
        data_seed=27300+seed-3405
        sx,se,sy=linear_pool(196,data_seed+12)
        # Fit data are generated and recorded but never used to pretrain this analytic model.
        fx,fe,fy=linear_pool(512,data_seed+11)
        tx,te,ty=linear_pool(1024,data_seed+20,target=True)
        np.savez_compressed(out/'linear_data.npz',source_query=sx.numpy(),source_eta=se.numpy(),source_labels=sy.numpy(),
            source_fit=fx.numpy(),fit_eta=fe.numpy(),fit_labels=fy.numpy(),target=tx.numpy(),target_eta=te.numpy(),target_labels=ty.numpy())
        for index,theta in enumerate(self.plan['D8']['linear_fixed_theta'][:1] if pilot else self.plan['D8']['linear_fixed_theta']):
            model=LinearBinary(theta).eval();original=model_hash(model)
            for block in range(1 if pilot else 8):
                block_seed=81000000+(seed-3405)*10000+index*100+block
                ids=sample_ids(range(1024),16,block_seed);a=tx[list(ids)]
                geo,kernels,kstats,profiles=self.synthetic_profiles(model,sx,a)
                token=f'linear_{seed}_{index}_{block}'
                seal={p['name']:tensor_hash(p['W']) for p in profiles}
                append_json(out/'synthetic_truth_access.jsonl',dict(event='sealed',token=token,readouts=seal,time=time.time()))
                append_json(out/'synthetic_truth_access.jsonl',dict(event='truth_access',token=token,scope='linear_known_truth',
                    A=list(ids),B='all1024_overlaps_A_not_independent',source_query_conditional=True,time=time.time()))
                base=dict(self.provenance(seed),layer='linear',scenario='LINEAR',snapshot_step=index,theta=theta,
                    block_id=block,block_seed=block_seed,token=token,target_A=list(ids),target_B_independent=False)
                self.synthetic_analyze(geo,kernels,kstats,profiles,sy,se,a,te[list(ids)],ty[list(ids)],
                    [tx],te,ty,fx[:16],fy[:16],dict(block_seed=block_seed),base,out,writer,block,False)
                if model_hash(model)!=original:raise ValueError('Linear model changed')

    def source_models(self,seed,out,writer,pilot=False):
        source={role:graph_pool(n,7300+seed-3405+self.plan['D8']['role_offsets'][role])
            for role,n in self.settings['source_counts'].items()}
        folder=self.stage/'checkpoints'/(f'pilot_{seed}' if pilot else str(seed));folder.mkdir(parents=True,exist_ok=False)
        for role,pool in source.items():write(out/f'generator_source_{role}.json',dict(seed=pool.input_seed,latents=pool.latents,
            eta=pool.eta.tolist(),labels=pool.labels.tolist()))
        model=GraphClassifier(dict(input_dim=2,edge_dim=1,hidden_dim=32,num_layers=2,readout='mean'),2).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=self.settings['warmup_lr'])
        generator=torch.Generator().manual_seed(seed+70000)
        checkpoints={}
        for step in range(1 if pilot else self.settings['warmup_steps']+1):
            if step:
                model.train();indices=torch.randperm(512,generator=generator)[:self.settings['warmup_batch']].tolist()
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(source['fit'].batch(indices)),source['fit'].labels[indices])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic warmup')
                loss.backward();optimizer.step();writer.add_scalar('warmup/loss',float(loss.detach()),step)
                append_json(out/'warmup_history.jsonl',dict(self.provenance(seed),step=step,loss=float(loss.detach()),fit_ids=indices))
            if step in [0,50,200]:
                model.eval();path=folder/f'step_{step}.pt'
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),encoder_spec=model.encoder.spec,
                    seed=seed,step=step,git_commit=self.commit,fit_seed=source['fit'].input_seed),path)
                checkpoints[step]=path
                snap=ParameterSnapshot(model)
                record=dict(seed=seed,step=step,path=str(path.relative_to(self.stage)),sha256=sha256(path))
                for role in ['validation','audit']:
                    pool=source[role];batches=[pool.batch(list(range(i,min(i+64,len(pool.graphs))))) for i in range(0,len(pool.graphs),64)]
                    record[role+'_conditional_CE']=soft_ce(snap,snap.flat,batches,pool.eta)
                append_json(out/'synthetic_checkpoint_manifest.jsonl',record)
        return source,checkpoints

    def graph(self,seed,out,writer,pilot=False):
        source,checkpoints=self.source_models(seed,out,writer,pilot)
        query=source['query'].batch(list(range(196)))
        targets={}
        for scenario in ['S1','S2','S3','S4']:
            targets[scenario]={role:graph_pool(self.settings['target_'+('construction' if role=='target_construction' else 'evaluation')+'_count'],
                7300+seed-3405+self.plan['D8']['role_offsets'][role],target=True,scenario=scenario)
                for role in ['target_construction','target_evaluation']}
            for role,pool in targets[scenario].items():write(out/f'generator_{scenario}_{role}.json',dict(seed=pool.input_seed,
                latents=pool.latents,eta=pool.eta.tolist(),labels=pool.labels.tolist(),used_only_for_synthetic_oracle=True))
        count=0
        for step,path in checkpoints.items():
            model=GraphClassifier(dict(input_dim=2,edge_dim=1,hidden_dim=32,num_layers=2,readout='mean'),2).double()
            state=torch.load(path,weights_only=False);model.load_state_dict(state['model']);model.eval();original=model_hash(model)
            shared={}
            for scenario in (['S1'] if pilot else ['S1','S2','S3','S4']):
                a_pool=targets[scenario]['target_construction'];b_pool=targets[scenario]['target_evaluation']
                batches=[b_pool.batch(list(range(i,min(i+64,1024)))) for i in range(0,1024,64)]
                b_gradient=None
                for block in range(1 if pilot else 8):
                    block_seed=82000000+(seed-3405)*10000+step*10+block
                    ids=sample_ids(range(1024),16,block_seed);fit_ids=sample_ids(range(512),16,block_seed+1000)
                    a=a_pool.batch(ids)
                    if scenario=='S3':geo,kernels,kstats,profiles=shared[block]
                    else:
                        geo,kernels,kstats,profiles=self.synthetic_profiles(model,query,a)
                        if scenario=='S1':shared[block]=(geo,kernels,kstats,profiles)
                    token=f'graph_{seed}_{step}_{scenario}_{block}'
                    hashes={p['name']:tensor_hash(p['W']) for p in profiles}
                    append_json(out/'synthetic_truth_access.jsonl',dict(event='sealed',token=token,readouts=hashes,time=time.time()))
                    append_json(out/'synthetic_truth_access.jsonl',dict(event='truth_access',token=token,scope='graph_known_truth',
                        A=list(ids),B='independent1024',source_query_conditional=True,time=time.time()))
                    if b_gradient is None:
                        b_gradient=soft_gradient(geo.snapshot,batches,b_pool.eta)
                        empirical=evaluate_flat(geo.snapshot,geo.snapshot.flat,batches,b_pool.labels)
                        append_json(out/'synthetic_baseline.jsonl',dict(self.provenance(seed),scenario=scenario,snapshot_step=step,
                            conditional_CE_B=soft_ce(geo.snapshot,geo.snapshot.flat,batches,b_pool.eta),BA_B=100*empirical['balanced_accuracy'],
                            AUROC_B=empirical['auroc']))
                    base=dict(self.provenance(seed),layer='graph',scenario=scenario,snapshot_step=step,block_id=block,
                        token=token,checkpoint=str(path.relative_to(self.stage)),checkpoint_sha256=sha256(path),
                        target_A=list(ids),fit_ids=list(fit_ids),target_B_independent=True)
                    self.synthetic_analyze(geo,kernels,kstats,profiles,source['query'].labels,source['query'].eta,
                        a,a_pool.eta[list(ids)],a_pool.labels[list(ids)],batches,b_pool.eta,b_pool.labels,
                        source['fit'].batch(fit_ids),source['fit'].labels[list(fit_ids)],dict(block_seed=block_seed,g_b=b_gradient),
                        base,out,writer,count,block in self.settings['synthetic_auxiliary_step_blocks'])
                    if model_hash(model)!=original:raise ValueError('Synthetic diagnostic changed model')
                    count+=1
                    write(out/'progress.json',dict(seed=seed,phase='synthetic_graph',blocks=count,scenario=scenario,step=step))
                    if block%4==3 or pilot:print(json.dumps(dict(event='synthetic_graph',seed=seed,scenario=scenario,step=step,block=block)),flush=True)

    def real(self,seed,out,writer,pilot=False):
        count=0
        for domain in ([1] if pilot else [1,2,3]):
            task=self.task(domain);_,_,dev,final=self.pools(domain)
            oracle=DevelopmentOracle(task,dev,final,out/'label_access.jsonl',enabled=True)
            for step in ([0] if pilot else [0,50,200]):
                model,row=self.load(domain,seed,step);original=model_hash(model);b_cache={}
                for block in range(1 if pilot else 8):
                    token=f'real_{domain}_{seed}_{step}_{block}'
                    rel=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
                    ids=read(self.p0/'logs'/str(seed)/f'{domain}_{seed}_{step}_{block}_indices.json')
                    geos={}
                    for candidate,root in [('H_JOINT',self.h_stage),('HA_JOINT',self.p0)]:
                        if sha256(root/rel)!=self.index[candidate][rel]['sha256']:raise ValueError('Real geometry changed')
                        with np.load(root/rel,allow_pickle=False) as d:t={k:torch.from_numpy(d[k].copy()) for k in d.files}
                        geos[candidate]=Geometry(ParameterSnapshot(model),t['V'],t['Q'],t['source_A'],t['target_A'],t['W'],
                            t['source_p'],t['source_h'],t['target_h'],{})
                    seal=hashlib.sha256(json.dumps({k:tensor_hash(g.readout) for k,g in geos.items()},sort_keys=True).encode()).hexdigest()
                    oracle.seal(token,ids['A'],theta_hash=tensor_hash(next(iter(geos.values())).snapshot.flat),readout_hash=seal)
                    a_labels=oracle.labels(ids['A'],token=token,purpose='real_residual_decomposition_A')
                    a=task.batch(ids['A'],'target');fit=task.batch(ids['fit'],'fit');snapshot=next(iter(geos.values())).snapshot
                    g_a=gradient_of_ce(snapshot,[a],a_labels);gs=gradient_of_ce(snapshot,[label_free(fit)],fit.y)
                    if not b_cache:
                        b_labels=oracle.labels(ids['B'],token=token,purpose='real_residual_decomposition_B_cached_snapshot')
                        b_batches=self.batches(task,ids['B'])
                        b_cache=dict(labels=b_labels,batches=b_batches,gradient=gradient_of_ce(snapshot,b_batches,b_labels),
                            baseline=evaluate_flat(snapshot,snapshot.flat,b_batches,b_labels))
                    g_b=b_cache['gradient'];source_labels=task.batch(ids['query'],'query').y
                    for candidate,geo in geos.items():
                        vectors,valid,diagnostics=decomposition_vectors(geo,source_labels,ids['block_seed'],
                            self.plan['D1']['permutation_offsets'],self.plan['D1']['random_offsets'],1e-12)
                        base=dict(self.provenance(seed),layer='real',task=f'0_to_{domain}',method=candidate,snapshot_step=step,
                            block_id=block,token=token,checkpoint_sha256=row['checkpoint_sha256'],
                            input_cache_sha256=self.index[candidate][rel]['sha256'],geometry_cache=rel,
                            W_hash=tensor_hash(geo.readout),indices_reference=f'{token}_indices.json')
                        write(out/f'{token}_indices.json',ids)
                        append_json(out/'real_residual_components.jsonl',dict(base,**diagnostics))
                        for arm,g in vectors.items():
                            m=gradient_metrics(g,g_a,g_b,gs,geo.basis,epsilon=1e-12)
                            append_json(out/'real_gradient_blocks.jsonl',dict(base,arm=arm,**m))
                        if block in self.settings['real_step_blocks']:
                            arms,validity=decomposition_steps(vectors,gs,g_a,beta=self.settings['real_beta'],epsilon=1e-12)
                            for arm,direction in arms.items():
                                flat=snapshot.flat-self.settings['real_eta']*direction
                                metrics=evaluate_flat(snapshot,flat,b_cache['batches'],b_cache['labels'])
                                append_json(out/'real_step_blocks.jsonl',dict(base,arm=arm,eta=self.settings['real_eta'],beta=self.settings['real_beta'],
                                    CE_B=metrics['ce'],BA_B=100*metrics['balanced_accuracy'],base_CE_B=b_cache['baseline']['ce'],
                                    first_order_CE_B=-self.settings['real_eta']*float(torch.dot(g_b,direction)),
                                    valid_comparison=validity[arm]))
                    if model_hash(model)!=original:raise ValueError('Real diagnostic changed model')
                    append_json(out/'temporary_step_checks.jsonl',dict(token=token,passed=True,original_model_unchanged=True))
                    count+=1
                    write(out/'progress.json',dict(seed=seed,phase='real_decomposition',blocks=count,domain=domain,step=step))
                print(json.dumps(dict(event='real_snapshot_complete',seed=seed,domain=domain,step=step)),flush=True)

    def worker(self,seed,pilot=False):
        seed_everything(seed,1,True)
        out=self.stage/'logs'/(f'pilot_{seed}' if pilot else str(seed));out.mkdir(exist_ok=False)
        writer=SummaryWriter(str(out/'tensorboard'));start=time.time()
        write(out/'RUN_INFO.json',dict(self.provenance(seed),started_at=start,pilot=pilot))
        try:
            self.linear(seed,out,writer,pilot)
            self.graph(seed,out,writer,pilot)
            self.real(seed,out,writer,pilot)
            write(out/'done.json',dict(self.provenance(seed),status='completed',started_at=start,finished_at=time.time(),
                elapsed_seconds=time.time()-start,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,pilot=pilot))
        except Exception:
            append_json(out/'monitor_events.jsonl',dict(event='failed',traceback=traceback.format_exc(),time=time.time()));raise
        finally:writer.close()

    def run(self):
        if not read(self.stage/'D0/engineering_checks.json')['passed']:raise ValueError('D0 required')
        pilot=read(self.stage/'logs/pilot_3405/done.json')
        estimate=pilot['elapsed_seconds']*96/self.settings['workers']*self.settings['compute_buffer']
        if estimate>self.settings['package_wall_cap_seconds']:raise ValueError('Pilot estimate exceeds cap')
        write(self.stage/'compute_estimate.json',dict(pilot_seconds=pilot['elapsed_seconds'],estimated_wall_seconds=estimate,
            note='Conservative pilot includes full generation/evaluation overhead per one graph block'))
        start=time.time();context=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=5,mp_context=context) as pool:
            pending={pool.submit(run_worker,self.config,seed):seed for seed in self.plan['seeds']}
            while pending:
                done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:f.result();del pending[f]
                states=[]
                for seed in self.plan['seeds']:
                    p=self.stage/'logs'/str(seed)/'progress.json'
                    if p.exists():states.append(read(p))
                elapsed=time.time()-start
                write(self.stage/'compute_ledger.json',dict(started_at=start,elapsed_wall_seconds=elapsed,gpu_hours=0,
                    status='running' if pending else 'completed',states=states,remaining_seeds=list(pending.values())))
                print(json.dumps(dict(event='progress',elapsed=round(elapsed,1),remaining=list(pending.values()),states=states)),flush=True)
                if elapsed>self.settings['package_wall_cap_seconds']:
                    for p in pool._processes.values():p.terminate()
                    raise TimeoutError('Mechanism package cap reached')


def run_worker(config,seed):ResearchExperiment(config).worker(seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','d0','pilot','run'])
    p.add_argument('--repo',required=True);p.add_argument('--stage',required=True);p.add_argument('--config',default='configs/mechanism_diagnostics.json')
    a=p.parse_args();e=ResearchExperiment(ExperimentConfig(a.repo,a.stage,str(Path(a.config).resolve())))
    torch.set_num_threads(1)
    if a.action=='prepare':e.prepare()
    elif a.action=='d0':
        run_d0(e)
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_mechanism_diagnostics','tests.test_p0_diagnostics','tests.test_d34_diagnostics','tests.test_h_only_diagnostics'])
        with (e.stage/'D0/unit_checks.log').open('w') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        if not result.wasSuccessful():raise ValueError('Mechanism unit acceptance failed')
        write(e.stage/'D0/mechanism_checks.json',dict(passed=True,unit_checks=result.testsRun))
    elif a.action=='pilot':e.worker(3405,pilot=True)
    else:e.run()

if __name__=='__main__':main()
