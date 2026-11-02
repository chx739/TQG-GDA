"""Frozen H-only label/prediction controls and D7-H query-count diagnostics."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import resource
import traceback
import unittest
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from p0_experiment import ExperimentConfig,ResearchExperiment as P0,read,write,run_d0
from tqg_gda.feedback_v2 import ParameterSnapshot
from tqg_gda.p0_diagnostics import DevelopmentOracle,Geometry,append_json,gradient_metrics,gradient_of_ce,evaluate_flat,label_free,model_hash,tensor_hash
from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from tqg_gda.query_label_diagnostics import label_prediction_vectors,label_prediction_steps,query_order,query_lambda,duplicate_check
from tqg_gda.utils import sha256,seed_everything


class ResearchExperiment(P0):
    def __init__(self,config):
        super().__init__(config)
        self.h_stage=Path(self.settings['h_stage'])
        self.cache_index={r['path']:r for r in read(self.h_stage/'local_geometry_manifest.json')['files']}
        self.protocol_sha=sha256(self.stage.parent/'execution_protocol.md')

    def ast_screen(self):
        super().ast_screen();findings=[]
        for path in ['label_query_experiment.py','label_query_summarize.py','tqg_gda/query_label_diagnostics.py']:
            for n in ast.walk(ast.parse((self.code/path).read_text())):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['eval','exec','compile']:findings.append(dict(path=path,line=n.lineno))
        write(self.stage/'label_query_ast_screen.json',dict(passed=not findings,findings=findings))
        if findings:raise ValueError('AST screen failed')

    def prepare(self):
        super().prepare();files=[]
        for seed in self.plan['seeds']:
            for domain in [1,2,3]:
                for step in self.settings['label_snapshots']:
                    for block in range(8):
                        rel=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
                        expected=self.cache_index[rel]['sha256']
                        if sha256(self.h_stage/rel)!=expected:raise ValueError('H input cache changed')
                        files.append(dict(path=rel,sha256=expected))
        write(self.stage/'input_geometry_manifest.json',dict(root=str(self.h_stage),files=files))
        m=read(self.stage/'experiment_manifest.json')
        for key in ['expected_D1_blocks','expected_D2_blocks','expected_D2_temporary_updates']:m.pop(key,None)
        m.update(scope='H-only LABEL/PRED split and D7-H A/B; no new training',execution_protocol_sha256=self.protocol_sha,
            expected=dict(label_blocks=360,label_gradient_rows=6120,label_temporary_updates=3420,
                query_blocks=120,query_logical_profiles=6000,query_physical_solves=4920,query_logical_steps=3000,query_physical_steps=2460))
        write(self.stage/'experiment_manifest.json',m)

    def geometry(self,model,domain,seed,step,block):
        rel=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
        if sha256(self.h_stage/rel)!=self.cache_index[rel]['sha256']:raise ValueError('H cache changed')
        with np.load(self.h_stage/rel,allow_pickle=False) as d:t={k:torch.from_numpy(d[k].copy()) for k in d.files}
        return Geometry(ParameterSnapshot(model),t['V'],t['Q'],t['source_A'],t['target_A'],t['W'],t['source_p'],t['source_h'],t['target_h'],{}),rel

    def provenance(self,row,block,idsref):
        base=self.base(row,block,idsref)
        base.update(candidate_id='H_LABEL_QUERY_DIAGNOSTIC',signature='H',execution_protocol_sha256=self.protocol_sha)
        return base

    def query_profiles(self,geo,ids,seed):
        kernels,kstats=kernel_family(geo.source_representation,geo.target_representation,geo.source_response,
            geo.target_response,torch.arange(196))
        h=kernels['H'];profiles=[];cache={}
        for rep in self.settings['query_repetitions']:
            ordered=query_order(ids['query'],seed,rep)
            for n in self.settings['query_sizes']:
                selected=sorted(ordered[:n]);positions=[ids['query'].index(i) for i in selected]
                idx=torch.tensor(positions)
                for convention in self.settings['lambda_conventions']:
                    lam=query_lambda(n,196,convention,self.settings['base_lambda']);key=(tuple(positions),lam)
                    reused=key in cache
                    if not reused:
                        sub=(h[0][idx][:,idx],h[1][idx],h[2])
                        w,info=solve_observed(sub,geo.target_response,'JOINT',lam,self.v2.calibration)
                        if n==196 and tensor_hash(w)!=tensor_hash(geo.readout):raise ValueError('D7 n196 does not replay H W')
                        cache[key]=dict(W=w,info=info,physical_id=len(cache))
                    profiles.append(dict(n=n,rep=rep,convention=convention,lambda_w=lam,positions=positions,query_ids=selected,
                        W=cache[key]['W'],info=cache[key]['info'],physical_id=cache[key]['physical_id'],reused=reused))
        if len(profiles)!=50 or len(cache)!=41:raise ValueError('Unexpected D7 reuse matrix')
        return profiles,kstats

    def worker(self,seed,pilot=False):
        seed_everything(seed,1,True);out=self.stage/'logs'/(f'pilot_{seed}' if pilot else str(seed));out.mkdir(exist_ok=False)
        writer=SummaryWriter(str(out/'tensorboard'));started=time.time();count=0
        write(out/'RUN_INFO.json',dict(seed=seed,git_commit=self.commit,plan_sha256=self.plan_sha,protocol_sha256=self.protocol_sha,
            started_at=started,pilot=pilot,hardware=self.hardware))
        try:
            for domain in ([1] if pilot else [1,2,3]):
                task=self.task(domain);_,_,dev,final=self.pools(domain)
                oracle=DevelopmentOracle(task,dev,final,out/'label_access.jsonl',enabled=True)
                for step in ([0] if pilot else self.settings['label_snapshots']):
                    model,row=self.load(domain,seed,step);original=model_hash(model);b_cache={}
                    for block in range(1 if pilot else 8):
                        geo,rel=self.geometry(model,domain,seed,step,block)
                        token=f'{domain}_{seed}_{step}_{block}'
                        ids=read(self.h_stage/'logs'/str(seed)/f'{token}_indices.json')
                        write(out/f'{token}_indices.json',ids)
                        profiles,kstats=self.query_profiles(geo,ids,seed) if step==0 else ([],None)
                        hashes={'LABEL_FULL':tensor_hash(geo.readout)}
                        hashes.update({f'D7/{p["rep"]}/{p["n"]}/{p["convention"]}':tensor_hash(p['W']) for p in profiles})
                        digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
                        append_json(out/'readout_seals.jsonl',dict(token=token,readouts=hashes,digest=digest,time=time.time()))
                        oracle.seal(token,ids['A'],theta_hash=tensor_hash(geo.snapshot.flat),readout_hash=digest)
                        ay=oracle.labels(ids['A'],token=token,purpose='label_prediction_D7_A_gradient')
                        a=task.batch(ids['A'],'target');fit=task.batch(ids['fit'],'fit');snapshot=geo.snapshot
                        ga=gradient_of_ce(snapshot,[a],ay);gs=gradient_of_ce(snapshot,[label_free(fit)],fit.y)
                        if not b_cache:
                            by=oracle.labels(ids['B'],token=token,purpose='label_prediction_D7_B_cached_snapshot')
                            bb=self.batches(task,ids['B'])
                            b_cache=dict(labels=by,batches=bb,gradient=gradient_of_ce(snapshot,bb,by),
                                metrics=evaluate_flat(snapshot,snapshot.flat,bb,by))
                        gb=b_cache['gradient'];labels=task.batch(ids['query'],'query').y
                        base=self.provenance(row,block,str((out/f'{token}_indices.json').relative_to(self.stage)))
                        base.update(token=token,geometry_sha256=self.cache_index[rel]['sha256'],W_hash=tensor_hash(geo.readout))
                        vectors,valid,stats=label_prediction_vectors(geo,labels,ids['block_seed'],self.plan['D1']['permutation_offsets'],
                            self.plan['D1']['random_offsets'],1e-12)
                        append_json(out/'label_components.jsonl',dict(base,**stats))
                        for arm,g in vectors.items():
                            metrics=gradient_metrics(g,ga,gb,gs,geo.basis)
                            append_json(out/'label_gradient_blocks.jsonl',dict(base,arm=arm,**metrics))
                        perform=block in self.settings['step_blocks']
                        if perform:
                            arms,validity=label_prediction_steps(vectors,gs,ga,self.settings['beta'],1e-12)
                            for arm,direction in arms.items():
                                m=evaluate_flat(snapshot,snapshot.flat-self.settings['eta']*direction,b_cache['batches'],b_cache['labels'])
                                append_json(out/'label_step_blocks.jsonl',dict(base,arm=arm,eta=self.settings['eta'],beta=self.settings['beta'],
                                    CE_B=m['ce'],BA_B=100*m['balanced_accuracy'],base_CE_B=b_cache['metrics']['ce'],
                                    direction_norm=float(direction.norm()),applied_feedback_norm=float(((direction-gs)/self.settings['beta']).norm()),
                                    first_order_CE_B=-self.settings['eta']*float(torch.dot(gb,direction)),valid_comparison=validity[arm]))
                        if profiles:
                            full_residual=((torch.nn.functional.one_hot(labels,2).double()-geo.source_probabilities)@geo.contrasts)
                            metrics_cache={};step_cache={}
                            source_ce=evaluate_flat(snapshot,snapshot.flat-self.settings['eta']*gs,b_cache['batches'],b_cache['labels'])['ce'] if perform else None
                            for p in profiles:
                                physical=p['physical_id']
                                if physical not in metrics_cache:
                                    residual=full_residual[p['positions']].reshape(-1)
                                    coeff=-(p['W']@residual);g=geo.basis@coeff
                                    m=gradient_metrics(g,ga,gb,gs,geo.basis)
                                    metrics_cache[physical]=(m,g,coeff)
                                m,g,coeff=metrics_cache[physical]
                                record=dict(base,experiment_id='D7_H_A',n=p['n'],rep=p['rep'],convention=p['convention'],
                                    lambda_w=p['lambda_w'],physical_id=physical,physical_solve_reused=p['reused'],
                                    selected_query_ids=p['query_ids'],coefficients=coeff.tolist(),solver=p['info'],kernel_statistics=kstats,
                                    profile_W_hash=tensor_hash(p['W']),source_class_counts=torch.bincount(labels[p['positions']],minlength=2).tolist(),**m)
                                append_json(out/'query_size_blocks.jsonl',record)
                                if perform:
                                    reused=physical in step_cache
                                    if not reused:
                                        mb=evaluate_flat(snapshot,snapshot.flat-self.settings['eta']*(gs+self.settings['beta']*g),b_cache['batches'],b_cache['labels'])
                                        step_cache[physical]=mb
                                    mb=step_cache[physical]
                                    append_json(out/'query_step_blocks.jsonl',dict(base,n=p['n'],rep=p['rep'],convention=p['convention'],lambda_w=p['lambda_w'],
                                        physical_id=physical,physical_step_reused=reused,CE_B=mb['ce'],BA_B=100*mb['balanced_accuracy'],
                                        source_CE_B=source_ce,base_CE_B=b_cache['metrics']['ce'],R_Q=(source_ce-mb['ce'])/b_cache['metrics']['ce']))
                            writer.add_scalar(f'D7/{domain}/n196/c_A',metrics_cache[profiles[-1]['physical_id']][0]['c_A'] or 0.,block)
                        if model_hash(model)!=original:raise ValueError('Diagnostic changed original model')
                        append_json(out/'temporary_step_checks.jsonl',dict(token=token,original_model_unchanged=True,passed=True))
                        count+=1
                        write(out/'progress.json',dict(seed=seed,blocks_complete=count,domain=domain,step=step,block=block,elapsed=time.time()-started))
                        if block%4==3 or pilot:print(json.dumps(dict(event='block',seed=seed,domain=domain,step=step,block=block)),flush=True)
            write(out/'done.json',dict(status='completed',seed=seed,git_commit=self.commit,blocks=count,started_at=started,finished_at=time.time(),
                elapsed_seconds=time.time()-started,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,pilot=pilot))
        except Exception:
            append_json(out/'monitor_events.jsonl',dict(event='failed',time=time.time(),traceback=traceback.format_exc()));raise
        finally:writer.close()

    def duplicate_engineering(self):
        cfg=self.settings['duplicate_engineering_config'];cases=[]
        torch.manual_seed(921)
        feat=torch.randn(9,7,dtype=torch.float64);gram=feat@feat.mT+.4*torch.eye(9,dtype=torch.float64)
        kernels=(gram[:5,:5],gram[:5,5:],gram[5:,5:]);response=torch.randn(4,2,3,dtype=torch.float64)
        for mode in ['MEAN','JOINT']:
            cases.append(dict(case='multiclass_PSD',**duplicate_check(kernels,response,mode,self.settings['duplicate_engineering_lambda'],cfg)))
        for domain in [1,2,3]:
            model,_=self.load(domain,3405,0);geo,_=self.geometry(model,domain,3405,0,0)
            family,_=kernel_family(geo.source_representation,geo.target_representation,geo.source_response,geo.target_response,torch.arange(196))
            h=family['H'];k=(h[0][:8,:8],h[1][:8],h[2])
            for mode in ['MEAN','JOINT']:
                cases.append(dict(case=f'real_0_to_{domain}_source8',**duplicate_check(k,geo.target_response,mode,self.settings['duplicate_engineering_lambda'],cfg)))
        passed=all(c['scaled_collapse_error']<=self.settings['duplicate_tolerance'] and c['fixed_collapse_vs_half_lambda_error']<=self.settings['duplicate_tolerance'] for c in cases)
        write(self.stage/'D0/duplicate_invariance.json',dict(passed=passed,cases=cases,tolerance=self.settings['duplicate_tolerance'],scope='D7-H-B engineering only; no new source information'))
        if not passed:raise ValueError('Duplicate normalization check failed')

    def run(self):
        if not read(self.stage/'D0/duplicate_invariance.json')['passed']:raise ValueError('D0/D7-B required')
        pilot=read(self.stage/'logs/pilot_3405/done.json')
        estimate=pilot['elapsed_seconds']*360/self.settings['workers']*self.settings['compute_buffer']
        if estimate>self.settings['package_wall_cap_seconds']:raise ValueError('Pilot estimate over cap')
        write(self.stage/'compute_estimate.json',dict(pilot_seconds=pilot['elapsed_seconds'],conservative_wall_estimate=estimate,blocks=360))
        start=time.time()
        with ProcessPoolExecutor(max_workers=self.settings['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(run_worker,self.config,seed):seed for seed in self.plan['seeds']}
            while pending:
                done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:f.result();del pending[f]
                elapsed=time.time()-start
                states=[]
                for seed in self.plan['seeds']:
                    p=self.stage/'logs'/str(seed)/'progress.json'
                    if p.exists():states.append(read(p))
                write(self.stage/'compute_ledger.json',dict(started_at=start,elapsed_wall_seconds=elapsed,status='running' if pending else 'completed',gpu_hours=0,states=states))
                print(json.dumps(dict(event='progress',elapsed=round(elapsed,1),remaining=list(pending.values()),states=states)),flush=True)
                if elapsed>self.settings['package_wall_cap_seconds']:
                    for p in pool._processes.values():p.terminate()
                    raise TimeoutError('Hard experiment budget cap')


def run_worker(config,seed):ResearchExperiment(config).worker(seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','d0','pilot','run']);p.add_argument('--repo',required=True);p.add_argument('--stage',required=True);p.add_argument('--config',default='configs/label_query_diagnostics.json')
    a=p.parse_args();e=ResearchExperiment(ExperimentConfig(a.repo,a.stage,str(Path(a.config).resolve())));torch.set_num_threads(1)
    if a.action=='prepare':e.prepare()
    elif a.action=='d0':
        run_d0(e)
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_query_label_diagnostics','tests.test_mechanism_diagnostics','tests.test_p0_diagnostics','tests.test_d34_diagnostics','tests.test_h_only_diagnostics'])
        with (e.stage/'D0/unit_checks.log').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        if not result.wasSuccessful():raise ValueError('Unit acceptance failed')
        e.duplicate_engineering();write(e.stage/'D0/label_query_checks.json',dict(passed=True,unit_checks=result.testsRun))
    elif a.action=='pilot':e.worker(3405,pilot=True)
    else:e.run()

if __name__=='__main__':main()
