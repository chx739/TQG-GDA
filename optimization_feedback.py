"""Local-bank diagnostics with sealed inputs, readouts and temporary-step logits."""
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

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from optimization_a4 import OptimizationA4
from assumptions_experiment import cpu_seconds,append
from p0_experiment import read,write
from tqg_gda.feedback_v2 import ParameterSnapshot
from tqg_gda.p0_diagnostics import (Geometry,DevelopmentOracle,gradient_of_ce,gradient_metrics,model_hash,tensor_hash,normalized)
from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from tqg_gda.query_label_diagnostics import query_order
from tqg_gda.optimization_diagnostics import closest_bank,effective_lambda,feedback_controls,binary_logit_metrics
from tqg_gda.utils import sha256,seed_everything


class OptimizationFeedback(OptimizationA4):
    phase='A2b'

    def prepare(self):
        (self.stage/self.phase).mkdir(exist_ok=True)
        findings=[]
        for rel in ['optimization_feedback.py','tqg_gda/optimization_diagnostics.py']:
            for node in ast.walk(ast.parse((self.code/rel).read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ['exec','eval','compile']:
                    findings.append(dict(file=rel,line=node.lineno))
        assert not findings
        write(self.stage/f'D0/ast_screen_{self.phase}.json',dict(passed=True,findings=findings))
        write(self.stage/self.phase/'manifest.json',dict(git_commit=self.commit,protocol_sha256=self.plan_sha,
            config=self.settings,hardware=self.hardware,expected_blocks=120,expected_profiles=6960,
            expected_step_profiles=3480,phase=self.phase,final_labels_allowed=False))

    def geometry(self,model,seed,domain,step,block):
        root=Path(self.settings['old_stage']);rel=f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz'
        index={r['path']:r['sha256'] for r in read(root/'local_geometry_manifest.json')['files']}
        assert sha256(root/rel)==index[rel]
        z={k:torch.from_numpy(v) for k,v in dict(np.load(root/rel,allow_pickle=False)).items()}
        geo=Geometry(ParameterSnapshot(model),z['V'],z['Q'],z['source_A'],z['target_A'],z['W'],z['source_p'],z['source_h'],z['target_h'],{})
        ids=read(root/f'logs/{seed}/{domain}_{seed}_{step}_{block}_indices.json')
        return geo,ids,index[rel]

    def profiles(self,geo,ids,labels,seed,domain,step,block,out,token):
        kernels,stats=kernel_family(geo.source_representation,geo.target_representation,geo.source_response,geo.target_response,torch.arange(196))
        result=[]
        for sig in self.settings['signatures']:
            k=kernels[sig]
            for count in self.settings['bank_sizes']:
                banks=[]
                for rep in self.settings['bank_repetitions']:
                    chosen=sorted(query_order(ids['query'],seed,rep)[:count]);ix=[ids['query'].index(i) for i in chosen]
                    banks.append((f'UNIFORM_{rep}','UNIFORM',rep,np.array(ix)))
                for selector in self.settings['signatures']:
                    ix=closest_bank(ids['query'],kernels[selector][1].numpy(),count)
                    banks.append((f'NEAR_{selector}',selector,-1,ix))
                for name,selector,rep,ix in banks:
                    for convention in self.settings['lambda_conventions']:
                        lam=effective_lambda(self.settings['base_joint_lambda'],count,196,convention)
                        w,info=solve_observed((k[0][ix][:,ix],k[1][ix],k[2]),geo.target_response,'JOINT',lam,self.settings['calibration'])
                        result.append(dict(profile=f'{sig}_{count}_{name}_{convention}',signature=sig,n=count,selector=selector,rep=rep,
                            convention=convention,lambda_w=lam,indices=ix,W=w,solver=info))
            w,info=solve_observed(k,geo.target_response,'JOINT',self.settings['base_joint_lambda'],self.settings['calibration'])
            refroot=Path(self.settings['old_stage'] if sig=='HA' else self.settings['old_h_stage'])
            ref=np.load(refroot/f'geometry_cache/{seed}/{domain}_{seed}_{step}_{block}.npz')['W']
            replay=float((w-torch.from_numpy(ref)).abs().max());assert replay<1e-8
            result.append(dict(profile=f'{sig}_FULL',signature=sig,n=196,selector='FULL',rep=-1,convention='BOTH',
                lambda_w=self.settings['base_joint_lambda'],indices=np.arange(196),W=w,solver=info,replay_error=replay))
        assert len(result)==58
        return result,dict(kernel_statistics=stats,selection='fixed grid, no target-label selection')

    def readout_vectors(self,profile,geo,labels,block_seed):
        ix=profile['indices'];p=geo.source_probabilities[ix];y=torch.nn.functional.one_hot(labels[ix],2).double();q=geo.contrasts
        apply=lambda residual:-(geo.basis@(profile['W']@residual.reshape(-1)))
        full=apply((y-p)@q);globalpart=apply((y.mean(0)-p.mean(0))[None,:].expand_as(p)@q)
        pred=apply(-(p-p.mean(0))@q);base=globalpart+pred
        permutations=[]
        for rep in [1,2,3]:
            order=torch.randperm(len(ix),generator=torch.Generator().manual_seed(block_seed+9100+rep))
            permutations.append(apply((y[order]-p)@q))
        pipeline=profile.get('pipeline_permutations',permutations)
        return feedback_controls(full,base,permutations,pipeline,geo.basis,block_seed,dict(GLOBAL=globalpart,PRED=pred))

    @staticmethod
    @torch.no_grad()
    def margins(snapshot,flat,batches):
        logits=torch.cat([snapshot.logits(batch,flat) for batch in batches])
        return (logits[:,1]-logits[:,0]).detach().numpy()

    def worker(self,seed,pilot=False):
        seed_everything(seed,1,True);out=self.stage/self.phase/(f'pilot_{seed}' if pilot else str(seed));out.mkdir(exist_ok=False)
        (out/'blocks').mkdir();writer=SummaryWriter(str(out/'tensorboard'))
        begun_cpu,begun=cpu_seconds(),time.time();block_count=profile_count=step_profiles=0
        meta=dict(seed=seed,git_commit=self.commit,protocol_sha256=self.plan_sha,config_name=self.settings['config_name'],
            hardware=self.hardware,gpu_hours=0.,event_flags='clean',phase=self.phase)
        write(out/'RUN_INFO.json',dict(meta,started_at=begun,pilot=pilot))
        try:
            for domain in ([1] if pilot else self.settings['domains']):
                task=self.old.task(domain);_,_,dev,final=self.old.pools(domain)
                oracle=DevelopmentOracle(task,dev,final,out/'label_access.jsonl',enabled=True)
                for step in self.snapshot_steps(pilot):
                    model,reference=self.old.load(domain,seed,step);before=model_hash(model);b_grad=None
                    for block in ([0] if pilot else self.settings['blocks']):
                        token=f'{domain}_{seed}_{step}_{block}';directory=out/'blocks'/token;directory.mkdir()
                        geo,ids,cache_sha=self.geometry(model,seed,domain,step,block)
                        labels=task.batch(ids['query'],'query').y;snapshot=geo.snapshot
                        assert ids['query']==sorted(task.query_ids) and not set(ids['query'])&set(task.fit_ids)
                        write(directory/'ids.json',dict(ids,checkpoint=reference,geometry_sha256=cache_sha))
                        profiles,additional=self.profiles(geo,ids,labels,seed,domain,step,block,directory,token)
                        batch_a=task.batch(ids['A'],'target');fit=task.batch(ids['fit'],'fit')
                        source_gradient=gradient_of_ce(snapshot,[fit],fit.y);batches=self.old.batches(task,ids['B'])
                        perform=block in self.settings['step_blocks'];vectors={};validity={};weights={};coeff={};details=[]
                        predictions={'INITIAL':self.margins(snapshot,snapshot.flat,batches)}
                        if perform:predictions['SOURCE']=self.margins(snapshot,snapshot.flat-self.settings['eta']*source_gradient,batches)
                        for profile in profiles:
                            name=profile['profile'];v,valid=self.profile_vectors(profile,geo,labels,ids['block_seed'])
                            vectors[name],validity[name]=v,valid
                            if 'W' in profile:weights[name]=profile['W'].numpy()
                            for arm,g in v.items():
                                coeff[name+'::'+arm]=(geo.basis.mT@g).numpy()
                                if perform:predictions[name+'::'+arm]=self.margins(snapshot,snapshot.flat-self.settings['eta']*(source_gradient+self.settings['beta']*g),batches)
                            ix=profile['indices'];safe={k:v for k,v in profile.items() if k not in ['W','indices','pipeline_permutations','vectors','validity']}
                            details.append(dict(safe,selected_query_ids=[ids['query'][int(i)] for i in ix],validity=valid,
                                source_class_counts=torch.bincount(labels[ix],minlength=2).tolist()))
                        np.savez_compressed(directory/'readouts.npz',V=geo.basis.numpy(),Q=geo.contrasts.numpy(),**weights)
                        np.savez_compressed(directory/'coefficients.npz',source_gradient=source_gradient.numpy(),**coeff)
                        np.savez_compressed(directory/'step_logits.npz',**predictions)
                        write(directory/'profiles.json',dict(meta,token=token,task=f'0_to_{domain}',snapshot_step=step,block=block,
                            profiles=details,additional=additional,performed_steps=perform,original_model_hash=before))
                        artifacts=[p for p in directory.iterdir() if p.is_file() and p.name!='seal.json']
                        seal=dict(files=[dict(path=p.name,sha256=sha256(p)) for p in sorted(artifacts)],time=time.time())
                        write(directory/'seal.json',seal)
                        oracle.seal(token,ids['A'],theta_hash=tensor_hash(snapshot.flat),readout_hash=sha256(directory/'seal.json'))
                        if not pilot:
                            ay=oracle.labels(ids['A'],token=token,purpose=f'{self.phase}_A_empirical_gradient_after_candidate_seal')
                            by=oracle.labels(ids['B'],token=token,purpose=f'{self.phase}_B_evaluation_after_logits_seal')
                            np.savez_compressed(directory/'evaluation_labels.npz',A=ay.numpy(),B=by.numpy())
                            ga=gradient_of_ce(snapshot,[batch_a],ay)
                            if b_grad is None:b_grad=gradient_of_ce(snapshot,batches,by)
                            baseline=binary_logit_metrics(predictions['INITIAL'],by.numpy())
                            common=dict(meta,task=f'0_to_{domain}',snapshot_step=step,block_id=block,token=token,
                                checkpoint_sha256=reference['checkpoint_sha256'],eta=self.settings['eta'],beta=self.settings['beta'])
                            if perform:
                                source_metrics=binary_logit_metrics(predictions['SOURCE'],by.numpy())
                                append(out/'step_blocks.jsonl',dict(common,profile='SOURCE',arm='SOURCE',CE_B=source_metrics['ce'],
                                    BA_B=100*source_metrics['balanced_accuracy'],base_CE_B=baseline['ce'],source_CE_B=source_metrics['ce'],R_Q=0.,valid=True))
                            oracle_predictions={}
                            for profile in profiles:
                                name=profile['profile'];v=vectors[name];valid=validity[name]
                                for arm,g in v.items():
                                    gm=gradient_metrics(g,ga,b_grad,source_gradient,geo.basis)
                                    append(out/'gradient_blocks.jsonl',dict(common,profile=name,arm=arm,control_valid=valid[arm],**gm))
                                    if perform:
                                        m=binary_logit_metrics(predictions[name+'::'+arm],by.numpy())
                                        append(out/'step_blocks.jsonl',dict(common,profile=name,arm=arm,CE_B=m['ce'],BA_B=100*m['balanced_accuracy'],
                                            base_CE_B=baseline['ce'],source_CE_B=source_metrics['ce'],R_Q=(source_metrics['ce']-m['ce'])/baseline['ce'],valid=valid[arm]))
                                if perform:
                                    og,ok=normalized(ga,float(v['FULL'].norm()),1e-12)
                                    margin=self.margins(snapshot,snapshot.flat-self.settings['eta']*(source_gradient+self.settings['beta']*og),batches)
                                    oracle_predictions[name]=margin;m=binary_logit_metrics(margin,by.numpy())
                                    append(out/'step_blocks.jsonl',dict(common,profile=name,arm='ORACLE_M',CE_B=m['ce'],BA_B=100*m['balanced_accuracy'],
                                        base_CE_B=baseline['ce'],source_CE_B=source_metrics['ce'],R_Q=(source_metrics['ce']-m['ce'])/baseline['ce'],valid=ok))
                            if perform:np.savez_compressed(directory/'oracle_step_logits.npz',**oracle_predictions)
                            np.savez_compressed(directory/'empirical_gradients.npz',A=ga.numpy(),B=b_grad.numpy())
                            self.extra_evaluation(directory,profiles,ids,ay,by,common,out)
                        if model_hash(model)!=before:raise ValueError('Temporary evaluation modified the model')
                        block_count+=1;profile_count+=len(profiles);step_profiles+=len(profiles)*perform
                        progress=dict(seed=seed,domain=domain,snapshot=step,block=block,blocks=block_count,profiles=profile_count,
                            step_profiles=step_profiles,cpu_seconds=cpu_seconds()-begun_cpu,elapsed=time.time()-begun,
                            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        append(out/'history.jsonl',dict(event='block_complete',model_unchanged=True,**progress));write(out/'progress.json',progress)
                        writer.add_scalar('compute/cpu_seconds',progress['cpu_seconds'],block_count)
                        if progress['cpu_seconds']>self.remaining_cpu()/5 or progress['peak_rss_kib']>self.settings['memory_cap_gib']*1024**2/5:
                            raise TimeoutError('Feedback worker exceeded remaining resource allocation')
                        print(json.dumps(dict(event=self.phase+'_block',**progress)),flush=True)
            result=dict(completed=True,pilot=pilot,seed=seed,blocks=block_count,profiles=profile_count,step_profiles=step_profiles,
                cpu_seconds=cpu_seconds()-begun_cpu,elapsed=time.time()-begun,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,git_commit=self.commit)
            write(out/'done.json',result);return result
        except Exception:
            write(out/'failed.json',dict(cpu_seconds=cpu_seconds()-begun_cpu,traceback=traceback.format_exc()));raise
        finally:writer.close()

    def profile_vectors(self,profile,geo,labels,block_seed):
        return self.readout_vectors(profile,geo,labels,block_seed)

    def snapshot_steps(self,pilot):return [0]

    def extra_evaluation(self,directory,profiles,ids,ay,by,common,out):pass

    def remaining_cpu(self):
        used=read(self.stage/'A4/done.json')['cpu_seconds_including_pilot_d0']
        p=self.stage/'A4/audit.json'
        if p.exists():used+=read(p)['cpu_seconds']
        return self.settings['cpu_hour_cap']*3600-used

    def run(self):
        root=self.stage/self.phase;pilot=read(root/'pilot_3405/done.json')
        expected=read(root/'manifest.json')['expected_blocks']
        estimate=pilot['cpu_seconds']*expected*self.settings['compute_buffer']
        if estimate>self.remaining_cpu():raise TimeoutError('Pilot forecast exceeds remaining package budget')
        write(root/'compute_estimate.json',dict(estimated_cpu_seconds=estimate,remaining_cpu_seconds=self.remaining_cpu(),
            pilot_cpu_seconds=pilot['cpu_seconds'],expected_blocks=expected,conservative_all_blocks_have_steps=True))
        started=time.time();results=[]
        with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(run_worker,str(self.repo),str(self.stage),self.config,seed,self.phase):seed for seed in self.settings['seeds']}
            while pending:
                done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:results.append(f.result());del pending[f]
                states=[]
                for seed in self.settings['seeds']:
                    p=root/str(seed)/'progress.json'
                    if p.exists():states.append(read(p))
                total=pilot['cpu_seconds']+sum(s['cpu_seconds'] for s in states)
                write(root/'compute_ledger.json',dict(status='running' if pending else 'completed',cpu_seconds=total,states=states,gpu_hours=0.))
                print(json.dumps(dict(event=self.phase+'_progress',pending=list(pending.values()),elapsed=time.time()-started,cpu_seconds=total)),flush=True)
                if total>self.remaining_cpu():
                    for process in pool._processes.values():process.terminate()
                    raise TimeoutError('Feedback phase hard CPU cap')
        write(root/'done.json',dict(completed=True,blocks=sum(r['blocks'] for r in results),profiles=sum(r['profiles'] for r in results),
             step_profiles=sum(r['step_profiles'] for r in results),cpu_seconds_including_pilot=pilot['cpu_seconds']+sum(r['cpu_seconds'] for r in results),workers=results))


def run_worker(repo,stage,config,seed,phase):
    if phase=='A6':
        from optimization_candidates import OptimizationCandidates
        return OptimizationCandidates(repo,stage,config).worker(seed)
    return OptimizationFeedback(repo,stage,config).worker(seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pilot','run']);p.add_argument('--repo',required=True)
    p.add_argument('--stage',required=True);p.add_argument('--config',required=True);a=p.parse_args();torch.set_num_threads(1)
    e=OptimizationFeedback(a.repo,a.stage,a.config)
    if a.action=='prepare':e.prepare()
    elif a.action=='pilot':e.worker(3405,True)
    else:e.run()


if __name__=='__main__':main()
