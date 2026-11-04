"""Source-selected LOCAL/KERNEL candidates and LR teachers, fixed A6 gates."""
import argparse
import ast
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from optimization_feedback import OptimizationFeedback
from p0_experiment import read,write
from assumptions_experiment import append
from tqg_gda.assumption_diagnostics import metrics,stable_seed
from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from tqg_gda.optimization_diagnostics import (bandwidth_kernel,effective_lambda,closest_bank,select_local,
    select_kernel,teacher_fit,feedback_controls)
from tqg_gda.p0_diagnostics import gradient_of_ce


def compressed_json(path,data):
    with gzip.open(path,'wt') as f:json.dump(data,f,allow_nan=False)


class OptimizationCandidates(OptimizationFeedback):
    phase='A6'

    def prepare(self):
        super().prepare()
        assert read(self.stage/'A2b/done.json')['completed']
        manifest=read(self.stage/'A6/manifest.json')
        manifest.update(expected_blocks=360,expected_profiles=3600,expected_step_profiles=1800,
            candidates=['LOCAL','KERNEL'],additional_reference='KERNEL_JOINT same selected parameters, not independently selected',
            teacher_projection='Same full-nonzero-rank A16 V; unprojected soft CE gradient independently verified')
        write(self.stage/'A6/manifest.json',manifest)
        findings=[]
        for node in ast.walk(ast.parse((self.code/'optimization_candidates.py').read_text())):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ['exec','eval','compile']:findings.append(node.lineno)
        assert not findings
        write(self.stage/'D0/ast_screen_A6_candidates.json',dict(passed=True,findings=findings))

    def snapshot_steps(self,pilot):return [0] if pilot else self.settings['snapshots']

    def remaining_cpu(self):
        used=read(self.stage/'A2b/done.json')['cpu_seconds_including_pilot']
        p=self.stage/'A2b/audit.json'
        if p.exists():used+=read(p).get('cpu_seconds',0)
        return super().remaining_cpu()-used

    def profiles(self,geo,ids,labels,seed,domain,step,block,out,token):
        n=len(labels);nt=len(ids['A']);kernels,kstats=kernel_family(geo.source_representation,geo.target_representation,
            geo.source_response,geo.target_response,torch.arange(n))
        fullkernels={sig:torch.cat((torch.cat((kernels[sig][0],kernels[sig][1]),1),
            torch.cat((kernels[sig][1].mT,kernels[sig][2]),1)),0).numpy() for sig in self.settings['signatures']}
        # Recompute normalized squared distances directly; do not invert an underflowed RBF.
        h=torch.cat((geo.source_representation,geo.target_representation))
        a=torch.cat((geo.source_response,geo.target_response)).flatten(1)
        dh=torch.cdist(h,h,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/kstats['scale_h_squared']
        da=torch.cdist(a,a,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/kstats['scale_a_squared']
        distances={'H':dh,'HA':dh+da};source_p=geo.source_probabilities[:,1].numpy()
        variants=[labels.numpy()]
        for rep in [1,2,3]:
            order=torch.randperm(n,generator=torch.Generator().manual_seed(ids['block_seed']+9100+rep))
            variants.append(labels[order].numpy())
        variants=np.array(variants)
        local_choices,local_cv=select_local(ids['query'],fullkernels,variants,source_p,self.settings)
        kernel_choices,kernel_cv=select_kernel(ids['query'],distances,variants,source_p,self.settings)
        compressed_json(out/'selection_cv.json.gz',dict(local=local_cv,kernel=kernel_cv,
            notes='Source labels only; each null independently reselects all options. Target B absent from choice functions.'))
        write(out/'selection.json',dict(local=local_choices,kernel=kernel_choices,variant_labels=variants.tolist()))
        result=[]

        def make(name,sig,ix,mode,lam,c=1.,**extra):
            k=bandwidth_kernel(distances[sig],c);ix=np.asarray(ix,dtype=int)
            w,info=solve_observed((torch.from_numpy(k[np.ix_(ix,ix)]),torch.from_numpy(k[ix,n:]),torch.from_numpy(k[n:,n:])),
                geo.target_response,mode,lam,self.settings['calibration'])
            return dict(profile=name,signature=sig,n=len(ix),indices=ix,W=w,solver=info,lambda_w=lam,mode=mode,bandwidth=c,
                attribution='fixed-selected-W decomposition plus whole-pipeline permutation',**extra)

        def applied(profile,y):
            ix=profile['indices'];yy=torch.from_numpy(np.asarray(y)[ix]).long()
            residual=(torch.nn.functional.one_hot(yy,2).double()-geo.source_probabilities[ix])@geo.contrasts
            return -(geo.basis@(profile['W']@residual.reshape(-1)))

        for sig in self.settings['signatures']:
            result.append(make('OLD_'+sig,sig,np.arange(n),'JOINT',.01,selector='FULL',convention='FIXED'))
            result.append(make('MEAN_'+sig,sig,np.arange(n),'MEAN',.005,selector='FULL',convention='FIXED'))
        local_profiles=[]
        for v,choice in enumerate(local_choices):
            ix=closest_bank(ids['query'],fullkernels[choice['selector']][:n,n:],choice['n'])
            lam=effective_lambda(.01,len(ix),n,choice['convention'])
            local_profiles.append(make('LOCAL','H',ix,'JOINT',lam,selector=choice['selector'],convention=choice['convention'],source_cv_nll=choice['cv_nll']))
        local_profiles[0]['pipeline_permutations']=[applied(local_profiles[v],variants[v]) for v in [1,2,3]]
        result.append(local_profiles[0])
        for mode,name,factor in [('MEAN','KERNEL',1),('JOINT','KERNEL_JOINT',2)]:
            profiles=[]
            for v,choice in enumerate(kernel_choices):
                profiles.append(make(name,choice['signature'],np.arange(n),mode,factor*choice['lambda_base'],choice['bandwidth'],
                    selector='FULL',convention=choice['convention'],source_cv_nll=choice['cv_nll']))
            profiles[0]['pipeline_permutations']=[applied(profiles[v],variants[v]) for v in [1,2,3]]
            result.append(profiles[0])

        # Only unlabelled B features are used for teacher predictions, never source selection.
        task=self.old.task(domain);snapshot=geo.snapshot;cache_key=(seed,domain,step)
        if not hasattr(self,'_teacher_cache') or self._teacher_cache['key']!=cache_key:
            hb=[];jb=[]
            for batch in self.old.batches(task,ids['B']):
                hb.append(snapshot.representation(batch).detach())
                jb.append(snapshot.jacobian(batch,geo.contrasts))
            self._teacher_cache=dict(key=cache_key,h=torch.cat(hb),jacobian=torch.cat(jb),B=ids['B'])
        cache=self._teacher_cache;assert cache['B']==ids['B']
        ab=(cache['jacobian']@geo.basis).reshape(len(ids['B']),1,-1)
        evaluation={
            'H':torch.cat((geo.target_representation,cache['h'])).numpy(),
            'HA':torch.cat((torch.cat((geo.target_representation,cache['h'])),
                           torch.cat((geo.target_response,ab)).flatten(1)),1).numpy()}
        sx={'H':geo.source_representation.numpy(),
            'HA':torch.cat((geo.source_representation,geo.source_response.flatten(1)),1).numpy()}
        a_batch=task.batch(ids['A'],'target');ap=snapshot.logits(a_batch).detach().softmax(-1)[:,1]
        np.savez_compressed(out/'teacher_features.npz',source_H=sx['H'],source_HA=sx['HA'],
            target_H=evaluation['H'],target_HA=evaluation['HA'],source_probability=source_p,
            target_A_probability=ap.numpy(),source_labels=labels.numpy())
        qdiff=float(geo.contrasts[1,0]-geo.contrasts[0,0])
        def teacher_gradient(probability):
            residual=(torch.from_numpy(probability[:nt])-ap)*qdiff
            return -(geo.basis@(geo.target_response[:,0,:].mT@residual/nt))

        for name,sig,ix in [('TEACHER_H','H',np.arange(n)),('TEACHER_HA','HA',np.arange(n)),
                            ('TEACHER_LOCAL','H',local_profiles[0]['indices'])]:
            ix=np.asarray(ix);bank_ids=[ids['query'][int(i)] for i in ix];true_y=labels[ix].numpy()
            conditional=[true_y]
            for rep in [1,2,3]:
                perm=torch.randperm(len(ix),generator=torch.Generator().manual_seed(ids['block_seed']+9100+rep)).numpy()
                conditional.append(true_y[perm])
            pred,states,choices,cv=teacher_fit(bank_ids,sx[sig][ix],np.array(conditional),evaluation[sig],
                self.settings['teacher_lambdas'],stable_seed(seed,token,name))
            stored={f'variant{v}_{key}':value for (v,_),state in states.items() for key,value in state.items()}
            metadata=dict(name=name,signature=sig,source_ids=bank_ids,source_labels=np.array(conditional).tolist(),selected=choices,inner_cv=cv)
            pipeline=[teacher_gradient(pred[v]) for v in [1,2,3]]
            if name=='TEACHER_LOCAL':
                pipeline=[];pipeline_meta=[]
                for v in [1,2,3]:
                    jx=local_profiles[v]['indices'];ji=[ids['query'][int(i)] for i in jx]
                    pp,ss,cc,rr=teacher_fit(ji,sx[sig][jx],variants[v:v+1,jx],evaluation[sig],self.settings['teacher_lambdas'],stable_seed(seed,token,name))
                    pipeline.append(teacher_gradient(pp[0]));pipeline_meta.append(dict(variant=v,source_ids=ji,selected=cc,inner_cv=rr))
                    stored.update({f'pipeline{v}_{key}':value for key,value in ss[0,0].items()});stored[f'pipeline{v}_predictions']=pp[0]
                metadata['whole_pipeline_null']=pipeline_meta
            full=teacher_gradient(pred[0]);base=teacher_gradient(np.full(len(pred[0]),true_y.mean()))
            vectors,valid=feedback_controls(full,base,[teacher_gradient(pred[v]) for v in [1,2,3]],pipeline,geo.basis,ids['block_seed'])
            soft=torch.from_numpy(np.c_[1-pred[0,:nt],pred[0,:nt]])
            raw_gradient=gradient_of_ce(snapshot,[a_batch],soft)
            projection_error=float((raw_gradient-full).abs().max())
            if projection_error>1e-8:raise ValueError('Full target Jacobian span does not reproduce soft teacher gradient')
            vectors['FULL_UNPROJECTED']=raw_gradient;valid['FULL_UNPROJECTED']=float(raw_gradient.norm())>1e-12
            np.savez_compressed(out/(name+'.npz'),predictions=pred,source_ids=np.array(bank_ids),A_ids=ids['A'],B_ids=ids['B'],
                full_gradient=raw_gradient.numpy(),projected_gradient=full.numpy(),**stored)
            compressed_json(out/(name+'.json.gz'),metadata)
            result.append(dict(profile=name,signature=sig,n=len(ix),indices=ix,mode='LR_TEACHER',selector='LOCAL' if name=='TEACHER_LOCAL' else 'FULL',
                convention='source_CV',vectors=vectors,validity=valid,teacher_file=name+'.npz',projection_max_error=projection_error,
                attribution='teacher output difference from prior; not a linear source-label decomposition'))
        assert len(result)==10
        return result,dict(kernel_statistics=kstats,local_selected=local_choices[0],kernel_selected=kernel_choices[0],
            candidate_choices='source-only CV; whole pipeline repeated for nulls',teacher_full_equals_projected_on_A='full numerical target rank retained')

    def profile_vectors(self,profile,geo,labels,block_seed):
        if 'vectors' in profile:return profile['vectors'],profile['validity']
        return super().profile_vectors(profile,geo,labels,block_seed)

    def extra_evaluation(self,directory,profiles,ids,ay,by,common,out):
        for profile in profiles:
            if 'teacher_file' not in profile:continue
            z=np.load(directory/profile['teacher_file'])
            for v in range(4):
                nt=len(ids['A'])
                for pool,labels,pred in [('A',ay.numpy(),z['predictions'][v,:nt]),('B',by.numpy(),z['predictions'][v,nt:])]:
                    append(out/'teacher_prediction_metrics.jsonl',dict(common,profile=profile['profile'],variant=v,pool=pool,
                        projection_max_error=profile['projection_max_error'],**metrics(labels,pred)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pilot','run']);p.add_argument('--repo',required=True)
    p.add_argument('--stage',required=True);p.add_argument('--config',required=True);a=p.parse_args();torch.set_num_threads(1)
    e=OptimizationCandidates(a.repo,a.stage,a.config)
    if a.action=='prepare':e.prepare()
    elif a.action=='pilot':e.worker(3405,True)
    else:e.run()


if __name__=='__main__':main()
