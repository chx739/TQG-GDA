"""Verify sealed feedback, source selection, label boundaries and real SGD replay."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import expit
import torch
from torch.nn import functional as F

from p0_experiment import ResearchExperiment,ExperimentConfig,read,write
from tqg_gda.feedback_v2 import ParameterSnapshot
from tqg_gda.p0_diagnostics import DevelopmentOracle,model_hash
from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from tqg_gda.optimization_diagnostics import bandwidth_kernel,effective_lambda,closest_bank
from tqg_gda.utils import sha256,seed_everything


def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def gzread(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def relative_error(left,right):
    return float(np.max(np.abs(np.asarray(left)-np.asarray(right)))/(1+np.max(np.abs(right))))


def normalize(v,norm,epsilon=1e-12):
    actual=np.linalg.norm(v)
    return v*(norm/actual) if actual>epsilon else np.zeros_like(v)


def replay_controls(full,base,conditional,pipeline,rank,seed,extra=None):
    label=full-base;fn=np.linalg.norm(full);ln=np.linalg.norm(label)
    result=dict(FULL=full,BASE=base,BASE_M=normalize(base,fn),LABEL=label,**(extra or {}))
    for i,(per,pipe) in enumerate(zip(conditional,pipeline),1):
        random=torch.randn(rank,generator=torch.Generator().manual_seed(seed+9200+i),dtype=torch.float64).numpy()
        cp=base+normalize(per-base,ln);cr=base+normalize(random,ln)
        result.update({f'CPL_{i}':cp,f'CRL_{i}':cr,f'CPL_M_{i}':normalize(cp,fn),f'CRL_M_{i}':normalize(cr,fn),
            f'FULL_PERM_M_{i}':normalize(pipe,fn),f'FULL_RAND_M_{i}':normalize(random,fn)})
    return result


def check_selection(selection,records,query):
    allowed=set(query);grouped=defaultdict(list)
    for r in records['kernel']:
        assert set(r['training_ids'])<=allowed and set(r['validation_ids'])<=allowed
        assert not set(r['training_ids'])&set(r['validation_ids'])
        grouped[(r['variant'],r['signature'],r['bandwidth'],r['convention'],r['lambda_base'])].append(r['nll'])
    for v,c in enumerate(selection['kernel']):
        candidates=[k for k in grouped if k[0]==v]
        assert all(len(grouped[k])==3 for k in candidates)
        best=min(candidates,key=lambda k:(sum(x/3 for x in grouped[k]),-k[4],-k[2],k[1]!='H',k[3]!='NORMALIZED'))
        assert best[1:]==(c['signature'],c['bandwidth'],c['convention'],c['lambda_base'])
    grouped=defaultdict(list)
    for r in records['local']:
        assert set(r['bank_ids'])<=allowed and set(r['validation_ids'])<=allowed
        assert not set(r['bank_ids'])&set(r['validation_ids'])
        grouped[(r['variant'],r['selector'],r['n'],r['convention'])].append(r['nll'])
    for v,c in enumerate(selection['local']):
        candidates=[k for k in grouped if k[0]==v];assert all(len(grouped[k])==4 for k in candidates)
        best=min(candidates,key=lambda k:(sum(x/4 for x in grouped[k]),-k[2],k[1]!='H',k[3]!='NORMALIZED'))
        assert best[1:]==(c['selector'],c['n'],c['convention'])


def check_teacher_cv(records,selected,ids):
    groups=defaultdict(list);allowed=set(ids)
    for r in records:
        assert set(r['train_ids'])<=allowed and set(r['validation_ids'])<=allowed and not set(r['train_ids'])&set(r['validation_ids'])
        groups[r['variant'],r['penalty']].append(r['nll'])
    for item in selected:
        v=item['variant'];keys=[k for k in groups if k[0]==v]
        assert all(len(groups[k])==3 for k in keys)
        chosen=min(keys,key=lambda k:(sum(x/3 for x in groups[k]),-k[1]))
        assert chosen[1]==item['penalty']


def audit_seed(repo,stage,phase,seed):
    seed_everything(seed,1,True);repo,stage=Path(repo),Path(stage);root=stage/phase;manifest=read(root/'manifest.json');cfg=manifest['config']
    out=root/str(seed);audit=root/'audit';audit.mkdir(exist_ok=True);started=time.process_time();counts=defaultdict(int);errors=defaultdict(float)
    backend=ResearchExperiment(ExperimentConfig(str(repo),cfg['old_stage'],str(repo/'configs/p0_diagnostics.json')))
    gradients={(r['token'],r['profile'],r['arm']):r for r in lines(out/'gradient_blocks.jsonl')}
    steps={(r['token'],r['profile'],r['arm']):r for r in lines(out/'step_blocks.jsonl')}
    teacher_metrics={(r['token'],r['profile'],r['variant'],r['pool']):r for r in lines(out/'teacher_prediction_metrics.jsonl')} if phase=='A6' else {}
    events=lines(out/'label_access.jsonl');cache={}
    def check(name,a,b):
        error=relative_error(a,b);errors[name]=max(errors[name],error)
        if error>1e-8:raise AssertionError((name,error))
    blockdirs=sorted((out/'blocks').iterdir(),key=lambda p:tuple(map(int,p.name.split('_'))))
    for directory in blockdirs:
        token=directory.name;domain,_,snapshot_step,block=map(int,token.split('_'));meta=read(directory/'profiles.json');ids=read(directory/'ids.json')
        assert meta['git_commit']==manifest['git_commit']
        seal=read(directory/'seal.json')
        for item in seal['files']:assert sha256(directory/item['path'])==item['sha256'];counts['sealed_files']+=1
        sealing=next(r for r in events if r['event']=='unlabeled_query_sealed' and r['token']==token)
        assert sealing['readout_hash']==sha256(directory/'seal.json')
        requests=[r for r in events if r['event']=='label_access' and r['token']==token]
        assert len(requests)==2 and all(r['time']>=sealing['time']>=seal['time'] for r in requests)
        assert requests[0]['ids']==ids['A'] and requests[1]['ids']==ids['B']
        if (domain,snapshot_step) not in cache:
            model,reference=backend.load(domain,seed,snapshot_step);task=backend.task(domain)
            _,_,dev,final=backend.pools(domain)
            oracle=DevelopmentOracle(task,dev,final,audit/f'{seed}_label_access.jsonl',enabled=True)
            cache[domain,snapshot_step]=dict(model=model,task=task,reference=reference,oracle=oracle,B_gradient=None)
        c=cache[domain,snapshot_step];model=c['model'];task=c['task'];oracle=c['oracle'];before=model_hash(model)
        assert ids['query']==sorted(task.query_ids) and not set(ids['query'])&set(task.fit_ids)
        apool,bpool,_,final=backend.pools(domain)
        assert set(ids['A'])<=set(apool) and ids['B']==list(bpool) and not(set(ids['A'])|set(ids['B']))&set(final)
        oracle.seal(token,ids['A'],theta_hash=before,readout_hash=sha256(directory/'seal.json'))
        ya=oracle.labels(ids['A'],token=token,purpose=phase+'_independent_audit_A')
        yb=oracle.labels(ids['B'],token=token,purpose=phase+'_independent_audit_B')
        truth=dict(np.load(directory/'evaluation_labels.npz'));np.testing.assert_array_equal(truth['A'],ya);np.testing.assert_array_equal(truth['B'],yb)
        qy=task.batch(ids['query'],'query').y.numpy();n=len(qy)
        geo_path=Path(cfg['old_stage'])/f'geometry_cache/{seed}/{token}.npz';assert sha256(geo_path)==ids['geometry_sha256']
        geo=dict(np.load(geo_path));v=geo['V'];q=geo['Q'];rank=v.shape[1]
        check('basis_orthogonality',v.T@v,np.eye(rank))
        arrays=dict(np.load(directory/'coefficients.npz'));observed=dict(np.load(directory/'empirical_gradients.npz'))
        with torch.no_grad():pa=model.head(torch.from_numpy(geo['target_h'])).softmax(-1).numpy()
        ar=(np.eye(2)[ya.numpy()]-pa)@q
        expected_ga=-v@(geo['target_A'][:,0,:].T@ar[:,0]/len(ya));check('A_gradient_from_response',expected_ga,observed['A'])
        fit=task.batch(ids['fit'],'fit');params=[p for p in model.parameters() if p.requires_grad]
        gs=torch.autograd.grad(F.cross_entropy(model(fit),fit.y),params,allow_unused=True)
        gs=np.concatenate([(torch.zeros_like(p) if g is None else g).detach().numpy().reshape(-1) for p,g in zip(params,gs)])
        check('ordinary_module_source_gradient',gs,arrays['source_gradient'])
        batches=backend.batches(task,ids['B'])
        if c['B_gradient'] is None:
            logits=torch.cat([model(b) for b in batches]);bg=torch.autograd.grad(F.cross_entropy(logits,yb),params,allow_unused=True)
            c['B_gradient']=np.concatenate([(torch.zeros_like(p) if g is None else g).detach().numpy().reshape(-1) for p,g in zip(params,bg)])
        check('ordinary_module_B_gradient',c['B_gradient'],observed['B'])
        kf,kstats=kernel_family(torch.from_numpy(geo['source_h']),torch.from_numpy(geo['target_h']),
            torch.from_numpy(geo['source_A']),torch.from_numpy(geo['target_A']),torch.arange(n))
        allkernels={sig:np.block([[kf[sig][0].numpy(),kf[sig][1].numpy()],[kf[sig][1].mT.numpy(),kf[sig][2].numpy()]]) for sig in ['H','HA']}
        h=torch.from_numpy(np.concatenate((geo['source_h'],geo['target_h'])));a=torch.from_numpy(np.concatenate((geo['source_A'],geo['target_A'])).reshape(n+len(ya),-1))
        dh=torch.cdist(h,h,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/kstats['scale_h_squared']
        da=torch.cdist(a,a,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/kstats['scale_a_squared']
        distances={'H':dh,'HA':dh+da};weights=dict(np.load(directory/'readouts.npz'));preds=dict(np.load(directory/'step_logits.npz'))
        oracle_preds=dict(np.load(directory/'oracle_step_logits.npz')) if meta['performed_steps'] else {}
        selection=None;features=None
        if phase=='A6':
            selection=read(directory/'selection.json');records=gzread(directory/'selection_cv.json.gz');check_selection(selection,records,ids['query'])
            counts['source_candidate_cv_records']+=len(records['kernel'])+len(records['local'])
            features=dict(np.load(directory/'teacher_features.npz'))
            np.testing.assert_array_equal(selection['variant_labels'][0],qy)
            for rep in [1,2,3]:
                order=torch.randperm(n,generator=torch.Generator().manual_seed(ids['block_seed']+9100+rep)).numpy()
                np.testing.assert_array_equal(selection['variant_labels'][rep],qy[order])
            check('teacher_source_features_H',features['source_H'],geo['source_h'])
            check('teacher_source_features_HA',features['source_HA'],np.c_[geo['source_h'],geo['source_A'].reshape(n,-1)])
        def solve(sig,ix,mode,lam,width):
            k=allkernels[sig] if phase=='A2b' else bandwidth_kernel(distances[sig],width)
            w,info=solve_observed((torch.from_numpy(k[np.ix_(ix,ix)]),torch.from_numpy(k[ix,n:]),torch.from_numpy(k[n:,n:])),
                    torch.from_numpy(geo['target_A']),mode,lam,cfg['calibration'])
            counts['readout_recomputations']+=1
            return w.numpy()
        def apply(w,ix,y):return -w@(((np.eye(2)[y[ix]]-geo['source_p'][ix])@q).reshape(-1))
        expected_all={}
        for profile in meta['profiles']:
            name=profile['profile'];ix=np.array([ids['query'].index(i) for i in profile['selected_query_ids']]);teacher='teacher_file' in profile
            if phase=='A6' and name in ['LOCAL','TEACHER_LOCAL']:
                chosen=selection['local'][0]
                np.testing.assert_array_equal(ix,closest_bank(ids['query'],allkernels[chosen['selector']][:n,n:],chosen['n']))
                if name=='LOCAL':
                    assert profile['signature']=='H' and profile['mode']=='JOINT'
                    check('local_selected_lambda',profile['lambda_w'],effective_lambda(.01,len(ix),n,chosen['convention']))
            if phase=='A6' and name in ['KERNEL','KERNEL_JOINT']:
                chosen=selection['kernel'][0]
                assert profile['signature']==chosen['signature'] and profile['bandwidth']==chosen['bandwidth']
                check('kernel_selected_lambda',profile['lambda_w'],chosen['lambda_base']*(2 if name=='KERNEL_JOINT' else 1))
            if not teacher:
                w=weights[name];replay=solve(profile['signature'],ix,profile.get('mode','JOINT'),profile['lambda_w'],profile.get('bandwidth',1.))
                check('readout_replay',w,replay);counts['finite_nonconverged_primary']+=not profile['solver']['converged']
                py=geo['source_p'][ix];yy=np.eye(2)[qy[ix]];full=apply(w,ix,qy)
                base=-w@(((yy.mean(0)[None,:]-py)@q).reshape(-1))
                globalpart=-w@(((yy.mean(0)-py.mean(0))[None,:].repeat(len(ix),0)@q).reshape(-1))
                predpart=-w@((-(py-py.mean(0))@q).reshape(-1));conditional=[];pipeline=[]
                for rep in [1,2,3]:
                    order=torch.randperm(len(ix),generator=torch.Generator().manual_seed(ids['block_seed']+9100+rep)).numpy()
                    conditional.append(-w@(((yy[order]-py)@q).reshape(-1)))
                    if phase=='A6' and name in ['LOCAL','KERNEL','KERNEL_JOINT']:
                        if name=='LOCAL':
                            choice=selection['local'][rep];jx=closest_bank(ids['query'],allkernels[choice['selector']][:n,n:],choice['n'])
                            lam=effective_lambda(.01,len(jx),n,choice['convention']);nw=solve('H',jx,'JOINT',lam,1)
                        else:
                            choice=selection['kernel'][rep];jx=np.arange(n);mode='MEAN' if name=='KERNEL' else 'JOINT'
                            nw=solve(choice['signature'],jx,mode,choice['lambda_base']*(1 if mode=='MEAN' else 2),choice['bandwidth'])
                        pipeline.append(apply(nw,jx,np.array(selection['variant_labels'][rep],dtype=int)))
                    else:pipeline.append(conditional[-1])
                expected=replay_controls(full,base,conditional,pipeline,rank,ids['block_seed'],dict(GLOBAL=globalpart,PRED=predpart))
            else:
                t=dict(np.load(directory/profile['teacher_file']));tm=gzread(directory/(name+'.json.gz'))
                check_teacher_cv(tm['inner_cv'],tm['selected'],tm['source_ids'])
                counts['teacher_cv_records']+=len(tm['inner_cv'])
                pred=t['predictions'];ex=features['target_'+profile['signature']];sx=features['source_'+profile['signature']]
                for variant in range(4):
                    mean=t[f'variant{variant}_mean'];scale=t[f'variant{variant}_scale'];weight=t[f'variant{variant}_weight']
                    check('teacher_train_only_mean',mean,sx[ix].mean(0));std=sx[ix].std(0);std=np.where(std>0,std,1)
                    check('teacher_train_only_scale',scale,std)
                    raw=expit(((ex-mean)/scale)@weight[:-1]+weight[-1]);check('teacher_prediction_replay',raw,pred[variant])
                    for pool,y,pp in [('A',ya.numpy(),raw[:len(ya)]),('B',yb.numpy(),raw[len(ya):])]:
                        row=teacher_metrics[token,name,variant,pool];clipped=np.clip(pp,1e-6,1-1e-6)
                        values=dict(n=len(y),n0=int((y==0).sum()),n1=int((y==1).sum()),
                            nll=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
                            brier=float(np.mean((clipped-y)**2)),residual_mse=float(np.mean((pp-y)**2)),
                            outside=float(np.mean((pp<0)|(pp>1))),
                            ba=float(np.mean([np.mean((pp[y==k]>=.5)==k) for k in [0,1]])) if len(np.unique(y))==2 else None)
                        for metric,value in values.items():
                            if value is None:assert row[metric] is None
                            else:check('teacher_direct_metrics',row[metric],value)
                        counts['teacher_prediction_rows']+=1
                    counts['teacher_selected_states']+=1
                    counts['teacher_refit_not_converged']+=not tm['selected'][variant]['optimization']['converged']
                    counts['teacher_inner_not_converged']+=sum(not r['optimization']['converged'] for r in tm['inner_cv'] if r['variant']==variant)
                qdiff=float(q[1,0]-q[0,0]);nt=len(ya);target_p=features['target_A_probability']
                tg=lambda p:-(geo['target_A'][:,0,:].T@((p[:nt]-target_p)*qdiff)/nt)
                full=tg(pred[0]);base=tg(np.full(len(pred[0]),qy[ix].mean()));conditional=[tg(pred[j]) for j in [1,2,3]];pipeline=conditional
                if name=='TEACHER_LOCAL':
                    pipeline=[]
                    for variant,item in enumerate(tm['whole_pipeline_null'],1):
                        check_teacher_cv(item['inner_cv'],item['selected'],item['source_ids']);counts['teacher_cv_records']+=len(item['inner_cv'])
                        w=t[f'pipeline{variant}_weight'];mean=t[f'pipeline{variant}_mean'];scale=t[f'pipeline{variant}_scale']
                        jx=np.array([ids['query'].index(i) for i in item['source_ids']])
                        check('pipeline_teacher_train_mean',mean,sx[jx].mean(0));std=sx[jx].std(0)
                        check('pipeline_teacher_train_scale',scale,np.where(std>0,std,1))
                        pp=expit(((ex-mean)/scale)@w[:-1]+w[-1]);check('pipeline_teacher_prediction_replay',pp,t[f'pipeline{variant}_predictions'])
                        pipeline.append(tg(pp));counts['teacher_selected_states']+=1
                        counts['teacher_refit_not_converged']+=not item['selected'][0]['optimization']['converged']
                        counts['teacher_inner_not_converged']+=sum(not r['optimization']['converged'] for r in item['inner_cv'])
                expected=replay_controls(full,base,conditional,pipeline,rank,ids['block_seed'])
                expected['FULL_UNPROJECTED']=v.T@t['full_gradient'];check('teacher_unprojected_projection',t['full_gradient'],v@full)
            for arm,coefficient in expected.items():
                stored=arrays[name+'::'+arm];check('coefficient_control_replay',stored,coefficient)
                expected_all[name,arm]=coefficient
                r=gradients[token,name,arm];assert r['git_commit']==manifest['git_commit']
                ga,gb,gs0=observed['A'],observed['B'],arrays['source_gradient'];gn=np.linalg.norm(stored);an=np.linalg.norm(ga);bn=np.linalg.norm(gb);sn=np.linalg.norm(gs0)
                ca=stored@(v.T@ga);cb=stored@(v.T@gb);cs=stored@(v.T@gs0);bproject=np.linalg.norm(v.T@gb)
                measures=dict(gradient_norm=gn,oracle_A_norm=an,oracle_B_norm=bn,source_norm=sn,
                    c_A=ca/(gn*an) if min(gn,an)>1e-12 else None,c_B=cb/(gn*bn) if min(gn,bn)>1e-12 else None,
                    e_A=np.sqrt(max(0,gn**2+an**2-2*ca))/an if an>1e-12 else None,a_A=ca/an**2 if an>1e-12 else None,
                    norm_ratio=gn/an if an>1e-12 else None,weighted_ratio=cfg['beta']*gn/sn if sn>1e-12 else None,
                    cosine_source=cs/(gn*sn) if min(gn,sn)>1e-12 else None,
                    c_B_projected=cb/(gn*bproject) if min(gn,bproject)>1e-12 else None,
                    B_space_outside_energy_fraction=max(0,1-bproject**2/bn**2) if bn>1e-12 else None)
                for metric,value in measures.items():
                    if value is None:assert r[metric] is None
                    else:check('gradient_metric_replay',r[metric],value)
                assert r['control_valid']==profile['validity'][arm]
                counts['gradient_rows']+=1
            counts['profiles']+=1
        def ce(margin):return float(np.mean(np.logaddexp(0,margin)-yb.numpy()*margin))
        if meta['performed_steps']:
            initial=ce(preds['INITIAL']);source=ce(preds['SOURCE'])
            selected_steps=[r for (t,p,a),r in steps.items() if t==token]
            for r in selected_steps:
                name,arm=r['profile'],r['arm']
                margin=preds['SOURCE'] if name=='SOURCE' else oracle_preds[name] if arm=='ORACLE_M' else preds[name+'::'+arm]
                loss=ce(margin);balanced=np.mean([np.mean((margin[yb.numpy()==label]>=0)==label) for label in [0,1]])*100
                check('step_metric_replay',r['CE_B'],loss);check('step_metric_replay',r['R_Q'],(source-loss)/initial)
                check('step_metric_replay',r['base_CE_B'],initial);check('step_metric_replay',r['source_CE_B'],source);check('step_metric_replay',r['BA_B'],balanced)
                counts['step_rows']+=1
            # All FULL and BASE transitions independently executed on an ordinary module copy.
            for profile in meta['profiles']:
                name=profile['profile']
                for arm in ['FULL','BASE']:
                    gradient=arrays['source_gradient']+cfg['beta']*(v@expected_all[name,arm])
                    temporary=copy.deepcopy(model);offset=0
                    with torch.no_grad():
                        for p in temporary.parameters():
                            if not p.requires_grad:continue
                            count=p.numel();p.add_(torch.from_numpy(gradient[offset:offset+count].reshape(p.shape)),alpha=-cfg['eta']);offset+=count
                        logits=torch.cat([temporary(b) for b in batches]);margin=(logits[:,1]-logits[:,0]).numpy()
                    check('ordinary_module_SGD_replay',margin,preds[name+'::'+arm]);counts['ordinary_SGD_forwards']+=1
        assert model_hash(model)==before;counts['blocks']+=1
        if counts['blocks']%8==0:print(json.dumps(dict(event=phase+'_audit',seed=seed,blocks=counts['blocks'])),flush=True)
    assert counts['gradient_rows']==len(gradients) and counts['step_rows']==len(steps)
    assert counts['teacher_prediction_rows']==len(teacher_metrics)
    result=dict(passed=True,seed=seed,phase=phase,counts=dict(counts),max_relative_errors=dict(errors),cpu_seconds=time.process_time()-started,
                final_labels_read=0,scope='All coefficients/CV choices/metrics; ordinary-module FULL and BASE steps, not every null parameter step')
    write(audit/f'{seed}.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--stage',required=True);p.add_argument('--phase',required=True,choices=['A2b','A6']);a=p.parse_args()
    stage=Path(a.stage);root=stage/a.phase;assert read(root/'done.json')['completed']
    assert not (root/'audit.json').exists()
    with ProcessPoolExecutor(max_workers=5) as pool:
        results=list(pool.map(audit_seed,[a.repo]*5,[a.stage]*5,[a.phase]*5,list(range(3405,3410))))
    counts=defaultdict(int);errors=defaultdict(float)
    for r in results:
        for k,v in r['counts'].items():counts[k]+=v
        for k,v in r['max_relative_errors'].items():errors[k]=max(errors[k],v)
    expected={'A2b':dict(blocks=120,profiles=6960,gradient_rows=167040,step_rows=87060),
              'A6':dict(blocks=360,profiles=3600,gradient_rows=85320,step_rows=44640)}[a.phase]
    for k,v in expected.items():assert counts[k]==v,(k,counts[k],v)
    write(root/'audit.json',dict(passed=True,phase=a.phase,counts=dict(counts),max_relative_errors=dict(errors),
        cpu_seconds=sum(r['cpu_seconds'] for r in results),final_labels_read=0,
        auditor_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),workers=results))
    print(json.dumps(dict(passed=True,counts=dict(counts),max_relative_errors=dict(errors))),flush=True)


if __name__=='__main__':main()
