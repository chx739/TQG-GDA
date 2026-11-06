"""Conditional 50-step matrix. All predictions are sealed before dev evaluation."""
import json
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from tqg_gda.adapt import Adapter
from tqg_gda.data import load_task
from tqg_gda.metrics import classification_metrics, evaluate_source
from tqg_gda.p0_diagnostics import DevelopmentOracle, append_json, model_hash
from tqg_gda.predict import load_predictor, model_checkpoint
from tqg_gda.pretrain import check_finite
from tqg_gda.utils import atomic_save, sha256, write_json, fingerprint
from tqg_gda.v2.query import ParameterSnapshot
from .diagnostics import input_batches, logits_at, permuted_labels
from .decisions import average, describe


class PermutedTask:
    def __init__(self,task,mapping):
        self.task,self.mapping=task,mapping

    def sample(self,role,*args,**kwargs):
        batch,ids=self.task.sample(role,*args,**kwargs)
        if role=='query':
            batch.y=torch.tensor([self.mapping[i] for i in ids],dtype=torch.long,device=batch.x.device)
        return batch,ids


def query_ce_step(adapter,task):
    model,cfg=adapter.model,adapter.config
    model.eval()
    p=next(model.parameters())
    fit,fi=task.sample('fit',cfg.adapt.fit_batch_size,adapter.generators['fit'],p.device,p.dtype)
    qry,qi=task.sample('query',cfg.adapt.query_batch_size,adapter.generators['query'],p.device,p.dtype)
    yf,fit.y=fit.y,None
    yq,qry.y=qry.y,None
    adapter.optimizer.zero_grad(set_to_none=True)
    sf,sq=F.cross_entropy(model(fit),yf),F.cross_entropy(model(qry),yq)
    loss=sf+cfg.adapt.feedback_weight*sq
    loss.backward()
    check_finite(model,loss)
    adapter.optimizer.step()
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise FloatingPointError('Nonfinite query-CE control state')
    adapter.steps+=1
    return dict(step=adapter.steps,source_ce=float(sf.detach()),query_ce=float(sq.detach()),
                total_loss=float(loss.detach()),batches=dict(fit=list(fi),query=list(qi)))


def run_m2(cache,output,manifest,source_rows,versions):
    from .run import ROOT,config_for,write_csv
    if not versions or any(v not in ('v1','v2') for v in versions):
        raise ValueError('M2 needs registered passing versions')
    output=Path(output)
    root=output/'M2'
    root.mkdir(exist_ok=True)
    tasks={d:load_task(config_for(3405,d,cache,output)) for d in (1,3)}
    records=[]
    # Source-independent baselines are trained once per seed.
    for source in source_rows:
        seed=source['seed']
        arms=[('FIT-CE','source_only',None,False),('FIT+QRY-CE','source_only',None,False)]
        arms += [(v+'-'+control,v,d,control=='PERM') for v in versions for d in (1,3) for control in ('TRUE','PERM')]
        for name,version,domain,permuted in arms:
            cfg=config_for(seed,domain or 1,cache,output,'v1')
            cfg=replace(cfg,version=version)
            folder=root/f'seed{seed}_{name}_domain{domain or "shared"}'
            if folder.exists():
                raise ValueError('M2 resume is not implemented; preserve incomplete trajectory')
            folder.mkdir()
            model,_=load_predictor(source['checkpoint'])
            before_buffers={n:v.clone() for n,v in model.named_buffers()}
            task=tasks[domain or 1]
            training_task=task
            if permuted:
                _,mapping=permuted_labels(task.query_ids,torch.tensor([task.labels[i] for i in task.query_ids]),seed,1)
                training_task=PermutedTask(task,mapping)
                write_json(folder/'permutation.json',mapping)
            adapter=Adapter(model,cfg)
            started=time.perf_counter()
            for step in range(51):
                if step:
                    row=query_ce_step(adapter,task) if name=='FIT+QRY-CE' else adapter.step(training_task)
                    append_json(folder/'history.jsonl',row)
                for key,value in model.named_buffers():
                    if not torch.equal(value,before_buffers[key]):
                        raise AssertionError('M2 BN buffer changed')
                if step in (0,10,50):
                    checkpoint=folder/f'step{step}.pt'
                    atomic_save(checkpoint,model_checkpoint(model,task,cfg,stage='minimal_M2',arm=name,
                        step=step,source_checkpoint_sha256=source['checkpoint_sha256']))
                    for d in ((domain,) if domain else (1,3)):
                        target=tasks[d]
                        snap=ParameterSnapshot(model)
                        logits=logits_at(snap,snap.flat,input_batches(target,target.target_ids))
                        prediction=folder/f'step{step}_domain{d}_logits.pt'
                        atomic_save(prediction,dict(graph_ids=list(target.target_ids),logits=logits,
                            checkpoint_sha256=sha256(checkpoint)))
                        records.append(dict(seed=seed,domain=d,arm=name,version=version,step=step,
                            checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                            prediction=str(prediction),prediction_sha256=sha256(prediction),
                            source_validation=evaluate_source(model,task)))
            write_json(folder/'completed.json',dict(status='completed',seconds=time.perf_counter()-started,
                source_checkpoint_sha256=source['checkpoint_sha256'],steps=50,arm=name))
            print(json.dumps(dict(event='M2_trajectory_complete',seed=seed,arm=name,domain=domain)),flush=True)
    write_json(root/'sealed_predictions.json',records)
    seal=fingerprint(records)
    metrics=[]
    for domain,task in tasks.items():
        role=manifest['targets'][str(domain)]
        dev=np.load(ROOT/role['development']['path'],allow_pickle=False).tolist()
        final=np.load(ROOT/role['final']['path'],allow_pickle=False).tolist()
        oracle=DevelopmentOracle(task,dev,final,output/'label_access.jsonl',enabled=True)
        oracle.seal('M2_all_arms',dev,theta_hash='see sealed checkpoint hashes',readout_hash=seal)
        labels=oracle.labels(dev,token='M2_all_arms',purpose='M2 all trajectories sealed; development metrics')
        for row in records:
            if row['domain']!=domain:
                continue
            payload=torch.load(row['prediction'],map_location='cpu',weights_only=False)
            if sha256(row['prediction'])!=row['prediction_sha256']:
                raise ValueError('Sealed prediction changed')
            positions={i:j for j,i in enumerate(payload['graph_ids'])}
            logits=payload['logits'][[positions[i] for i in dev]]
            metrics.append(dict(**row,**classification_metrics(logits,labels)))
    write_json(root/'development_metrics.json',metrics)
    write_csv(root/'development_metrics.csv',metrics)
    decisions=[]
    for version in versions:
        for domain in (1,3):
            subset=[m for m in metrics if m['domain']==domain]
            pairs=[]
            for seed in (3405,3406,3407):
                rows=[m for m in subset if m['seed']==seed]
                arm={m['arm']:m for m in rows if m['step']==50}
                frozen=next(m for m in rows if m['step']==0)
                true=arm[version+'-TRUE']
                controls=dict(FROZEN=frozen,FIT=arm['FIT-CE'],QRY=arm['FIT+QRY-CE'],PERM=arm[version+'-PERM'])
                for name,control in controls.items():
                    pairs.append(dict(seed=seed,control=name,ce_gain=control['ce']-true['ce'],
                        ba_gain_pp=100*(true['balanced_accuracy']-control['balanced_accuracy'])))
            comparisons={name:dict(ce=describe([p['ce_gain'] for p in pairs if p['control']==name]),
                ba=describe([p['ba_gain_pp'] for p in pairs if p['control']==name]),
                ce_positive_seeds=sum(p['ce_gain']>0 for p in pairs if p['control']==name))
                for name in ('FROZEN','FIT','QRY','PERM')}
            numerical=[]
            for folder in root.glob(f'*{version}-*_domain{domain}'):
                entries=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
                qs=[e['query'] for e in entries]
                solves=[q['calibration'] for q in qs if 'calibration' in q]
                numerical.append(sum('numerical' in q['status'] for q in qs)/50<=.1 and
                    (version=='v1' or bool(solves) and sum(not c['converged'] for c in solves)/len(solves)<=.1))
            passed=all(c['ce']['mean']>0 and c['ce_positive_seeds']>=2 for c in comparisons.values())
            passed=passed and comparisons['FIT']['ba']['mean']>=.5 and all(
                comparisons[n]['ba']['mean']>=0 for n in ('FROZEN','QRY','PERM')) and all(numerical)
            decisions.append(dict(version=version,domain=domain,expand=passed,comparisons=comparisons,
                                  numerical_passed=all(numerical)))
    write_json(root/'decision.json',decisions)
    write_json(output/'M2_status.json',dict(status='completed',versions=versions,trajectories=6+12*len(versions),
        updates=50*(6+12*len(versions)),final_labels_read=False))
