"""Sealed two-phase diagnostics. Source phase never requests target labels.

--phase source first constructs label-free responses, scores source labels and seals
all cached target predictions. --phase target requires both completed training and
that source seal, then reads only development target labels. Neither phase trains.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F
from torch_geometric.data.separate import separate
import yaml

from tqg_gda.data import GraphDomainTask
from tqg_gda.diagnostics import heldout_response_features
from tqg_gda.feedback import ProbeConfig, Snapshot, build_feedback, fit_residual
from tqg_gda.training import load_model
from tqg_gda.utils import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def seal(path, payload):
    path = Path(path)
    if path.exists():
        if read(path) != payload:
            raise ValueError('Changed sealed evidence: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')


@torch.no_grad()
def prediction_metrics(logits, labels, prediction):
    residual = F.one_hot(labels, logits.shape[1]).to(logits) - logits.softmax(-1)
    error = (residual-prediction).square().sum(-1).mean()
    zero = residual.square().sum(-1).mean()
    ce = F.cross_entropy(logits, labels)
    return dict(residual_mse=float(error), zero_prediction_mse=float(zero),
        mse_improvement=float(zero-error), prediction_rms=float(prediction.square().sum(-1).mean().sqrt()),
        direct_logit_ce_decrease_alpha_0_1=float(ce-F.cross_entropy(logits+.1*prediction, labels)),
        direct_logit_ce_decrease_alpha_1_0=float(ce-F.cross_entropy(logits+prediction, labels)))


def aggregate(rows, grouping):
    """Collapse blocks/label permutations within each model seed before SD."""
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in grouping)
        grouped.setdefault(key, {}).setdefault(row['seed'], []).append(row['metrics'])
    out = []
    for key, seeds in sorted(grouped.items()):
        per_seed = []
        for seed, values in sorted(seeds.items()):
            metric_names = sorted(set.intersection(*(set(v) for v in values)))
            means = {k: statistics.mean(v[k] for v in values if v[k] is not None)
                     if any(v[k] is not None for v in values) else None for k in metric_names}
            per_seed.append(dict(seed=seed, observations=len(values), **means))
        names = [k for k in per_seed[0] if k not in ('seed', 'observations')]
        summary = {}
        for name in names:
            values = [r[name] for r in per_seed if r[name] is not None]
            summary[name] = dict(mean=statistics.mean(values) if values else None,
                seed_std=statistics.stdev(values) if len(values)>1 else None,
                model_seeds=len(values))
        out.append(dict(zip(grouping,key), per_seed=per_seed, metrics=summary))
    return out


def parameter_metrics(model, query, held, labels, predictions):
    """Same full-parameter space for both gradients; tentative steps are stateless."""
    named = tuple(model.named_parameters()); params = tuple(p for _,p in named)
    encoder_indices = [i for i,(n,_) in enumerate(named) if n.startswith('encoder.')]
    def grads(loss, retain=False):
        values = torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
        return tuple(torch.zeros_like(p) if v is None else v.detach() for p,v in zip(params,values))
    def norm(values):
        value = math.sqrt(sum(float(v.double().square().sum()) for v in values))
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite diagnostic parameter or gradient norm')
        return value
    oracle_loss = F.cross_entropy(model(held),labels)
    if not torch.isfinite(oracle_loss):
        raise FloatingPointError('Nonfinite target diagnostic CE')
    oracle = grads(oracle_loss); oracle_norm = norm(oracle)
    oracle_encoder = norm([oracle[i] for i in encoder_indices])
    baseline = float(oracle_loss.detach()); scale = max(1.,norm(params))
    saved = {n:p.detach().clone() for n,p in named}
    buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    query_logits = model(query); base = query_logits.detach()
    results = []
    for item in predictions:
        correction = item['target_query_prediction']
        if not torch.isfinite(correction).all():
            raise FloatingPointError('Nonfinite cached correction')
        gradient = grads(.5*(query_logits-(base+correction)).square().sum(-1).mean(),retain=True)
        gn = norm(gradient); ge = norm([gradient[i] for i in encoder_indices])
        def cosine(indices, denominator):
            if denominator[1] <= 1e-12:
                return None
            if denominator[0] <= 1e-12:
                return 0.
            return max(-1.,min(1.,sum(float((gradient[i]*oracle[i]).sum()) for i in indices)
                /(denominator[0]*denominator[1])))
        metrics = dict(gradient_cosine_full=cosine(range(len(params)),(gn,oracle_norm)),
            gradient_cosine_encoder=cosine(encoder_indices,(ge,oracle_encoder)),
            feedback_gradient_norm=gn,oracle_gradient_norm=oracle_norm,
            zero_feedback_fraction=float(gn<=1e-12))
        for fraction,label in ((1e-4,'0_0001'),(1e-3,'0_001')):
            if gn<=1e-12:
                after = baseline
            else:
                candidate = {n:saved[n]-(fraction*scale/gn)*g for (n,_),g in zip(named,gradient)}
                with torch.no_grad():
                    z = functional_call(model,(candidate,{n:b.clone() for n,b in buffers.items()}),(held,),strict=True)
                    after = float(F.cross_entropy(z,labels))
                    if not math.isfinite(after):
                        raise FloatingPointError('Nonfinite tentative-step CE')
            metrics['parameter_ce_decrease_fraction_'+label] = baseline-after
        results.append(metrics)
    if any(not torch.equal(saved[n],p) for n,p in named):
        raise ValueError('Diagnostic mutated model parameters')
    return results


class Diagnostics:
    def __init__(self, execution_root):
        self.root = Path(execution_root).resolve(); self.run = self.root.parent
        self.plan_path = self.run/'stage2_method/experiment_plan.yaml'
        self.plan = yaml.safe_load(self.plan_path.read_text())
        self.data = self.plan['datasets'][0]
        self.roles_path = Path(self.data['target_split_manifest']['path'])
        if sha256(self.roles_path)!=self.data['target_split_manifest']['sha256']:
            raise ValueError('Target split manifest changed')
        self.roles = read(self.roles_path)
        self.manifest_path = self.root/'training_manifest.json'
        self.manifest = read(self.manifest_path)
        self.require_training_seal()
        self.out = self.root/'diagnostics'; self.out.mkdir(exist_ok=True)
        self.limit = self.root/'compute_ledger.json'
        torch.set_num_threads(1)
        self.contract = dict(training_manifest_sha256=sha256(self.manifest_path),
            plan_sha256=sha256(self.plan_path), script_sha256=sha256(__file__),
            target_roles_sha256=sha256(self.roles_path),
            training_seal_sha256=sha256(self.training_seal_path))

    def require_training_seal(self):
        self.training_seal_path = self.root/'training_seal.json'
        training_seal = read(self.training_seal_path)
        if not training_seal.get('all_70_training_jobs_completed',False):
            raise PermissionError('All pilot training must be sealed before diagnostics')
        if training_seal.get('manifest_sha256') != sha256(self.manifest_path):
            raise ValueError('Training manifest differs from seal')
        if len(training_seal.get('jobs',[])) != 70:
            raise PermissionError('Training seal must contain all 70 jobs')
        for job in training_seal['jobs']:
            for output in job['outputs'].values():
                if sha256(output['path']) != output['sha256']:
                    raise ValueError('Training output changed '+output['path'])
        records = self.manifest.get('records',[])
        if len(records)!=60:
            raise PermissionError('Training seal must register all sixty adaptation runs')
        for item in records:
            for key in ('final_checkpoint','warmup_checkpoint'):
                path = Path(item[key])
                if not path.is_file():
                    raise PermissionError('Missing sealed checkpoint '+str(path))
                digest = item.get(key+'_sha256')
                if digest and sha256(path)!=digest:
                    raise ValueError('Sealed checkpoint changed '+str(path))

    def check_budget(self):
        # Runner ledger is authoritative. No training or unbounded retry here.
        if self.limit.exists():
            ledger = read(self.limit)
            deadline = ledger.get('deadline_unix')
            if deadline is not None and time.time()>=deadline:
                raise TimeoutError('Approved CPU wall-clock cap reached')

    def states(self):
        records = [r for r in self.manifest['records'] if r['arm']=='original_TQG_target_probes']
        if len(records)!=10:
            raise ValueError('Need 2 tasks x 5 original TQG model seeds')
        for record in sorted(records,key=lambda r:(r['task'],r['seed'])):
            for state,key in [('source_warmup','warmup_checkpoint'),('fixed_final_original_TQG','final_checkpoint')]:
                yield record,state,Path(record[key])

    def task(self, record, config):
        domain = record['task'].split('_to_')[-1]; role = self.roles['domains'][domain]
        for item in (role['development'],role['final']):
            if sha256(item['path'])!=item['sha256']:
                raise ValueError('Target split IDs changed')
        if sha256(role['index_path'])!=role['index_sha256']:
            raise ValueError('Target domain index changed')
        dev = tuple(map(int,np.load(role['development']['path'],allow_pickle=False)))
        final = set(map(int,np.load(role['final']['path'],allow_pickle=False)))
        if set(dev)&final:
            raise ValueError('Target development/final overlap')
        task = GraphDomainTask(Path(self.data['root'])/'PROTEINS/processed/data.pt',
            Path(self.data['root'])/self.data['source_protocol'],target_index=role['index_path'],
            target_policy='transductive-unlabeled',fit_fraction=config['split']['fit_fraction'],
            split_seed=config['split']['seed'],use_node_attributes=False,preprocess_pool='fit')
        return task,dev

    def source(self):
        all_rows=[]; artifacts=[]
        for record,state,checkpoint in self.states():
            model,saved=load_model(checkpoint); task,dev=self.task(record,saved['config'])
            snapshot=Snapshot(model)
            folder=self.out/record['task']/str(record['seed'])/state; folder.mkdir(parents=True,exist_ok=True)
            metrics_path=folder/'source_metrics.jsonl'
            if metrics_path.exists():
                raise FileExistsError('Do not overwrite partial diagnostics: '+str(metrics_path))
            checkpoint_sha=sha256(checkpoint)
            model_parameters={n:p.detach().clone() for n,p in model.named_parameters()}
            with metrics_path.open('x') as stream:
                for block in range(20):
                    self.check_budget()
                    sampler=torch.Generator().manual_seed(700000+block)
                    order=torch.randperm(len(task.query_ids),generator=sampler).tolist()
                    fitting=tuple(task.query_ids[i] for i in order[:32]); held_ids=tuple(task.query_ids[i] for i in order[32:48])
                    order=torch.randperm(len(dev),generator=sampler).tolist()
                    query_ids=tuple(dev[i] for i in order[:16]); target_held_ids=tuple(dev[i] for i in order[16:48])
                    held=task.batch(held_ids,'query'); tq=task.batch(query_ids,'target'); th=task.batch(target_held_ids,'target')
                    with torch.no_grad():
                        held_z=model(held).detach(); query_z=model(tq).detach(); target_held_z=model(th).detach()
                    cached=[]
                    for n in (16,32):
                        source=task.batch(fitting[:n],'query')
                        with torch.no_grad(): source_z=model(source).detach()
                        residual=F.one_hot(source.y,source_z.shape[1]).to(source_z)-source_z.softmax(-1)
                        for direction in ('target','random','source'):
                            probe=ProbeConfig(direction=direction)
                            fb=build_feedback(model,source,source.y,tq,probe,
                                torch.Generator().manual_seed(900000+block))
                            hs=heldout_response_features(snapshot,held,fb,probe)
                            ht=heldout_response_features(snapshot,th,fb,probe)
                            conditions=[('true_source_labels',None)]+[('within_batch_label_permutation',p) for p in (113,227,349)]
                            for condition,perm in conditions:
                                labels=source.y
                                if perm is not None:
                                    permutation=torch.randperm(n,generator=torch.Generator().manual_seed(perm+block))
                                    labels=labels[permutation]
                                rr=F.one_hot(labels,source_z.shape[1]).to(source_z)-source_z.softmax(-1)
                                a=fit_residual(fb.source_features,rr,.01)
                                item=dict(task=record['task'],seed=record['seed'],state=state,block=block,
                                    direction=direction,query_count=n,label_condition=condition,permutation_seed=perm,effective_permutation_seed=perm+block if perm is not None else None,
                                    coefficient_norm=float(a.norm()),invalid_direction_fraction=1-len(fb.directions)/4)
                                for domain,z,y,phi in [('source_fitted',source_z,labels,fb.source_features),('source_heldout',held_z,held.y,hs)]:
                                    row=dict(item,domain=domain,metrics=prediction_metrics(z,y,phi@a))
                                    stream.write(json.dumps(row,allow_nan=False)+'\n');all_rows.append(row)
                                cached.append(dict(item,target_query_prediction=fb.target_features@a,target_heldout_prediction=ht@a))
                        for control in ('zero','source_query_mean'):
                            b=residual.mean(0) if control=='source_query_mean' else residual.new_zeros(residual.shape[1])
                            item=dict(task=record['task'],seed=record['seed'],state=state,block=block,
                                direction=control,query_count=n,label_condition='true_source_labels',permutation_seed=None,
                                coefficient_norm=0.,invalid_direction_fraction=0.)
                            for domain,z,y in [('source_fitted',source_z,source.y),('source_heldout',held_z,held.y)]:
                                row=dict(item,domain=domain,metrics=prediction_metrics(z,y,b.expand_as(z)))
                                stream.write(json.dumps(row,allow_nan=False)+'\n');all_rows.append(row)
                            cached.append(dict(item,target_query_prediction=b.expand_as(query_z),target_heldout_prediction=b.expand_as(target_held_z)))
                    path=folder/f'block_{block:02d}.pt'
                    torch.save(dict(contract=self.contract,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
                        source_fitting_ids=fitting,source_heldout_ids=held_ids,target_query_ids=query_ids,target_heldout_ids=target_held_ids,
                        target_query_logits=query_z,target_heldout_logits=target_held_z,predictions=cached),path)
                    artifacts.append(dict(path=str(path),sha256=sha256(path)))
            if any(not torch.equal(p,model_parameters[n]) for n,p in model.named_parameters()):
                raise ValueError('Source diagnostics mutated model')
            artifacts.append(dict(path=str(metrics_path),sha256=sha256(metrics_path)))
            print(json.dumps(dict(event='source_diagnostic_state_completed',task=record['task'],seed=record['seed'],state=state)),flush=True)
        groups=aggregate(all_rows,['task','state','direction','query_count','label_condition','domain'])
        write_json(self.out/'source_summary.json',dict(groups=groups,aggregation='blocks and permutation replicates within model seed first'))
        artifacts.append(dict(path=str(self.out/'source_summary.json'),sha256=sha256(self.out/'source_summary.json')))
        seal(self.root/'source_evidence.json',dict(contract=self.contract,complete=True,source_row_count=len(all_rows),
            artifacts=artifacts,target_labels_read=False,final_target_labels_read=False,
            source_mse_is_not_a_necessary_target_gain_gate=True))

    def target(self):
        source_path=self.root/'source_evidence.json'; evidence=read(source_path)
        if not evidence.get('complete') or evidence['contract']!=self.contract:
            raise PermissionError('Missing or incompatible sealed source evidence')
        for artifact in evidence['artifacts']:
            if sha256(artifact['path'])!=artifact['sha256']:
                raise ValueError('Sealed source artifact changed '+artifact['path'])
        all_rows=[]; artifacts=[]
        for record,state,checkpoint in self.states():
            model,saved=load_model(checkpoint);task,dev=self.task(record,saved['config'])
            folder=self.out/record['task']/str(record['seed'])/state
            path=folder/'target_metrics.jsonl'
            with path.open('x') as stream:
                for block in range(20):
                    self.check_budget(); cache=torch.load(folder/f'block_{block:02d}.pt',map_location='cpu',weights_only=False)
                    if cache['contract']!=self.contract or sha256(checkpoint)!=cache['checkpoint_sha256']:
                        raise ValueError('Cached prediction contract mismatch')
                    if set(cache['target_query_ids']) & set(cache['target_heldout_ids']):
                        raise ValueError('Diagnostic target query and heldout overlap')
                    query=task.batch(cache['target_query_ids'],'target');held=task.batch(cache['target_heldout_ids'],'target')
                    def labels(ids):
                        if not set(ids)<=set(dev):raise PermissionError('Final target labels forbidden')
                        values=[]
                        for i in ids:
                            graph=separate(task._raw.__class__,task._raw,i,task.slices,decrement=False)
                            if graph.y is None or graph.y.numel()!=1 or int(graph.y.item()) not in task.class_map:
                                raise ValueError('Invalid closed-set target label')
                            values.append(task.class_map[int(graph.y.item())])
                        return torch.tensor(values,dtype=torch.long)
                    query_labels=labels(cache['target_query_ids']);held_labels=labels(cache['target_heldout_ids'])
                    gradients=parameter_metrics(model,query,held,held_labels,cache['predictions'])
                    for item,gradient in zip(cache['predictions'],gradients):
                        metadata={k:v for k,v in item.items() if not k.endswith('_prediction')}
                        for domain,z,y,pred in [('target_query',cache['target_query_logits'],query_labels,item['target_query_prediction']),
                                               ('target_heldout',cache['target_heldout_logits'],held_labels,item['target_heldout_prediction'])]:
                            metrics=prediction_metrics(z,y,pred)
                            if domain=='target_heldout':metrics.update(gradient)
                            row=dict(metadata,domain=domain,metrics=metrics)
                            stream.write(json.dumps(row,allow_nan=False)+'\n');all_rows.append(row)
            artifacts.append(dict(path=str(path),sha256=sha256(path)))
            print(json.dumps(dict(event='target_diagnostic_state_completed',task=record['task'],seed=record['seed'],state=state)),flush=True)
        groups=aggregate(all_rows,['task','state','direction','query_count','label_condition','domain'])
        write_json(self.out/'target_summary.json',dict(groups=groups,aggregation='blocks and permutation replicates within model seed first'))
        seal(self.out/'target_evidence.json',dict(complete=True,contract=self.contract,source_evidence_sha256=sha256(source_path),
            target_row_count=len(all_rows),artifacts=artifacts,final_target_labels_read=False,
            target_labels_used_to_train_or_select=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['source','target'],required=True)
    parser.add_argument('--execution-root',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args();obj=Diagnostics(args.execution_root)
    getattr(obj,args.phase)()


if __name__=='__main__':main()
