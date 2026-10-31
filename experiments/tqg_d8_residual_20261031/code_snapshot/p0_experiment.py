"""P0 entrypoint: prepare, D0, pilot, then five fixed-seed diagnostic workers."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
import csv
from dataclasses import dataclass
import importlib.metadata as md
import json
import multiprocessing
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from tqg_gda.calibration_v2 import CalibrationConfig, solve_calibration
from tqg_gda.data import GraphDomainTask
from tqg_gda.feedback_v2 import (ParameterSnapshot, V2Config, build_gradient_feedback,
    parameter_feedback_loss, signature_kernels, target_space)
from tqg_gda.models import center_logits
from tqg_gda.p0_diagnostics import (DevelopmentOracle, append_json, controls,
    development_pools, evaluate_flat, fixed_isotropic, gradient_metrics, gradient_of_ce,
    label_free, model_hash, one_step_arms, sample_ids, tensor_hash, unlabeled_geometry)
from tqg_gda.training import load_model
from tqg_gda.utils import sha256, seed_everything


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    temp.replace(path)


@dataclass(frozen=True)
class ExperimentConfig:
    repo: str
    stage: str
    config: str


class ResearchExperiment:
    def __init__(self, config):
        self.config = config
        self.repo, self.stage = Path(config.repo).resolve(), Path(config.stage).resolve()
        self.code = Path(__file__).resolve().parent
        self.settings = read(config.config)
        self.plan_dir = self.stage.parent / 'stage2_method'
        self.plan = read(self.plan_dir / 'experiment_plan.yaml')
        self.plan_sha = sha256(self.plan_dir / 'experiment_plan.yaml')
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.code, text=True).strip()
        self.archive = self.repo / self.settings['archive']
        self.role_dir = self.archive / 'protocol/data_roles'
        self._tasks = {}
        self.v2 = V2Config(calibration=self.plan['fixed_method']['calibration'])
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), platform.processor())
        self.hardware = f'cpu:{cpu};threads={self.settings["threads_per_worker"]};FP64'
        if self.settings['final_labels_allowed'] or self.settings['device'] != 'cpu':
            raise ValueError('This frozen execution uses CPU and no final labels')

    def task(self, domain):
        if domain not in self._tasks:
            task = GraphDomainTask(self.plan['datasets'][0]['cache'], self.role_dir / 'source_protocol.json',
                target_index=self.role_dir / f'0_to_{domain}/target.npy',
                target_policy='transductive-unlabeled', fit_fraction=.8, split_seed=431)
            if (len(task.fit_ids), len(task.query_ids), len(task.validation_ids), task.input_dim, task.edge_dim) != (779,196,109,14,3):
                raise ValueError('Archived source roles or model dimensions changed')
            self._tasks[domain] = task
        return self._tasks[domain]

    def snapshot_row(self, domain, seed, step):
        rows = read(self.stage / 'snapshot_manifest.json')['references']
        return next(r for r in rows if (r['domain'],r['seed'],r['snapshot_step']) == (domain,seed,step))

    def load(self, domain, seed, step):
        row = self.snapshot_row(domain, seed, step)
        path = self.repo / row['checkpoint']
        if sha256(path) != row['checkpoint_sha256']:
            raise ValueError('Snapshot changed after freezing')
        model, state = load_model(path)
        task = self.task(domain)
        for field, ids in [('source_fit_ids',task.fit_ids),('source_query_ids',task.query_ids),
                           ('source_validation_ids',task.validation_ids)]:
            if state['identity'][field] != list(ids):
                raise ValueError('Checkpoint source identity mismatch: ' + field)
        if state['encoder_spec'] != dict(input_dim=14,edge_dim=3,hidden_dim=32,num_layers=2,readout='mean'):
            raise ValueError('Checkpoint encoder does not match frozen plan')
        model.double().eval()
        return model, row

    def pools(self, domain):
        original = np.load(self.role_dir/f'0_to_{domain}/development.npy', allow_pickle=False)
        final = np.load(self.role_dir/f'0_to_{domain}/final.npy', allow_pickle=False)
        a, b = development_pools(original, domain)
        if set(a)&set(b) or (set(a)|set(b))&set(final.tolist()):
            raise ValueError('Target role overlap')
        return a, tuple(sorted(b)), original.tolist(), final.tolist()

    def batches(self, task, ids):
        n = self.settings['evaluation_microbatch']
        return [task.batch(ids[start:start+n], 'target') for start in range(0, len(ids), n)]

    def base(self, row, block, ids_ref):
        return dict(task=f'0_to_{row["domain"]}', model_seed=row['seed'], seed=row['seed'],
            snapshot_step=row['snapshot_step'], checkpoint_sha256=row['checkpoint_sha256'],
            checkpoint=row['checkpoint'], block_id=block, indices_reference=ids_ref,
            config_name=self.settings['config_name'], git_commit=self.commit,
            execution_commit=self.commit, plan_sha256=self.plan_sha, gpu_hours=0.,
            hardware=self.hardware, event_flags='clean', label_scope='development_oracle',
            beta=self.plan['D2']['beta'], lambda_w=self.v2.calibration['lambda_w'],
            signature=self.v2.signature, candidate_id='V2_ORIGINAL', eta=0.)

    def prepare(self):
        start = time.time()
        for record in read(self.plan_dir/'plan_manifest.json')['files']:
            if sha256(self.plan_dir/record['path']) != record['sha256']:
                raise ValueError('Frozen plan document changed: '+record['path'])
        archived = {f['path']:f['sha256'] for f in read(self.archive/'manifest.json')['files']}
        jobs = list(csv.DictReader((self.archive/'job_index.csv').open()))
        references = []
        for domain in range(1,4):
            for seed in self.plan['seeds']:
                source = next(j for j in jobs if j['phase']=='source_pretraining' and j['arm']=='fit' and int(j['seed'])==seed)
                fit = next(j for j in jobs if j['phase']=='adaptation' and j['arm']=='B_fit'
                           and j['task']==f'0_to_{domain}' and int(j['seed'])==seed)
                for step in self.plan['snapshots']['times']:
                    rel = source['selected_checkpoint'] if not step else str(Path(fit['last_checkpoint']).parent/f'step_{step}.pt')
                    path = self.archive/rel
                    digest = sha256(path)
                    if digest != archived[rel]:
                        raise ValueError('Archived hash mismatch: '+rel)
                    references.append(dict(domain=domain,seed=seed,snapshot_step=step,
                        checkpoint=str(path.relative_to(self.repo)), checkpoint_sha256=digest,
                        historical_training_commit=(source if not step else fit)['training_commit']))
                last_dir = (self.archive/fit['last_checkpoint']).parent
                s200 = torch.load(last_dir/'step_200.pt',map_location='cpu',weights_only=False)['model']
                last = torch.load(last_dir/'last.pt',map_location='cpu',weights_only=False)['model']
                if s200.keys()!=last.keys() or any(not torch.equal(s200[k],last[k]) for k in s200):
                    raise ValueError('B_fit step200 differs from last')
            for role in ('development','final','target'):
                path = self.role_dir/f'0_to_{domain}/{role}.npy'
                if sha256(path)!=archived[str(path.relative_to(self.archive))]:
                    raise ValueError('Archived input role hash mismatch')
            a,b,original,final=self.pools(domain)
            dest=self.stage/'data_roles'/f'0_to_{domain}'
            dest.mkdir(exist_ok=True)
            np.save(dest/'A.npy',np.array(a,dtype=np.int64))
            np.save(dest/'B.npy',np.array(b,dtype=np.int64))
            write(dest/'identity.json',dict(original_development=original,final_ids=final,
                A_sha256=sha256(dest/'A.npy'), B_sha256=sha256(dest/'B.npy'), split_uses_labels=False))
        write(self.stage/'snapshot_manifest.json',dict(references=references,
              step200_matches_last=True, unique_files=len({r['checkpoint'] for r in references})))
        with (self.stage/'job_matrix.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(references[0]));writer.writeheader();writer.writerows(references)
        versions={p:md.version(p) for p in ('torch','torch-geometric','numpy','scipy','matplotlib','tensorboard')}
        write(self.stage/'experiment_manifest.json',dict(git_commit=self.commit,plan_sha256=self.plan_sha,
            settings=self.settings, hardware=self.hardware,python=sys.version,versions=versions,
            gpu_available=torch.cuda.is_available(),no_continuous_adaptations=True,
            target_final_labels_allowed=False,expected_D1_blocks=360,expected_D2_blocks=180,
            expected_D2_temporary_updates=6480,prepared_at=start))
        write(self.stage/'compute_ledger.json',dict(started_at=start,cap_seconds=self.settings['package_wall_cap_seconds'],
            status='preparing',gpu_hours=0.,new_training_jobs=0))
        self.ast_screen()
        print(json.dumps(dict(event='prepared',snapshots=len(references),hardware=self.hardware)),flush=True)

    def ast_screen(self):
        findings=[]
        for path in [self.code/'p0_experiment.py',self.code/'tqg_gda/p0_diagnostics.py',self.code/'p0_summarize.py']:
            if not path.exists():
                continue
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('exec','eval','compile'):
                    findings.append(dict(path=str(path),line=node.lineno,kind='dynamic_execution'))
                if isinstance(node,ast.Import) and any(x.name.split('.')[0] in ('requests','urllib','socket') for x in node.names):
                    findings.append(dict(path=str(path),line=node.lineno,kind='network_import'))
        write(self.stage/'ast_screen.json',dict(passed=not findings,findings=findings,
              local_subprocess_justification='git metadata and local worker coordination only; no network'))
        if findings:
            raise ValueError('AST screen failed')

    def geometry(self, model, source_inputs, target_inputs, domain, seed, step, block):
        """Candidate hook; the original P0 path remains unchanged."""
        return unlabeled_geometry(model, source_inputs, target_inputs, self.v2)

    def worker_entrypoint(self):
        return run_worker

    def worker(self, seed, *, pilot=False):
        seed_everything(seed,self.settings['threads_per_worker'],True)
        name=f'pilot_{seed}' if pilot else str(seed)
        out=self.stage/'logs'/name
        out.mkdir(exist_ok=False)
        cache=self.stage/'geometry_cache'/name
        cache.mkdir(parents=True,exist_ok=False)
        writer=SummaryWriter(str(out/'tensorboard'))
        write(out/'RUN_INFO.json',dict(seed=seed,git_commit=self.commit,plan_sha256=self.plan_sha,
              hardware=self.hardware,config_name=self.settings['config_name'],started_at=time.time(),pilot=pilot))
        started=time.perf_counter()
        count=0
        try:
            for domain in ([1] if pilot else [1,2,3]):
                task=self.task(domain)
                a_pool,b_pool,development,final=self.pools(domain)
                query_ids=tuple(sorted(task.query_ids))
                query=task.batch(query_ids,'query')
                source_labels=query.y.clone()
                source_inputs=label_free(query)
                b_batches=self.batches(task,b_pool)
                oracle=DevelopmentOracle(task,development,final,out/'label_access.jsonl',
                                         enabled=self.settings['allow_development_oracle'])
                for step in ([0] if pilot else self.plan['snapshots']['times']):
                    model,row=self.load(domain,seed,step)
                    original_hash=model_hash(model)
                    g_b=base_b=b_labels=None
                    for block in range(1 if pilot else self.plan['D1']['blocks_per_snapshot']):
                        block_started=time.perf_counter()
                        block_seed=61000000+domain*100000+(seed-3405)*10000+step*10+block
                        a_ids=sample_ids(a_pool,self.plan['roles']['target_query_from_A'],block_seed)
                        fit_ids=sample_ids(task.fit_ids,self.plan['roles']['source_fit_per_block'],block_seed+1000)
                        a_input=task.batch(a_ids,'target')
                        fit=task.batch(fit_ids,'fit')
                        fit_labels=fit.y.clone()
                        fit_input=label_free(fit)
                        token=f'{domain}_{seed}_{step}_{block}'
                        index_path=out/f'{token}_indices.json'
                        write(index_path,dict(A=list(a_ids),B=list(b_pool),fit=list(fit_ids),query=list(query_ids),block_seed=block_seed))
                        t=time.perf_counter()
                        geo=self.geometry(model,source_inputs,a_input,domain,seed,step,block)
                        query_seconds=time.perf_counter()-t
                        cache_path=cache/f'{token}.npz'
                        cache_started=time.perf_counter()
                        np.savez_compressed(cache_path,**{k:v.detach().numpy() for k,v in dict(
                            V=geo.basis,Q=geo.contrasts,W=geo.readout,source_A=geo.source_response,
                            target_A=geo.target_response,source_p=geo.source_probabilities,
                            source_h=geo.source_representation,target_h=geo.target_representation).items()})
                        cache_seconds=time.perf_counter()-cache_started
                        oracle.seal(token,a_ids,theta_hash=tensor_hash(geo.snapshot.flat),readout_hash=tensor_hash(geo.readout))
                        gradients,validity,residual=controls(geo,source_labels,block_seed,
                            self.plan['D1']['permutation_offsets'],self.plan['D1']['random_offsets'],
                            self.settings['epsilon_gradient'])
                        a_labels=oracle.labels(a_ids,token=token,purpose='D1_oracle_A_and_D2_evaluation')
                        t=time.perf_counter()
                        g_a=gradient_of_ce(geo.snapshot,[a_input],a_labels)
                        oracle_A_seconds=time.perf_counter()-t
                        if g_b is None:
                            b_labels=oracle.labels(b_pool,token=token,purpose='D1_oracle_B_and_D2_evaluation_cached_within_snapshot')
                            t=time.perf_counter()
                            g_b=gradient_of_ce(geo.snapshot,b_batches,b_labels)
                            base_b=evaluate_flat(geo.snapshot,geo.snapshot.flat,b_batches,b_labels)
                            append_json(out/'timings.jsonl',dict(token=token,oracle_B_seconds=time.perf_counter()-t))
                        g_source=gradient_of_ce(geo.snapshot,[fit_input],fit_labels)
                        gradients.update({'ORACLE-A':g_a,'SOURCE':g_source})
                        base=self.base(row,block,str(index_path.relative_to(self.stage)))
                        base.update(rank=geo.basis.shape[1],source_query_graphs=len(query_ids),
                                    geometry_cache=str(cache_path.relative_to(self.stage)),
                                    geometry_sha256=sha256(cache_path),
                                    solver_status=geo.statistics['calibration']['termination'],
                                    solver_gap=geo.statistics['calibration']['gap'],
                                    readout_norm=float(geo.readout.norm()),
                                    source_residual_norm=float(residual.norm()),
                                    feedback_gradient_norm=float(gradients['TRUE'].norm()))
                        for arm,gradient in gradients.items():
                            metrics=gradient_metrics(gradient,g_a,g_b,g_source,geo.basis,
                                epsilon=self.settings['epsilon_gradient'],beta=self.plan['D2']['beta'])
                            record=dict(base,experiment_id='D1',arm=arm,null_rep_id=arm.rsplit('-',1)[-1] if arm.startswith(('PERM-','RAND-')) else '',
                                metric_name='c_A',metric_value=metrics['c_A'],metric_direction='max',**metrics)
                            append_json(out/'gradient_blocks.jsonl',record)
                            for key in ('c_A','e_A','c_B','gradient_norm'):
                                if metrics[key] is not None:
                                    writer.add_scalar(f'D1/{domain}/{arm}/{key}',metrics[key],count)
                        append_json(out/'geometry_stats.jsonl',dict(base,statistics=geo.statistics,
                                    source_residual_norm=float(residual.norm()),query_seconds=query_seconds,
                                    oracle_A_seconds=oracle_A_seconds,cache_IO_seconds=cache_seconds))
                        logged_cosine=gradient_metrics(gradients['TRUE'],g_a,g_b,g_source,geo.basis)['c_A']
                        if logged_cosine is not None:
                            writer.add_scalar(f'D1/{domain}/cosine_TRUE',logged_cosine,count)
                        if block in self.plan['D2']['first_blocks']:
                            base_a=evaluate_flat(geo.snapshot,geo.snapshot.flat,[a_input],a_labels)
                            arms,arm_validity=one_step_arms(gradients,g_a,g_source,self.plan['D2']['beta'],self.settings['epsilon_gradient'])
                            def save(arm,eta,flat,direction,valid,step_seconds=0.):
                                eval_started=time.perf_counter()
                                ma=evaluate_flat(geo.snapshot,flat,[a_input],a_labels) if arm!='F' else base_a
                                eval_a_seconds=time.perf_counter()-eval_started
                                eval_started=time.perf_counter()
                                mb=evaluate_flat(geo.snapshot,flat,b_batches,b_labels) if arm!='F' else base_b
                                eval_b_seconds=time.perf_counter()-eval_started
                                first=-eta*float(torch.dot(g_b,direction))
                                append_json(out/'one_step_blocks.jsonl',dict(base,experiment_id='D2',arm=arm,eta=eta,
                                    null_rep_id=arm[-1] if arm.startswith(('S+P-','S+R-')) else '',
                                    metric_name='CE_B',metric_value=mb['ce'],metric_direction='min',
                                    valid_comparison=valid,invalid_reason='' if valid else 'norm_match_unavailable',
                                    CE_A=ma['ce'],CE_B=mb['ce'],BA_A=100*ma['balanced_accuracy'],BA_B=100*mb['balanced_accuracy'],
                                    AUROC_A=ma['auroc'],AUROC_B=mb['auroc'],class_recall_B=mb['class_recall'],
                                    base_CE_A=base_a['ce'],base_CE_B=base_b['ce'],
                                    change_CE_B=mb['ce']-base_b['ce'],first_order_CE_B=first,
                                    first_order_remainder=mb['ce']-base_b['ce']-first,
                                    direction_norm=float(direction.norm()),
                                    flat_copy_step_seconds=step_seconds,eval_A_seconds=eval_a_seconds,eval_B_seconds=eval_b_seconds,
                                    elapsed_seconds=time.perf_counter()-block_started))
                                writer.add_scalar(f'D2/{domain}/{eta}/{arm}/CE_B',mb['ce'],count)
                            save('F',0.,geo.snapshot.flat,torch.zeros_like(g_a),True)
                            for eta in self.plan['D2']['etas']:
                                for arm,direction in arms.items():
                                    # A separate flat tensor for every arm; no optimizer state is reused.
                                    step_started=time.perf_counter()
                                    flat=geo.snapshot.flat-eta*direction
                                    save(arm,eta,flat,direction,arm_validity[arm],time.perf_counter()-step_started)
                            if model_hash(model)!=original_hash:
                                raise ValueError('Temporary step mutated original model or buffers')
                            append_json(out/'temporary_step_checks.jsonl',dict(token=token,passed=True,original_hash=original_hash,
                                        updates=len(arms)*len(self.plan['D2']['etas']),original_model_unchanged=True))
                        count+=1
                        elapsed=time.perf_counter()-block_started
                        append_json(out/'timings.jsonl',dict(token=token,total_block_seconds=elapsed,query_seconds=query_seconds))
                        write(out/'progress.json',dict(seed=seed,blocks_complete=count,domain=domain,step=step,block=block,
                              elapsed_seconds=time.perf_counter()-started,last_block_seconds=elapsed,status='running'))
                        if count%4==0 or pilot:
                            print(json.dumps(dict(event='block',seed=seed,domain=domain,step=step,block=block,seconds=round(elapsed,3))),flush=True)
            write(out/'done.json',dict(status='completed',seed=seed,blocks=count,elapsed_seconds=time.perf_counter()-started,
                  git_commit=self.commit,plan_sha256=self.plan_sha,
                  peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  oracle_label_accesses=sum(len(event['ids']) for event in
                      (json.loads(line) for line in (out/'label_access.jsonl').read_text().splitlines())
                      if event['event']=='label_access')))
        except Exception:
            append_json(out/'monitor_events.jsonl',dict(event='failed',traceback=traceback.format_exc(),time=time.time()))
            raise
        finally:
            writer.close()
        return dict(seed=seed,blocks=count,seconds=time.perf_counter()-started)

    def run(self):
        if not read(self.stage/'D0/engineering_checks.json')['passed']:
            raise RuntimeError('D0 must pass before D1/D2')
        pilot=read(self.stage/'logs/pilot_3405/done.json')
        # Pilot has the expensive D1 geometry and the full 36-arm D2 block.
        estimate=pilot['elapsed_seconds']*180/self.settings['workers']*self.settings['compute_buffer']
        estimate+=pilot['elapsed_seconds']*180/self.settings['workers']*.5
        if estimate>self.settings['package_wall_cap_seconds']:
            raise RuntimeError('Runtime estimate exceeds package cap; do not silently shrink the plan')
        write(self.stage/'compute_estimate.json',dict(pilot_seconds=pilot['elapsed_seconds'],
              conservative_estimated_wall_seconds=estimate,workers=self.settings['workers'],buffer=self.settings['compute_buffer'],
              cap_seconds=self.settings['package_wall_cap_seconds'],scope='D1+D2, 360 and 180 blocks'))
        start=time.time()
        finished_blocks=0
        print(json.dumps(dict(event='starting_full_P0',estimated_seconds=estimate)),flush=True)
        context=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=self.settings['workers'],mp_context=context) as pool:
            pending={pool.submit(self.worker_entrypoint(),self.config,seed):seed for seed in self.plan['seeds']}
            warned=set()
            while pending:
                completed,_=wait(pending,timeout=self.settings['progress_interval_seconds'],return_when=FIRST_COMPLETED)
                for future in completed:
                    result=future.result()
                    finished_blocks+=result['blocks']
                    print(json.dumps(dict(event='seed_complete',**result)),flush=True)
                    del pending[future]
                elapsed=time.time()-start
                states=[]
                for seed in pending.values():
                    path=self.stage/'logs'/str(seed)/'progress.json'
                    if path.exists(): states.append(read(path))
                write(self.stage/'compute_ledger.json',dict(status='running' if pending else 'completed',
                      elapsed_wall_seconds=elapsed,gpu_hours=0.,workers=self.settings['workers'],
                      cap_seconds=self.settings['package_wall_cap_seconds'],new_training_jobs=0))
                print(json.dumps(dict(event='progress',elapsed_seconds=round(elapsed,1),remaining_seeds=list(pending.values()),
                                      completed_blocks=finished_blocks+sum(s['blocks_complete'] for s in states))),flush=True)
                for fraction in (.5,.8):
                    if elapsed>=fraction*self.settings['package_wall_cap_seconds'] and fraction not in warned:
                        warned.add(fraction)
                        append_json(self.stage/'logs/monitor_events.jsonl',dict(event='budget_warning',fraction=fraction,time=time.time()))
                if elapsed>=self.settings['package_wall_cap_seconds']:
                    for process in pool._processes.values(): process.terminate()
                    raise TimeoutError('P0 hard budget cap reached')


def run_worker(config,seed):
    return ResearchExperiment(config).worker(seed)


class Tiny(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2,4),nn.BatchNorm1d(4),nn.Tanh(),nn.Dropout(.5))
        self.head=nn.Linear(4,classes)

    def forward(self,batch):
        return center_logits(self.head(self.encoder(batch)))


def run_d0(experiment):
    e=experiment
    seed_everything(3405,e.settings['threads_per_worker'],True)
    out=e.stage/'D0'
    out.mkdir(exist_ok=False)
    checks=[]
    permissions=[]

    def verify(name,actual,expected,*,tolerance=None):
        difference=float((actual-expected).norm())
        reference=float(expected.norm())
        relative=difference/max(reference,1e-12)
        allowed=e.settings['relative_tolerance'] if tolerance is None else tolerance
        passed=(difference<=e.settings['absolute_tolerance']+allowed*reference
                and (relative<=allowed if reference>1e-12 else difference<=e.settings['absolute_tolerance']))
        checks.append(dict(name=name,absolute_error=difference,relative_error=relative,passed=passed))
        if not passed:
            raise AssertionError(checks[-1])

    def algebra(model,source,source_y,target,target_y,fit,fit_y,name):
        model.eval()
        before=model_hash(model)
        geometry=unlabeled_geometry(model,source,target,e.v2)
        gradient,residual=geometry.gradient(source_y)
        direct=gradient_of_ce(geometry.snapshot,[target],target_y)
        count=len(target_y)
        ry=(F.one_hot(target_y,model.head.out_features).double()-geometry.snapshot.logits(target).softmax(-1))@geometry.contrasts
        reconstructed=-geometry.basis@torch.einsum('bqr,bq->r',geometry.target_response,ry)/count
        verify(name+'_target_gradient',reconstructed,direct)
        # The explicit Jacobian is intentionally restricted to a few source graphs.
        small_source=Batch.from_data_list(source.to_data_list()[:3]) if isinstance(source,Batch) else source[:3]
        explicit=(geometry.snapshot.jacobian(small_source,geometry.contrasts)@geometry.basis).reshape_as(geometry.source_response[:3])
        verify(name+'_source_JVP',geometry.source_response[:3],explicit)
        production=build_gradient_feedback(model,source,source_y,target,e.v2)
        verify(name+'_production_readout',geometry.readout,production.readout)
        verify(name+'_production_gradient',gradient,production.gradient)
        changed=build_gradient_feedback(model,source,source_y.roll(1),target,e.v2)
        verify(name+'_labels_do_not_change_W',production.readout,changed.readout,tolerance=e.settings['readout_label_tolerance'])
        params=tuple(model.parameters())
        auxiliary=torch.cat([x.reshape(-1) for x in torch.autograd.grad(parameter_feedback_loss(model,production),params)])
        verify(name+'_auxiliary_gradient',auxiliary,gradient)
        source_gradient=gradient_of_ce(geometry.snapshot,[fit],fit_y)
        clone=copy.deepcopy(model).eval()
        optimizer=torch.optim.SGD(clone.parameters(),lr=e.plan['D2']['primary_eta'])
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(clone(fit),fit_y)+e.plan['D2']['beta']*parameter_feedback_loss(clone,production)
        loss.backward();optimizer.step()
        new=torch.cat([p.detach().reshape(-1) for p in clone.parameters()])
        expected=-e.plan['D2']['primary_eta']*(source_gradient+e.plan['D2']['beta']*gradient)
        verify(name+'_actual_SGD_delta',new-geometry.snapshot.flat,expected)
        if model_hash(model)!=before or any(m.training for m in model.modules()):
            raise AssertionError('D0 changed model state or training flags')
        checks.append(dict(name=name+'_state_unchanged',passed=True))

    try:
        for classes in (2,3):
            torch.manual_seed(1234+classes)
            model=Tiny(classes).double().eval()
            source=torch.randn(5,2,dtype=torch.float64)
            target=torch.randn(3,2,dtype=torch.float64)
            source_y=torch.arange(5)%classes
            target_y=torch.arange(3)%classes
            algebra(model,source,source_y,target,target_y,source,source_y,f'C{classes}')
        for domain in range(1,4):
            task=e.task(domain)
            a_pool,b_pool,dev,final=e.pools(domain)
            seed=61000000+domain*100000
            ids=sample_ids(a_pool,16,seed)
            target=task.batch(ids,'target')
            model,row=e.load(domain,3405,0)
            source=task.batch(tuple(sorted(task.query_ids)),'query')
            fit=task.batch(sample_ids(task.fit_ids,16,seed+1000),'fit')
            oracle=DevelopmentOracle(task,dev,final,out/'label_access.jsonl',enabled=True)
            geo=unlabeled_geometry(model,label_free(source),target,e.v2)
            token=f'D0_{domain}'
            oracle.seal(token,ids,theta_hash=tensor_hash(geo.snapshot.flat),readout_hash=tensor_hash(geo.readout))
            labels=oracle.labels(ids,token=token,purpose='D0_gradient_reconstruction')
            algebra(model,label_free(source),source.y,target,labels,label_free(fit),fit.y,f'real_{domain}')
            for enabled,denied_ids,seal in [(False,ids,token),(True,[final[0]],token),(True,ids,'unknown')]:
                oracle.enabled=enabled
                try:
                    oracle.labels(denied_ids,token=seal,purpose='D0_permission_negative_test')
                except PermissionError:
                    permissions.append(dict(domain=domain,enabled=enabled,token_known=seal==token,denied=True))
                else:
                    raise AssertionError('Oracle permission test accepted forbidden access')
            if target.y is not None:
                raise AssertionError('Training target inputs contain y')
            checks.append(dict(name=f'real_{domain}_input_has_no_y',passed=True))
        torch.manual_seed(678)
        dtype=torch.float64
        x=torch.randn(5,5,dtype=dtype)
        kss=x@x.mT
        hst=torch.randn(10,3,dtype=dtype)
        htt=hst.mT@torch.linalg.solve(torch.kron(kss,torch.eye(2,dtype=dtype)),hst)+torch.eye(3,dtype=dtype)
        lam=e.v2.calibration['lambda_w']
        mean,_=solve_calibration(kss,hst,htt,CalibrationConfig(mode='mean_only',lambda_w=lam/2))
        verify('ISO_equals_MEAN_lambda_half',fixed_isotropic(kss,hst,lam),mean)
        zero_w,_=solve_calibration(torch.eye(3,dtype=dtype),torch.zeros(6,2,dtype=dtype),torch.eye(2,dtype=dtype))
        verify('no_support_W_zero',zero_w,torch.zeros_like(zero_w))
        verify('no_support_feedback_zero',zero_w@torch.randn(6,dtype=dtype),torch.zeros(2,dtype=dtype))
        sh,th=torch.randn(5,4,dtype=dtype),torch.randn(3,4,dtype=dtype)
        sa,ta=torch.randn(5,2,3,dtype=dtype),torch.randn(3,2,3,dtype=dtype)
        first=signature_kernels(sh,th,sa,ta)
        second=signature_kernels(sh,th,7*sa,7*ta)
        for i in range(3): verify(f'scale_invariance_kernel_{i}',first[i],second[i],tolerance=1e-10)
        v,stats=target_space(torch.zeros(3,5,dtype=dtype))
        zero=torch.zeros(5,dtype=dtype)
        m=gradient_metrics(zero,zero,zero,zero,v)
        if stats['rank']!=0 or m['c_A'] is not None or m['weighted_ratio'] is not None or m['e_A'] is not None:
            raise AssertionError('Near-zero boundaries fabricated a metric')
        checks.append(dict(name='zero_rank_gradient_source_boundaries',passed=True))
        write(out/'permission_checks.json',dict(passed=True,checks=permissions,final_labels_read=0))
        write(out/'engineering_checks.json',dict(passed=True,checks=checks,git_commit=e.commit,
              plan_sha256=e.plan_sha,real_tasks=3,synthetic_classes=[2,3],research_efficacy_not_tested=True))
        print(json.dumps(dict(event='D0_passed',checks=len(checks),permission_denials=len(permissions))),flush=True)
    except Exception:
        write(out/'engineering_checks.json',dict(passed=False,checks=checks,traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','d0','pilot','run'])
    parser.add_argument('--repo',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--config',default='configs/p0_diagnostics.json')
    args=parser.parse_args()
    config=ExperimentConfig(args.repo,args.stage,str(Path(args.config).resolve()))
    experiment=ResearchExperiment(config)
    torch.set_num_threads(experiment.settings['threads_per_worker'])
    if args.action=='prepare': experiment.prepare()
    elif args.action=='d0': run_d0(experiment)
    elif args.action=='pilot':
        if not read(experiment.stage/'D0/engineering_checks.json')['passed']: raise RuntimeError('D0 failed')
        experiment.worker(3405,pilot=True)
    else: experiment.run()


if __name__=='__main__':
    main()
