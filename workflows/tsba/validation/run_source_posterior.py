"""Frozen matched source-posterior intervention and post-seal diagnostics."""
import argparse
import copy
import json
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
import traceback
import torch
from tsba.classifier import Coordinates,ParameterMap
from tsba.config import Config
from .archive import digest,manifest,provenance,save_tensor,write_json
from .distributions import oracles
from .run_controls import make_head
from .run_full_source import verify_archive
from .run_replay import replay_cell
from .source_posterior import ARMS,analytic_source_posterior,expected_ce,adapt_posterior,audit_posterior_event


def read(p):return json.loads(p.read_text())


def run_cell(parent,out,dim,seed,plan,log):
    start=time.perf_counter();out.mkdir()
    for name in ('inputs.pt','config.json','schedule.pt','evaluation_labels.pt','initial_predictions.pt','population-score.pt'):
        shutil.copy2(parent/name,out/name)
    inputs=torch.load(out/'inputs.pt',weights_only=True);cfg=Config(**read(out/'config.json'))
    schedule=torch.load(out/'schedule.pt',weights_only=True)
    if cfg.seed!=seed or inputs['source'].shape[1]!=dim or cfg.source_adapt_batch!=512:raise ValueError('wrong parent cell/config')
    coords=Coordinates(inputs['coordinates']['mean'],inputs['coordinates']['scale'])
    posterior,log_posterior=analytic_source_posterior(inputs['source'],coords)
    save_tensor(out/'source-posterior.pt',dict(probabilities=posterior,log_probabilities=log_posterior,raw_source=inputs['source']*coords.scale+coords.mean))
    initial=make_head(inputs);initial_vector=ParameterMap(initial).vector()
    population,_=oracles(inputs['target'],coords,inputs['sigma'])
    saved=torch.load(out/'population-score.pt',weights_only=True)
    for key,value in population.state_dict().items():torch.testing.assert_close(value,saved['state_dict'][key],atol=0,rtol=0)
    with torch.no_grad():
        initial_logits=initial(inputs['evaluation'])
        torch.testing.assert_close(initial_logits,torch.load(out/'initial_predictions.pt',weights_only=True),atol=0,rtol=0)
        error=float((initial(inputs['source']).log_softmax(-1)-log_posterior).abs().max())
        if error>1e-11:raise ValueError('analytic posterior and source Bayes initialization differ')
    summary=dict(dim=dim,seed=seed,source_posterior_initial_log_error=error,config_exactly_unchanged=True,arms={})
    for arm in ARMS:
        dest=out/arm;dest.mkdir();head=copy.deepcopy(initial);events=[];begin=time.perf_counter()
        def observe(e):
            events.append(e);audit_posterior_event(head,inputs,posterior,schedule[e['record']['round']],e)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>plan['resources']['rss_bytes']:raise MemoryError('RSS budget exceeded')
        try:
            trace=adapt_posterior(head,inputs['source'],posterior,inputs['target'],population,inputs['sigma'],cfg,schedule,arm,
                                  observer=observe,logger=lambda m:log(f'd={dim} seed={seed} arm={arm} {m}'))
        finally:
            save_tensor(dest/'rounds.pt',events);save_tensor(dest/'classifier.pt',head.state_dict())
        with torch.no_grad():
            pred=head(inputs['evaluation']);save_tensor(dest/'predictions.pt',pred)
            final_ce=float(expected_ce(head(inputs['source']),posterior).mean())
        drift=float((ParameterMap(head).vector()-initial_vector).abs().max())
        logit_drift=float((pred-initial_logits).abs().max())
        # This is a diagnostic assertion after the unchanged optimization, not a
        # new skip rule, gradient clamp, line-search constraint or model choice.
        stationary=arm=='population' or (drift<=plan['control_stationarity']['parameter_atol'] and logit_drift<=plan['control_stationarity']['logit_atol'])
        summary['arms'][arm]=dict(accepted=trace['accepted'],stop_reason=trace['stop_reason'],rounds=len(events),
                                 seconds=time.perf_counter()-begin,source_posterior_ce=final_ce,
                                 max_parameter_drift=drift,max_evaluation_logit_drift=logit_drift,
                                 stationary_check_passed=stationary,
                                 source_gradient_max_norm=max((e['record'].get('source_gradient_norm',0.) for e in events),default=0.))
        write_json(dest/'adaptation.json',trace)
        if not stationary:raise ValueError(f'{arm} exceeded predeclared stationary-control tolerance')
    summary.update(seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(out/'cell_summary.json',summary);log(f'CELL COMPLETE d={dim} seed={seed} seconds={summary["seconds"]:.1f}')


def diagnostics(root,reference,completed,log):
    # All adaptation predictions exist before any new target-label decoding.
    seal=manifest(root,exclude=('run.log',));write_json(root/'prediction-seal.json',seal)
    out=root/'diagnostics';out.mkdir();rows=[];audits=[];failures=[]
    replay_plan=read(reference/'replay-plan.json');write_json(out/'replay-plan.json',replay_plan)
    for dim,seed in completed:
        name=f'd{dim}_seed{seed}';parent=root/name
        try:
            for arm in ARMS:
                rel=f'{name}/{arm}/predictions.pt'
                if digest(root/rel)!=seal[rel]['sha256']:raise ValueError('sealed predictions changed')
            r,a=replay_cell(parent,out/name,dim,seed,replay_plan,log)
            current=torch.load(out/name/'diagnostic_inputs.pt',weights_only=True)
            old=torch.load(reference/name/'diagnostic_inputs.pt',weights_only=True)
            for key in ('source_raw','source_y','source_log_posterior','nuisance'):
                torch.testing.assert_close(current[key],old[key],atol=0,rtol=0)
            posterior=torch.load(parent/'source-posterior.pt',weights_only=True)['probabilities']
            inputs=torch.load(parent/'inputs.pt',weights_only=True);head=make_head(inputs);params=ParameterMap(head)
            for arm in ARMS:
                vectors=torch.load(out/name/arm/'checkpoint_vectors.pt',weights_only=True)
                for item in [x for x in r if x['arm']==arm]:
                    with torch.no_grad():item['source_posterior_train_ce']=float(expected_ce(params.logits(vectors[item['round']],inputs['source']),posterior).mean())
            rows.extend(r);audits.append(dict(dim=dim,seed=seed,diagnostic_inputs_match_previous=True,**a))
        except Exception as exc:
            failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()));log(f'DIAGNOSTIC FAILED {name}: {exc}')
    write_json(out/'metrics.json',rows);write_json(out/'audits.json',audits);write_json(out/'failures.json',failures)
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    repo=Path(__file__).resolve().parents[3];parent=args.parent.resolve();reference=args.replay.resolve();root=args.output.resolve()
    planpath=repo/'docs/tsba/experiments/source-posterior-plan.json';plan=read(planpath)
    expected=repo/'docs/tsba/experiments/e2-full-source-20261110/run-manifest.json'
    expected_replay=repo/'docs/tsba/experiments/e2-replay-20261110/run-manifest.json'
    integrity=dict(parent=verify_archive(parent,expected),replay=verify_archive(reference,expected_replay))
    root.mkdir(parents=True,exist_ok=False);write_json(root/'parent-integrity.json',integrity)
    write_json(root/'provenance.json',provenance(repo));shutil.copy2(planpath,root/'source-posterior-plan.json')
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(root/'execution-source.tar.gz'),'HEAD'],cwd=repo,check=True)
    with (root/'python-packages.json').open('x') as f:subprocess.run([__import__('sys').executable,'-m','pip','list','--format=json'],stdout=f,check=True)
    failures=[];completed=[];started=time.perf_counter()
    def timeout(signum,frame):raise TimeoutError('predeclared cell time budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    with (root/'run.log').open('x',buffering=1) as f:
        def log(message):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+message;print(line,flush=True);f.write(line+'\n')
        for dim in plan['dimensions']:
            for seed in plan['seeds']:
                signal.alarm(plan['resources']['cell_seconds'])
                try:
                    if time.perf_counter()-started>plan['resources']['total_seconds']:raise TimeoutError('global time budget exceeded')
                    name=f'd{dim}_seed{seed}';run_cell(parent/name,root/name,dim,seed,plan,log);completed.append((dim,seed))
                except Exception as exc:
                    failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()));log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                finally:signal.alarm(0)
        write_json(root/'failures.json',failures)
        log('ADAPTATION SEALED; starting passive diagnostics')
        diagnostic_failures=diagnostics(root,reference,completed,log)
        after=dict(parent=verify_archive(parent,expected),replay=verify_archive(reference,expected_replay))
        if after!=integrity:raise ValueError('parent artifacts changed')
        write_json(root/'execution.json',dict(seconds=time.perf_counter()-started,completed_cells=len(completed),failures=len(failures),diagnostic_failures=len(diagnostic_failures),parents_unchanged=True))
        log(f'COMPLETE cells={len(completed)}/10 failures={len(failures)} diagnostic_failures={len(diagnostic_failures)} seconds={time.perf_counter()-started:.1f}')
    write_json(root/'manifest.json',manifest(root))
    if failures or diagnostic_failures:raise SystemExit(1)


if __name__=='__main__':main()
