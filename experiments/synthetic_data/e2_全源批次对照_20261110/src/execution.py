"""Change only the source adaptation batch from 128 to all 512 archived samples."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
import torch
from torch.nn import functional as F
from tsba.adapt import adapt_classifier
from tsba.classifier import Coordinates, ParameterMap
from tsba.config import Config
from experiments.common.io import digest, manifest, provenance, save_tensor, write_json
from experiments.synthetic_data.e2_目标请求对照_20261109.src.controls import adapt_control
from experiments.synthetic_data.e1_e2_符号几何与适配_20261109.src.distributions import oracles
from experiments.common.metrics import classification
from experiments.synthetic_data.e2_目标请求对照_20261109.src.execution import make_head

ARMS = ('source_ce_ggn', 'zero_zeta_geometry', 'population')


def read(path):return json.loads(path.read_text())


def verify_archive(root, expected):
    if digest(root/'manifest.json') != digest(expected):raise ValueError('parent manifest mismatch')
    items=read(expected)
    for name,r in items.items():
        p=root/name
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError(f'parent changed: {p}')
    return dict(path=str(root),manifest_sha256=digest(expected),verified_files=len(items))


def full_source_config(parent, count):
    if count != 512:raise ValueError('this protocol requires exactly 512 source samples')
    cfg=replace(parent,source_adapt_batch=count)
    delta={k:(parent.to_dict()[k],v) for k,v in cfg.to_dict().items() if parent.to_dict()[k]!=v}
    if delta!={'source_adapt_batch':(128,512)}:raise ValueError(f'unexpected configuration change: {delta}')
    return cfg


def full_source_schedule(parent_events, old_cfg, source_count, target_count):
    """Changing the slice length consumes no RNG. Target IDs remain identical."""
    if len(parent_events)!=old_cfg.adapt_rounds:raise ValueError('incomplete parent schedule')
    rng=torch.Generator().manual_seed(old_cfg.seed+201)
    result=[]
    for parent in parent_events:
        source_ids=torch.randperm(source_count,generator=rng)
        target_ids=torch.randperm(target_count,generator=rng)[:old_cfg.target_batch]
        if not torch.equal(source_ids[:old_cfg.source_adapt_batch],parent['source_ids']):raise ValueError('parent source RNG mismatch')
        if not torch.equal(target_ids,parent['target_ids']):raise ValueError('parent target RNG mismatch')
        result.append(dict(source_ids=source_ids,target_ids=target_ids))
    return result


@torch.no_grad()
def audit_full_source_event(head, inputs, schedule_row, event):
    for name in ('source_ids','target_ids'):
        if not torch.equal(event[name],schedule_row[name]):raise ValueError(f'{name} schedule mismatch')
    ids=event['source_ids']
    if not torch.equal(ids.sort().values,torch.arange(len(inputs['source']))):raise ValueError('source set is not complete')
    params=ParameterMap(head)
    def ce(vector):return float(F.cross_entropy(params.logits(vector,inputs['source']),inputs['source_y']))
    before=ce(event['before']);after=ce(event['after'])
    rec=event['record'];errors=[]
    if 'before' in rec:errors.append(abs(before-rec['before']['source_ce']))
    for trial in rec.get('trials',[]):
        candidate=event['before']+trial['alpha']*event['delta']
        errors.append(abs(ce(candidate)-trial['source_ce']))
    error=max(errors,default=0.)
    if error>1e-11:raise ValueError(f'logged acceptance CE is not full-source CE: {error}')
    event['full_source_ce_audit']=dict(before=before,after=after,change=after-before,
                                     max_logged_ce_error=error,trials_checked=len(rec.get('trials',[])))


def run_cell(original,controls,out,dim,seed,log):
    started=time.perf_counter();out.mkdir()
    for name in ('inputs.pt','evaluation_labels.pt','initial_predictions.pt'):
        if digest(original/name)!=digest(controls/name):raise ValueError('parents do not share identical inputs/initialization')
        shutil.copy2(original/name,out/name)  # labels are opaque bytes until evaluation
    shutil.copy2(original/'config.json',out/'parent-config.json')
    inputs=torch.load(out/'inputs.pt',weights_only=True)
    old_cfg=Config(**read(out/'parent-config.json'))
    cfg=full_source_config(old_cfg,len(inputs['source']))
    if cfg.seed!=seed or inputs['source'].shape[1]!=dim:raise ValueError('cell identity mismatch')
    write_json(out/'config.json',cfg.to_dict())
    parents=torch.load(original/'population'/'rounds.pt',weights_only=True)
    schedule=full_source_schedule(parents,old_cfg,len(inputs['source']),len(inputs['target']))
    save_tensor(out/'schedule.pt',schedule)
    write_json(out/'configuration-audit.json',dict(only_changed_key='source_adapt_batch',before=128,after=512,
                                                  target_schedule_exact=True,source_shuffle_replayed=True))
    # Save the existing minibatch counterparts; these are not re-run.
    prior_summary={}
    for arm in ARMS:
        parent=original if arm=='population' else controls
        events=torch.load(parent/arm/'rounds.pt',weights_only=True)
        if len(events)!=len(schedule):raise ValueError('incomplete comparator trajectory')
        for e,p in zip(events,parents):
            for key in ('source_ids','target_ids'):
                if not torch.equal(e[key],p[key]):raise ValueError('minibatch comparator IDs mismatch')
        shutil.copy2(parent/arm/'predictions.pt',out/f'minibatch_{arm}_predictions.pt')
        prior_summary[arm]=read(parent/'cell_summary.json')['arms'][arm]
    write_json(out/'minibatch-summary.json',prior_summary)
    initial=make_head(inputs)
    with torch.no_grad():
        torch.testing.assert_close(initial(inputs['evaluation']),torch.load(out/'initial_predictions.pt',weights_only=True),atol=0,rtol=0)
        initial_ce=float(F.cross_entropy(initial(inputs['source']),inputs['source_y']))
    c=inputs['coordinates'];coordinates=Coordinates(c['mean'],c['scale'])
    population,_=oracles(inputs['target'],coordinates,inputs['sigma'])
    save_tensor(out/'population-score.pt',dict(state_dict=population.state_dict(),sigma=inputs['sigma']))
    # Check the restored oracle at the exact old first-round boundary queries.
    b=parents[0]['boundaries']
    with torch.no_grad():
        gamma=(population(b['points'])*b['normals']).sum(-1)
        zeta=(cfg.tau*inputs['sigma']**2*gamma).clamp(-cfg.request_multiplier*inputs['sigma'],cfg.request_multiplier*inputs['sigma'])
        torch.testing.assert_close(zeta,b['zeta'],atol=0,rtol=0)
    summary=dict(dim=dim,seed=seed,initial_source_ce=initial_ce,arms={})
    for arm in ARMS:
        head=copy.deepcopy(initial);events=[];dest=out/arm;dest.mkdir();begin=time.perf_counter()
        def observe(payload):
            events.append(payload)
            audit_full_source_event(head,inputs,schedule[payload['record']['round']],payload)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4294967296:raise MemoryError('4 GiB RSS budget exceeded')
        logger=lambda message:log(f'd={dim} seed={seed} arm={arm} {message}')
        try:
            if arm=='population':
                trace=adapt_classifier(head,inputs['source'],inputs['source_y'],inputs['target'],population,
                                       inputs['sigma'],cfg,logger=logger,observer=observe)
            else:
                trace=adapt_control(head,inputs['source'],inputs['source_y'],inputs['target'],inputs['sigma'],
                                    cfg,schedule,arm,observer=observe,logger=logger)
        finally:
            save_tensor(dest/'rounds.pt',events);save_tensor(dest/'classifier.pt',head.state_dict())
        seconds=time.perf_counter()-begin
        with torch.no_grad():
            save_tensor(dest/'predictions.pt',head(inputs['evaluation']))
            source_ce=float(F.cross_entropy(head(inputs['source']),inputs['source_y']))
        write_json(dest/'adaptation.json',trace)
        changes=[e['full_source_ce_audit']['change'] for e in events]
        summary['arms'][arm]=dict(accepted=trace['accepted'],stop_reason=trace['stop_reason'],
                                 rounds=len(trace['history']),seconds=seconds,source_ce=source_ce,
                                 full_source_ce_increase_rounds=sum(x>1e-10 for x in changes),
                                 max_full_source_ce_increase=max(changes,default=0.))
    summary['seconds']=time.perf_counter()-started
    summary['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write_json(out/'cell_summary.json',summary)
    log(f'CELL COMPLETE d={dim} seed={seed} seconds={summary["seconds"]:.1f}')


def evaluate(root,cells):
    seal=manifest(root,exclude=('run.log',));write_json(root/'prediction-seal.json',seal)
    results=[];access=[]
    for dim,seed in cells:
        directory=root/f'd{dim}_seed{seed}'
        labels=torch.load(directory/'evaluation_labels.pt',weights_only=True)
        access.append(dict(path=str((directory/'evaluation_labels.pt').relative_to(root)),
                           time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                           purpose='paired synthetic evaluation after sealing all full-source predictions'))
        initial=classification(torch.load(directory/'initial_predictions.pt',weights_only=True),labels)
        for batch in (128,512):
            for arm in ARMS:
                path=directory/f'minibatch_{arm}_predictions.pt' if batch==128 else directory/arm/'predictions.pt'
                if digest(path)!=seal[str(path.relative_to(root))]['sha256']:raise ValueError('prediction seal mismatch')
                metrics=classification(torch.load(path,weights_only=True),labels)
                results.append(dict(dim=dim,seed=seed,arm=arm,source_batch=batch,initial=initial,final=metrics,
                                    delta_ba_pp=100*(metrics['balanced_accuracy']-initial['balanced_accuracy']),
                                    delta_nll=metrics['nll']-initial['nll']))
    write_json(root/'label-access.json',access);write_json(root/'evaluation.json',results)
