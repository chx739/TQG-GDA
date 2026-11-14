"""Post-seal evaluation of the entire fixed grid; B never selects a model."""
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from experiments.common.io import digest, manifest
from experiments.mutagenicity.r1_20261111.src.data import canonical_sha
from experiments.common.statistics import classification_metrics, write_csv as _write_csv; from experiments.mutagenicity.r1_20261111.src.evaluate import seed_summary
from experiments.mutagenicity.r2_20261111.src.common import write_json as _write_json, seal, load, verify, Broker, jobs, job_key


def write_json(path,value):
    """Recompute deterministic evaluation products and verify existing records."""
    path=Path(path);normalized=json.loads(json.dumps(value,allow_nan=False))
    if path.exists():
        if json.loads(path.read_text())!=normalized:raise PermissionError('Evaluation record changed: '+str(path))
    else:_write_json(path,value)


def write_csv(path,rows):
    path=Path(path);temp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    _write_csv(temp,rows)
    with temp.open('rb') as f:os.fsync(f.fileno())
    if path.exists():
        if path.read_bytes()!=temp.read_bytes():raise PermissionError('Evaluation CSV changed: '+str(path))
    else:os.link(temp,path)
    temp.unlink()


def evidence_manifest(root):
    """Exclude this manifest and its own interrupted-publication hardlinks.

    Other partial artifacts remain evidence. A crash after publishing the
    manifest but before unlinking its temporary name must not make the manifest
    appear as a newly discovered input on the next recovery attempt.
    """
    root=Path(root)
    excluded=('manifest.json',*[p.name for p in root.glob('manifest.json.partial-*')])
    return manifest(root,exclude=excluded)


def evaluate(root,plan):
    root=Path(root);verify(root,'OBS');out=root/'evaluation'
    if (root/'seals/FINAL.json').exists():
        verify(root,'FINAL')
        write_json(root/'manifest.json',evidence_manifest(root))
        return
    broker=Broker(root);roles=json.loads((root/'deps/roles.json').read_text());labels={}
    for d in plan['domains']:
        labels[f'B{d}']=broker.read(roles['targets'][str(d)]['B'],'target_B','Fixed R2 complete-grid evaluation after OBS',domain=d)
    for seed in plan['seeds']:
        s=json.loads((root/f'deps/roles/seed{seed}.json').read_text())
        labels[(seed,'source_fit')]={int(k):v for k,v in json.loads((root/f'source-labels/seed{seed}.json').read_text()).items()}
        for role in ('validation','diagnostic'):
            labels[(seed,'source_'+role)]=broker.read(s[role+'_ids'],'source_'+role,'Fixed R2 source stability evaluation after OBS',seed=seed)
    write_json(out/'labels.json',{str(k):v for k,v in labels.items()})
    records=[];mechanisms=[];diagnostics=[]
    for spec in jobs(plan):
        directory=root/'trajectories'/job_key(spec);complete=json.loads((directory/'complete.json').read_text())
        for attempt in plan['prediction_attempts']:
            path=directory/f'round{attempt:03d}.pt'
            state=load(path) if path.exists() else None
            for pool in (f"B{spec['domain']}",'source_fit','source_validation','source_diagnostic'):
                base=dict(**spec,attempt=attempt,pool=pool,trajectory_status=complete['status'],available=state is not None)
                if state is not None:
                    p=state['predictions'][pool];mapping=labels[pool] if pool.startswith('B') else labels[(spec['seed'],pool)]
                    metrics=classification_metrics(p['probabilities'],[mapping[i] for i in p['ids']],p['log_probabilities'])
                    base.update(metrics)
                records.append(base)
        allrows=[load(p) for p in sorted(directory.glob('round[0-9][0-9][0-9].pt'))[1:]]
        for p in allrows:
            row=dict(**spec,attempt=p['attempt'],**{k:v for k,v in p['record'].items() if k not in ('trials','solver','before','after')})
            for phase in ('before','after'):
                row.update({phase+'_'+k:v for k,v in p['record'].get(phase,{}).items() if k in ('source_ce','move_loss','objective')})
            comp=p.get('current_score_comparison')
            if comp is not None:row.update({'current_score_'+k:v for k,v in comp['all'].items()})
            mechanisms.append(row)
        diagnostic=root/f"diagnostics/seed{spec['seed']}_domain{spec['domain']}/lambda{spec['lam']}"/(job_key(spec).split('/')[-1]+'.pt')
        state=load(diagnostic)
        for pool,p in state['predictions'].items():
            row=dict(**spec,pool=pool,alpha=state['alpha'],available=state['available'])
            if state['available']:
                mapping=labels[pool] if pool.startswith('B') else labels[(spec['seed'],pool)]
                row.update(classification_metrics(p['probabilities'],[mapping[i] for i in p['ids']],p['log_probabilities']))
            diagnostics.append(row)
    write_csv(out/'metrics.csv',records);write_json(out/'metrics.json',records)
    write_csv(out/'mechanisms.csv',mechanisms);write_json(out/'mechanisms.json',mechanisms)
    write_csv(out/'first-round.csv',diagnostics)
    # All comparisons are paired within seed/task, including the original head.
    lookup={(r['seed'],r['domain'],r['kind'],r['bandwidth'],r['updates'],r['lam'],r['sign'],r['attempt'],r['pool']):r for r in records}
    comparisons=[]
    for r in records:
        if r['sign']!=1:continue
        prefix=(r['seed'],r['domain']);tail=(r['attempt'],r['pool'])
        refs=dict(geometry=lookup[(*prefix,'geometry',0,0,r['lam'],0,*tail)],
            original=lookup[(*prefix,r['kind'],r['bandwidth'],r['updates'],r['lam'],1,0,r['pool'])],
            reverse=lookup[(*prefix,r['kind'],r['bandwidth'],r['updates'],r['lam'],-1,*tail)])
        row={k:r[k] for k in ('seed','domain','kind','bandwidth','updates','lam','attempt','pool')}
        for metric in ('nll','balanced_accuracy','accuracy'):
            row[metric]=r.get(metric)
            for name,ref in refs.items():row['delta_'+metric+'_vs_'+name]=r[metric]-ref[metric] if metric in r and metric in ref else None
        comparisons.append(row)
    write_csv(out/'paired-comparisons.csv',comparisons);write_json(out/'paired-comparisons.json',comparisons)
    summaries=[]
    keys=sorted({(r['domain'],r['kind'],r['bandwidth'],r['updates'],r['lam'],r['attempt'],r['pool']) for r in comparisons})
    for key in keys:
        names=('domain','kind','bandwidth','updates','lam','attempt','pool');rows=[r for r in comparisons if tuple(r[n] for n in names)==key]
        rows.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rows]==plan['seeds']
        result=dict(zip(names,key))
        for name in ('nll','balanced_accuracy','accuracy',*[f'delta_{metric}_vs_{ref}' for metric in ('nll','balanced_accuracy','accuracy') for ref in ('geometry','original','reverse')]):
            result[name]=seed_summary([r[name] for r in rows])
        summaries.append(result)
    write_json(out/'summary.json',summaries)
    flat=[]
    for r in summaries:
        flat.append({**{k:v for k,v in r.items() if not isinstance(v,dict)},**{k+'_'+s:v[s] for k,v in r.items() if isinstance(v,dict) for s in ('mean','sample_sd','available_count')}})
    write_csv(out/'summary.csv',flat)
    # Prior target-supervised references are report-only aggregates, authenticated
    # only after R2 predictions were sealed. No old labels enter R2 workers.
    old=Path(plan['dependency']['root']);final=json.loads((old/'seals/FINAL.json').read_text())
    assert canonical_sha({k:final[k] for k in ('stage','files')})==final['digest']
    rel='evaluation/warm_summary.json'
    assert digest(old/rel)==final['files'][rel]
    write_json(out/'prior-target-reference.json',dict(source=str(old/rel),sha256=digest(old/rel),values=json.loads((old/rel).read_text())))
    verify(root,'OBS')
    completion=dict(status='complete',trajectories=630,trajectory_failures=sum(r['status']!='complete' for r in json.loads((root/'adapt-index.json').read_text())['results']),
        metric_rows=len(records),comparison_rows=len(comparisons),mechanism_attempts=len(mechanisms),
        final_labels_parsed=False,new_target_supervised_training=False,primary_attempt=30,primary_metric='B NLL',B_is_development=True)
    write_json(root/'completion.json',completion)
    seal(root,'FINAL',['evaluation','completion.json','logs','seals/INPUT.json','seals/OBS.json'])
    write_json(root/'manifest.json',evidence_manifest(root));print(json.dumps(completion),flush=True)
