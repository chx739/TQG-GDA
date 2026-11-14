"""Post-FINAL descriptive analysis; no model fitting or endpoint selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.statistics import write_csv; from experiments.mutagenicity.r1_20261111.src.evaluate import seed_summary
from experiments.mutagenicity.r2_20261111.src.common import load, jobs, job_key, write_json, verify


def analyze(root,out):
    root=Path(root);out=Path(out);verify(root,'OBS');verify(root,'FINAL')
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'execution-plan.json').read_text())
    summary=json.loads((root/'evaluation/summary.json').read_text())
    target=[r for r in summary if r['attempt']==30 and r['pool'].startswith('B')]
    counts={}
    for kind in ('learned','kde'):
        rows=[r for r in target if r['kind']==kind]
        counts[kind]=dict(cells=len(rows),
            nll_better_than_original=sum(r['delta_nll_vs_original']['mean']<0 for r in rows),
            nll_better_than_geometry=sum(r['delta_nll_vs_geometry']['mean']<0 for r in rows),
            nll_better_than_reverse=sum(r['delta_nll_vs_reverse']['mean']<0 for r in rows),
            BA_better_than_original=sum(r['delta_balanced_accuracy_vs_original']['mean']>0 for r in rows),
            lambda_10_100_normal_worse_than_reverse=sum(r['lam']>=10 and r['delta_nll_vs_reverse']['mean']>0 for r in rows))
    write_json(out/'grid-counts.json',counts)
    mechanism=json.loads((root/'evaluation/mechanisms.json').read_text());by_spec=defaultdict(list)
    names=('seed','domain','kind','bandwidth','updates','lam','sign')
    for r in mechanism:by_spec[tuple(r[k] for k in names)].append(r)
    perseed=[]
    fields=('alpha','request_rms_sigma','request_clipping_rate','target_source_norm_ratio',
        'target_margin_rms_sigma','executed_target_margin_rms_sigma','actual_margin_rms_sigma','proposal_cosine')
    for key,rows in sorted(by_spec.items()):
        if key[-1]!=1:continue
        value=dict(zip(names,key));assert len(rows)==30
        for field in fields:
            v=[r[field] for r in rows if r.get(field) is not None]
            value[field]=float(np.mean(v)) if len(v)==30 else None
        perseed.append(value)
    grouped=defaultdict(list)
    for row in perseed:grouped[tuple(row[k] for k in names[1:])].append(row)
    mech_summary=[]
    for key,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:r['seed'])
        result=dict(zip(names[1:],key));result.update({f:seed_summary([r[f] for r in rows]) for f in fields});mech_summary.append(result)
    write_csv(out/'mechanism-per-seed.csv',perseed);write_json(out/'mechanism-summary.json',mech_summary)
    # Post hoc explanation of the source-control failure: subsets are fixed by
    # the original head, so changing membership cannot manufacture the effect.
    labels=json.loads((root/'evaluation/labels.json').read_text());confidence=[]
    for spec in jobs(plan):
        if spec['kind']!='geometry':continue
        directory=root/'trajectories'/job_key(spec);pool=f"B{spec['domain']}"
        initial=load(directory/'round000.pt')['predictions'][pool]
        y=torch.tensor([labels[pool][str(i)] for i in initial['ids']])
        wrong=initial['probabilities'].argmax(1)!=y
        initial_nll=-initial['log_probabilities'][torch.arange(len(y)),y]
        for attempt in (0,30):
            p=load(directory/f'round{attempt:03d}.pt')['predictions'][pool]
            assert p['ids']==initial['ids']
            nll=-p['log_probabilities'][torch.arange(len(y)),y]
            contribution_wrong=float((nll[wrong]-initial_nll[wrong]).sum()/len(y))
            contribution_correct=float((nll[~wrong]-initial_nll[~wrong]).sum()/len(y))
            assert abs(contribution_wrong+contribution_correct-float(nll.mean()-initial_nll.mean()))<1e-12
            confidence.append(dict(seed=spec['seed'],domain=spec['domain'],lam=spec['lam'],attempt=attempt,n=len(y),
                initially_wrong=int(wrong.sum()),argmax_changes=int((p['probabilities'].argmax(1)!=initial['probabilities'].argmax(1)).sum()),
                nll=float(nll.mean()),mean_max_probability=float(p['probabilities'].max(1).values.mean()),
                mean_abs_logit_margin=float((p['log_probabilities'][:,0]-p['log_probabilities'][:,1]).abs().mean()),
                initial_error_subset_nll=float(nll[wrong].mean()),initial_correct_subset_nll=float(nll[~wrong].mean()),
                delta_nll_from_initial_errors=contribution_wrong,delta_nll_from_initial_correct=contribution_correct))
    write_csv(out/'posthoc-source-control-confidence.csv',confidence)
    write_json(out/'posthoc-source-control-confidence.json',confidence)
    write_json(out/'scope.json',dict(stage='post-FINAL descriptive analysis',
        classification_endpoint_unchanged=True,new_training=False,new_label_queries=False,
        confidence_analysis='Unplanned descriptive decomposition on geometry controls, fixed original-error/correct subsets; no new causal intervention',
        counts_note='Grid cells are configurations, not independent replicates; inferential unit remains the five seeds'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.root, a.output)
