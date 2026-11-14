"""Replay the archived full-source trajectories; never train or select a model."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
import traceback
import torch
from torch.nn import functional as F
from tsba.classifier import Coordinates,ParameterMap
from experiments.common.synthetic.archive import digest, manifest, provenance, save_tensor, write_json
from experiments.common.synthetic.distributions import sample_gmm
from experiments.common.synthetic.replay import posterior_log_prob, source_metrics, target_metrics, boundary_probe
from experiments.common.synthetic.run_controls import make_head
from experiments.common.synthetic.run_full_source import ARMS, verify_archive


def read(path):return json.loads(path.read_text())


def frozen_head(inputs):
    head=make_head(inputs)
    for p in head.parameters():p.requires_grad_(False)
    return head


def replay_cell(parent,out,dim,seed,plan,log):
    begin=time.perf_counter();out.mkdir()
    for name in ('inputs.pt','config.json','evaluation_labels.pt','initial_predictions.pt'):
        shutil.copy2(parent/name,out/name)
    inputs=torch.load(out/'inputs.pt',weights_only=True)
    coordinates=Coordinates(inputs['coordinates']['mean'],inputs['coordinates']['scale'])
    head=frozen_head(inputs);params=ParameterMap(head);initial=params.vector()
    raw,source_y=sample_gmm(plan['source_evaluation']['count'],dim,plan['source_evaluation']['seed_offset']+seed,.5)
    source=coordinates(raw).detach();true_logp=posterior_log_prob(raw,.5)
    rng=torch.Generator().manual_seed(plan['boundary']['seed_offset']+seed)
    nuisance=torch.randn(plan['boundary']['probes'],dim-1,generator=rng,dtype=torch.float64)
    save_tensor(out/'diagnostic_inputs.pt',dict(source_raw=raw,source_y=source_y,source_log_posterior=true_logp,
                                               nuisance=nuisance,source_seed=plan['source_evaluation']['seed_offset']+seed,
                                               probe_seed=plan['boundary']['seed_offset']+seed))
    # All parameters were fixed and sealed in the parent run before this replay.
    access=dict(path=str(parent/'evaluation_labels.pt'),sha256=digest(out/'evaluation_labels.pt'),
                time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),purpose='post-hoc fixed-checkpoint trajectory diagnostics; no selection')
    write_json(out/'label-access.json',access)
    labels=torch.load(out/'evaluation_labels.pt',weights_only=True)
    with torch.no_grad():
        source_initial=head(source);target_initial=head(inputs['evaluation'])
        torch.testing.assert_close(target_initial,torch.load(out/'initial_predictions.pt',weights_only=True),rtol=0,atol=0)
        posterior_error=float((source_initial.log_softmax(-1)-true_logp).abs().max())
        if posterior_error>1e-11:raise ValueError('initial head is not the source Bayes posterior')
    rows=[];audit=dict(initial_source_posterior_log_error=posterior_error,states=0,final_logits_bitwise=0,unchanged_after_observation=0)
    for arm in ARMS:
        dest=out/arm;dest.mkdir();events=torch.load(parent/arm/'rounds.pt',weights_only=True)
        if len(events)!=30:raise ValueError('incomplete parent trajectory')
        vectors=[initial]
        previous=initial
        for e in events:
            torch.testing.assert_close(e['before'],previous,atol=0,rtol=0)
            vectors.append(e['after']);previous=e['after']
        saved=torch.stack(vectors);save_tensor(dest/'checkpoint_vectors.pt',saved)
        shutil.copy2(parent/arm/'classifier.pt',dest/'classifier.pt')
        head.load_state_dict(torch.load(dest/'classifier.pt',weights_only=True))
        torch.testing.assert_close(params.vector(),saved[-1],atol=0,rtol=0)
        with torch.no_grad():torch.testing.assert_close(head(inputs['evaluation']),torch.load(parent/arm/'predictions.pt',weights_only=True),rtol=0,atol=0)
        audit['final_logits_bitwise']+=1
        predictions=[];geometry=[];arm_rows=[]
        try:
            for round_id,vector in enumerate(vectors):
                params.commit(vector)
                with torch.no_grad():
                    source_logits=head(source);target_logits=head(inputs['evaluation']);predictions.append(target_logits.detach().clone())
                    train_ce=float(F.cross_entropy(head(inputs['source']),inputs['source_y']))
                sr=source_metrics(source_logits,source_y,true_logp,source_initial)
                tr=target_metrics(target_logits,labels,target_initial)
                bp=plan['boundary']
                br,ba=boundary_probe(head,coordinates,nuisance,bp['lower'],bp['upper'],bp['grid_points'],bp['bisection_steps'])
                torch.testing.assert_close(params.vector(),vector,atol=0,rtol=0)
                if any(p.grad is not None or p.requires_grad for p in head.parameters()):raise ValueError('parameter gradients unexpectedly enabled')
                if round_id:
                    prev=geometry[-1];valid=torch.isfinite(prev['normals']).all(-1)&torch.isfinite(ba['normals']).all(-1)
                    br['previous_normal_pairs']=int(valid.sum())
                    br['previous_normal_angle_mean_deg']=float(torch.acos((prev['normals'][valid]*ba['normals'][valid]).sum(-1).clamp(-1,1)).mean()*180/torch.pi) if valid.any() else None
                else:br.update(previous_normal_pairs=br['valid_normals'],previous_normal_angle_mean_deg=0.)
                geometry.append(ba)
                row=dict(dim=dim,seed=seed,arm=arm,round=round_id,source_train_ce=train_ce,source=sr,target=tr,boundary=br)
                arm_rows.append(row);rows.append(row);audit['states']+=1;audit['unchanged_after_observation']+=1
                if round_id%10==0:log(f'd={dim} seed={seed} arm={arm} replay={round_id}/30 roots={br["unique_detected"]}/{bp["probes"]}')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>plan['resources']['rss_bytes']:raise MemoryError('RSS budget exceeded')
        finally:
            save_tensor(dest/'target_logits.pt',torch.stack(predictions) if predictions else [])
            save_tensor(dest/'boundary_records.pt',geometry)
            write_json(dest/'metrics.json',arm_rows)
    write_json(out/'audit.json',audit)
    write_json(out/'cell_summary.json',dict(dim=dim,seed=seed,seconds=time.perf_counter()-begin,
                                           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    log(f'CELL COMPLETE d={dim} seed={seed} seconds={time.perf_counter()-begin:.1f}')
    return rows,audit
