"""Exact R1b Adam/RNG continuation; immutable endpoints, never DSM selection."""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from tsba.score import ScoreNetwork, denoising_loss
from tsba.utils import finite
from experiments.common.runtime import rss_bytes
from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.steps import EmpiricalScore, old_boundaries, displacement_request
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.common import write_json, load, atomic_tensor, model_from_frozen


def restore(state):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model=ScoreNetwork(state['model_state']['net.0.weight'].shape[1],64).double()
    model.load_state_dict(state['model_state'])
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    opt.load_state_dict(state['optimizer_state'])
    rng=torch.Generator();rng.set_state(state['training_rng_state'])
    torch.random.set_rng_state(state['torch_rng_state'])
    return model,opt,rng


def epoch_update(model,opt,rng,x,h,batch=128):
    total=0.
    for ids in torch.randperm(len(x),generator=rng).split(batch):
        noise=torch.randn(x[ids].shape,generator=rng,dtype=x.dtype)
        loss=denoising_loss(model,x[ids],h,noise);finite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        finite(*(p.grad for p in model.parameters()));opt.step()
        finite(*model.parameters());total+=float(loss.detach())*len(ids)
    return total/len(x)


@torch.no_grad()
def validation(model,x,h,noise):
    result=sum(float(denoising_loss(model,xb,h,eb))*len(xb)
        for xb,eb in zip(x.split(128),noise.split(128)))/len(x)
    finite(result);return result


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for u,v in zip(a,b):assert_nested_equal(u,v)
    else:assert a==b,(a,b)


def comparison(empirical,learned,normals,sigma0):
    eg,er=displacement_request(empirical,normals,sigma0)
    lg,lr=displacement_request(learned,normals,sigma0)
    near=(sigma0*eg).abs()<=1e-6
    def metrics(mask):
        n=int(mask.sum())
        if not n:return dict(n=0,sign_agreement=None,normal_rmse_sigma=None,request_rmse_sigma=None,magnitude_rmse_sigma=None)
        return dict(n=n,sign_agreement=float((eg[mask].sign()==lg[mask].sign()).double().mean()),
            normal_rmse_sigma=float((sigma0*(lg[mask]-eg[mask])).square().mean().sqrt()),
            magnitude_rmse_sigma=float((sigma0*(lg[mask].abs()-eg[mask].abs())).square().mean().sqrt()),
            request_rmse_sigma=float(((lr[mask]-er[mask])/sigma0).square().mean().sqrt()))
    return dict(all=metrics(torch.ones_like(near)),near_zero=metrics(near),away_zero=metrics(~near),
        near_zero_threshold_sigma=1e-6,empirical_gamma=eg,learned_gamma=lg,
        empirical_request=er,learned_request=lr,empirical_score=empirical,learned_score=learned)


def continue_score(root,spec,plan):
    root=Path(root);seed,domain,b=spec['seed'],spec['domain'],spec['bandwidth']
    directory=root/f'scores/seed{seed}_domain{domain}/b{b}'
    if (directory/'complete.json').exists():return json.loads((directory/'complete.json').read_text())
    directory.mkdir(parents=True,exist_ok=True);(directory/'losses').mkdir(exist_ok=True);started=time.monotonic()
    old=root/f'deps/r1b/scores/seed{seed}_domain{domain}/b{b}'
    f=load(root/f'deps/source/seed{seed}/frozen.pt')
    bank=load(root/f'deps/geometry/seed{seed}_domain{domain}/banks/source.pt')
    x=f['x'][bank['target_ids']].detach();sigma0=bank['sigma'];h=b*sigma0
    assert (len(x)+127)//128==9
    initial=load(old/'epochs/epoch099.pt');prior=load(old/'epochs/epoch098.pt')
    assert initial['updates_completed']==900 and initial['epoch']==99 and initial['sigma']==h
    # Reproduce the last old epoch, including Adam moments and training RNG.
    model,opt,rng=restore(prior)
    train=epoch_update(model,opt,rng,x,h)
    assert_nested_equal(model.state_dict(),initial['model_state'])
    assert_nested_equal(opt.state_dict(),initial['optimizer_state'])
    assert_nested_equal(rng.get_state(),initial['training_rng_state'])
    assert train==initial['history'][-1]['train_loss']
    assert validation(model,x,h,initial['validation_noise'])==initial['history'][-1]['noise_validation_loss']
    bounds,_=old_boundaries(bank,model_from_frozen(f),sigma0)
    with torch.no_grad():score900=model(bounds.points)
    proposal=load(root/f'deps/r1b/steps/seed{seed}_domain{domain}/learned_b{b}/proposal.pt')
    assert_nested_equal(score900,proposal['score'])
    if not (directory/'restore-check.json').exists():write_json(directory/'restore-check.json',dict(
        last_epoch_replayed_exactly=True,model=True,adam=True,training_rng=True,old_query_scores=True,
        updates=900,source_checkpoint='epoch099.pt',not_selected_checkpoint=True))
    # A purpose-separated seed fixes an independent second noise draw. Shared
    # across bandwidths within each seed/task, but independent of the old noise.
    noise_seed=int.from_bytes(hashlib.sha256(f'tsba-r2-second-noise|{seed}|{domain}'.encode()).digest()[:8],'little')%(2**63-1)
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(noise_seed),dtype=x.dtype)
    if not (directory/'second-noise.pt').exists():atomic_tensor(directory/'second-noise.pt',dict(seed=noise_seed,noise=noise))
    assert_nested_equal(load(directory/'second-noise.pt')['noise'],noise)
    if not (directory/'checkpoints/u00900.pt').exists():atomic_tensor(directory/'checkpoints/u00900.pt',initial)
    checkpoints=sorted((directory/'checkpoints').glob('u*.pt'))
    state=load(checkpoints[-1]);model,opt,rng=restore(state)
    history=copy.deepcopy(state['history']);best=state['best_validation_loss']
    best_epoch=state['selected_epoch'];best_state=copy.deepcopy(state['best_model_state'])
    for epoch in range(state['epoch']+1,800):
        train=epoch_update(model,opt,rng,x,h)
        val=validation(model,x,h,initial['validation_noise'])
        row=dict(epoch=epoch,train_loss=train,noise_validation_loss=val);history.append(row)
        rowpath=directory/f'losses/epoch{epoch:03d}.json'
        if rowpath.exists():assert json.loads(rowpath.read_text())==row
        else:write_json(rowpath,row)
        if val<best:best,best_epoch,best_state=val,epoch,copy.deepcopy(model.state_dict())
        updates=9*(epoch+1)
        if (epoch+1)%plan['score_recovery_epochs']==0 or updates in plan['score_updates']:
            checkpoint=dict(epoch=epoch,updates_completed=updates,model_state=copy.deepcopy(model.state_dict()),
                optimizer_state=copy.deepcopy(opt.state_dict()),training_rng_state=rng.get_state(),
                validation_rng_state=initial['validation_rng_state'],torch_rng_state=torch.random.get_rng_state(),
                validation_noise=initial['validation_noise'],history=copy.deepcopy(history),
                selected_epoch=best_epoch,best_validation_loss=best,best_model_state=best_state,
                sigma=h,config={**initial['config'],'score_epochs':800})
            atomic_tensor(directory/f'checkpoints/u{updates:05d}.pt',checkpoint)
        if rss_bytes()>plan['resources']['max_worker_rss_bytes']:raise MemoryError('score RSS limit')
    previous=None;rows=[]
    empirical=EmpiricalScore(x,h)(bounds.points)
    for updates in plan['score_updates']:
        state=load(directory/f'checkpoints/u{updates:05d}.pt');model,_,_=restore(state)
        with torch.no_grad():value=model(bounds.points)
        metrics=comparison(empirical,value,bounds.normals,sigma0)
        metrics.update(updates=updates,primary_noise_loss=state['history'][-1]['noise_validation_loss'],
            second_noise_loss=validation(model,x,h,noise),sigma0=sigma0,h=h,
            adjacent_request_rmse_sigma=(float(((metrics['learned_request']-previous)/sigma0).square().mean().sqrt())
                if previous is not None and len(bounds) else None))
        previous=metrics['learned_request']
        path=directory/f'static/u{updates:05d}.pt'
        if path.exists():assert_nested_equal(load(path),metrics)
        else:atomic_tensor(path,metrics)
        rows.append({k:v for k,v in metrics.items() if not isinstance(v,torch.Tensor)})
    if not (directory/'history.json').exists():write_json(directory/'history.json',history)
    result=dict(**spec,status='complete',endpoints=rows,best_DSM_epoch=best_epoch,
        best_DSM_updates=(best_epoch+1)*9,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=rss_bytes())
    write_json(directory/'complete.json',result);return result
