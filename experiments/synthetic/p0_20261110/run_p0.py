"""Extend the archived 2D/8192 score checkpoints and analyze a scalar oracle."""
from experiments.common.paths import repository_root
import argparse
import copy
import json
from pathlib import Path
import resource
import shutil
import time
import traceback
import torch
from tsba.config import Config
from tsba.score import denoising_loss
from tsba.utils import finite
from experiments.common.synthetic.archive import manifest, provenance, save_tensor, write_json, digest
from experiments.common.synthetic.run_full_source import verify_archive
from experiments.common.synthetic.static_score import schedules, new_model, evaluate, audit_scores, tensor_hash
from experiments.synthetic.p0_20261110.p0_oracle import analyze


def read(p):return json.loads(p.read_text())


def cell(parent, original, root, seed, log):
    old=parent/f'd2_seed{seed}'; out=root/old.name; out.mkdir()
    cfg=Config(**read(old/'config.json'))
    fixed=torch.load(old/'fixed-inputs.pt',weights_only=True)
    assert fixed['target'].shape==(8192,2)
    sigma=fixed['baseline_sigma']; scale=float(fixed['coordinates']['scale'])
    shutil.copy2(old/'fixed-inputs.pt',out/'fixed-inputs.pt')
    shutil.copy2(old/'config.json',out/'config.json')
    masks=torch.load(old/'common-masks.pt',weights_only=True)
    save_tensor(out/'common-masks.pt',masks)
    schedule=schedules(2,seed,updates=6400)
    previous=torch.load(old/'schedule.pt',weights_only=True)
    for n in (512,2048,8192):
        torch.testing.assert_close(schedule['ids'][n][:400],previous['ids'][n],rtol=0,atol=0)
    torch.testing.assert_close(schedule['noise'][:400],previous['noise'],rtol=0,atol=0)
    torch.testing.assert_close(schedule['validation_noise'],previous['validation_noise'],rtol=0,atol=0)
    save_tensor(out/'schedule.pt',dict(ids=schedule['ids'][8192],noise=schedule['noise'],validation_noise=schedule['validation_noise']))
    oldfit=old/'n8192_s1'
    checkpoint=torch.load(oldfit/'checkpoint.pt',weights_only=True)
    model=new_model(2,cfg); model.load_state_dict(checkpoint['final_state'])
    opt=torch.optim.Adam(model.parameters(),lr=cfg.score_lr); opt.load_state_dict(checkpoint['optimizer'])
    history=copy.deepcopy(checkpoint['train_history']); validations=copy.deepcopy(checkpoint['validation_history'])
    best=min(v['noise_validation_loss'] for v in validations)
    best_step=checkpoint['selected_update']; best_state=copy.deepcopy(checkpoint['selected_state'])
    initial=checkpoint['initial_state']
    shutil.copy2(oldfit/'checkpoint.pt',out/'checkpoint-400.pt')
    values=torch.load(oldfit/'per-query.pt',weights_only=True)
    save_tensor(out/'fixed-reference-scores.pt',{bank:{k:v for k,v in arrays.items() if k in ('score_empirical','score_population','score_zero')} for bank,arrays in values.items()})
    input_hash=digest(out/'fixed-inputs.pt')
    def observe(step,selected_state):
        selected=new_model(2,cfg).requires_grad_(False).eval(); selected.load_state_dict(selected_state)
        restored=new_model(2,cfg).requires_grad_(False).eval()
        restored.load_state_dict(torch.load(out/f'checkpoint-{step}.pt',weights_only=True)['final_state'])
        metrics={}; arrays={}
        for name,bank in fixed['queries'].items():
            q=bank['points']; normals=bank.get('normals')
            scores={k:values[name]['score_'+k] for k in ('empirical','population','zero')}
            scores.update(learned=evaluate(model,q),selected=evaluate(selected,q))
            torch.testing.assert_close(evaluate(restored,q),scores['learned'],rtol=0,atol=0)
            metrics[name],arrays[name]=audit_scores(q,normals,scores,sigma,scale,cfg,masks.get(name),
                q[:,0]*scale+fixed['coordinates']['mean'][0])
            if step==400:
                torch.testing.assert_close(scores['learned'],values[name]['score_learned'],rtol=0,atol=0)
        write_json(out/f'metrics-{step}.json',metrics); save_tensor(out/f'queries-{step}.pt',arrays)
    observe(400,best_state)
    try:
        for step in range(401,6401):
            ids=schedule['ids'][8192][step-1]; noise=schedule['noise'][step-1]
            loss=denoising_loss(model,fixed['target'][ids],sigma,noise);finite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();finite(*(p.grad for p in model.parameters()))
            opt.step();finite(*model.parameters())
            history.append(dict(update=step,loss=float(loss.detach())))
            if step%4==0:
                with torch.no_grad():
                    val=sum(float(denoising_loss(model,xb,sigma,eb))*len(xb) for xb,eb in
                            zip(fixed['target'][:512].split(128),schedule['validation_noise'].split(128)))/512
                finite(val);validations.append(dict(update=step,noise_validation_loss=val))
                if val<best:best,best_step,best_state=val,step,copy.deepcopy(model.state_dict())
            if step%400==0:log(f'seed={seed} update={step} DSM={float(loss.detach()):.8g} validation={val:.8g}')
            if step in (1600,6400):
                save_tensor(out/f'checkpoint-{step}.pt',dict(initial_state=initial,final_state=model.state_dict(),
                    optimizer=opt.state_dict(),selected_state=best_state,selected_update=best_step,updates=step,
                    train_history=history,validation_history=validations,dim=2,hidden=cfg.score_hidden,sigma=sigma))
                observe(step,best_state)
    finally:
        write_json(out/'training.json',dict(history=history,validation_history=validations))
        save_tensor(out/'last-live-state.pt',dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),completed_updates=len(history)))
    assert digest(out/'fixed-inputs.pt')==input_hash
    old_inputs=torch.load(original/f'd2_seed{seed}'/'inputs.pt',weights_only=True)
    torch.testing.assert_close(old_inputs['target'],fixed['target'][:512],rtol=0,atol=0)
    raw_source=(old_inputs['source'][:,0]*scale+fixed['coordinates']['mean'][0]).numpy()
    oracle,curves,inputs=analyze(scale,sigma*scale,cfg,raw_source)
    write_json(out/'oracle.json',oracle)
    save_tensor(out/'oracle-records.pt',{k:{kk:torch.as_tensor(vv) for kk,vv in v.items()} for k,v in dict(curves=curves,inputs=inputs).items()})
    write_json(out/'audit.json',dict(parent_checkpoint_sha256=digest(oldfit/'checkpoint.pt'),
        checkpoint400_copied_bitwise=digest(out/'checkpoint-400.pt')==digest(oldfit/'checkpoint.pt'),
        prefix_schedule_exact=True,queries_fixed=True,head_fixed=True,input_sha256=input_hash,
        optimizer_steps=[400,1600,6400],noise_sha256=tensor_hash(schedule['noise']),learned_closed_loop=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);repo=repository_root()
    write_json(root/'provenance.json',provenance(repo))
    shutil.copy2(repo/'experiments/synthetic/p0_20261110/p0-plan.json',root/'plan.json')
    parents={}
    for label,path,published in [('static',args.parent,'e2_static_score_20261110'),('original',args.original,'e1_e2_20261109')]:
        parents[label]=verify_archive(path,repo/f'experiments/synthetic/{published}/run-manifest.json')
    write_json(root/'parents-before.json',parents)
    start=time.perf_counter();failures=[]
    with (root/'run.log').open('x',buffering=1) as file:
        def log(s):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+s
            print(line,flush=True);file.write(line+'\n')
        for seed in range(5):
            try:
                cell(args.parent,args.original,root,seed,log)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4294967296:raise MemoryError('RSS limit')
                log(f'COMPLETE seed={seed}')
            except Exception as exc:
                failures.append(dict(seed=seed,error=repr(exc),traceback=traceback.format_exc()));log(f'FAILED seed={seed} {exc}')
        log(f'COMPLETE failures={len(failures)} seconds={time.perf_counter()-start:.2f}')
    for label,path,published in [('static',args.parent,'e2_static_score_20261110'),('original',args.original,'e1_e2_20261109')]:
        verify_archive(path,repo/f'experiments/synthetic/{published}/run-manifest.json')
    write_json(root/'parents-after.json',parents)
    write_json(root/'failures.json',failures)
    write_json(root/'completion.json',dict(seeds=5-len(failures),failures=len(failures),seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        newly_executed_training_updates=sum(max(0, len(read(p)["history"])-400) for p in root.glob("d2_seed*/training.json")),learned_closed_loop=False,target_evaluation_labels_read=False))
    write_json(root/'manifest.json',manifest(root))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
