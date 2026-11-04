"""Offline D1 target evaluation, strictly restricted to fixed development IDs."""
import json

import torch
from torch.nn import functional as F

from revision_common import Context, STAGE, Budget, read, atomic_json
from revision_evaluation import DevelopmentLabels
from revision_execution_diagnostics import dense_centered_jacobian, gradient
from tqg_gda.utils import sha256


def main():
    ctx=Context()
    out=STAGE/'diagnostics/D1'
    seal=read(out/'source_seal.json')
    if not seal['complete'] or seal['contract']['git_commit'] != ctx.commit:
        raise ValueError('Need compatible completed D1 source seal')
    for item in seal['artifacts']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Sealed D1 artifact changed')
    budget=Budget('D1_offline_target_evaluation')
    status='failed'
    try:
        for domain in (2,3):
            task=ctx.task(domain)
            labels=DevelopmentLabels(ctx,domain,task,out/'target_label_access.jsonl')
            for seed in ctx.plan['seeds']:
                model,state,path=ctx.source(domain,seed)
                params=tuple(model.parameters())
                encoder_mask=torch.cat([torch.full((p.numel(),),name.startswith('encoder.'),dtype=torch.bool)
                                        for name,p in model.named_parameters()])
                for block in range(20):
                    budget.check()
                    folder=out/f'0_to_{domain}'/str(seed)
                    target_path=folder/f'target_{block:02d}.json'
                    if target_path.exists():
                        if read(target_path)['commit']!=ctx.commit:
                            raise ValueError('Incompatible cached D1 target evaluation')
                        continue
                    cache=torch.load(folder/f'block_{block:02d}.pt',map_location='cpu',weights_only=False)
                    query=task.batch(cache['ids']['query'],'target')
                    held=task.batch(cache['ids']['held'],'target')
                    yh=labels(cache['ids']['held'])
                    zh=cache['target_heldout_logits']
                    residual=F.one_hot(yh,zh.shape[1]).to(zh)-zh.softmax(-1)
                    _,jac=dense_centered_jacobian(model,query)
                    true_grad=gradient(F.cross_entropy(model(held),yh),params)
                    base_ce=float(F.cross_entropy(zh,yh))
                    zero=float(residual.square().sum(-1).mean())
                    rows=[]
                    for pred in cache['predictions']:
                        dQ,dH=pred['d_query'],pred['d_held']
                        g=-torch.einsum('ncp,nc->p',jac,dQ)/len(dQ)
                        def cosine(mask=None):
                            l,r=(g,true_grad) if mask is None else (g[mask],true_grad[mask])
                            denominator=float(l.norm()*r.norm())
                            return float((l@r/denominator).clamp(-1,1)) if denominator>1e-12 else None
                        error=float((residual-dH).square().sum(-1).mean())
                        metrics=dict(target_residual_mse=error,zero_residual_mse=zero,target_residual_gain=zero-error,
                                     gradient_cosine=cosine(),gradient_cosine_encoder=cosine(encoder_mask),
                                     feedback_gradient_norm=float(g.norm()),
                                     direct_logit_CE_gain=base_ce-float(F.cross_entropy(zh+.1*dH,yh)))
                        rows.append(dict(key=pred['key'],family=pred['family'],direction=pred['direction'],
                                         n=pred['n'],requested_columns=pred['requested_columns'],ridge_rule=pred['ridge_rule'],
                                         selected_ridge=pred['ridge'],metrics=metrics))
                    atomic_json(target_path,dict(commit=ctx.commit,domain=domain,seed=seed,block=block,rows=rows))
                    print(json.dumps(dict(event='D1_offline_block',domain=domain,seed=seed,block=block)),flush=True)
        status='completed'
    finally:
        budget.finish(status)


if __name__=='__main__':
    main()
