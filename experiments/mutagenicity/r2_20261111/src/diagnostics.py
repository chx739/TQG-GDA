"""Passive common-state first-round small steps reconstructed from proposals."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from tsba.boundary import BoundaryBatch
from tsba.classifier import ParameterMap
from tsba.problem import TsbaProblem
from experiments.mutagenicity.r2_20261111.src.adapt import settings
from experiments.mutagenicity.r2_20261111.src.common import write_json, load, atomic_tensor, model_from_frozen, jobs, job_key, pools, predictions


def first_round(root,seed,domain,lam,plan):
    root=Path(root);directory=root/f'diagnostics/seed{seed}_domain{domain}/lambda{lam}'
    if (directory/'complete.json').exists():return json.loads((directory/'complete.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    f=load(root/f'deps/source/seed{seed}/frozen.pt');model=model_from_frozen(f)
    base=ParameterMap(model).vector();torch_rng=torch.random.get_rng_state().clone()
    rows=[s for s in jobs(plan) if (s['seed'],s['domain'],s['lam'])==(seed,domain,lam)]
    payloads=[load(root/'trajectories'/job_key(s)/'round001.pt') for s in rows]
    reference=payloads[0];sigma0=reference['sigma0']
    failures=[dict(spec=s,reason=p['record'].get('reason')) for s,p in zip(rows,payloads)
        if p['record'].get('status')=='failed' or 'boundaries' not in p]
    if failures:
        # Projection may fail before any BoundaryBatch exists. Keep the whole
        # common-step diagnostic unavailable, rather than blocking OBS/evaluation
        # or silently dropping the failed seed from the registered grid.
        b=None
    else:b=BoundaryBatch(**reference['boundaries'])
    alpha=1/16;reason='predeclared_start';largest=0.;matrix=reference.get('matrix_sha256')
    for p in payloads:
        torch.testing.assert_close(p['before_vector'],base,rtol=0,atol=0)
        if failures:alpha=None;reason='first_round_numerical_failure';continue
        for key in ('points','pairs','normal_norm','normals','old_margin','input_indices'):
            torch.testing.assert_close(p['boundaries'][key],reference['boundaries'][key],rtol=0,atol=0)
        if p.get('matrix_sha256')!=matrix:raise AssertionError('First-round systems differ')
        if not len(b):alpha=None;reason='no_valid_boundary';break
        if p.get('delta') is None or not p['record'].get('solver',{}).get('valid',False):
            alpha=None;reason='unavailable_proposal';break
        largest=max(largest,float((p['D']@p['delta']).abs().max()))
    if alpha is not None:
        while alpha*largest>.1*sigma0 and alpha>1/32768:alpha/=2;reason='motion_guard_halving'
        if alpha*largest>.1*sigma0:alpha=None;reason='small_step_floor_exhausted'
    batch=reference['batch'];labels=json.loads((root/f'source-labels/seed{seed}.json').read_text())
    y=torch.tensor([labels[str(i)] for i in batch['source_ids']]);cfg=settings(seed,lam)
    ids=pools(root,seed,domain);paths=[]
    for spec,p in zip(rows,payloads):
        key=job_key(spec).split('/')[-1];dest=directory/(key+'.pt')
        if dest.exists():paths.append(dest.name);continue
        problem=TsbaProblem(model,f['x'][batch['source_ids']],y,BoundaryBatch(**p['boundaries']),cfg) if b is not None else None
        candidate=base if alpha is None else base+alpha*p['delta']
        # Functional evaluation then a fresh copy for prediction: parent stays fixed.
        view=model_from_frozen(f);ParameterMap(view).commit(candidate)
        actual=problem.movement(candidate).detach() if b is not None and len(b) else torch.empty(0,dtype=torch.float64)
        atomic_tensor(dest,dict(spec=spec,alpha=alpha,reason=reason,available=alpha is not None,
            model_state=view.state_dict(),predictions=predictions(view,f['x'],ids),
            actual_margin_movement=actual,predicted_margin_movement=(alpha*p['D']@p['delta'] if alpha is not None else None),
            metrics=problem.evaluate(candidate) if b is not None and len(b) else None,matrix_sha256=matrix))
        paths.append(dest.name)
    torch.testing.assert_close(ParameterMap(model).vector(),base,rtol=0,atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(),torch_rng,rtol=0,atol=0)
    result=dict(seed=seed,domain=domain,lam=lam,alpha=alpha,reason=reason,groups=len(rows),
        scope='all 10 estimator/bandwidth/update combinations, both signs, and one geometry control; same lambda',
        max_full_predicted_movement_sigma=largest/sigma0,passive_model_and_rng_verified=True,files=paths,failures=failures)
    write_json(directory/'complete.json',result);return result
