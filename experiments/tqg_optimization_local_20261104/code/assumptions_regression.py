"""Check generalized probe code against already-published Mutagenicity predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqg_gda.assumption_diagnostics import fit_grid, ProbeConfig, stable_seed


def run(repo,output):
    repo=Path(repo);archive=repo/'experiments/tqg_assumptions_20261103'
    z=dict(np.load(archive/'features/3405_1.npz'));ids=z['ids'].tolist();directory=archive/'logs/3405'
    old_predictions=dict(np.load(directory/'1_sealed_predictions.npz'))
    ns=196;h=torch.from_numpy(z['h']);a=torch.from_numpy(z['a'].reshape(len(ids),-1))
    dh=torch.cdist(h,h,compute_mode='donot_use_mm_for_euclid_dist').square()/torch.quantile(torch.pdist(h[:ns]).square(),.5).clamp_min(1e-16)
    da=torch.cdist(a,a,compute_mode='donot_use_mm_for_euclid_dist').square()/torch.quantile(torch.pdist(a[:ns]).square(),.5).clamp_min(1e-16)
    checks=[]
    for sig in ['H','HA']:
        role=json.loads((directory/f'1_0_{sig}_S_ids.json').read_text());tr=[ids.index(i) for i in role['train']]
        ev=[ids.index(i) for i in role['source_audit']+role['B']]
        x=z['h'] if sig=='H' else np.c_[z['h'],z['a'].reshape(len(ids),-1)]
        k=torch.exp(-(dh if sig=='H' else dh+da)/2).numpy()
        for family in ['KRR','LR','MLP']:
            key=f'1_0_{sig}_S_{family}_0';model=np.load(directory/f'models/{key}.npz')
            pred,_,_=fit_grid(family,x[tr],model['training_y'][None,:],z['p'][tr,1],x[ev],z['p'][ev,1],k[np.ix_(tr,tr)],k[np.ix_(ev,tr)],z['q'],[float(model['penalty'])],stable_seed('init',3405,0,'refit'),ProbeConfig(),paired=True)
            error=float(np.max(np.abs(pred[0,0]-old_predictions[key])))
            assert error<1e-8,(key,error)
            checks.append(dict(signature=sig,family=family,max_absolute_prediction_error=error,passed=True))
    Path(output).write_text(json.dumps(dict(passed=True,scope='six published selected predictions, no new target labels',checks=checks),indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);run(args.repo,args.output)
