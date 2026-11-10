"""Replay P0 fixed endpoints and summarize paired budgets and scalar oracle."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import torch
import numpy as np
from tsba.config import Config
from .archive import write_json, digest, manifest
from .static_score import new_model,evaluate,audit_scores,schedules
from .summarize_replay import stats,flat,close_tree,write_csv
from .summarize_static_score import check_manifest
from .p0_oracle import analyze


def read(p):return json.loads(p.read_text())

def aggregate(rows,keys):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,items in grouped.items():
        row=dict(zip(keys,key))
        for field in items[0]:
            if field in keys or field=='seed':continue
            values=[r[field] for r in items]
            if all(isinstance(v,(float,int)) for v in values):row[field]=stats(values)
        result.append(row)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=args.run;out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    count=check_manifest(root);metrics=[];fits=[];oracles=[];gates=[];replays=0
    for seed in range(5):
        d=root/f'd2_seed{seed}';fixed=torch.load(d/'fixed-inputs.pt',weights_only=True)
        masks=torch.load(d/'common-masks.pt',weights_only=True);cfg=Config(**read(d/'config.json'))
        scale=float(fixed['coordinates']['scale']);sigma=fixed['baseline_sigma']
        sched=torch.load(d/'schedule.pt',weights_only=True);expected=schedules(2,seed,6400)
        for name in ('noise','validation_noise'):torch.testing.assert_close(sched[name],expected[name],rtol=0,atol=0)
        torch.testing.assert_close(sched['ids'],expected['ids'][8192],rtol=0,atol=0)
        previous=None
        for steps in (400,1600,6400):
            ck=torch.load(d/f'checkpoint-{steps}.pt',weights_only=True);arrays=torch.load(d/f'queries-{steps}.pt',weights_only=True)
            saved=read(d/f'metrics-{steps}.json')
            assert len(ck['train_history'])==steps and len(ck['validation_history'])==steps//4
            assert all(int(x['step'])==steps for x in ck['optimizer']['state'].values())
            best=min(ck['validation_history'],key=lambda r:r['noise_validation_loss'])
            assert ck['selected_update']==best['update']
            if previous:
                assert ck['train_history'][:len(previous['train_history'])]==previous['train_history']
                assert ck['validation_history'][:len(previous['validation_history'])]==previous['validation_history']
            previous=ck
            init=new_model(2,cfg)
            for k,v in init.state_dict().items():torch.testing.assert_close(v,ck['initial_state'][k],rtol=0,atol=0)
            fits.append(dict(seed=seed,updates=steps,selected_update=best['update'],validation_final=ck['validation_history'][-1]['noise_validation_loss'],validation_selected=best['noise_validation_loss'],training_last100_mean=np.mean([r['loss'] for r in ck['train_history'][-100:]])))
            for bank,q in fixed['queries'].items():
                a=arrays[bank]
                for label,state in [('learned','final_state'),('selected','selected_state')]:
                    m=new_model(2,cfg);m.load_state_dict(ck[state]);torch.testing.assert_close(evaluate(m,q['points']),a['score_'+label],rtol=0,atol=0);replays+=1
                values={k:a['score_'+k] for k in ('learned','selected','empirical','population','zero')}
                recomputed,_=audit_scores(q['points'],q.get('normals'),values,sigma,scale,cfg,masks.get(bank),q['points'][:,0]*scale+fixed['coordinates']['mean'][0])
                close_tree(recomputed,saved[bank])
                for label in ('learned','selected','empirical','zero'):
                    c=saved[bank]['comparisons'][label+'_vs_population']
                    row=dict(seed=seed,updates=steps,bank=bank,estimator=label,**flat(c))
                    row.update({'signal_'+k:v for k,v in saved[bank]['signals'].get(label,{}).items()});metrics.append(row)
                if 'normals' in q:
                    for label in values:
                        gamma=torch.sum(a['score_'+label]*q['normals'],dim=1)
                        zeta=torch.minimum(torch.maximum(cfg.tau*sigma*sigma*gamma,torch.full_like(gamma,-cfg.request_multiplier*sigma)),torch.full_like(gamma,cfg.request_multiplier*sigma))
                        torch.testing.assert_close(gamma,a['gamma_'+label],atol=0,rtol=0);torch.testing.assert_close(zeta,a['zeta_'+label],atol=1e-15,rtol=1e-15)
            gate=dict(seed=seed,updates=steps)
            for bank in ('initial_boundaries','auxiliary_planes'):
                c=saved[bank]['comparisons'];sg=saved[bank]['signals']
                gate[bank+'_normal_better_zero']=c['learned_vs_population']['normal_rmse_scaled']<c['zero_vs_population']['normal_rmse_scaled']
                gate[bank+'_request_better_zero']=c['learned_vs_population']['request_rmse_sigma']<c['zero_vs_population']['request_rmse_sigma']
                gate[bank+'_oracle_toward_target']=sg['population']['toward_target_boundary']==1
                gate[bank+'_learned_majority_toward_target']=sg['learned']['toward_target_boundary']>.5
            signs=saved['auxiliary_planes']['comparisons']['learned_vs_population']['common_sign']
            gate['both_sign_recalls_above_half']=all(signs[k]>.5 for k in ('positive_recall','negative_recall'))
            gate['passes_all']=all(v for k,v in gate.items() if k not in ('seed','updates'));gates.append(gate)
        record=torch.load(d/'oracle-records.pt',weights_only=True);oracle=read(d/'oracle.json')
        recalculated,curves,inputs=analyze(scale,sigma*scale,cfg,record['inputs']['finite_source'].numpy());close_tree(oracle,recalculated)
        for k,v in curves.items():torch.testing.assert_close(record['curves'][k],torch.as_tensor(v),atol=0,rtol=0)
        base=oracle['reference']['initial_ba']
        for variant in ('population_source','finite512_posterior','density_only'):
            a=oracle[variant];t=a if variant=='density_only' else a['trajectory']
            row=dict(seed=seed,variant=variant,raw_sigma=sigma*scale,scale=scale,initial_ba=base,target_bayes_ba=oracle['reference']['target_bayes_ba'],b30=t['at30']['b'],gain30_pp=100*(t['at30']['target_ba']-base),final_b=t['final']['b'],final_gain_pp=100*(t['final']['target_ba']-base),final_source_kl=t['final']['source_kl'],steps=t['final']['step'],stop_reason=t['stop_reason'])
            if variant!='density_only':
                row.update(equilibrium_b=a['equilibrium_b'],equilibrium_gain_pp=100*(a['equilibrium_ba']-base),equilibrium_residual=a['equilibrium_residual'])
                assert abs(a['equilibrium_residual'])<1e-10
                for required in a['required_lambda']:row['lambda_b'+str(required['b'])]=required['lambda_for_stationary_balance']
            oracles.append(row)
    paired=[]
    for row in metrics:
        if row['updates']==400:continue
        baseline=next(x for x in metrics if x['seed']==row['seed'] and x['updates']==400 and x['bank']==row['bank'] and x['estimator']==row['estimator'])
        paired.append({k:(v-baseline[k] if isinstance(v,(int,float)) and k not in ('seed','updates') else v) for k,v in row.items()})
    for name,rows in [('metrics',metrics),('fits',fits),('oracle',oracles),('gates',gates),('paired',paired)]:write_csv(out/(name+'.csv'),rows)
    write_json(out/'summary.json',dict(metrics=aggregate(metrics,('updates','bank','estimator')),paired=aggregate(paired,('updates','bank','estimator')),fits=aggregate(fits,('updates',)),oracle=aggregate(oracles,('variant',)),gates=aggregate(gates,('updates',))))
    write_json(out/'audit.json',dict(run_files_verified=count,checkpoint_bank_replays=replays,optimizer_counters_verified=15,paired_schedules_exact=5,oracle_recomputations=5,metrics_recomputed=60,independent_normal_and_request_checks=True,learned_closed_loop=False))
    plot(out,metrics,oracles)
    write_json(out/'manifest.json',manifest(out))


def plot(out,metrics,oracles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for j,(key,title,mult) in enumerate([('normal_rmse_scaled','Auxiliary normal RMSE (sigma units)',1),('common_sign.balanced','Auxiliary balanced sign agreement (%)',100)]):
        for label in ('learned','selected','empirical','zero'):
            ys=[];errs=[]
            for budget in (400,1600,6400):
                values=[r[key]*mult for r in metrics if r['updates']==budget and r['bank']=='auxiliary_planes' and r['estimator']==label]
                ys.append(np.mean(values));errs.append(np.std(values,ddof=1))
            axes[j].errorbar([400,1600,6400],ys,yerr=errs,marker='o',label=label,capsize=3)
        axes[j].set_xscale('log',base=4);axes[j].set_xticks([400,1600,6400],['400','1600','6400']);axes[j].set_xlabel('Training updates');axes[j].set_title(title);axes[j].legend(fontsize=8)
    for variant,label in [('population_source','Population CE'),('finite512_posterior','Finite posterior CE'),('density_only','Density only')]:
        vals=[r['final_gain_pp'] for r in oracles if r['variant']==variant]
        axes[2].bar(label,np.mean(vals),yerr=np.std(vals,ddof=1),capsize=3)
    axes[2].set_title('Scalar oracle final gain (percentage points)');axes[2].tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=170);fig.savefig(out/'overview.svg');plt.close(fig)

if __name__=='__main__':main()
