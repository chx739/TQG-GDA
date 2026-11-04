"""Independent artifact/normal-equation replay and descriptive A4 aggregation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from p0_experiment import read,write,ExperimentConfig,ResearchExperiment
from tqg_gda.assumption_diagnostics import ProbeLabelBoundary,metrics
from tqg_gda.optimization_diagnostics import bandwidth_kernel,effective_lambda,choose_kernel
from tqg_gda.utils import sha256


def export(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def audit_and_summarize(repo,stage):
    repo,stage=Path(repo),Path(stage);manifest=read(stage/'A4/manifest.json');cfg=manifest['config']
    done=read(stage/'A4/done.json');assert done['completed'] and done['units']==960 and done['logical_fits']==368640
    backend=ResearchExperiment(ExperimentConfig(str(repo),cfg['old_stage'],str(repo/'configs/p0_diagnostics.json')))
    audit_dir=stage/'A4/audit';audit_dir.mkdir(exist_ok=False)
    begun=time.process_time();max_normal=max_prediction=max_metric=max_cv=0.;counts=defaultdict(int)
    grouped=defaultdict(list);references=defaultdict(list);source_commit=manifest['git_commit']
    def unit_key(r):return (int(r['task'][-1]),int(r['fold']),r['train_domain'],int(r['n']),r['signature'])
    for seed in cfg['seeds']:
        out=stage/f'A4/{seed}';cv={};info=read(out/'done.json');assert info['git_commit']==source_commit
        with (out/'inner_cv.csv').open() as f:
            for r in csv.DictReader(f):
                key=unit_key(r)
                if key not in cv:cv[key]=np.zeros((2,4,3,4))
                co=cfg['lambda_conventions'].index(r['convention']);v=int(r['variant'])
                ci=cfg['bandwidth_multipliers'].index(float(r['bandwidth']));li=cfg['kernel_lambdas'].index(float(r['lambda_base']))
                cv[key][co,v,ci,li]+=float(r['nll'])/3;counts['inner_cv_rows']+=1
                assert r['git_commit']==source_commit
        labels={};features={};roles={};distances={}
        for domain in cfg['domains']:
            feature=Path(cfg['feature_stage'])/f'features/{seed}_{domain}.npz';role=read(feature.with_suffix('.json'))
            assert sha256(feature)==role['feature_sha256'];d=dict(np.load(feature));features[domain]=d;roles[domain]=role
            h,a=torch.from_numpy(d['h']),torch.from_numpy(d['a'].reshape(len(d['ids']),-1))
            hs=float(torch.quantile(torch.pdist(h[:196]).square(),.5).clamp_min(1e-16))
            aa=float(torch.quantile(torch.pdist(a[:196]).square(),.5).clamp_min(1e-16))
            dh=torch.cdist(h,h,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/hs
            da=torch.cdist(a,a,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/aa
            distances[domain]={'H':dh,'HA':dh+da}
            seal_path=out/f'{domain}_sealed_artifacts.json';seal=read(seal_path)
            with (out/'inner_cv.csv').open('rb') as f:digest=hashlib.sha256(f.read(seal['cv_prefix_bytes'])).hexdigest()
            assert digest==seal['cv_prefix_sha256']
            for item in seal['models']:assert sha256(out/item['path'])==item['sha256'];counts['sealed_files']+=1
            events=[json.loads(line) for line in (out/'label_access.jsonl').read_text().splitlines()]
            s=next(e for e in events if e['event']=='prediction_sealed' and e['sha256']==sha256(seal_path))
            b=next(e for e in events if e['event']=='label_access' and e['role']=='B_evaluation' and e['prediction_sha256']==s['sha256'])
            assert b['time']>=s['time'] and b['ids']==role['B']
            boundary=ProbeLabelBoundary(backend.task(domain),role['A'],role['B'],role['final'],audit_dir/'label_access.jsonl')
            boundary.seal(sha256(seal_path),s['models']);labels[domain]=boundary.read('B_evaluation',role['B'])
            np.savez_compressed(audit_dir/f'{seed}_{domain}_labels.npz',ids=role['B'],labels=labels[domain])
            counts['seals']+=1
        with (out/'curves.csv').open() as f:
            for key,iterator in itertools.groupby(csv.DictReader(f),unit_key):
                domain,fold,td,n,sig=key;rows=list(iterator);d=features[domain];role=roles[domain];pos={int(i):j for j,i in enumerate(d['ids'])}
                name=f'{domain}_{fold}_{td}_{n}_{sig}';path=out/f'models/{name}.npz';z=dict(np.load(path));m=read(path.with_suffix('.json'))
                max_cv=max(max_cv,float(np.abs(z['cv']-cv[key]).max()));assert np.allclose(z['cv'],cv[key],atol=1e-12,rtol=0)
                tr=[pos[int(i)] for i in z['train_ids']];ev=[pos[int(i)] for i in z['eval_ids']]
                source_labels=dict(zip(role['query'],d['source_y'].tolist()));extra=m['source_audit'];by=labels[domain]
                assert not set(z['train_ids'])&set(extra)
                if td=='S':assert set(z['train_ids'])<=set(role['query'])
                else:assert set(z['train_ids'])<=set(role['A'])
                assert not set(z['train_ids'])&set(role['B'])
                for ci,c in enumerate(cfg['bandwidth_multipliers']):
                    k=bandwidth_kernel(distances[domain][sig],c);kt=k[np.ix_(tr,tr)];ke=k[np.ix_(ev,tr)]
                    for co,convention in enumerate(cfg['lambda_conventions']):
                        for li,base in enumerate(cfg['kernel_lambdas']):
                            lam=effective_lambda(base,n,147,convention);alpha=z['alpha'][co,:,ci,li]
                            residual=z['training_labels']-d['p'][tr,1][None,:]
                            err=float(np.abs(alpha@kt+lam*alpha-residual).max());max_normal=max(max_normal,err)
                            pred=d['p'][ev,1][None,:]+alpha@ke.T
                            err=float(np.abs(pred-z['predictions'][co,:,ci,li]).max());max_prediction=max(max_prediction,err)
                for selected in m['selected']:
                    co=cfg['lambda_conventions'].index(selected['convention']);v=selected['variant']
                    choice=choose_kernel(cv[key][co,v],cfg['bandwidth_multipliers'],cfg['kernel_lambdas'])
                    assert choice==(selected['bandwidth_index'],selected['lambda_index'])
                for stat in m['readout_statistics']:
                    counts['joint_readout_observers']+=1
                    counts['joint_not_converged']+=not stat['joint_solver']['converged']
                for r in rows:
                    assert r['git_commit']==source_commit
                    co=cfg['lambda_conventions'].index(r['convention']);v=int(r['variant'])
                    ci=cfg['bandwidth_multipliers'].index(float(r['bandwidth']));li=cfg['kernel_lambdas'].index(float(r['lambda_base']))
                    raw=z['predictions'][co,v,ci,li];pool=r['pool']
                    if pool=='TRAIN':ix=np.arange(n);truth=z['training_labels'][v]
                    elif pool=='S_audit':ix=np.arange(n,n+len(extra));truth=np.array([source_labels[i] for i in extra])
                    else:assert pool=='B';ix=np.arange(n+len(extra),len(raw));truth=by
                    measured=metrics(truth,raw[ix])
                    for metric in ['nll','brier','outside','residual_mse','ba']:
                        if measured[metric] is None:assert r[metric]==''
                        else:max_metric=max(max_metric,abs(float(r[metric])-measured[metric]))
                    selected=m['selected'][co*4+v]
                    assert (r['selected']=='True')==(ci==selected['bandwidth_index'] and li==selected['lambda_index'])
                    group=(r['task'],seed,n,sig,td,r['convention'],v,pool)
                    if r['selected']=='True':grouped[group].append(r)
                    if ci==1 and li==1:references[group].append(r)
                    counts['curve_rows']+=1
                counts['models']+=1
        print(json.dumps(dict(event='A4_audit_seed',seed=seed,models=counts['models'],max_normal=max_normal,max_metric=max_metric)),flush=True)
    assert counts['models']==960 and counts['curve_rows']==230400 and counts['inner_cv_rows']==276480 and counts['seals']==15
    assert max(max_normal,max_prediction,max_metric,max_cv)<1e-8
    seedrows=[]
    for key,rs in sorted(grouped.items()):
        task,seed,n,sig,td,conv,v,pool=key;assert len(rs)==4
        avg=lambda k:float(np.mean([float(r[k]) for r in rs]))
        seedrows.append(dict(task=task,seed=seed,n=n,signature=sig,train_domain=td,convention=conv,variant=v,pool=pool,
            nll=avg('nll'),brier=avg('brier'),residual_mse=avg('residual_mse'),outside=avg('outside'),
            reference_nll=float(np.mean([float(r['nll']) for r in references[key]])),prior_nll=avg('prior_nll'),
            effective_dimension=avg('effective_dimension'),contrast_rkhs_norm=avg('contrast_rkhs_norm'),
            git_commit=source_commit,config_name=cfg['config_name'],hardware=manifest['hardware'],gpu_hours=0.,event_flags='clean'))
    export(stage/'A4/results.csv',seedrows)
    summary=[]
    keys=sorted({(r['task'],r['n'],r['signature'],r['train_domain'],r['convention'],r['pool']) for r in seedrows})
    for task,n,sig,td,conv,pool in keys:
        rs=[r for r in seedrows if (r['task'],r['n'],r['signature'],r['train_domain'],r['convention'],r['pool'])==(task,n,sig,td,conv,pool)]
        true=[r for r in rs if r['variant']==0];assert len(true)==5
        null=[np.mean([x['nll'] for x in rs if x['seed']==r['seed'] and x['variant']!=0]) for r in true]
        summary.append(dict(task=task,n=n,signature=sig,train_domain=td,convention=conv,pool=pool,
            nll=float(np.mean([r['nll'] for r in true])),nll_sd=float(np.std([r['nll'] for r in true],ddof=1)),
            prior_nll=float(np.mean([r['prior_nll'] for r in true])),shuffle_nll=float(np.mean(null)),
            reference_nll=float(np.mean([r['reference_nll'] for r in true])),
            selected_vs_reference_wins=sum(r['nll']<r['reference_nll'] for r in true),
            prior_and_shuffle_wins=int(sum(r['nll']<r['prior_nll'] and r['nll']<v for r,v in zip(true,null))),
            outside=float(np.mean([r['outside'] for r in true]))))
    export(stage/'A4/summary.csv',summary)
    audit=dict(passed=True,counts=dict(counts),normal_equation_max_error=max_normal,prediction_replay_max_error=max_prediction,
        metric_replay_max_error=max_metric,cv_replay_max_error=max_cv,final_labels_read=0,
        extra_logged_B_audit_labels=15*271,cpu_seconds=time.process_time()-begun)
    write(stage/'A4/audit.json',audit)
    write_report(stage,cfg,summary,audit)


def write_report(stage,cfg,summary,audit):
    counts=audit['counts']
    write(stage/'A4/results_summary.json',dict(status='A4_complete_A2b_A6_pending',summary=summary,audit=audit,scope='Frozen-feature diagnostics, not adaptation'))
    lines=['# A4-a核读取学习曲线：阶段结果','',
        '已完成三任务、五模型种子、四外折/三内折，368640次逻辑拟合。没有更新GNN。全部源端选择由源内CV完成；T→T是诊断，不能当UDA算法。',
        '完整960份模型/预测、15次封存、276480条CV记录与230400条曲线指标通过独立回放。下表只摘录n147，不能替代全部学习曲线。',
        '', '| 任务 | 签名 | 正则约定 | S→S NLL | S→T NLL | T→T NLL | S→T越界率 |', '|---|---|---|---:|---:|---:|---:|']
    for task in ['0_to_1','0_to_2','0_to_3']:
        for sig in cfg['signatures']:
            for conv in cfg['lambda_conventions']:
                def pick(td,pool):return next(r for r in summary if (r['task'],r['n'],r['signature'],r['convention'],r['train_domain'],r['pool'])==(task,147,sig,conv,td,pool))
                ss,st,tt=pick('S','S_audit'),pick('S','B'),pick('T','B')
                lines.append(f"| {task} | {sig} | {conv} | {ss['nll']:.4f} | {st['nll']:.4f} | {tt['nll']:.4f} | {100*st['outside']:.2f}% |")
    lines+=['','NLL越低越好。内层选参改变后的预测改善不直接等于反馈有效；下一步仍执行预登记LOCAL/KERNEL及A6。两种正则约定完整保留，不按B挑最好一列。',
        '源H的n147数据/模型在三任务共享，不能当三份独立源证据；目标B历史已被观察，结论属于开发集研究。',
        f"JOINT读出观察器未收敛记录：{counts['joint_not_converged']}/{counts['joint_readout_observers']}；保留原标志。",'',
        '详细指标见 results.csv、summary.csv；审计见 audit.json。A2-b、A6、连续适配和PROTEINS新复验尚未完成。']
    (stage/'A4/结果解读.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--report-only',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.report_only:
        stage=Path(a.stage);audit=read(stage/'A4/audit.json');assert audit['passed']
        cfg=read(stage/'A4/manifest.json')['config'];summary=[]
        with (stage/'A4/summary.csv').open() as f:
            for r in csv.DictReader(f):
                tags={'task','signature','train_domain','convention','pool'}
                integers={'n','selected_vs_reference_wins','prior_and_shuffle_wins'}
                summary.append({k:(v if k in tags else int(v) if k in integers else float(v)) for k,v in r.items()})
        write_report(stage,cfg,summary,audit)
        write(stage/'A4/postprocess_recovery.json',dict(reason='numpy int64 in report summary serialization',
            corrected='explicit Python integer conversion',existing_audit_passed=True,model_refits=0,new_label_reads=0))
    else:audit_and_summarize(a.repo,a.stage)
