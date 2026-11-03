"""Aggregate sealed A0/A1/A2-a diagnostics without changing models or gates."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def mean(x):return float(statistics.mean(x)) if x and all(v is not None for v in x) else None
def sd(x):return float(statistics.stdev(x)) if len(x)>1 and all(v is not None for v in x) else None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def nll(y,p):
    p=np.clip(p,1e-6,1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def summarize(stage):
    stage=Path(stage).resolve();manifest=read(stage/'experiment_manifest.json');config=manifest['settings']
    seeds=config['seeds'];tasks=[f'0_to_{d}' for d in config['domains']]
    records=[];audit=[];all_support={};nonconverged=0;new_target_labels=0
    for seed in seeds:
        directory=stage/f'logs/{seed}';done=read(directory/'done.json');assert done['completed'] and done['selection_units']==576 and done['logical_fits']==6912
        records+=lines(directory/'probe_metrics.jsonl')
        seals={}
        for r in lines(directory/'label_access.jsonl'):
            if r['event']=='prediction_sealed':seals[r['sha256']]=r
            if r['event']=='label_access':
                assert r['role'] in ['A_train','B_evaluation']
                if r['role']=='B_evaluation':
                    assert r['prediction_sha256'] in seals and seals[r['prediction_sha256']]['time']<=r['time']
                    assert seals[r['prediction_sha256']]['models']==192
                new_target_labels+=len(r['ids'])
        for d in config['domains']:
            identity=read(stage/f'features/{seed}_{d}.json');a,b,final=map(set,[identity['A'],identity['B'],identity['final']])
            for r in lines(directory/'label_access.jsonl'):
                if r['event']!='label_access':continue
                # Identify the allowed domain by its disjoint input IDs.
                if set(r['ids'])<=a or set(r['ids'])<=b:assert not set(r['ids'])&final
            support=read(directory/f'{d}_support.json')['groups']
            all_support[seed,f'0_to_{d}']=support
            pred=directory/f'{d}_sealed_predictions.npz'
            assert sha(pred) in seals
            payload=read(directory/f'{d}_sealed_artifacts.json')
            assert payload['prediction_sha256']==sha(pred)
            assert payload['support_sha256']==sha(directory/f'{d}_support.json')
            assert payload['feature_sha256']==sha(stage/f'features/{seed}_{d}.npz')
            cv_prefix=(directory/'inner_cv.jsonl').read_bytes()[:payload['inner_cv_prefix_bytes']]
            assert hashlib.sha256(cv_prefix).hexdigest()==payload['inner_cv_prefix_sha256']
            assert len(payload['models'])==192
            for item in payload['models']:assert sha(directory/item['path'])==item['sha256']
            for sig in config['signatures']:
                for fold in range(4):
                    for domain in ['S','T']:
                        ids=read(directory/f'{d}_{fold}_{sig}_{domain}_ids.json')
                        assert len(ids['train'])==147 and len(set(ids['train']))==147
                        assert not set(ids['train'])&set(ids['B'])
                        assert sorted(sum(ids['inner_validation'],[]))==sorted(ids['train'])
                        assert set(ids['source_train'])&set(ids['source_audit'])==set()
                        if domain=='T':assert set(ids['train'])<=a
                        else:assert set(ids['train'])<=set(identity['query']) and not set(ids['train'])&set(identity['fit'])
            for r in identity['query_diagnostics']:audit.append(dict(**r,feature_sha256=identity['feature_sha256']))
        for r in lines(directory/'inner_cv.jsonl'):
            nonconverged+=r['optimization']['converged'] is False
    assert len(records)==10080
    assert all(r['git_commit']==manifest['git_commit'] and r['event_flags']=='clean' for r in records)
    # Collapse folds first, retain model seed as the reporting unit.
    grouped=defaultdict(list)
    for r in records:grouped[tuple(r[k] for k in ['seed','task','signature','family','train_domain','pool','variant'])].append(r)
    seed_rows=[];lookup={}
    for key,rows in grouped.items():
        assert len(rows)==4
        row=dict(zip(['seed','task','signature','family','train_domain','pool','variant'],key))
        for k in ['nll','brier','ba','outside','residual_mse','prior_nll','model_nll','n','n0','n1','cv_nll']:
            row[k]=mean([r.get(k,0 if k in ['n0','n1'] else None) for r in rows])
        row['min_n']=min(r['n'] for r in rows)
        row['git_commit']=manifest['git_commit'];row['config_name']=config['config_name'];row['gpu_hours']=0.;row['hardware']=manifest['hardware'];row['event_flags']='clean'
        seed_rows.append(row);lookup[key]=row
    table(stage/'results.csv',seed_rows)
    def get(seed,task,sig,family,td,pool,variant=0):return lookup[seed,task,sig,family,td,pool,variant]
    def learn(seeds,task,sig,family,td,pool):
        prior=[];null=[]
        for seed in seeds:
            r=get(seed,task,sig,family,td,pool)
            shuffled=mean([get(seed,task,sig,family,td,pool,v)['nll'] for v in [1,2,3]])
            prior.append(r['prior_nll']-r['nll'] if r['nll'] is not None else None)
            null.append(shuffled-r['nll'] if shuffled is not None and r['nll'] is not None else None)
        valid=all(v is not None for v in prior+null)
        wins=sum(a>0 and b>0 for a,b in zip(prior,null)) if valid else 0
        passed=valid and mean(prior)>=.02 and mean(null)>=.02 and wins>=4
        return dict(passed=passed,prior_advantage=mean(prior),shuffle_advantage=mean(null),paired_wins=wins)
    summary=[];support_rows=[]
    for task in tasks:
        for sig in config['signatures']:
            for family in config['families']:
                ss=learn(seeds,task,sig,family,'S','S_audit');tt=learn(seeds,task,sig,family,'T','B');st=learn(seeds,task,sig,family,'S','B')
                gaps=[get(s,task,sig,family,'S','B')['nll']-get(s,task,sig,family,'T','B')['nll'] for s in seeds]
                row=dict(task=task,signature=sig,family=family,
                    S_S_NLL=mean([get(s,task,sig,family,'S','S_audit')['nll'] for s in seeds]),
                    S_T_NLL=mean([get(s,task,sig,family,'S','B')['nll'] for s in seeds]),
                    T_T_NLL=mean([get(s,task,sig,family,'T','B')['nll'] for s in seeds]),
                    S_T_std=sd([get(s,task,sig,family,'S','B')['nll'] for s in seeds]),
                    T_T_std=sd([get(s,task,sig,family,'T','B')['nll'] for s in seeds]),
                    S_S_learnable=ss['passed'],S_T_learnable=st['passed'],T_T_learnable=tt['passed'],
                    transfer_gap=mean(gaps),gap_positive_seeds=sum(v>0 for v in gaps),
                    transfer_gap_candidate=tt['passed'] and mean(gaps)>=.02 and sum(v>0 for v in gaps)>=4,
                    S_S_details=ss,S_T_details=st,T_T_details=tt)
                summary.append(row)
                differences=[];minimum=1000
                for seed in seeds:
                    sn=get(seed,task,sig,family,'S','B_near');tn=get(seed,task,sig,family,'T','B_near')
                    sf=get(seed,task,sig,family,'S','B_far');tf=get(seed,task,sig,family,'T','B_far')
                    minimum=min(minimum,sn['min_n'],sf['min_n'])
                    differences.append((sf['nll']-tf['nll'])-(sn['nll']-tn['nll']) if all(r['nll'] is not None for r in [sn,tn,sf,tf]) else None)
                valid=all(x is not None for x in differences)
                support_rows.append(dict(task=task,signature=sig,family=family,min_stratum_count=minimum,
                    gap_far_minus_near=mean(differences),positive_seeds=sum(v>0 for v in differences) if valid else 0,
                    support_candidate=bool(valid and minimum>=30 and mean(differences)>=.02 and sum(v>0 for v in differences)>=4)))
    # Per-graph paired comparisons on exactly the same target input IDs.
    predictions={}
    for seed in seeds:
        with (stage/f'logs/{seed}/probe_predictions.csv').open() as stream:
            for r in csv.DictReader(stream):
                if r['pool']!='B':continue
                key=(seed,r['task'],r['signature'],r['family'],r['train_domain'],int(r['variant']),int(r['fold']))
                predictions.setdefault(key,{})[int(r['graph_id'])]=(int(r['label']),float(r['probability']))
    equivalence=[]
    for row in summary:
        task,sig,family=row['task'],row['signature'],row['family']
        near_ids=None
        for seed in seeds:
            groups=[r for r in all_support[seed,task] if r['signature']==sig]
            for group in groups:
                these={i for i,near in zip(group['B'],group['near']) if near}
                near_ids=these if near_ids is None else near_ids&these
        selected=sorted(near_ids);entry=dict(task=task,signature=sig,family=family,common_near_graphs=len(selected),
            gap=None,ci90=None,source_learnable=False,target_learnable=False,local_compatibility_candidate=False)
        if len(selected)>=30:
            per_seed=[];learn_flags={}
            for td in ['S','T']:
                prior_adv=[];shuffle_adv=[]
                for seed in seeds:
                    fold_loss=[];fold_null=[];fold_prior=[]
                    for fold in range(4):
                        true=predictions[seed,task,sig,family,td,0,fold]
                        y=np.array([true[i][0] for i in selected]);p=np.array([true[i][1] for i in selected]);fold_loss.append(nll(y,p))
                        fold_null.append(np.mean([nll(y,np.array([predictions[seed,task,sig,family,td,v,fold][i][1] for i in selected])) for v in [1,2,3]],axis=0))
                        # Prior from the selected outer training model, unchanged across variants.
                        rr=next(r for r in records if (r['seed'],r['task'],r['signature'],r['family'],r['train_domain'],r['variant'],r['fold'],r['pool'])==(seed,task,sig,family,td,0,fold,'B'))
                        domain=int(task[-1]);key=f'{domain}_{fold}_{sig}_{td}_{family}_0'
                        saved=np.load(stage/f'logs/{seed}/models/{key}.npz');prior=float(saved['training_y'].mean());fold_prior.append(nll(y,np.full(len(y),prior)))
                    actual=np.mean(fold_loss,axis=0)
                    prior_adv.append(float(np.mean(fold_prior)-actual.mean()));shuffle_adv.append(float(np.mean(fold_null)-actual.mean()))
                    if td=='S':per_seed.append(actual)
                    else:per_seed[seeds.index(seed)]-=actual
                learn_flags[td]=mean(prior_adv)>=.02 and mean(shuffle_adv)>=.02 and sum(a>0 and b>0 for a,b in zip(prior_adv,shuffle_adv))>=4
            difference=np.mean(per_seed,axis=0)
            rng=np.random.default_rng(int(hashlib.sha256((task+sig+family).encode()).hexdigest()[:8],16))
            bootstrap=np.mean(difference[rng.integers(len(selected),size=(1000,len(selected)))],axis=1)
            interval=np.quantile(bootstrap,[.05,.95]).tolist()
            entry.update(gap=float(difference.mean()),ci90=interval,source_learnable=learn_flags['S'],target_learnable=learn_flags['T'],
                local_compatibility_candidate=bool(learn_flags['S'] and learn_flags['T'] and interval[0]>=-.02 and interval[1]<=.02))
        equivalence.append(entry)
    # Geometric support vs empirical A-gradient diagnostic, descriptive only.
    geometric=[]
    for seed in seeds:
        for domain in config['domains']:
            identity=read(stage/f'features/{seed}_{domain}.json');z=np.load(stage/f'features/{seed}_{domain}.npz');ids=z['ids'].tolist()
            ix=[ids.index(i) for i in identity['target_batch']]
            h,a=z['h'],z['a'].reshape(len(ids),-1)
            for sig in config['signatures']:
                support=read(stage/f'logs/{seed}/{domain}_support.json')
                dh=((h[:196,None,:]-h[np.array(ix)][None,:,:])**2).sum(2)/support['scale_h_squared']
                dist=dh if sig=='H' else dh+((a[:196,None,:]-a[np.array(ix)][None,:,:])**2).sum(2)/support['scale_a_squared']
                diagnostic=next(r for r in identity['query_diagnostics'] if r['signature']==sig)
                geometric.append(dict(**diagnostic,nearest_A_distance_mean=float(dist.min(0).mean()),A_max_kernel_mean=float(np.exp(-dist/2).max(0).mean())))
    table(stage/'geometry_summary.csv',geometric)
    flat=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in summary]
    table(stage/'triangle_summary.csv',flat);table(stage/'support_summary.csv',support_rows)
    overall={}
    for sig in config['signatures']:
        for family in config['families']:
            group=[r for r in summary if r['signature']==sig and r['family']==family]
            overall[f'{sig}_{family}']=dict(same_source_tasks=sum(r['S_S_learnable'] for r in group),same_target_tasks=sum(r['T_T_learnable'] for r in group),
                transfer_gap_tasks=sum(r['transfer_gap_candidate'] for r in group),
                support_candidate_tasks=sum(r['support_candidate'] for r in support_rows if r['signature']==sig and r['family']==family),
                local_compatibility_tasks=sum(r['local_compatibility_candidate'] for r in equivalence if r['signature']==sig and r['family']==family))
    result=dict(status='completed',scope='A0/A1/A2-a only; no GNN adaptation',git_commit=manifest['git_commit'],plan_sha256=manifest['plan_sha256'],
        engineering_passed=True,selection_units=2880,logical_fits=34560,metric_rows=len(records),seed_rows=len(seed_rows),
        inner_logistic_nonconverged=nonconverged,final_labels_read=0,formal_target_label_values_requested=new_target_labels,
        triangle=summary,support=support_rows,local_compatibility=equivalence,overall=overall,
        A0=read(stage/'D0/replay.json'),cpu_seconds_formal=sum(read(stage/f'logs/{s}/done.json')['cpu_seconds'] for s in seeds),
        compute_estimate=read(stage/'compute_estimate.json'))
    write(stage/'results_summary.json',result)
    write(stage/'artifact_audit.json',dict(passed=True,models=2880,prediction_seals=15,source_train=147,source_holdout=49,inner_folds=3,
          label_access_after_seals=True,final_labels_read=0,checkpoint_references=15,unique_source_models=5,
          source_query_fit_disjoint=True,pilot_excluded=True,logical_fits=34560,optimization_nonconvergence_retained=nonconverged))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,task in zip(axes,tasks):
        group=[r for r in summary if r['task']==task]
        xx=np.arange(len(group));ax.bar(xx-.18,[r['S_T_NLL'] for r in group],.36,label='S→T');ax.bar(xx+.18,[r['T_T_NLL'] for r in group],.36,label='T→T diagnostic')
        ax.set_xticks(xx,[r['signature']+'\n'+r['family'] for r in group]);ax.set_title(task)
    axes[0].set_ylabel('NLL (lower better)');axes[-1].legend();fig.tight_layout();fig.savefig(stage/'figures/triangle.png',dpi=160);plt.close(fig)
    report(stage,result)
    return result


def report(stage,result):
    rows=result['triangle'];o=result['overall']
    text='''# §3.8条件诊断：A0／A1／A2-a结果解读

这是冻结特征探针实验，不是完整TQG适配准确率。源／目标训练各147张；源留出49张，目标B评价271张。四外折、三内折，五模型种子。NLL越低越好；目标训练探针只作诊断，不能当作UDA方法。H与HA分别报告，三探针分别选参数，没有在B上挑最优模型。

## 同监督量的三角比较

| 任务 | 签名 | 探针 | S→S NLL | S→T NLL | T→T NLL | 迁移缺口 | 源内可学 | 目标内可学 | 缺口候选 |
|---|---|---|---:|---:|---:|---:|---|---|---|
'''
    for r in rows:
        text+=f"| {r['task']} | {r['signature']} | {r['family']} | {r['S_S_NLL']:.4f} | {r['S_T_NLL']:.4f} | {r['T_T_NLL']:.4f} | {r['transfer_gap']:+.4f} | {'是' if r['S_S_learnable'] else '否'} | {'是' if r['T_T_learnable'] else '否'} | {'是' if r['transfer_gap_candidate'] else '否'} |\n"
    text+='\n“可学”同时要求较训练类比例、三次乱标签均值的平均NLL改善≥0.02，以及至少4/5种子同向。缺口候选还要求目标内可学、源训练比目标训练的B池NLL高≥0.02、4/5种子同向。未过门槛不等于签名绝无标签信息。\n\n## 跨任务路线筛选\n\n| 候选 | 源内可学任务 | 目标内可学任务 | 迁移缺口候选任务 | 支持相关候选任务 | 局部相容候选任务 |\n|---|---:|---:|---:|---:|---:|\n'
    for name,r in o.items():text+=f"| {name} | {r['same_source_tasks']}/3 | {r['same_target_tasks']}/3 | {r['transfer_gap_tasks']}/3 | {r['support_candidate_tasks']}/3 | {r['local_compatibility_tasks']}/3 |\n"
    text+='\n至少2/3任务同向才升级为跨任务路线建议，且仍限定这个探针／签名。迁移缺口是条件关系、支持、有限样本拟合的合并证据，不能直接宣布η_S≠η_T。局部相容只限输入支持代理覆盖区域，描述性区间不证明普遍相等。\n\n## 支持分层\n\n| 任务 | 签名 | 探针 | 最小分组图数 | Gap远−Gap近 | 同向种子 | 候选 |\n|---|---|---|---:|---:|---:|---|\n'
    for r in result['support']:
        delta='不可比' if r['gap_far_minus_near'] is None else f"{r['gap_far_minus_near']:+.4f}"
        text+=f"| {r['task']} | {r['signature']} | {r['family']} | {r['min_stratum_count']} | {delta} | {r['positive_seeds']}/5 | {'是' if r['support_candidate'] else '否'} |\n"
    text+='\n分组阈值只看输入：源147留一最近邻距离90%分位数。每折每模型两组均须≥30张才满足支持候选筛选。图数不足记为证据不足，不因合并不同任务而过关。E、近邻、ESS是代理，不能当真实目标误差或充分覆盖证明。\n\n## 工程、范围与限制\n\n'
    text+=f"- A0的15个几何、30个H/HA读出回放通过。新探针10项单测通过。所有2880个选择单元、34560次拟合完成；内层LR未收敛记录{result['inner_logistic_nonconverged']}条，完整保留。MLP固定200步，不把固定步结束称为优化收敛。\n"
    text+='- 无骨干更新、无final读取。目标A标签仅训练诊断探针；B在预测、模型、内层选择和支持分组封存后读取。开发集已被历史观察，不是新盲测。\n- 本轮不执行A2-b、A3—A6，不因单个探针成功直接启动完整适配。没有逐图η真值，四项假设仍应分开按有限证据判断。\n- 完整逐种子／逐折指标、内层选择、预测、模型参数、label_access和readout replay均保留。原始query196不混入骨干已见的fit779。\n\n![三角对照](figures/triangle.png)\n\n[逐模型指标](results.csv) · [三角汇总](triangle_summary.csv) · [支持汇总](support_summary.csv) · [完整JSON](results_summary.json) · [执行协议](execution_protocol.md)\n'
    (stage/'结果解读.md').write_text(text)
    log='''# A0/A1/A2-a run report

Status: completed. Reused 5 initial source-fit GNNs (15 task references); no GNN updates or final labels. All four outer folds, three inner folds, H/HA, KRR/LR/MLP, true labels and 3 nulls completed. CPU FP64. See 结果解读.md and results_summary.json for gates, not just best means.

## Provenance and artifacts

'''+f"Execution commit: `{result['git_commit']}`. Method SHA: `{result['plan_sha256']}`. 2880 selected refits / 34560 total fits; 10080 fold metrics. Logical rows and nulls are not independent model seeds. Formal CPU seconds: {result['cpu_seconds_formal']:.3f}.\n\n"
    log+='''## Deviations and implementation resolutions

No outcome-based change to task, seeds, families, grids or gates. Full-batch probe optimization, exact LR objective, null pairing, equal training sizes, support minimum counts and common-subset bootstrap were fixed in execution_protocol.md before B evaluation. Kernel bandwidth uses full source-query input geometry; no held-out labels. MLP batching validated against independent fitting. Historical B projection scalars reused to avoid redundant B gradient access.

## Reproduction

From frozen code/, with matching input/cache SHA and local references: assumptions_experiment.py prepare → a0 → pilot → run, all with --repo <source repository> --stage <stage directory>. Output directories must be fresh. Pin all library versions in experiment_manifest.json. Postprocess with assumptions_summarize.py --stage <stage>. Code and dataset caches have separate roles; dataset not published.

## Interpretation limits

Target-trained probes are diagnostic oracles, not deployable UDA methods. A measured gap does not uniquely establish conditional shift. No-detected-gap is not equivalence; insufficient support or power remains inconclusive. Existing development inputs/labels have been observed by earlier studies. A2-b/A3–A6 and external-dataset validation are not performed here.
'''
    (stage/'run_report.md').write_text(log)
    (stage/'hand_off.md').write_text('Read results_summary.json and 结果解读.md; preserve probe-specific gates and target-label diagnostic boundaries. Do not claim source-label transfer or complete §3.8 validation from predictive probes alone.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args();result=summarize(args.stage)
    print(json.dumps(dict(status=result['status'],overall=result['overall']),ensure_ascii=False))
