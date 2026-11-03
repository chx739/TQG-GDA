"""Read-only sensitivity summaries; does not alter gates or formal numerical results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time


def extend(stage):
    stage=Path(stage)
    seeds=list(csv.DictReader((stage/'one_step_seed_summary.csv').open()))
    grad=list(csv.DictReader((stage/'gradient_snapshot_seed_summary.csv').open()))
    step=list(csv.DictReader((stage/'one_step_snapshot_seed_summary.csv').open()))
    results=[];snapshots=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        rows=[r for r in seeds if r['task']==task and float(r['eta'])==.0003]
        results.append(dict(task=task,seed_wins={k:sum(float(r[k])>0 for r in rows)
            for k in ['R_Q','advantage_PERM','advantage_RAND']}))
        for t in [0,50,200]:
            g=[r for r in grad if r['task']==task and int(r['snapshot_step'])==t]
            s=[r for r in step if r['task']==task and int(r['snapshot_step'])==t and float(r['eta'])==.0003]
            snapshots.append(dict(task=task,snapshot_step=t,
                **{k:statistics.mean(float(r[k]) for r in g) for k in ['c_TRUE','c_PERM','c_RAND','c_B_TRUE']},
                **{k:statistics.mean(float(r[k]) for r in s) for k in ['R_Q','advantage_PERM','advantage_RAND','oracle_extra']}))
    data=dict(scope='Descriptive seed/snapshot stability; original D1/D2 gates unchanged',seed_wins=results,snapshots=snapshots)
    (stage/'stability_diagnostics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    table='\n'.join(f"| {r['task'].replace('_to_','→')} | {r['seed_wins']['R_Q']}/5 | {r['seed_wins']['advantage_PERM']}/5 | {r['seed_wins']['advantage_RAND']}/5 |" for r in results)
    st='\n'.join(f"| {r['task'].replace('_to_','→')} | {r['snapshot_step']} | {r['c_TRUE']:.4f} | {r['c_PERM']:.4f} | {r['c_B_TRUE']:.4f} | {100*r['R_Q']:.6f} | {100*r['advantage_PERM']:.6f} |" for r in snapshots)
    section=f'''## 进一步看“稳定”：单步过门槛，但不能只看均值

D2在0→2、0→3通过了原筛选门槛，这是相对原HA版本的积极结果：在这批冻结位置，真实源标签能够提供平均独立单步收益。**D1仍为0/3；D2为2/3。不能把本轮概括成“完全没有有用信号”，也不能说“稳定超过两类null已经成立”。**

D2原判据要求源训练之外的收益至少3/5种子为正，但对两类null只要求跨种子平均优势为正。额外公开下面的配对种子胜数作稳定性描述，不事后改门槛：

| 任务 | 胜过源训练 | 胜过等幅乱标签 | 胜过等幅随机 |
|---|---:|---:|---:|
{table}

0→2虽然平均优于乱标签，但只有2/5种子胜出，且种子3405贡献较大；0→3为3/5种子胜过乱标签、4/5胜过随机，仍不是普遍有效。0→3方向余弦真实0.1885、乱标签0.1668：逐图标签对应带来的优势仅0.0217，未达预设0.05。这里不是说两者严格等价。

### 分快照检查

以下每行是五个模型种子的平均，快照0代表fit best，50/200为源训练步。最后两列仍为相对CE百分比，主η=0.0003。D1每快照8块，D2每快照前4块，二者不逐块混作同一指标。

| 任务 | 快照 | 真实c_A | 乱标签c_A | 真实c_B | 对源训练额外降CE（%） | 对乱标签额外降CE（%） |
|---|---:|---:|---:|---:|---:|---:|
{st}

0→2的c_A偏弱而c_B为正，提示有限A批梯度与全B池方向可能不同；这是观察到的现象，不能仅凭这张表判定其唯一原因。0→3在200步快照上，对乱标签的单步优势变负，不能挑初始快照替代总体结果。

主步长下，等幅Oracle相对源训练在三个任务平均均有正收益，特别0→1实际反馈负、Oracle正，说明这个位置并非“任何额外目标反馈都无效”。Oracle仍依赖A标签，不能作为无监督训练结果。

综合：保留H-only作为较有希望的简化候选，暂不进入长期完整训练；先做D8已知条件规律控制，以及源平均残差／中心化残差分解，检查逐图监督关系是否提供稳定增量。

'''
    old='当前H-only尚不能稳定证明真实源标签比乱标签和随机反馈更有用，因此暂停完整训练。不能仅凭它在D4优于HA就宣布修复成功。'
    new='H-only的D2单步收益得到两任务支持，但D1未通过，且对乱标签的种子稳定性不足。暂停完整训练，同时保留这个局部积极结果；不能仅凭它在D4优于HA就宣布修复成功。'
    for name in ['结果解读.md','decision.md','run_report.md']:
        p=stage/name;t=p.read_text();assert '## 进一步看“稳定”' not in t
        t=t.replace(old,new).replace('## 如何理解、下一步怎么做',section+'## 如何理解、下一步怎么做')
        p.write_text(t)
    p=stage/'run_report.md';t=p.read_text();t+='''\n## 后处理记录\n\n首次汇总已生成数值表，但绘图输出目录未创建，导致绘图失败；创建figures后仅重跑汇总，原始实验未重跑、无种子删除。错误日志summary.log与成功重试summary_retry.log均保留。此为报告后处理错误，不伪装成训练重启；正式结果event_flags保持clean。\n\ncompute_ledger的started_at补自最早正式worker的RUN_INFO时间，另标明来源；原driver elapsed_wall_seconds保持99.07921075820923秒。\n\n补充稳定性解释来自h_only_interpretation.py，只读取既有CSV，未改变预设门槛、原始记录或主数值。\n''';p.write_text(t)
    with (stage/'logs/postprocessing_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='report_directory_missing_recovered',phase='postprocessing',experiment_rerun=False,raw_metrics_changed=False,logs=['summary.log','summary_retry.log'],time=time.time()))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();extend(a.stage)
