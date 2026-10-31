"""Aggregate the H-only replication using unchanged D1/D2 research gates."""
import argparse
import csv
import json
from pathlib import Path
import time
from p0_summarize import summarize,save


def report(stage):
    stage=Path(stage)
    summarize(stage)
    summary=json.loads((stage/'results_summary.json').read_text())
    settings=json.loads((stage/'experiment_manifest.json').read_text())['settings']
    old=Path(settings['d34_stage'])
    expected={}
    for seed in range(3405,3410):
        for line in (old/'logs'/str(seed)/'signature_target_blocks.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['arm']=='H':expected[(row['task'],seed,row['snapshot_step'],row['block_id'])]=row
    maxerr=0.;n=0
    for row in csv.DictReader((stage/'gradient_blocks.csv').open()):
        if row['arm']=='TRUE':
            prior=expected[(row['task'],int(row['seed']),int(row['snapshot_step']),int(row['block_id']))]
            for key in ['c_A','c_B','e_A','gradient_norm']:
                maxerr=max(maxerr,abs(float(row[key])-prior[key]))
            assert row['candidate_id']=='V2_H_ONLY' and row['signature']=='H'
            n+=1
    assert n==360 and maxerr<1e-10,(n,maxerr)
    summary.update(candidate_id='V2_H_ONLY',H_replay_blocks=n,D4_H_max_metric_difference=maxerr,
        candidate_selected_after_observing_D4=True,exploratory_development_replication=True)
    save(stage/'results_summary.json',summary)
    d1,d2=summary['D1'],summary['D2']
    rows1='\n'.join(f"| {r['task'].replace('_to_','→')} | {r['c_TRUE']:.4f} | {r['c_PERM']:.4f} | {r['c_RAND']:.4f} | {r['advantage_PERM']:.4f} | {r['advantage_RAND']:.4f} | {r['PERM_seed_wins']}/5、{r['RAND_seed_wins']}/5 | {'通过' if r['passed'] else '未通过'} |" for r in d1)
    rows2='\n'.join(f"| {r['task'].replace('_to_','→')} | {r['R_Q']*100:.6f} | {r['advantage_PERM']*100:.6f} | {r['advantage_RAND']*100:.6f} | {r['oracle_extra']*100:.6f} | {r['seed_wins']}/5 | {'通过' if r['passed'] else '未通过'} |" for r in d2)
    proceed=summary['eligible_for_D9']
    text=f'''# H-only：真实标签能否超过乱标签与随机反馈

**D1：{'通过' if summary['D1_passed'] else '未通过'}；D2：{'通过' if summary['D2_passed'] else '未通过'}；{'满足进入完整训练的筛选条件，但需另行登记训练方案' if proceed else '不建议进入完整训练'}。**

本轮只把相似度从HA改为H（图表示），仍保留目标Jacobian探测、源响应读出和JOINT校准λ=0.01。H不使用响应距离，不代表删除目标探测或源响应。

## 设置

- Mutagenicity，SEGA按边数构造的规模域0→1/2/3，非官方自然域。
- 五模型种子3405—3409，源fit779／query196／验证109；每块fit16、目标A16、独立B完整271。
- 同一45个任务×种子×快照引用（35个不同checkpoint）；快照为fit best、源SGD50／200步。
- D1共360块；D2用每快照前4块，共180块、6480次独立临时更新。不新增预训练、不连续适配。
- β=0.1，步长0.0001／0.0003／0.001；主结果固定0.0003，不事后挑选。FP64、CPU。
- 先对每个模型种子内各块、快照等权平均，再对五种子汇总。块、快照和三次null重复不当作新增种子。
- 相同θ、目标方向V、核与W下，PERM只打乱query标签，保留各图预测p；RAND在同一V空间随机取向。各用预定三次重复，不挑选。
- D1保留各臂原始幅度误差；D2将乱标签、随机、Oracle反馈匹配到真实反馈范数，源梯度保持相同。
- 所有W在开发标签读取前固定；目标A标签仅用于梯度诊断／Oracle，B标签仅作独立评价。最终测试标签未读取。

## D1：方向是否有真正的监督信息

余弦比较反馈与目标A真实经验梯度；越大越同向。乱标签／随机各先平均三次。最后一列为冻结门槛。

| 任务 | 真实标签余弦 | 乱标签余弦 | 随机余弦 | 对乱标签优势 | 对随机优势 | 配对种子胜数（乱／随机） | 门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
{rows1}

要求至少2/3任务同时满足：真实标签余弦>0，对两类null各至少0.05优势，各至少3/5种子为正，成对非零覆盖≥80%。结果有{sum(r['passed'] for r in d1)}/3任务通过。幅度误差及分快照、分种子结果见原始CSV，角度好不等于幅度准确。

## D2：迈一步，另一批图是否真的变好

所有数值为**额外降低B池CE的相对百分比**，正数为真实反馈更好，**不是balanced accuracy百分点**。“相对源训练”即S+Q对S；Oracle在同一A批取真实梯度并匹配反馈范数，仅作诊断上限参照。

| 任务 | 相对源训练 | 相对等幅乱标签 | 相对等幅随机 | 等幅Oracle相对源训练 | 真实反馈正收益种子 | 门槛 |
|---|---:|---:|---:|---:|---:|---|
{rows2}

要求至少2/3任务同时满足：相对源训练收益≥0.01%（R_Q≥1e-4），至少3/5种子收益>0，对乱标签／随机均值均有正优势，覆盖≥80%。结果有{sum(r['passed'] for r in d2)}/3任务通过。

## 如何理解、下一步怎么做

{'当前证据支持按预定门槛进入下一阶段；单步CE不是长期分类收益，需要独立完整训练方案。' if proceed else '当前H-only尚不能稳定证明真实源标签比乱标签和随机反馈更有用，因此暂停完整训练。不能仅凭它在D4优于HA就宣布修复成功。'}

H-only是看过D4后选出的候选，开发标签也已被反复观察；这轮是固定判据下的探索性复核，不是新盲测或显著性证明。源标签均值／残差总体偏差可能带来全局校准，不能把小幅结果直接归于逐图标签关系。Oracle取有限A批，能否推广到B也受批次噪声影响。

建议后续用D8已知条件规律检查读取器在友好条件下能否恢复正确方向；再分解源平均残差与中心化残差，确认逐图信息的增量，不先增加训练轮数。

![H-only方向与单步](figures/p0_primary.png)

## 完整性与复现

- 与D4 H臂360块逐项重放，最大梯度指标差{maxerr:.3e}。
- 非收敛校准块：{summary['nonconverged_calibration_blocks']}；原θ与buffer不变；临时步不保存为训练起点。
- 原方案及D1/D2门槛不改写；独立候选说明见candidate_protocol.md。
- 原始记录、TensorBoard、种子／快照汇总、代码快照与哈希随归档保存；数据和高维几何缓存留本机。
'''
    (stage/'结果解读.md').write_text(text)
    (stage/'decision.md').write_text(text)
    (stage/'hand_off.md').write_text(f"# H-only交接\n\nD1={summary['D1_passed']}，D2={summary['D2_passed']}，eligible_for_D9={proceed}。本轮无完整训练、无final标签。详见结果解读.md。\n")
    manifest=json.loads((stage/'experiment_manifest.json').read_text())
    ledger=json.loads((stage/'compute_ledger.json').read_text())
    (stage/'run_report.md').write_text(text+f'''\n## 执行信息\n\n- git_commit：{manifest['git_commit']}\n- plan_sha256：{manifest['plan_sha256']}\n- candidate_protocol_sha256：{manifest['candidate_protocol_sha256']}\n- hardware：{manifest['hardware']}\n- 正式执行耗时：{ledger['elapsed_wall_seconds']:.2f}秒；GPU时：0。\n- 命令：先运行`h_only_experiment.py prepare/d0/pilot/run --repo ... --stage ... --config configs/h_only_diagnostics.json`，再运行`h_only_report.py --stage ...`。\n- 对原方案的显式修订：仅新增H-only候选；原始P0及方案文件不覆盖。几何读取已封存P0无标签缓存，H读出逐块精确回放D4；D1/D2所有null、Oracle、步长、规模、门槛保持原样。\n''')
    save(stage.parent/'stage_3_done',dict(stage=3,scope='H-only D1/D2',started_at=ledger['started_at'],finished_at=time.time(),gpu_hours_consumed_so_far=0,status='complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);args=p.parse_args();report(args.stage)
