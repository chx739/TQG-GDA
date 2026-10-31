import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics

root=Path('/home/hxchen/runs/2026-10-31-tqg-v2-diagnostic-plan')
stage=root/'stage3_execution'
read=lambda p:json.loads(p.read_text())
s=read(stage/'results_summary.json')
manifest=read(stage/'experiment_manifest.json')
ledger=read(stage/'compute_ledger.json')
d0=read(stage/'D0/engineering_checks.json')
rows=list(csv.DictReader((stage/'gradient_seed_summary.csv').open()))
d2=[r for r in s['D2']]
done=[read(stage/'logs'/str(seed)/'done.json') for seed in range(3405,3410)]
worker_seconds=sum(r['elapsed_seconds'] for r in done)
wall=ledger['elapsed_wall_seconds']
fmt=lambda x:f'{x:.4f}' if x is not None else 'NA'
tf=lambda x:'通过' if x else '未通过'
d1_table='| 任务 | TRUE余弦（均值±SD） | 打乱标签余弦 | 随机反馈余弦 | TRUE−打乱 | TRUE−随机 | D1 |\n|---|---:|---:|---:|---:|---:|---|\n'
for r in s['D1']:
 d1_table+=f'| {r["task"].replace("_to_","→")} | {fmt(r["c_TRUE"])} ± {fmt(r["cosine_SD"])} | {fmt(r["c_PERM"])} | {fmt(r["c_RAND"])} | {fmt(r["advantage_PERM"])} | {fmt(r["advantage_RAND"])} | {tf(r["passed"])} |\n'
d2_table='| 任务 | 真实反馈额外降CE（%） | 五种子正增益数 | 等幅度Oracle额外降CE（%） | 有害更新率（%） | D2 |\n|---|---:|---:|---:|---:|---|\n'
for r in d2:
 d2_table+=f'| {r["task"].replace("_to_","→")} | {r["R_Q"]*100:.6f} | {r["seed_wins"]}/5 | {r["oracle_extra"]*100:.6f} | {r["harmful"]*100:.2f} | {tf(r["passed"])} |\n'
details='| 任务 | t=0 TRUE余弦 | 源SGD50步 TRUE余弦 | 源SGD200步 TRUE余弦 |\n|---|---:|---:|---:|\n'
snapshot=list(csv.DictReader((stage/'gradient_snapshot_seed_summary.csv').open()))
for task in ['0_to_1','0_to_2','0_to_3']:
 vals=[statistics.mean(float(r['c_TRUE']) for r in snapshot if r['task']==task and int(r['snapshot_step'])==t) for t in (0,50,200)]
 details+='| '+task.replace('_to_','→')+' | '+' | '.join(fmt(v) for v in vals)+' |\n'

if not s['D1_passed']:
 interpretation='当前真实源标签没有在至少两个任务上同时达到对打乱／随机反馈的预定方向优势。不能据此把旧实验的小幅分类变化归因于稳定的监督迁移，也不能从这次有限批次诊断直接推论所有TQG变体无效。'
 next_step='下一步优先D3：先匹配有效正则，比较mean/ISO/joint；再用D4分开检查h、A、h+A签名。若源内模拟仍差，再执行D8已知条件梯度的合成实验。暂不扩batch、epoch或完整适配网格。'
elif not s['D2_passed']:
 interpretation='方向诊断达到了预定标准，但在另一批目标图上的单步收益未达到完整门槛。方向有信息与一步有实用幅度是两件事；先看同范数Oracle、源目标冲突和原始／等幅度对照，不能直接称持续适配有效。'
 next_step='下一步按D2的Oracle结果定位；D6检查反馈幅度／源目标混合，D3排除正则效应。改动仍需用同一D1/D2门槛重新验证，不直接进入D9。'
else:
 interpretation='当前版本同时达到D1/D2的预定内部筛选标准，支持继续验证其读取信号与单步作用；这不是长期分类收益或投稿就绪证明。'
 next_step='可按D3/D4厘清校准和签名贡献，冻结最多两个候选，再另按D9执行完整适配。当前P0不自动启动D9。'

white=f'''# P0实验结果白话解读

本轮D0：{tf(s['D0_passed'])}；D1：{tf(s['D1_passed'])}；D2：{tf(s['D2_passed'])}。{interpretation}

## 做了什么

- Mutagenicity的SEGA边数规模域0→1/2/3，非官方自然域。源fit779、query196、source validation109；源fit每步16，目标查询16。
- 复用5个fit预训练模型及配对源SGD的50/200步快照，共45个任务×种子×快照引用，实际35个不同文件。不重新预训练，不连续适配。
- 每个目标域原542个development ID按无标签哈希分A/B各271；A每次抽16，B每次完整评价271。B仍属于已观察的开发集，不是新盲测。final标签全程未取值。
- D1共360块；每块用同一θ/V/W对比真实源标签、3次只置乱标签、3次同空间等范数随机反馈、零反馈、Oracle和源梯度。
- D2取前4块，共180块；每块12种更新×3个学习率，共6480个独立临时单步。每臂从原θ开始，原模型及buffer不变。
- 以下按模型种子内先平均快照／块／置乱，再汇总5种子。±为模型种子样本SD，不把多个块当成独立种子。

## D0：先确认不是新诊断脚本算错

50项代数／状态检查通过，9次权限负测试拒绝；真实任务的CE梯度重构相对误差约1.3—1.4×10⁻¹⁵。实际普通SGD增量、源JVP、辅助梯度、固定Ω与KRR等价、无支持零反馈和统一缩放不变性均通过容差。工程通过只允许解释后续结果，不证明跨域迁移。

## D1：源标签给出的方向是否更有用

余弦越接近1越同向，0表示接近正交，负值意味着局部一阶上可能反向。打乱只改变Y与图的对应，预测p、V、核和W都不变；没有把整个残差乱排。

{d1_table}
预定门槛：至少2/3任务同时满足TRUE平均余弦>0、对两种null的余弦优势各≥0.05、两个比较各≥3/5种子为正、有效覆盖≥80%。正余弦本身不足以说明源标签比无关反馈有额外价值。

{details}
以上分阶段表仅用于解释，主判据仍等权使用三个快照，不事后只挑较好阶段。

## D2：实际迈一步，另一批目标是否改善

主学习率0.0003，反馈β0.1。这里“额外降CE”是相对同块源SGD多降低的B池CE，再除以更新前CE，乘100；**不是balanced accuracy百分点**。正数有帮助，负数拖后腿。Oracle用A标签的真实CE梯度，并缩放到与实际反馈同范数，只作诊断参考。

{d2_table}
预定门槛为R_Q≥0.0001，即表中额外降CE≥0.01%，至少2/3任务、各≥3/5种子为正，且真实反馈优于等幅度打乱和随机对照，覆盖≥80%。有害指比匹配源SGD的B池CE高超过1e-8；同时保存连续数值，不能只看这个二元事件。

同范数Oracle不是B池真实梯度，它来自A的16张图；其不改善B不能直接证明没有目标改进空间。0.0001／0.001学习率结果和所有逐块数值都在原始表中，不能用其中较好者替换主学习率。

## 进一步定位

- 加权反馈范数β||g_hat||/||g_S||在三个任务平均约31.4%／28.1%／27.1%，并非反馈几乎为零。源标签估计梯度的平均相对误差为2.407／1.846／1.429；零估计在非零真实梯度上的相对误差为1。因此不能只看执行了反馈或范数不小就认为监督有效。
- 单独反馈Q在0→1、0→2同样平均有害；同范数Oracle单独及混合更新平均都改善B。源损失混合未表现为本轮的首要解释。
- 使用已保存数据补充描述性检查：主学习率180个块的实际增量与一阶预测同号，三个任务中位相对余项约1.44%／0.45%／0.39%。这支持优先检查估计方向；不是对所有步长、模型和执行错误的排除。
- 结构代理误差与真实梯度相对误差的种子内Spearman相关没有稳定正向关系（见secondary_diagnostics.json），不能把代理目标当作已校准置信度。相关性只作描述，无独立样本显著性主张。

## 下一步

{next_step}

这是开发集上的探索性机制诊断；经验标签梯度含噪声，一步普通CE变化不能替代完整训练的balanced accuracy结果。没有新增的最终集分数。
'''
white+='\n![P0主要诊断](figures/p0_primary.png)\n\n左图：真实标签、打乱标签与随机反馈的平均余弦，误差线为五种子样本SD；右图：每个种子的额外B池相对CE改善，黑线为平均值。\n'
(stage/'结果解读.md').write_text(white)
report=f'''# Run Report

D0通过；D1{tf(s['D1_passed'])}，D2{tf(s['D2_passed'])}。{interpretation}

## Executed as planned

- 冻结原计划与哈希；D0→D1→D2，不运行后续阶段。
- 3任务×5模型种子×3快照×8块＝360个D1块；D2前4块共180，6480临时步、180个冻结参考。
- 无标签几何先封存，之后显式development Oracle；最终标签不读取；临时模型不保存或作为训练起点。
- FP64、CPU、5个单线程worker；共同输入、读取器、随机数、源fit梯度；原模型和buffer未变。
- 本轮新增6项单元测试及既有16项v2数学回归测试通过；D0另有50项验收及9个权限拒绝记录。
- 所有逐块指标、5种子汇总、TensorBoard、源模型SHA、代码历史与计划保持可追溯。pilot独立，不计正式样本。

## Deviations from plan

- 没有改变数据、种子、批次、臂、学习率、阈值和执行精度。GPU驱动不可用，按方案CPU默认配置执行。
- 现有TensorBoard缺失，补齐2.20.0；实际依赖版本固定在执行副本requirements.txt，不冒充原环境已具备。
- pilot单worker外推约{read(stage/'compute_estimate.json')['conservative_estimated_wall_seconds']:.1f}秒，正式5worker墙钟{wall:.1f}秒；外推未充分覆盖并发竞争／加载。未超8小时包上限；后续预算应用并发pilot改进计时，不改变本轮样本数。
- 使用functional独立参数张量执行临时步，D0用真实torch.optim.SGD核对执行等式；不保存数千个临时Oracle checkpoint。

## Failed runs

- 无正式失败作业、数值回退或种子删除。一次汇总入口的相对路径调用失败，保留aggregation_relative_path_failure.log；改用绝对stage路径成功汇总，原始实验与判据未变。主仓库已修复后续入口。有限但未收敛的校准块为{s['nonconverged_calibration_blocks']}/360，完整记录，未混称精确最优解。
- 研究判据未通过与运行错误分开；失败／不充分结果保留，不修改阈值。

## Compute consumed

- 规划硬上限8小时；GPU实际0小时。
- D1/D2正式墙钟{wall:.1f}秒（{wall/60:.2f}分钟）。5个单线程worker累计墙钟{worker_seconds:.1f}秒，约{worker_seconds/3600:.4f} CPU-worker小时（含I/O，非纯CPU user-time）。
- 固定pilot另计{read(stage/'logs/pilot_3405/done.json')['elapsed_seconds']:.3f}秒；D0／预检／文档归档不计入上述正式D1/D2墙钟。
- 每worker峰值RSS约{max(r['peak_memory_bytes'] for r in done)/1024**2:.1f}MiB；高维几何缓存在本地保留，公开指标与SHA，避免无限上传向量。

## Headline number

D1达到门槛任务{sum(r['passed'] for r in s['D1'])}/3；D2达到主学习率门槛任务{sum(r['passed'] for r in s['D2'])}/3。是否允许当前版本进入D9：{s['eligible_for_D9']}。

{d1_table}
{d2_table}

## Reproducibility manifest

- 冻结执行Git提交：{manifest['git_commit']}；公开代码副本与execution_history.bundle共同保存该历史。
- 源仓库实现提交：{read(stage/'code/CODE_PROVENANCE.json')['source_repository_commit']}。
- 计划SHA256：{manifest['plan_sha256']}；protocol中全部文档保持原始哈希。
- 硬件：{manifest['hardware']}；种子3405—3409；配置configs/p0_diagnostics.json。
- 输入：snapshot_manifest.json的45引用／35个独立文件，已有归档checkpoint按SHA核对，不重复发布源模型。
- 命令：在code目录依次运行p0_experiment.py prepare/d0/pilot/run --repo /home/hxchen/TQG-GDA --stage ../，再运行p0_summarize.py --stage /home/hxchen/runs/2026-10-31-tqg-v2-diagnostic-plan/stage3_execution；运行前OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1。
- 原始：gradient_blocks.csv、one_step_blocks.csv、logs/*/*.jsonl及TensorBoard；汇总：results.csv、results_summary.json；简明解释：结果解读.md。
- 限制：非盲测、非长期适配结论、非条件期望梯度证明、非统计显著性结论。
'''
(stage/'run_report.md').write_text(report)
(stage/'README.md').write_text('''# TQG v2 P0完整诊断归档

[结果白话解读](结果解读.md) · [完整执行报告](run_report.md) · [汇总JSON](results_summary.json) · [源快照与哈希](snapshot_manifest.json) · [原始D1](gradient_blocks.csv) · [原始D2](one_step_blocks.csv)

本归档只包含D0/D1/D2，不是新增的连续适配实验。最终标签未读取。protocol保存不可变原方案；code_snapshot为执行代码，execution_history.bundle可恢复实际执行Git历史。

源模型引用主仓库已有experiments/mutagenicity_target_batch196_20261028/checkpoints；未重复上传35个输入模型。geometry_cache只在原始本地run中保存，local_geometry_manifest.json给出文件SHA和大小，CSV中的geometry_cache路径相对原始source_run。公开数据包括完整指标与代码，不包含原始数据集或高维梯度缓存。
''')
now=datetime.now(timezone.utc).isoformat()
marker={'stage':3,'scope':'P0 D0/D1/D2 only','started_at':datetime.fromtimestamp(manifest['prepared_at'],timezone.utc).isoformat(),'finished_at':now,'gpu_hours_consumed_so_far':0.,'status':'completed','D1_passed':s['D1_passed'],'D2_passed':s['D2_passed']}
(root/'stage_3_done').write_text(json.dumps(marker,indent=2)+'\n')
meta=read(root/'run.yaml')
access_files=[stage/'D0/label_access.jsonl',*list((stage/'logs').glob('*/label_access.jsonl'))]
access_total=sum(len(event['ids']) for path in access_files for event in (json.loads(line) for line in path.read_text().splitlines()) if event['event']=='label_access')
meta.update(status='P0_executed',stage3_execution='P0 completed; D3-D9 not executed',target_labels_read=access_total,development_oracle_only=True,final_labels_read=0,final_labels_allowed=False)
(root/'run.yaml').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report_written':True,'D1_passed':s['D1_passed'],'D2_passed':s['D2_passed'],'wall_seconds':wall}))
