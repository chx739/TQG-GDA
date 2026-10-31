"""Render D3/D4 reports from frozen primary summaries and labeled diagnostics."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import statistics


def read(path):return json.loads(path.read_text())
def fmt(x):return f'{x:.4f}' if x is not None else 'NA'
def mark(x):return '通过' if x else '未通过'


def render(stage):
    stage=Path(stage).resolve();s=read(stage/'results_summary.json');manifest=read(stage/'experiment_manifest.json')
    extra=read(stage/'numerical_supplement.json');ledger=read(stage/'compute_ledger.json')
    step_seed=list(csv.DictReader((stage/'step_seed_summary.csv').open()))
    def setting(experiment,task,arm,alpha=None):
        return next(r for r in s['task_settings'] if r['experiment_id']==experiment and r['task']==task and r['arm']==arm and r['alpha']==alpha)
    def criteria(experiment,task):return next(r for r in s['criteria'] if r['experiment_id']==experiment and r['task']==task)
    tasks=['0_to_1','0_to_2','0_to_3'];kernels=['H','A','HA','CONST','HA_PERM']
    d3='| 任务 | MEAN／ISO余弦 | JOINT余弦 | JOINT−ISO | 同幅单步额外降CE（%） | D3 |\n|---|---:|---:|---:|---:|---|\n'
    for task in tasks:
        c=criteria('D3',task)
        d3+=f'| {task.replace("_to_","→")} | {fmt(setting("D3",task,"MEAN",.005)["c_A"])} | {fmt(setting("D3",task,"JOINT",.005)["c_A"])} | {fmt(c["cosine_advantage"])} | {c["norm_matched_B_advantage"]*100:.6f} | {mark(c["passed"])} |\n'
    d4='| 相似度核 | 0→1余弦 | 0→2余弦 | 0→3余弦 | 源内留出余弦 |\n|---|---:|---:|---:|---:|\n'
    for kernel in kernels:
        d4+='| '+kernel+' | '+' | '.join(fmt(setting('D4',task,kernel)['c_A']) for task in tasks+['source_audit'])+' |\n'
    gain='| 任务 | H原幅度 | HA原幅度 | H等范数 | HA等范数 | D4 |\n|---|---:|---:|---:|---:|---|\n'
    for task in tasks:
        vals=[]
        for scale in ['raw','norm_matched']:
            for arm in ['H','HA']:
                vals.append(statistics.mean(float(r['R_Q'])*100 for r in step_seed if r['experiment_id']=='D4' and r['task']==task and r['scale']==scale and r['arm']==arm))
        gain+=f'| {task.replace("_to_","→")} | '+' | '.join(f'{v:.6f}' for v in vals)+f' | {mark(criteria("D4",task)["passed"])} |\n'
    width='| 指标 | 0→1 | 0→2 | 0→3 |\n|---|---:|---:|---:|\n'
    for key,title in [('feedback_cosine_vs_reference','JOINT对匹配MEAN的反馈余弦'),('omega_deviation','Ω偏离均匀权重的Frobenius范数')]:
        width+='| '+title+' | '+' | '.join(fmt(setting('D3',task,'JOINT',.005)[key]) for task in tasks)+' |\n'
    white=f'''# D3校准对照与D4相似度诊断：结果解读

**D3：{mark(s['D3_passed'])}；D4：{mark(s['D4_passed'])}。** 复杂校准没有稳定增量，当前加入响应相似度也没有稳定优于只看表示。这里的“未通过”指冻结方案的研究门槛；主比较均已正常执行且收敛，不是工程崩溃。

## 统一设置与阅读方法

- Mutagenicity，SEGA边数规模域0→1/2/3，非官方自然域。源fit779、query196、validation109；fit每块16，目标A每块16，B池完整271。
- 复用P0的45个任务×种子×快照引用、实际35个文件，以及360份无标签几何缓存；不新增预训练或连续适配。
- 五模型种子3405—3409；快照为fit best、源SGD50/200步。先平均种子内快照／块，再汇总五种子；源内两折也不算额外模型种子。
- 全部FP64、普通SGD单步η0.0003、β0.1。每个临时更新从同一原θ开始，原模型／buffer不变。
- 余弦越接近1越同向，接近0说明接近正交；相对误差越小越好。表中“额外降CE（%）”=相对源SGD多降低的B池CE／更新前CE×100，**不是balanced accuracy百分点**。没有最终测试集分数。

## D3：先公平比较复杂校准和简单核回归

每个α∈{{0.001,0.005,0.01,0.05,0.1}}比较MEAN λ=α、ISO λ=2α、JOINT λ=2α。ISO是固定Ω=I/r的条件解，与MEAN代数等价。共360×15=5400逻辑配置；主α固定0.005，其他只作敏感性分析。

{d3}
预设要求：至少2/3任务的余弦增量≥0.03、各至少3/5种子为正、同范数独立B池增益为正、覆盖≥80%。本轮三个任务均未达完整门槛。ISO/MEAN最大相对W误差为{s['ISO_MEAN_max_relative_error']:.3e}，说明正则匹配对照算对了。

**不是“最大方向项没有生效”。** Ω在主配置几乎集中到一个方向，JOINT与匹配MEAN的反馈平均余弦约0.89—0.92，方向确实改变；但这个变化没有带来预定的真实任务增益。结构代理的最坏方向不自动等于标签任务需要的方向。

{width}
正则曲线显示，更强正则通常降低相对误差／反馈幅度，但余弦没有跨任务稳定转好；不能把“缩小错误反馈”称为成功恢复监督。所有α都公开，不挑事后最优值替换主比较。MEAN更简单，但当前HA+MEAN也未被证明能work。

![D3正则曲线](figures/regularization_curves.png)

## D4：表示相近、响应相近，到底有没有用

- H：只比较图表示h。
- A：只比较当前目标方向上的响应A。
- HA：原版两种距离合并。
- CONST：源query核值全为1，不按query图选择权重；目标Jacobian仍照常构造。
- HA_PERM：把源签名重新分配给源ID，标签和残差保持原ID；同时置换源源核两轴和源目标核源行，保证完整Gram一致。

五核都固定joint λ0.01。本轮不根据D3结果临时给不同核换读取器。

{d4}
主比较是HA对H。真实目标三个任务的同幅单步均未超过H；特别在0→3，加入响应后的平均方向余弦从0.1885降到0.0307。

{gain}
H等范数列与原幅度相同，因为H本身就是参考幅度。HA缩放到同块H范数后仍未获得增量，因而不能只用“HA反馈变小了”解释差别。三个任务均未达到D4预定门槛：余弦增量≥0.03、至少3/5种子为正、同幅B池改善为正、覆盖≥80%。

## 源内正对照也不强

源query196按输入ID哈希分98/98，两折交换bank与audit；audit内每块A16、B82。5种子×3快照×2折×4块=120块，模型只用原fit监督，query未用于这些快照的反馈训练。

源内H余弦0.0453、HA为0.0896；HA角度有弱增量，但同幅独立B池效果几乎持平，平均差约−2.89×10⁻⁸（相对CE量），没有形成强正对照。经验标签梯度含噪声；不能因此宣称条件期望梯度也必然不可恢复。

近邻标签一致率说明相似度并非完全无信息：源内H约62.5%、HA约62.8%，对应类比例基线约49.6%；真实源bank的HA约66%，基线约49.8%。**“邻居标签常相同”没有自动变成可靠的残差／梯度读取。** 图的标签、当前模型误差以及它们的跨域关系是不同问题。

源内bank98与真实bank196数量不同，源内和目标的差别只作机制定位，不构成严格匹配样本量的域因果比较。

![D4相似度诊断](figures/signature_comparison.png)

## 常数核对照带来的提醒

CONST的真实目标余弦比HA高，但在本轮追加读取阶段，它只使用源query标签总体比例与源预测总体偏差。二分类时，令u=目标响应平均、n=源bank大小，最优读出为W=c·u·1ᵀ，c=(r+1)/(n(r+1)+λ)。因此它读取的是源残差的总和；**只打乱固定query池中的query标签对应关系，不会改变这个总和**。源fit预训练仍已使用图与标签的对应；这里仅说明，CONST的追加反馈不能被解释为利用了query逐图标签关系。

CONST的原幅度单步收益很小；不能只看较好余弦就宣称分类有效。它提示后续需要区分“全局平均残差校准”与“逐图监督信息”各自的作用。

2400个D4逻辑求解中有68个CONST控制未达容差（67次迭代上限、1次回溯上限），均为真实目标组；H/A/HA/HA_PERM及D3主比较全部收敛。未收敛记录和标志完整保留，没有改写为成功。

补充的**无标签、二分类解析数值审计**显示，CONST读出相对精确解最大误差为{extra['CONST_max_relative_weight_error']:.3e}，最小矩阵余弦为{extra['CONST_min_weight_cosine']:.15f}。该检查不更换正式结果或收敛状态，也不影响主H/HA结论。

![原幅度与等幅度单步](figures/raw_vs_matched_steps.png)

## 下一步建议

1. 优先把H-only作为一个独立候选，按原D1/D2补齐真实／打乱标签／等范数随机及Oracle，而不是直接长期训练。H的单步更好不等于已经证明源监督有效。
2. 做D8已知条件规律的正负对照，分开“经验标签噪声很大”与“读取器在友好条件下也估不准”。目前源内诊断不足以充当强正对照。
3. 后续另登记一次全局／逐图残差分解：r_i=平均残差＋中心化残差，比较两部分的方向和独立单步。先确认逐图标签关系提供额外价值，再考虑D5局部bank／门控。
4. 暂不把最大方向项当作已证实贡献；需要保留时应提供增量证据。简化读取可减少复杂度，但不是修复迁移失配的保证。

本轮仅完成D3/D4，不自动运行上述后续项目，不读取final。结果属于已观察开发集上的探索性机制证据，不是新盲测、一般不可行证明或投稿就绪结论。
'''
    (stage/'结果解读.md').write_text(white)
    finished=[read(stage/'logs'/str(seed)/'done.json') for seed in range(3405,3410)]
    worker_seconds=sum(r['elapsed_seconds'] for r in finished)
    report=f'''# Run Report

D3{mark(s['D3_passed'])}、D4{mark(s['D4_passed'])}。主配置的复杂校准与响应签名均未获得预定增量；主比较全部收敛。详见结果解读.md。

## Executed as planned

- 原计划／数据／快照／P0缓存哈希固定；3任务、5模型种子、3快照，360真实目标块和120源内块。
- D3为5400逻辑校准配置；D4为1800真实目标、600源内梯度记录。原幅度／等范数实际功能参数更新共8400次，均独立从原θ开始。
- 每块完整封存全部W后才读取Oracle标签；development与源query-audit分别记日志；最终标签未取值。
- 源bank98和audit98原ID排序；A16、B82；源fit16、源侧快照canonical0→1；不重复计三个任务。
- D3/D4新增及核心回归28项测试通过，3个真实任务旧W/预测/响应精确回放。独立CONST解析观测器另有1项最优解一阶条件／收敛解测试通过。
- D3原JOINT0.01回放P0、ISO与MEAN每α等价逐块核对；未换目标方向、丢种子、减批次或改变门槛。

## Deviations from plan

- 原方案未细列D3单步子矩阵，本次在新标签读取前固定为D2前4块、主η、两幅度、全部15臂；D4五核均joint，不启动可选MEAN矩阵。执行细节见experiment_manifest.json，未改研究门槛。
- 正式solve_seconds含内部source分解，未逐次单列分解计时。补充在相同360块上无标签单worker测source eig，中央値{extra['source_eigh_median_seconds']:.6f}秒；这不是正式并发时长拆分或效率优势证明。
- 发现68个CONST控制未收敛后，另做二分类解析解核对；只作数值审计，不重跑或替换正式指标，不改变任何求解器上限。
- D4 HA真实目标与D3主JOINT的360个相同求解复用并明确标记；逻辑配置数与物理求解耗时分开。源几何构造器另外有120次内部HA计算，不计成新的独立条件。

## Failed runs

- 无正式异常退出、非有限回退或种子缺失。
- D3非收敛{s['D3_nonconverged']}/5400；D4非收敛{s['D4_nonconverged']}/2400，全部为CONST真实目标组。有限近似权重、gap和termination原样保存，不称精确最优。
- 研究判据未通过与运行失败分开。不能按alpha／snapshot事后选较好组改判。

## Compute consumed

- 规划8小时硬上限；正式D3/D4墙钟{ledger['elapsed_wall_seconds']:.1f}秒（{ledger['elapsed_wall_seconds']/60:.2f}分钟）。
- 五个单线程worker累计墙钟{worker_seconds:.1f}秒，约{worker_seconds/3600:.4f} CPU-worker小时（含I/O，不是纯CPU user-time）；GPU0小时。
- 固定pilot另计{read(stage/'logs/pilot_3405/done.json')['elapsed_seconds']:.3f}秒，补充无标签数值计时另计{extra['elapsed_seconds']:.3f}秒；工程验收与汇总／归档不计入正式墙钟。
- 最大worker RSS约{max(r['peak_memory_bytes'] for r in finished)/1024**2:.1f}MiB。

## Headline number

D3过门槛任务0/3，D4过门槛任务0/3；不据此进入D9。Ω已强烈专门化但收益不稳定，HA等范数单步在三个真实任务上均未超过H。

{d3}
{d4}
{gain}

## Reproducibility manifest

- 冻结执行提交：{manifest['git_commit']}；code_snapshot与execution_history.bundle保存完整历史。
- 源仓库实现提交：{read(stage/'code/CODE_PROVENANCE.json')['source_repository_commit']}。
- 原计划SHA256：{manifest['plan_sha256']}；protocol保持逐字节一致。
- 配置configs/d34_diagnostics.json；{manifest['hardware']}；种子3405—3409；依赖锁定见code_snapshot/requirements.txt。
- source模型45引用／35独立文件见snapshot_manifest；360份P0缓存及索引SHA见p0_cache_manifest，不重复发布模型／几何向量。
- code目录依次运行d34_experiment.py prepare/d0/pilot/run --repo /home/hxchen/TQG-GDA --stage ../，再运行d34_summarize.py --stage ../；线程环境OMP/MKL/OPENBLAS_NUM_THREADS均为1。
- 补充分析源代码与来源见analysis_code；正式结果代码版本不被后续报告／归档提交替换。
- 原始表：calibration_grid_blocks.csv、signature_target_blocks.csv、source_audit_blocks.csv、step_blocks.csv；配对与种子汇总同目录。results.csv每行按模型种子汇总，有代码／硬件／配置／事件标记。
- 开发标签已经观察；源内与目标bank数量不同；经验梯度含噪声；增量实验不替代候选D1/D2与长期分类验证。
'''
    (stage/'run_report.md').write_text(report)
    (stage/'README.md').write_text('# D3/D4完整诊断归档\n\n[白话结果](结果解读.md) · [执行报告](run_report.md) · [汇总](results_summary.json) · [原始D3](calibration_grid_blocks.csv) · [原始D4目标](signature_target_blocks.csv) · [源内审计](source_audit_blocks.csv) · [单步表](step_blocks.csv)\n\n不包含原始数据集、源checkpoint或P0高维几何缓存，按输入清单与SHA引用既有归档。全部最终标签未读取；没有持续适配模型。执行代码副本和Git bundle可恢复实际版本。\n')
    (stage/'hand_off.md').write_text('D3/D4均未达到预定增量门槛（真实任务各0/3）。联合Ω确实集中到单方向，反馈却未稳定更准；HA在同幅单步上未超过H。CONST更好余弦只涉及全局平均残差，且有68个有限未收敛控制，数值解析审计单列。优先H-only候选D1/D2及D8已知条件梯度，再考虑全局／逐图残差分解；不启动D9，不把本结果包装成长程无监督收益。\n')
    now=datetime.now(timezone.utc).isoformat()
    write_marker=dict(stage=3,scope='D3 and D4 only',started_at=datetime.fromtimestamp(manifest['prepared_at'],timezone.utc).isoformat(),
        finished_at=now,gpu_hours_consumed_so_far=0.,status='completed',D3_passed=s['D3_passed'],D4_passed=s['D4_passed'])
    (stage.parent/'stage_3_done').write_text(json.dumps(write_marker,indent=2)+'\n')
    meta=read(stage.parent/'run.yaml');meta.update(status='D3_D4_executed',D3_passed=s['D3_passed'],D4_passed=s['D4_passed'],final_labels_read=0)
    (stage.parent/'run.yaml').write_text(json.dumps(meta,indent=2)+'\n')
    shutil.copy2(stage.parent/'stage_3_done',stage/'stage3_marker.json')
    shutil.copy2(stage.parent/'run.yaml',stage/'run_record.json')
    print(json.dumps(dict(report_written=True,D3_passed=s['D3_passed'],D4_passed=s['D4_passed'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);render(parser.parse_args().stage)
