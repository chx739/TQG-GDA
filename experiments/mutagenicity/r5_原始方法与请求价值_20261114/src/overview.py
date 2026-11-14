"""Generate cross-setting presentation from independently audited, sealed metrics."""
import json
import statistics
from .archive import SETTINGS, read_csv
from .shared import output

LABELS = {'mutagenicity_edge':'Mutagenicity 边数（历史）', 'mutagenicity_node':'Mutagenicity 节点数（新增）',
          'proteins_edge':'PROTEINS 边数（新增）'}


def number(value, scale=1., signed=False, precision=5):
    if value is None or value == '':
        return 'NA'
    return format(float(value)*scale, ('+' if signed else '') + f'.{precision}f')


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|'+'|'.join('---' for _ in header)+'|',
            *['| ' + ' | '.join(map(str,row)) + ' |' for row in rows], '']


def overview(experiment):
    result=experiment/'results'
    evidence=json.loads((result/'evidence.json').read_text())
    summaries={s:{(r['method_or_contrast'],r['scope']):r for r in json.loads((result/s/'summary.json').read_text())} for s in SETTINGS}
    lines=['# R5 原始 TSBA 与密度请求价值：三设置完整结果', '',
           '状态：已完成。以下分别呈现三个设置，不合并成一个跨设置总分。全部主终点为第 30 次尝试。', '',
           'B0 是源模型；B1 是 M00 零请求，保留持续源 CE、边界响应矩阵与数值保护；A0 是原始 M00＋学习 score。', '',
           '## 整体效果', '',
           '先在每个 seed 内等权平均 12 个方向，再对 5 个 seed 求均值、样本 SD 和胜/平/负。表中 Accuracy、BA、F1 的差为百分点，NLL 保持原量纲。', '']
    rows=[]
    for s in SETTINGS:
        r=summaries[s]
        rows.append([LABELS[s],*[f"{number(r[m,'Avg']['accuracy']['mean'],100)} ± {number(r[m,'Avg']['accuracy']['sd'],100)}" for m in ('B0','B1','A0')]])
    lines+=table(['设置','B0 Accuracy (%)','B1 Accuracy (%)','A0 Accuracy (%)'], rows)
    rows=[]
    for s in SETTINGS:
        for c in ('A0−B1','A0−B0','B1−B0'):
            r=summaries[s][c,'Avg'];a=r['accuracy']
            rows.append([LABELS[s],c,number(a['mean'],100,True),number(a['sd'],100),f"{a['wins']}/{a['ties']}/{a['losses']}",
                         number(r['balanced_accuracy']['mean'],100,True),number(r['macro_f1']['mean'],100,True),number(r['nll']['mean'],signed=True)])
    lines+=table(['设置','配对','Δ Accuracy','SD','胜/平/负','Δ BA','Δ Macro-F1','Δ NLL'],rows)
    lines+=['[全部逐方向表与曲线见各设置报告](#分设置材料)。[独立复算的总表](results/aggregate.csv)与[五个 seed 原始值](results/aggregate-seeds.csv)。', '',
            '![配对与种子](results/cross-setting.png)', '',
            '彩色点为五个 seed 各自的方向等权配对值，黑方块与误差条为均值±样本 SD。', '',
            '## 同状态旁路：请求本身是否有价值', '',
            '第 1/30 次尝试之前，在 A0 实际状态上使用相同边界、源批次、矩阵与共同步长；候选不提交。NLL 差小于零有利，Accuracy 差大于零有利。', '']
    rows=[]
    for s in SETTINGS:
        all_rows=read_csv(result/s/'side_summary.csv')
        selected={(r['attempt'],r['contrast'],r['metric']):r for r in all_rows if r['scope']=='Avg' and r['pool']=='target'}
        for attempt in ('1','30'):
            for c,label in (('normal_minus_0','正常−零'),('normal_minus_-1','正常−反向')):
                n=selected[attempt,c,'nll'];a=selected[attempt,c,'accuracy']
                rows.append([LABELS[s],attempt,label,number(n['mean'],signed=True,precision=8),f"{n['wins']}/{n['ties']}/{n['losses']}",
                             number(a['mean'],100,True),f"{n['available_pairs']}/{n['planned_pairs']}"])
    lines+=table(['设置','尝试前','配对','Δ NLL','NLL 胜/平/负','Δ Accuracy','有效任务×seed'],rows)
    lines+=['旁路完整均值只在每个 seed 的 12 个方向均有可用配对时计算；不完整时保留 NA 和覆盖分母。', '',
            '## 执行幅度与边界支持', '',
            '以下为可用旁路块的描述性均值，不把锚点当独立重复。请求幅度为 RMS(ζ)/σ；请求增量执行量为 RMS(αD(δ+−δ0))/σ；总执行量为 RMS(αDδ+)/σ。', '']
    motion=[]
    for s in SETTINGS:
        sides=read_csv(result/s/'side.csv')
        for attempt in ('1','30'):
            rows=[r for r in sides if r.get('pool')=='target' and r.get('direction')=='1' and r['attempt']==attempt and r['status']=='available']
            row=dict(setting=s,attempt=int(attempt),available_blocks=len(rows),planned_blocks=60)
            for field in ('alpha','request_rms_sigma','increment_motion_rms_sigma','total_motion_rms_sigma','actual_vs_zero_rms_sigma',
                          'normalized_margin_change_rms','root_rmse_vs_proxy_sigma'):
                values=[float(r[field]) for r in rows if r[field]!='']
                row[field]=statistics.mean(values) if values else None
            row['single_roots']=sum(int(r['single_root_count']) for r in rows)
            row['relocation_anchors']=sum(int(r['relocation_roots']) for r in rows)
            motion.append(row)
    output.write_csv(result/'side-motion-summary.csv',motion)
    lines+=table(['设置','尝试前','块数','共同 α','请求 RMS/σ','请求增量 RMS/σ','总执行 RMS/σ','单根/全部锚点'],
                  [[LABELS[r['setting']],r['attempt'],f"{r['available_blocks']}/60",number(r['alpha'],precision=7),number(r['request_rms_sigma'],precision=7),
                    number(r['increment_motion_rms_sigma'],precision=7),number(r['total_motion_rms_sigma'],precision=7),
                    f"{r['single_roots']}/{r['relocation_anchors']}"] for r in motion])
    lines+=['[更多几何字段](results/side-motion-summary.csv)。归一化 margin 变化不是精确边界距离；重定位误差只在单根上定义，失败锚点保留在分母中。', '',
            '## 源域表现与固定概率缩放诊断', '',
            '固定 B0 logit 倍率 0.5/0.75 仅作概率诊断，没有用于选模或调参。每个池内仍是方向等权、seed 等权。', '']
    rows=[]
    for s in SETTINGS:
        endpoint=read_csv(result/s/'endpoints.csv')
        scaled=read_csv(result/s/'calibration.csv')
        for pool in ('target','source_fit','source_validation'):
            def avg(records,method,metric,attempt):
                chosen=[float(r[metric]) for r in records if r['method']==method and r['pool']==pool and r['attempt']==attempt]
                if len(chosen)!=60:
                    raise ValueError('Missing pooled endpoint')
                return statistics.mean(chosen)
            rows.append([LABELS[s],pool,*[number(avg(endpoint,m,'nll','30')) for m in ('B0','B1','A0')],
                         *[number(avg(scaled,m,'nll','0')) for m in ('B0_scale_0.5','B0_scale_0.75')]])
    lines+=table(['设置','池','B0 NLL','B1 NLL','A0 NLL','B0×0.5 NLL','B0×0.75 NLL'],rows)
    lines+=['## 完整性与存档', '',
            '总计 360 条轨迹、10,800 次适配尝试，其中新增 240 条/7,200 次，历史复用 120 条/3,600 次。'
            '全部 11,160 个状态预测（含各轨迹起点）已重载核验；没有工程失败或数值失败任务被删除。', '']
    rows=[]
    for entry in evidence['settings']:
        r=entry['replay']
        rows.append([LABELS[entry['setting']],r['sources'],r['scores'],r['trajectories'],r['attempts'],r['predictions_replayed'],
                     f"{r['side_statuses'].get('available',0)}/120",r['side_predictions_replayed']])
    lines+=table(['设置','源模型','score','轨迹','尝试','重载状态','可用旁路','重载候选'],rows)
    lines+=['执行代码：`'+evidence['execution_commits'][0]+'`。运行参数和方法没有在观察结果后改变。新增发布工具只读取封存后的记录，独立复算汇总并核验原文件哈希。', '',
            '[证据记录](results/evidence.json) · [档案索引与封存摘要](results/archive-index.json) · [实际配置](configs/executed-configs.json) · '
            '[源选模与 score 完成记录](results/model-completion.json) · [最终事件](results/final-events.jsonl) · [启动时快照](results/startup.md)', '',
            '完整日志、checkpoints、逐轮状态和预测保存在 `/home/hxchen/research/GDA/TSBA/experiments/r5/<setting>/attempt-001/`。'
            '仓库发布逐设置 CSV/图表、最终日志、封存清单和索引；旧 R4 档案没有覆盖。', '',
            '## 分设置材料', '',
            *[f"- [{LABELS[s]}](results/{s}/report.md)：[旁路种子值](results/{s}/side_seeds.csv)、[旁路分组配对](results/{s}/side_paired.csv)、[全部端点](results/{s}/endpoints.csv)、[源/目标域汇总](results/{s}/summary.csv)、[最终日志](results/final-logs/{s}.log)。" for s in SETTINGS], '',
            '## 结论边界', '',
            '完整目标域参与 score 拟合与适配，属于传导式开发评价。Mutagenicity 两种划分高度重合；PROTEINS 是另一数据设置，但本轮未登记为独立最终测试。'
            '五个 seed 描述固定域划分下的训练随机性，12 个方向共享图和源模型，不是 60 个独立域样本。'
            'KDE 仅为无标签诊断参照，不是真实总体 score；本轮结论针对当前表示、估计器、请求幅度和更新规则的组合。', '']
    interpretation=experiment/'results/interpretation.md'
    if interpretation.exists():
        position=lines.index('## 整体效果')
        lines[position:position]=['## 核心判断','',interpretation.read_text().strip(),'']
    (experiment/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True)
    seeds=read_csv(result/'aggregate-seeds.csv')
    for ax,s in zip(axes,SETTINGS):
        for x,c in enumerate(('A0−B1','A0−B0','B1−B0')):
            points=[float(r['accuracy'])*100 for r in seeds if r['setting']==s and r['method_or_contrast']==c]
            ax.scatter([x+(i-2)*.035 for i in range(5)],points,alpha=.7)
            ax.errorbar(x,statistics.mean(points),yerr=statistics.stdev(points),fmt='ks',capsize=4)
        ax.axhline(0,color='grey',linewidth=.8)
        ax.set(xticks=range(3),xticklabels=['A0-B1','A0-B0','B1-B0'],title=s,ylabel='Accuracy difference (pp)')
    fig.savefig(result/'cross-setting.png',dpi=170)
    plt.close(fig)
