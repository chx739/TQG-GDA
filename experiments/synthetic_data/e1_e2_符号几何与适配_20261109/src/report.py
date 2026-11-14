"""Deterministic evidence tables for the archived E1/E2 report."""
import argparse
import json
from pathlib import Path
import statistics


def mean(values):
    values=[x for x in values if x is not None]
    return statistics.mean(values) if values else None


def fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def render(s):
    lines=['# TSBA E1/E2 合成验证结果（2026-11-09）','',
           '> 固定表示上的机制实验；没有训练图编码器，没有读取真实图任务的目标标签。', '',
           '## 执行与证据范围','',
           f"执行代码：[`{s['execution_commit'][:7]}`](https://github.com/chx739/TQG-GDA/tree/{s['execution_commit']})。方案和代码在运行前冻结；2个维度×5个种子×4臂，每臂30轮。全部预测封存后统一读取合成评价标签；不按标签选epoch、轮次或种子。",'',
           f"本地原始档案：`{s['archive_path']}`。文件校验通过 {s['integrity_files_verified']} 项；执行失败记录 {len(s['failures'])} 项。原始checkpoint、逐轮记录、完整日志在本地持久保留；GitHub发布代码、报告、表格和校验清单，不包含模型运行二进制。",'',
           '生成先验为等权二分类GMM；每域512张量样本，独立评价10,000点。源均值整体右移0.5，使用解析源Bayes线性头及生产坐标/残差初始化，U2=0。这里的样本是表示向量，不是真实图。score使用100 epochs，适配使用默认Config、CPU单线程、float64。','',
           '## E1：方向与风险反例','',
           '| 场景（完整局部请求） | Δlog density | Δ错误率（百分点） |','| --- | ---: | ---: |']
    for r in s['analytic']['cases']:
        if r['alpha']==1:
            lines.append(f"| {r['case']} | {fmt(r['delta_log_density'],6)} | {fmt(100*r['delta_risk'],6)} |")
    conflict=s['analytic']['scalar_conflict']
    lines.extend(['',f"线性固定斜率几何恒等式的绝对误差为 {s['analytic']['linear_geometry_error']:.3g}，低于1e-8。源监督与密度请求的固定斜率、总体积分专例得到局部平衡根 b={conflict['roots'][0]['b']:.6f}；源阈值为0.5，目标密度谷为0。这个专例不能证明非线性生产算法的固定点。",'',
                  '非零U2探针的误差曲线如下。曲线使用五种步长均能重定位的共同锚点；全量有效/缺失计数仍在CSV中保留。网格括根可能漏掉切触根，多根不强行选择。','',
                  '![E1几何误差](E1-geometry.png)','',
                  '## E2：全部固定预算轨迹','',
                  '| 维度 | score臂 | 完成 | ΔBA均值±样本SD（百分点） | 胜/平/负 | ΔNLL均值 | 平均接受轮数 |',
                  '| --- | --- | ---: | ---: | --- | ---: | ---: |'])
    for dim in ['2','16']:
        for arm,r in s['grouped'][dim].items():
            lines.append(f"| {dim} | {arm} | {r['completed']}/5 | {fmt(r['delta_ba_pp']['mean'])} ± {fmt(r['delta_ba_pp']['sd'])} | {r['wins']}/{r['ties']}/{r['losses']} | {fmt(r['delta_nll']['mean'],6)} | {fmt(r['accepted']['mean'],1)} |")
    lines.extend(['','Δ均相对同种子的源初始化；BA越大越好，NLL越小越好。五种子是重复单元，锚点与轮次不充当独立样本。失败轨迹不混入正常完成均值；完整逐种子数值见[evaluation.csv](evaluation.csv)。','',
                  '![E2逐种子风险变化](E2-risk.png)','',
                  '## score可靠性与可达覆盖','',
                  '| 维度 | 初始边界 learned→population 法向RMSE | 初始边界 zero→population RMSE | 辅助双面 learned 符号平衡一致率 | 辅助双面 empirical 符号平衡一致率 |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for dim in [2,16]:
        def metric(bank,pair,key):
            return mean([r.get(key) for r in s['score_queries'] if r['dim']==dim and r['bank']==bank and r['pair']==pair])
        lines.append(f"| {dim} | {fmt(metric('initial_boundaries','learned_vs_population','normal_rmse'))} | {fmt(metric('initial_boundaries','zero_vs_population','normal_rmse'))} | {fmt(metric('auxiliary_planes','learned_vs_population','balanced_sign_agreement'))} | {fmt(metric('auxiliary_planes','empirical_vs_population','balanced_sign_agreement'))} |")
    lines.extend(['','法向误差均乘σ；表中先按每个种子的共同查询点计算，再跨种子等权。实际初始边界只有一种总体score方向，因此其平衡符号一致率为NA；双方向值来自原坐标b=±0.5两张辅助查询面，不是主模型的真实边界。完整拟合/统计/总体误差在[score_queries.csv](score_queries.csv)；轨迹查询按来源臂/轮次回放，见本地observations.json。','',
                  '| 维度 | 原坐标σ范围 | learned有效锚点/输入 | learned裁剪比例（先按轮再按种子平均） |',
                  '| --- | ---: | ---: | ---: |'])
    for dim in [2,16]:
        cells=[r for r in s['cells'] if r['dim']==dim]
        traj=[r for r in s['trajectories'] if r['dim']==dim and r['arm']=='learned']
        sigmas=[r['raw_sigma'] for r in cells]
        lines.append(f"| {dim} | {min(sigmas):.5f}–{max(sigmas):.5f} | {fmt(mean([r['coverage'] for r in traj]))} | {fmt(mean([r['clipping_round_mean'] for r in traj]))} |")
    lines.extend(['','覆盖不足或零请求按原定义记录，不调整带宽/半径补齐。E2几何每轮固定取前8个有效锚点；全部锚点均有score和margin响应记录。重定位只是观察，不参与验收。','',
                  '![score验证与边界误差](E2-score-validation.png)','', '图中每点对应一个种子所选score checkpoint；灰叉为同点零score误差。这里只展示配对诊断，不声称五点建立了可靠相关关系。','', '## 固定参考的源监督与请求分解','',
                  '| 维度 | score臂 | source响应RMS均值 | request响应RMS均值 |',
                  '| --- | --- | ---: | ---: |'])
    for dim in [2,16]:
        for arm in ['learned','empirical','population']:
            rows=[r for r in s['decompositions'] if r['dim']==dim and r['arm']==arm]
            lines.append(f"| {dim} | {arm} | {fmt(mean([r['source_response_rms'] for r in rows]),6)} | {fmt(mean([r['request_response_rms'] for r in rows]),6)} |")
    lines.extend(['','每条轨迹只在第一个活动轮的同一A下计算δS=−A⁻¹gS和δT=A⁻¹(λB/M)Dᵀζ，表中为DδS、DδT的RMS。它解释该轮候选的来源，不能当作跨轮风险改善的因果分解。固定参考ζ=0保留DᵀD，并单独验收；它不是生产zero-score臂，后者跳过更新。','',
                  '## 限制与档案入口','',
                  '- E2没有完整的源继续训练轨迹，因此任何相对Source-initial的收益都不能单独归因于密度信息。E3/E4方案已补充固定参考ζ=0、源GGN与尝试轮掩码控制。',
                  '- E3/E4真实图实验尚未执行；真实表示可读性H1没有在本轮检验。PL/entropy的700步面板已预先登记，但没有在本轮运行。',
                  '- 维度变化同时按生产默认改变adapter宽度；跨维度差异不能全部归因于带宽或维数。',
                  '- 合成反例不证明所有真实任务失败；局部几何近似通过也不证明最终分类有效。',
                  '- 原始日志、checkpoint、输入生成参数、逐轮批次/锚点/状态、封存、标签访问与完整文件SHA256见本地run目录；归档路径和校验信息见[archive-index.json](archive-index.json)。',
                  '- 发布的[summary.json](summary.json)保留全量种子结果与主要过程统计；[checkpoint-audit.json](checkpoint-audit.json)给出重载复核结果。',''])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with args.output.open('x') as f:f.write(render(json.loads(args.summary.read_text())))
