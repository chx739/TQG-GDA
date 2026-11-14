"""Present every fixed step and time, without selecting a favorable one."""
import json
from collections import Counter
from .shared import output
from .summarize import summaries
from .protocol import blocks


def collect(root,plan):
    tables={name:[] for name in ('candidates','paired','derivatives','interactions','coverage')}
    failures=[]
    for task,step in blocks(plan):
        path=root/'evaluation/blocks'/task.key/f'before{step:02d}.json'
        data=json.loads(path.read_text())
        for name in tables:tables[name].extend(data[name])
        if data.get('error'):failures.append(dict(source=task.source,target=task.target,seed=task.seed,attempt=step,error=data['error']))
    for name,rows in tables.items():output.write_csv(root/'evaluation'/f'{name}.csv',rows)
    output.write_json(root/'evaluation/failures.json',failures)
    statuses=dict(Counter(r['status'] for r in tables['coverage'] if r['k']==1.))
    seed_rows,rows=summaries(plan,tables)
    output.write_csv(root/'evaluation/seed_summary.csv',seed_rows)
    output.write_csv(root/'evaluation/summary.csv',rows)
    output.write_json(root/'evaluation/summary.json',rows)
    selection={(r['family'],r['attempt'],r['k'],r['arm'],r['pool'],r['metric']):r for r in rows
               if r['scope']=='Avg' and r['subset']=='all_finite'}
    fmt=lambda v: 'NA' if v is None else f'{v:+.8g}'
    lines=[f"# R6 {plan['setting']['id']}", '',
           'R5 固定状态的事后机制诊断。新增训练与闭环适配为零；主时点为尝试 1 前、主步长 k=1。', '',
           f'评价状态块：{statuses}。逐块错误见 failures.json 和 blocks；候选阶段错误保留在原 blocks checkpoint。', '',
           '| 时点 | k | 对比/量 | 池 | 均值 | SD | 完整 seed | 有效块 |',
           '|---|---|---|---|---|---|---|---|']
    for attempt in plan['states']['attempts_before']:
        for k in plan['steps']['multipliers']:
            for family,arm,pool,metric in [('paired','U+−N','target','nll'),('paired','U+−U-','target','nll'),
                    *[('paired','S−N',p,'nll') for p in ('source_fit','source_validation','target')],
                    ('paired','SU+−S','target','nll'),('interactions','I','target','interaction'),
                    ('derivatives','s/u','target','d_s'),('derivatives','s/u','target','d_u')]:
                r=selection[family,attempt,k,arm,pool,metric]
                lines.append(f"| {attempt} | {k:g} | {arm}:{metric} | {pool} | {fmt(r['mean'])} | {fmt(r['sd'])} | {r['complete_seeds']}/5 | {r['available_blocks']}/{r['planned_blocks']} |")
    lines += ['', 'NLL 差及方向导数为负有利；交互 I 为正表示有限步长组合使正常请求的边际效果变差。方向导数的一阶项可加，没有另加一阶交互。', '',
              '缺失方向使该 seed 全方向值为 NA，任一 seed 不完整则完整五 seed 均值为 NA。conditional_* 是注明覆盖的条件统计，不能替代主均值。', '',
              '主表保留全部有限候选，包括越界候选；common_motion_feasible 敏感性表使用六组共同可行的同一块集合。', '',
              '[候选](candidates.csv) · [配对](paired.csv) · [方向导数](derivatives.csv) · [交互](interactions.csv) · [覆盖](coverage.csv) · [种子值](seed_summary.csv) · [全部汇总](summary.csv)', '',
              'Accuracy/BA/Macro-F1 原值为 0–1 比例，配对 *_pp 列为百分点；NLL 为原量纲。目标生成图补集仍参与 R5 score 拟合。', '']
    focus=plan['evaluation']['posthoc_focus']
    if plan['setting']['id']==focus['setting']:
        lines += ['', '## R5 观察后指定的案例', '', '以下三方向是事后选取的解释案例，主时点/主步长固定为尝试 1 前、k=1；全部方向和其他时点、步长仍列于 summary.csv。', '',
                  '| 方向 | 对比 | 池 | 均值 | SD | 完整 seed |', '|---|---|---|---|---|---|']
        for source,target in focus['directions']:
            scope=f'M{source}→M{target}'
            for arm,pool in [('U+−N','target'),('U+−U-','target'),('S−N','source_fit'),('S−N','source_validation'),('S−N','target'),('SU+−S','target')]:
                r=next(r for r in rows if r['family']=='paired' and r['attempt']==1 and r['k']==1 and r['arm']==arm and r['pool']==pool and r['metric']=='nll' and r['scope']==scope and r['subset']=='all_finite')
                lines.append(f"| {scope} | {arm} | {pool} | {fmt(r['mean'])} | {fmt(r['sd'])} | {r['complete_seeds']}/5 |")
    output.write_json(root/'evaluation/completion.json',dict(block_statuses=statuses,rows={k:len(v) for k,v in tables.items()},seed_rows=len(seed_rows),summary_rows=len(rows)))
    path=root/'evaluation/report.md';text='\n'.join(lines)
    if path.exists() and path.read_text()!=text:raise ValueError('Report differs on recovery')
    if not path.exists():path.write_text(text)
    return tables
