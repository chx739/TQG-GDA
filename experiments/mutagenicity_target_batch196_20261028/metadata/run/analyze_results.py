"""Post-training descriptive analysis; no dataset loading or target label access."""
import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent;S=R/'stage3_execution'
BATCHES=('B16','B32','B64','B196')
ARMS=('B_frozen','B_fit',*BATCHES)

def main():
 assert (R/'stage_3_done').exists()
 z=json.loads((S/'results_summary.json').read_text());d=json.loads((S/'training_diagnostics.json').read_text())
 a=json.loads((S/'independent_audit.json').read_text());assert a['passed']
 text='# Mutagenicity：目标batch196追加实验\n\n'
 text+='只新增目标batch196，0→1/2/3各五种子，共15次新适配。16/32/64及源fit SGD复用已核对配置、数据、模型哈希和数值代码的历史实验。源fit每步16、query全部196、相同预训练模型、适配200步、SGD lr0.0003、beta0.1、lambda_W0.01、FP64。\n\n'
 text+='196是本轮目标batch的新对照值，不是目标域全部图数。目标域1/2/3共有1084/1084/1085图；源query196来自另一域。源0域1084图=975训练+109验证，975=779fit+196query。域由SEGA图规模划分提供，不是Mutagenicity官方域标签。\n\n'
 text+='主指标：固定200步的balanced accuracy（%），五种子均值±样本标准差。\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---:|---:|---:|\n'
 for arm in ARMS:
  cells=[]
  for domain in (1,2,3):
   m=z['metrics'][f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else 'fixed_200']['balanced_accuracy']
   cells.append(f"{100*m['mean']:.2f} ± {100*m['sd']:.2f}")
  text+='| '+arm+' | '+' | '.join(cells)+' |\n'
 text+='\n## B196相对其他batch的配对差\n\n单位百分点，括号为五个种子胜/平/负。主要预设比较为B196对B16；对32/64为描述性比较。\n\n| 参照 | 0→1 | 0→2 | 0→3 |\n|---|---|---|---|\n'
 with (S/'results.csv').open() as stream:results=list(csv.DictReader(stream))
 paired={};supported=[]
 for base in BATCHES[:-1]:
  cells=[];paired[base]={}
  for domain in (1,2,3):
   ix={(r['seed'],r['arm']):float(r['balanced_accuracy']) for r in results if r['task']==f'0_to_{domain}' and r['selection']=='fixed_200'}
   v=[100*(ix[(str(seed),'B196')]-ix[(str(seed),base)]) for seed in range(3405,3410)]
   p=dict(mean_difference_pp=statistics.mean(v),seed_differences_pp=v,wins=sum(x>1e-10 for x in v),ties=sum(abs(x)<=1e-10 for x in v),losses=sum(x< -1e-10 for x in v))
   paired[base][str(domain)]=p
   cells.append(f"{p['mean_difference_pp']:+.3f} ({p['wins']}/{p['ties']}/{p['losses']})")
   if base=='B16' and p['mean_difference_pp']>0 and p['wins']>=3:supported.append(domain)
  text+='| '+base+' | '+' | '.join(cells)+' |\n'
 text+=f'\n预设条件要求至少2/3任务平均差为正且至少3/5种子获胜。B196有{len(supported)}/3任务满足，'+('达到' if len(supported)>=2 else '未达到')+'预设支持条件。五个种子、固定划分下的探索性结果不等于统计显著性或跨数据集结论。\n'
 fields=('feedback_norm_mean','feedback_norm_sd','feedback_norm_cv','feedback_norm_p95','feedback_source_ratio_median','step_seconds_median','wall_seconds')
 rows=[]
 for domain in (1,2,3):
  for arm in BATCHES:
   g=[r for r in d['jobs'] if r['domain']==domain and r['arm']==arm];assert len(g)==5
   row=dict(task=f'0_to_{domain}',arm=arm,seeds=5)
   for f in fields:
    values=[r[f] for r in g];row[f+'_mean']=statistics.mean(values);row[f+'_sd']=statistics.stdev(values)
   for f in ('calibration_calls','nonconverged','numerical_fallbacks'):row[f]=sum(r[f] for r in g)
   cover=[r['coverage_fraction'] for r in a['target_coverage'] if r['domain']==domain and r['arm']==arm]
   row['unique_target_coverage_mean']=statistics.mean(cover)
   rows.append(row)
 with (S/'diagnostics_summary.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 text+='\n## 幅度波动、反馈强度、耗时与覆盖\n\n先对每个作业200步求统计量，再对五个种子取平均。CV为反馈梯度范数的标准差/均值，只描述幅度波动，不代表方向正确。耗时为每作业训练步耗时中位数的跨种子均值，历史负载不同，仅作参考。\n\n| 任务 | batch | 范数CV | 平均范数 | 加权反馈/源梯度比 | 秒/步 | 不同目标图覆盖率 | 未收敛/校准次数 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for r in rows:
  text+=f"| {r['task']} | {r['arm'][1:]} | {r['feedback_norm_cv_mean']:.3f} | {r['feedback_norm_mean_mean']:.3f} | {r['feedback_source_ratio_median_mean']:.3f} | {r['step_seconds_median_mean']:.3f} | {100*r['unique_target_coverage_mean']:.2f}% | {r['nonconverged']}/{r['calibration_calls']} |\n"
 text+='\n固定200步，目标图使用次数含重复为3200/6400/12800/39200，源fit各3200。二分类目标空间秩最多16/32/64/196，保留完整有效秩，不手动截断。每批内无重复、跨步可重复，四档使用同一目标随机排列的不同长度前缀；fit/query逐步序列一致。方向数、目标使用次数、核结构及正则相对影响共同变化，因此不是等计算预算比较，也不能把效果单独归因于采样噪声。\n'
 text+='\n## CE和AUROC\n\n| 任务 | batch | CE | AUROC |\n|---|---:|---:|---:|\n'
 for domain in (1,2,3):
  for arm in BATCHES:
   m=z['metrics'][f'0_to_{domain}'][arm]['fixed_200'];text+=f"| 0→{domain} | {arm[1:]} | {m['ce']['mean']:.6f} | {m['auroc']['mean']:.6f} |\n"
 text+='\n## 实现与边界\n\n'
 text+=f"- 独立审计通过；新增3000步，复用12000步。参数反馈等式最大误差{a['max_gradient_identity_error']:.3g}；SGD执行等式最大误差{a['max_sgd_step_identity_error']:.3g}。这些等式不证明构造的反馈等于真实目标梯度。\n"
 text+='- 开发标签在全部训练封存后读取，final标签未读。开发集此前已被历史实验观察；本轮是探索性实验。\n- 未改变源fit batch、优化器、反馈权重或核正则；未执行按epoch洗牌、等计算预算或真实目标梯度方向诊断。\n- 核心tqg_gda数值代码与已验证实现逐字节相同；仅扩展实验控制、评价和归档。\n'
 (S/'结果解读.md').write_text(text)
 payload=dict(primary=dict(supported_tasks=supported,preset_criterion_met=len(supported)>=2),paired_B196_minus=paired,no_target_label_reader=True,no_new_training=True)
 (S/'descriptive_analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
 (S/'hand_off.md').write_text(f'目标batch196追加15次训练，源fit16/query196/200步不变。B196对16在{len(supported)}/3任务满足预设条件。配对BA差（百分点）为'+', '.join(f"0→{i}: {paired['B16'][str(i)]['mean_difference_pp']:+.3f}" for i in (1,2,3))+'。final标签未读。幅度波动不能代替方向正确性，固定步数不等计算预算。\n')
 for f,title,ylabel in [('feedback_norm_cv','Feedback magnitude variation','Norm coefficient of variation'),('step_seconds_median','Runtime (16/32/64 historical)','Step seconds')]:
  fig,axes=plt.subplots(1,3,figsize=(11,3.8))
  for domain,ax in zip((1,2,3),axes):
   g=[next(r for r in rows if r['task']==f'0_to_{domain}' and r['arm']==arm) for arm in BATCHES]
   ax.bar(np.arange(4),[r[f+'_mean'] for r in g],yerr=[r[f+'_sd'] for r in g],capsize=3,color=['#6b9bc1','#448aab','#df8a42','#b95a59'])
   ax.set_xticks(np.arange(4),['16','32','64','196']);ax.set_xlabel('Target batch size');ax.set_ylabel(ylabel);ax.set_title(f'0 → {domain}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  fig.suptitle(title);fig.tight_layout()
  for ext in ('png','pdf'):fig.savefig(S/'figures'/f'{f}.{ext}',dpi=180,bbox_inches='tight')
  plt.close(fig)
 print(json.dumps(payload))
if __name__=='__main__':main()
