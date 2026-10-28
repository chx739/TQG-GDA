"""Finalize only a completed batch196 run; no training or target-label reader."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path(__file__).resolve().parent;S=R/'stage3_execution';REPO=Path('/home/hxchen/TQG-GDA')
NAME='mutagenicity_target_batch196_20261028'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 state=json.loads((R/'live_state.json').read_text());assert state['status']=='COMPLETE'
 assert json.loads((R/'stage_3_done').read_text())['new_adaptations']==15
 for script in ('post_run_audit.py','analyze_results.py'):
  with (S/(script+'.log')).open('x') as f:subprocess.run([sys.executable,str(R/script)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
 (R/'CONTINUATION.md').write_text('COMPLETE: 15 B196 jobs, 60 reused adaptations, 165 evaluations. No relaunch. Audit/analysis complete. Check GitHub sync status before reporting final completion.\n')
 dest=REPO/'experiments'/NAME
 subprocess.run([sys.executable,str(REPO/'scripts/archive_target_batch.py'),'--run',str(R),'--destination',str(dest)],cwd=REPO,check=True)
 stats=json.loads((dest/'manifest.json').read_text())['statistics']
 (dest/'README.md').write_text('# Mutagenicity：目标batch196追加对照\n\n新增15次B196训练；60次适配对照及5次源预训练模型复用，保留各自原训练提交。0→1/2/3各五种子，源fit每步16、query全部196、相同预训练起点、适配200步。\n\n[结果解读](results/结果解读.md) · [完整报告](results/run_report.md) · [评价与模型对应](evaluation_index.csv) · [作业索引](job_index.csv)\n\n'
  +f"本归档包含{stats['history_files']}份JSONL日志、{stats['checkpoint_files']}份checkpoint、{stats['evaluation_records']}条评价记录；来源文件共{stats['total_bytes']:,}字节（约{stats['total_bytes']/1024**2:.1f} MiB）。\n\n"
  +'`manifest.json`与`SHA256SUMS`记录原始字节、来源路径及哈希。复用结果同时核对训练代码、配置、样本角色、模型起点与旧指标。没有在归档时训练或读取标签。\n\n目标域图数为1084/1084/1085，196是每步目标抽样数，不是全量目标图。域来自SEGA图规模划分。目标标签只在全部训练封存后的开发评价阶段读取；final标签未读。数据缓存与TensorBoard二进制事件留在本机，JSONL及执行代码快照保留。\n\n校验：`python scripts/verify_experiment_archive.py experiments/'+NAME+'`。模型文件均小于1 MiB，不需要Git LFS。\n')
 docs=REPO/'docs/experiments'/NAME;docs.mkdir(exist_ok=False)
 wanted=['run_report.md','结果解读.md','results.csv','results_summary.json','training_diagnostics.json','diagnostics_summary.csv','descriptive_analysis.json','independent_audit.json','completion_audit.json','hand_off.md']
 for f in wanted:shutil.copy2(S/f,docs/f)
 shutil.copy2(R/'stage2_method/experiment_plan.yaml',docs/'experiment_plan.yaml')
 shutil.copytree(S/'figures',docs/'figures')
 (docs/'README.md').write_text('# Mutagenicity：目标batch196追加实验\n\n[结果解读](结果解读.md) · [完整报告](run_report.md) · [原始日志和模型](../../../experiments/'+NAME+'/README.md)\n\n15次新适配，60次复用适配；固定200步为主结果，源验证CE选点为次要结果。开发标签只在封存后评价，final标签未读。\n')
 files=[dict(path=str(p.relative_to(docs)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(docs.rglob('*')) if p.is_file()]
 (docs/'artifact_manifest.json').write_text(json.dumps(dict(source_run=str(R),execution_commit=json.loads((S/'experiment_manifest.json').read_text())['git_commit'],files=files),indent=2)+'\n')
 print(json.dumps(dict(complete=True,archive=str(dest),docs=str(docs),statistics=stats)))
if __name__=='__main__':main()
