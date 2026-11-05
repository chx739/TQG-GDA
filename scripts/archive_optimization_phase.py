"""Publish only a completed, independently audited optimization phase."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def publish(repo,stage,phase,name):
    source=stage/phase;done=read(source/'done.json');audit=read(source/'audit.json')
    assert done['completed'] and audit['passed']
    if phase=='A4':
        assert done['logical_fits']==368640 and audit['counts']['curve_rows']==230400
        code=stage/'code_a4';code_source=stage/'source_a4.json'
    elif phase in ['A2b','A6']:
        assert read(source/'summary_audit.json')['passed']
        assert audit['counts']['blocks']==(120 if phase=='A2b' else 360)
        code=stage/('code_feedback' if phase=='A2b' else 'code_candidates')
        code_source=stage/('source_feedback.json' if phase=='A2b' else 'source_candidates.json')
    else:raise ValueError('Publication for this phase is not yet implemented')
    dest=repo/'experiments'/name;dest.mkdir(exist_ok=False)
    for p in source.iterdir():
        if p.name.endswith('.tmp'):continue
        if p.is_dir():shutil.copytree(p,dest/p.name)
        else:shutil.copy2(p,dest/p.name)
    shutil.copytree(stage.parent/'stage2_method',dest/'protocol')
    shutil.copytree(code,dest/'code',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copy2(code_source,dest/'code_provenance.json')
    subprocess.run(['git','bundle','create',str(dest/'execution_code.bundle'),'--all'],cwd=code,check=True)
    (dest/'D0').mkdir()
    checks=['unit_checks_a4.log','mean_identity.json','ast_screen_a4.json'] if phase=='A4' else ['feedback_checks.log','selection_checks.log','mean_identity.json','ast_screen_'+phase+'.json']
    if phase=='A6':checks.append('ast_screen_A6_candidates.json')
    for filename in checks:
        shutil.copy2(stage/'D0'/filename,dest/'D0'/filename)
    (dest/'publication_code').mkdir()
    publications=['workflows/v2/optimization/optimization_a4_report.py','workflows/v2/optimization/optimization_plot.py'] if phase=='A4' else ['workflows/v2/optimization/optimization_feedback_audit.py','workflows/v2/optimization/optimization_feedback_report.py','workflows/v2/optimization/optimization_summary_audit.py','workflows/v2/optimization/optimization_feedback_documents.py','tests/test_optimization_report.py']
    for filename in publications+['scripts/archive_optimization_phase.py']:
        target=dest/'publication_code'/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/filename,target)
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    write(dest/'publication_provenance.json',dict(repo_commit=code_commit,source_run=str(stage.parent),phase=phase,
        feature_archive='experiments/tqg_assumptions_20261103/features',original_feature_stage=read(source/'manifest.json')['config']['feature_stage'],
        raw_graph_data_uploaded=False,existing_GNN_checkpoints_duplicated=False))
    write(dest/'history.jsonl.index.json',dict(shards=[str(p.relative_to(dest)) for p in sorted(dest.glob('*/history.jsonl'))]))
    if phase=='A4':(dest/'README.md').write_text('# A4-a核读取学习曲线原始归档\n\n'+
        '本归档仅覆盖已完成的A4-a；LOCAL、A6和本轮整体目标仍在继续。\n\n'+
        '[白话结果](结果解读.md) · [完整指标](results.csv) · [独立审计](audit.json) · [冻结协议](protocol/method.md)\n\n'+
        '包含960份参数/全部网格预测、CV、曲线指标、日志、TensorBoard、代码快照与bundle。原始图数据和历史GNN不重复上传；输入特征可追溯到上轮assumptions归档。\n\n'+
        '368640次是小矩阵读出拟合，不是GNN训练次数。存在1条JOINT观察器有限未收敛，保留原记录。报告首次导出整数序列化错误及恢复记录保留，未重做拟合。\n')
    else:
        (dest/'README.md').write_text(f'# {phase}：已完成实验原始归档\n\n'+
            '[白话结果](结果解读.md) · [逐种子指标](results.csv) · [数值审计](audit.json) · [聚合审计](summary_audit.json) · [冻结协议](protocol/method.md)\n\n'+
            '包含全部读出、教师参数、CV选择、系数、封存前logits、开发评价标签、单步/梯度记录、日志、TensorBoard、样本ID、代码及Git bundle。没有原始图数据；不重复历史GNN。封存文件按原字节保存，seal.json与artifact_manifest.json分别验证块和整个归档。\n\n'+
            '原checkpoint/几何路径和SHA保留在blocks/*/ids.json；其来源为历史P0/H-only归档，可按SHA映射，而不是声称绝对路径可在任意机器直接运行。当前独立单步不是连续适配轨迹。\n\n'+
            '没有候选达到多任务继续门槛。A2-b误差聚合口径敏感性完整公开；A6一个主读出和一个教师内折有限未收敛保留。publication_compute_ledger.json区分已计量CPU与未完全计量的编排开销。\n')
    files=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file():
            assert p.stat().st_size<90*1024**2,str(p)
            files.append(dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=digest(p)))
    assert sum(r['bytes'] for r in files)<1024**3
    write(dest/'artifact_manifest.json',dict(files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),phase=phase))
    for item in files:assert digest(dest/item['path'])==item['sha256']
    docs=repo/'docs/v2/results'/name;docs.mkdir(exist_ok=False)
    links=['结果解读.md','results.csv','results_summary.json','audit.json','README.md','run_report.md','hand_off.md','compute_ledger.json']
    links+=['summary.csv'] if phase=='A4' else ['summary_audit.json','publication_compute_ledger.json']
    for filename in links:
        relative=Path('../../../../experiments')/name/filename
        (docs/filename).symlink_to(relative)
    (docs/'figures').symlink_to(Path('../../../../experiments')/name/'figures')
    (docs/'protocol').symlink_to(Path('../../../../experiments')/name/'protocol')
    print(json.dumps(dict(archive=str(dest),files=len(files),bytes=sum(r['bytes'] for r in files))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--phase',default='A4');p.add_argument('--name',default='tqg_optimization_a4_20261104');a=p.parse_args()
    publish(Path(a.repo),Path(a.stage),a.phase,a.name)
