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
    for filename in ['unit_checks_a4.log','mean_identity.json','ast_screen_a4.json']:
        shutil.copy2(stage/'D0'/filename,dest/'D0'/filename)
    (dest/'publication_code').mkdir()
    for filename in ['optimization_a4_report.py','optimization_plot.py','scripts/archive_optimization_phase.py']:
        shutil.copy2(repo/filename,dest/'publication_code'/Path(filename).name)
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    write(dest/'publication_provenance.json',dict(repo_commit=code_commit,source_run=str(stage.parent),phase=phase,
        feature_archive='experiments/tqg_assumptions_20261103/features',original_feature_stage=read(source/'manifest.json')['config']['feature_stage'],
        raw_graph_data_uploaded=False,existing_GNN_checkpoints_duplicated=False))
    write(dest/'history.jsonl.index.json',dict(shards=[str(p.relative_to(dest)) for p in sorted(dest.glob('*/history.jsonl'))]))
    (dest/'README.md').write_text('# A4-a核读取学习曲线原始归档\n\n'+
        '本归档仅覆盖已完成的A4-a；LOCAL、A6和本轮整体目标仍在继续。\n\n'+
        '[白话结果](结果解读.md) · [完整指标](results.csv) · [独立审计](audit.json) · [冻结协议](protocol/method.md)\n\n'+
        '包含960份参数/全部网格预测、CV、曲线指标、日志、TensorBoard、代码快照与bundle。原始图数据和历史GNN不重复上传；输入特征可追溯到上轮assumptions归档。\n\n'+
        '368640次是小矩阵读出拟合，不是GNN训练次数。存在1条JOINT观察器有限未收敛，保留原记录。报告首次导出整数序列化错误及恢复记录保留，未重做拟合。\n')
    files=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file():
            assert p.stat().st_size<90*1024**2,str(p)
            files.append(dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=digest(p)))
    assert sum(r['bytes'] for r in files)<1024**3
    write(dest/'artifact_manifest.json',dict(files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),phase=phase))
    for item in files:assert digest(dest/item['path'])==item['sha256']
    docs=repo/'docs/v2/results'/name;docs.mkdir(exist_ok=False)
    for filename in ['结果解读.md','results.csv','summary.csv','results_summary.json','audit.json','README.md','run_report.md','hand_off.md','compute_ledger.json']:
        relative=Path('../../../../experiments')/name/filename
        (docs/filename).symlink_to(relative)
    (docs/'figures').symlink_to(Path('../../../../experiments')/name/'figures')
    (docs/'protocol').symlink_to(Path('../../../../experiments')/name/'protocol')
    print(json.dumps(dict(archive=str(dest),files=len(files),bytes=sum(r['bytes'] for r in files))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--phase',default='A4');p.add_argument('--name',default='tqg_optimization_a4_20261104');a=p.parse_args()
    publish(Path(a.repo),Path(a.stage),a.phase,a.name)
