"""Freeze and score the registered R7 refit on strong source anchors."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from experiments.common import io
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from experiments.mutagenicity.r14_节点类型与分类更新_20261121.src import experiment as r14
from experiments.mutagenicity.r16_目标图结构与监督分数蒸馏_20261121.src import inputs as r16
from tsba.shared_update import config as shared_config, predict
from ..src import (anchor_solver, decisions, fit, head_crossover, head_crossover_scoring, paired, r5_inputs,
                   seals, strong_anchor_refit as refit, strong_anchor_scoring as scoring)


CODE_MODULES = (io,r8_inputs,composition,pilot,wl,structural,structural_observation,inputs,prior_run,
                roc_score,roc_transfer,historical,observation,r14,r16,shared_config,predict,anchor_solver,
                decisions,fit,head_crossover,head_crossover_scoring,paired,r5_inputs,seals,refit,scoring)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',required=True,choices=('freeze','score'))
    parser.add_argument('--obs-sha256')
    args = parser.parse_args()
    if (args.phase=='score')!=(args.obs_sha256 is not None):
        parser.error('--obs-sha256 is required only in score phase')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    protocols = []
    for field,pin in (('protocol','registration_commit'),('teacher_protocol','teacher_registration_commit')):
        protocol = here/cfg[field]
        if inputs.sha(git('show',f"{cfg[pin]}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
            raise PermissionError('Registered strong-anchor protocol differs')
        protocols.append(protocol)
    revision = git('rev-parse','HEAD').decode().strip()
    files = [config,*protocols,Path(__file__),*(Path(m.__file__) for m in CODE_MODULES)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in files}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or config differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    execution = dict(code_commit=revision,command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
                     started_utc=datetime.now(timezone.utc).isoformat())
    if args.phase=='freeze':
        output.mkdir(parents=True,exist_ok=False)
        io.write_json(output/'executed-config.json',dict(**execution,code_sha256=hashes,plan=cfg,
                      torch_threads=1,label_phases={'freeze':[],'score':['holdout']},
                      versions={m:importlib.metadata.version(m) for m in ('torch','torch-geometric','numpy','scipy')}))
    else:
        old = json.loads((output/'executed-config.json').read_bytes())
        if old['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('Scoring requires exactly the frozen code and a pending OBS')

    phase = 'authenticate_historical_inputs'
    try:
        r10 = roc_transfer.stage(Path(cfg['structural_archive']),'FIT',cfg['structural_fit_sha256'])
        plan = json.loads(prior_run.read_member(Path(cfg['structural_archive']),r10['files'],'executed-config.json'))['plan']
        c = historical.CArchive(dict(cfg,r7_archive=plan['r7_archive']))
        if args.phase=='freeze':
            phase = 'fit_and_freeze_without_labels'
            obs = refit.freeze(output,repo,cfg,c)
            print('R7 strong-anchor OBS',obs,flush=True)
        else:
            phase = 'score_frozen_h'
            final = scoring.score(output,repo,cfg,c,args.obs_sha256,execution=dict(**execution,obs_sha256=args.obs_sha256))
            print('R7 strong-anchor FINAL',final,flush=True)
    except Exception:
        try:
            io.write_json(output/f'execution-failure-{args.phase}.json',dict(phase=phase,traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise


if __name__=='__main__':
    main()
