"""Freeze the fixed R12 reader on V/A/H, then score the original H roles."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    head_crossover, head_crossover_scoring, r5_inputs, seals)
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer, thresholds
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from ..src import (conditional_votes, fixed_reader_inputs, fixed_reader_scoring, fixed_reader_transfer,
                   reader_ranges, readouts, vote_mixture)

CODE = (io,head_crossover,head_crossover_scoring,r5_inputs,seals,composition,r8_inputs,pilot,wl,structural,
        structural_observation,inputs,prior_run,roc_score,roc_transfer,thresholds,historical,observation,
        conditional_votes,fixed_reader_inputs,fixed_reader_scoring,fixed_reader_transfer,reader_ranges,readouts,vote_mixture)


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
    old = here/cfg['historical_config']
    historical_cfg = json.loads(old.read_bytes())
    protocol = here/cfg['protocol']

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('The registered R12 fixed-reader protocol differs')
    revision = git('rev-parse','HEAD').decode().strip()
    files = [config,old,protocol,Path(__file__),*(Path(m.__file__) for m in CODE)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in files}
    if any(inputs.sha(git('show',f'{revision}:{path}'))!=sha for path,sha in hashes.items()):
        raise PermissionError('Actual code or configuration differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    execution = dict(code_commit=revision,command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
                     started_utc=datetime.now(timezone.utc).isoformat())
    if args.phase=='freeze':
        output.mkdir(parents=True,exist_ok=False)
        io.write_json(output/'executed-config.json',dict(**execution,code_sha256=hashes,plan=cfg,
                      historical_config=historical_cfg,training_updates=0,label_phases={'freeze':['validation'],'score':['holdout']},
                      threads=1,versions={m:importlib.metadata.version(m) for m in ('torch','torch-geometric','numpy','scipy')}))
    else:
        recorded = json.loads((output/'executed-config.json').read_bytes())
        if recorded['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('R12 scoring requires the frozen revision and a pending OBS')
    phase = 'authenticate_historical_readers'
    try:
        archive = fixed_reader_inputs.Archive(repo,here,cfg,historical_cfg)
        if args.phase=='freeze':
            phase = 'freeze_V_A_H_readouts_and_actions'
            obs = fixed_reader_transfer.freeze(output,cfg,historical_cfg,archive)
            print('R12 fixed-reader OBS',obs,flush=True)
        else:
            phase = 'score_frozen_H'
            final = fixed_reader_scoring.score(output,repo,cfg,historical_cfg,archive,args.obs_sha256,
                                               execution=dict(**execution,obs_sha256=args.obs_sha256))
            print('R12 fixed-reader FINAL',final,flush=True)
    except Exception:
        try:
            io.write_json(output/f'execution-failure-{args.phase}.json',dict(phase=phase,traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise


if __name__=='__main__':
    main()
