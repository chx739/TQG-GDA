"""Freeze the registered R17 paired readouts, then score the original E roles."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    geometry_models, reference_geometry, residual_inputs, true_rate_replay, weighted_residuals, wl)
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from experiments.mutagenicity.r15_新局部结构与图类别_20261121.src import structure
from ..src import readout, screen

CODE = (io,r5_inputs,seals,composition,r8_inputs,pilot,geometry_models,reference_geometry,residual_inputs,
        true_rate_replay,weighted_residuals,wl,structural,structural_observation,inputs,prior_run,roc_score,
        roc_transfer,source_inputs,structure,readout,screen)


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
    protocol = here/cfg['protocol']

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('The registered R17 protocol differs')
    revision = git('rev-parse','HEAD').decode().strip()
    files = [config,protocol,Path(__file__),*(Path(m.__file__) for m in CODE)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in files}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or configuration differs from the recorded Git revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    execution = dict(code_commit=revision,command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
                     started_utc=datetime.now(timezone.utc).isoformat())
    if args.phase=='freeze':
        output.mkdir(parents=True,exist_ok=False)
        io.write_json(output/'executed-config.json',dict(**execution,code_sha256=hashes,plan=cfg,
                      label_phases={'freeze':['D'],'score':['E']},threads=1,
                      versions={m:importlib.metadata.version(m) for m in ('torch','torch-geometric','numpy','scipy')}))
    else:
        old = json.loads((output/'executed-config.json').read_bytes())
        if old['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('R17 scoring needs the frozen revision and a pending OBS')
    phase = 'authenticate_source_data'
    try:
        prepared = screen.prepare(repo,here,cfg)
        if args.phase=='freeze':
            phase = 'fit_D_and_freeze_predictions'
            obs = screen.freeze(output,cfg,prepared)
            print('R17 paired readout OBS',obs,flush=True)
        else:
            phase = 'score_E_only'
            final = screen.score(output,cfg,prepared,args.obs_sha256,
                                 execution=dict(**execution,obs_sha256=args.obs_sha256))
            print('R17 paired readout FINAL',final,flush=True)
    except Exception:
        try:
            io.write_json(output/f'execution-failure-{args.phase}.json',dict(phase=phase,traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise


if __name__=='__main__':
    main()
