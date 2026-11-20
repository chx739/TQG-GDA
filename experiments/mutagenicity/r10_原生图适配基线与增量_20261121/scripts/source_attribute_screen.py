"""Run the registered single-seed PROTEINS source attribute screen."""
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
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from ..src import source_attribute_screen as runner, source_attributes, structural, structural_observation


CODE_MODULES = (
    io, r5_inputs, seals, composition, r8_inputs, pilot, geometry_models, reference_geometry,
    residual_inputs, true_rate_replay, weighted_residuals, wl, inputs, prior_run, roc_score,
    roc_transfer, source_inputs, runner, source_attributes, structural, structural_observation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=('all','freeze','score'), default='all')
    parser.add_argument('--obs-sha256')
    args = parser.parse_args()
    if (args.phase=='score')!=(args.obs_sha256 is not None):
        parser.error('--obs-sha256 is required only in the score phase')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())
    protocol = here/cfg['protocol']

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    protocol_path = protocol.relative_to(repo)
    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol_path}"))!=io.digest(protocol):
        raise PermissionError('Registered node attribute protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    checked = [config,protocol,Path(__file__),*(Path(m.__file__) for m in CODE_MODULES)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in checked}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or config differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()

    def execution():
        return dict(code_commit=revision, command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
                    started_utc=datetime.now(timezone.utc).isoformat())

    if args.phase=='score':
        old = json.loads((output/'executed-config.json').read_bytes())
        if old['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('Scoring requires the exact frozen code and a pending OBS')
    else:
        output.mkdir(parents=True,exist_ok=False)
        io.write_json(output/'executed-config.json',dict(**execution(),code_sha256=hashes,plan=cfg,
            label_phases={'freeze':['D','C'],'score':['E']},torch_threads=1,
            versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')}))

    phase = 'authenticate_source_inputs'
    try:
        prepared = runner.prepare(repo,here,cfg)
        obs = args.obs_sha256
        if args.phase!='score':
            phase = 'fit_D_and_calibrate_C'
            obs = runner.freeze(output,cfg,prepared)
            print('R10 PROTEINS attributes OBS',obs,flush=True)
        if args.phase!='freeze':
            phase = 'score_E_only'
            final = runner.score(output,cfg,prepared,obs,execution=dict(**execution(),obs_sha256=obs))
            print('R10 PROTEINS attributes FINAL',final,flush=True)
    except Exception:
        failure = output/f'execution-failure-{args.phase}.json'
        try:
            io.write_json(failure,dict(phase=phase,traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise


if __name__=='__main__':
    main()
