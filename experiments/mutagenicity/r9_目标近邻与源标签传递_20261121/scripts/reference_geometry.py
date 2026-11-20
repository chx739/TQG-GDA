"""Freeze and score the registered R9 landmark-geometry source screen."""
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
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from ..src import geometry_models, reference_geometry, residual_inputs, true_rate_replay, weighted_residuals, wl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=('all','freeze','score'), default='all')
    parser.add_argument('--obs-sha256')
    args = parser.parse_args()
    if (args.phase=='score') != (args.obs_sha256 is not None):
        parser.error('--obs-sha256 is required only in the score phase')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())
    protocol = here/cfg['protocol']

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    protocol_path = protocol.relative_to(repo)
    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol_path}"))!=io.digest(protocol):
        raise PermissionError('The registered geometry protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    modules = (io,r5_inputs,seals,composition,structural,structural_observation,inputs,prior_run,
               roc_score,roc_transfer,source_inputs,geometry_models,reference_geometry,
               residual_inputs,true_rate_replay,weighted_residuals,wl)
    checked = [config,protocol,Path(__file__),*(Path(m.__file__) for m in modules)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in checked}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or config differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    if args.phase != 'score':
        output.mkdir(parents=True, exist_ok=False)
        io.write_json(output/'executed-config.json',dict(code_commit=revision, code_sha256=hashes, plan=cfg,
            command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
            label_phases={'freeze':['D','C'],'score':['E']},torch_threads=1,
            started_utc=datetime.now(timezone.utc).isoformat(),
            versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')}))
    else:
        executed = json.loads((output/'executed-config.json').read_bytes())
        if executed['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('Scoring requires the exact frozen code and a pending OBS')

    phase = 'authenticate_source_inputs'
    try:
        reference = reference_geometry.authenticate(here,cfg)
        archive, features = reference_geometry.prepare(repo,reference)
        obs = args.obs_sha256
        if args.phase != 'score':
            phase = 'fit_D_and_calibrate_C'
            obs = reference_geometry.freeze(output,cfg,reference,archive,features)
            print('R9 reference geometry OBS',obs,flush=True)
        if args.phase != 'freeze':
            phase = 'score_E_only'
            io.write_json(output/'score-execution.json',dict(code_commit=revision, obs_sha256=obs,
                command=[sys.executable,*sys.orig_argv[1:]], cwd=str(Path.cwd()),
                started_utc=datetime.now(timezone.utc).isoformat()))
            final = reference_geometry.score(output,cfg,reference,archive,features,obs)
            print('R9 reference geometry FINAL',final,flush=True)
    except Exception:
        name = f'execution-failure-{args.phase}.json'
        io.write_json(output/name,dict(phase=phase,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
