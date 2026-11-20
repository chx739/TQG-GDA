"""Run the registered R9 source-size residual screen and seal each outcome."""
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
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_rate, roc_score, roc_transfer, thresholds
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from ..src import bridge, residual_inputs, residual_models, weighted_residuals, wl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())
    if (cfg['model_C'],cfg['neighbor_count'],cfg['regularization_per_calibration_graph'],cfg['max_relative_weight'],
            cfg['minimum_u_mae_gain_fraction'],cfg['minimum_e_accuracy_gain_fraction']) != (1.,8,.01,4.,[1,100],[1,200]):
        raise PermissionError('Config differs from the registered residual screen')

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    protocol = (here/cfg['protocol']).relative_to(repo)
    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol}"))!=io.digest(repo/protocol):
        raise PermissionError('Registered R9 residual protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    modules = (io,r5_inputs,seals,composition,structural,structural_observation,inputs,prior_run,roc_rate,
               roc_score,roc_transfer,thresholds,source_inputs,bridge,wl,residual_inputs,residual_models,weighted_residuals)
    checked = [config,Path(__file__),repo/protocol,*(Path(m.__file__) for m in modules)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in checked}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or configuration differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    io.write_json(output/'executed-config.json',dict(code_commit=revision,code_sha256=hashes,plan=cfg,
        command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),torch_threads=1,
        git_status=git('status','--short').decode().splitlines(),started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')}))

    phase = 'authenticate_source_inputs'
    try:
        archive = residual_inputs.Archive(repo,cfg)
        roles,features,topology = archive.prepare()
        phase = 'fit_D_and_calibrate_C'
        obs = weighted_residuals.freeze(output,cfg,archive,roles,features,topology)
        print('R9 weighted source residuals OBS',obs,flush=True)
        phase = 'score_U_and_E'
        final = weighted_residuals.score(output,cfg,archive,roles,topology,obs)
        print('R9 weighted source residuals FINAL',final,flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json',dict(phase=phase,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
