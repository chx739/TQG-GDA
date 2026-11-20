"""Run the R7 frozen E/J head crossover on the registered C development grid."""
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
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from tsba.shared_update import predict
from ..src import decisions, fit, head_crossover, head_crossover_scoring, paired, r5_inputs, seals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    protocol = (here/cfg['protocol']).relative_to(repo)
    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol}"))!=io.digest(repo/protocol):
        raise PermissionError('The registered R7 reference-head protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    modules = (io,inputs,prior_run,roc_score,roc_transfer,historical,observation,predict,
               decisions,fit,head_crossover,head_crossover_scoring,paired,r5_inputs,seals)
    checked = [config,Path(__file__),repo/protocol,*(Path(m.__file__) for m in modules)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in checked}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or config differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    io.write_json(output/'executed-config.json',dict(code_commit=revision,code_sha256=hashes,plan=cfg,
        command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),training_updates=0,torch_threads=1,
        label_phases={'freeze':[], 'score':['holdout']},started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')}))

    phase = 'authenticate_and_freeze_crossovers'
    try:
        c = historical.CArchive(cfg)
        obs = head_crossover.freeze(output,repo,cfg,c)
        print('R7 reference head crossover OBS',obs,flush=True)
        phase = 'score_frozen_h'
        final = head_crossover_scoring.score(output,repo,cfg,c,obs)
        print('R7 reference head crossover FINAL',final,flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json',dict(phase=phase,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
