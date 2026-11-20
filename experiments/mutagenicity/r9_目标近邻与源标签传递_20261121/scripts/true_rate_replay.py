"""Run the registered three-way exact rate diagnostic from frozen R9 scores."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common import io
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_rate, roc_score, roc_transfer, thresholds
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from ..src import residual_inputs, residual_models, true_rate_replay, weighted_residuals


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
        raise PermissionError('Registered R9 rate replay protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    modules = (io,inputs,prior_run,roc_rate,roc_score,roc_transfer,thresholds,source_inputs,
               residual_inputs,residual_models,weighted_residuals,true_rate_replay)
    files = [config,Path(__file__),repo/protocol,*(Path(m.__file__) for m in modules)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in files}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or config differs from the recorded revision')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    io.write_json(output/'executed-config.json',dict(code_commit=revision,code_sha256=hashes,plan=cfg,
        command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),training_updates=0,
        label_phases={'freeze':[], 'score':['E']},started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')}))
    phase = 'authenticate_parent'
    try:
        reference = true_rate_replay.authenticate(here,cfg)
        phase = 'freeze_24_directions'
        obs = true_rate_replay.freeze(output,cfg,reference)
        print('R9 true-rate OBS',obs,flush=True)
        phase = 'score_E_only'
        final = true_rate_replay.score(output,repo,cfg,reference,obs)
        print('R9 true-rate FINAL',final,flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json',dict(phase=phase,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
