"""Freeze and score the registered replay of the R7 two-teacher weight difference."""
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
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from .strong_anchor_refit import CODE_MODULES
from ..src import centered_target_update as runner


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
    plan = json.loads(config.read_bytes())
    parent_config = here/plan['anchor_config']
    cfg = json.loads(parent_config.read_bytes())
    protocol = here/plan['protocol']

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{plan['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('The registered centered replay protocol differs')
    revision = git('rev-parse','HEAD').decode().strip()
    files = [config,parent_config,protocol,Path(__file__),Path(__file__).with_name('strong_anchor_refit.py'),
             Path(runner.__file__),*(Path(m.__file__) for m in CODE_MODULES)]
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
        io.write_json(output/'executed-config.json',dict(**execution,code_sha256=hashes,plan=plan,
                      anchor_config=cfg,training_updates=0,label_phases={'freeze':[],'score':['holdout']},
                      versions={m:importlib.metadata.version(m) for m in ('torch','numpy','scipy')}))
    else:
        old = json.loads((output/'executed-config.json').read_bytes())
        if old['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('Scoring requires exactly the frozen code and a pending OBS')

    phase = 'authenticate_historical_inputs'
    try:
        r10_path = Path(cfg['structural_archive'])
        r10 = roc_transfer.stage(r10_path,'FIT',cfg['structural_fit_sha256'])
        historical_plan = json.loads(prior_run.read_member(r10_path,r10['files'],'executed-config.json'))['plan']
        c = historical.CArchive(dict(cfg,r7_archive=historical_plan['r7_archive']))
        if args.phase=='freeze':
            phase = 'freeze_centered_actions'
            obs = runner.freeze(output,here,plan,cfg,c)
            print('R7 centered-target OBS',obs,flush=True)
        else:
            phase = 'score_frozen_h'
            final = runner.score(output,repo,here,plan,cfg,c,args.obs_sha256,
                                 execution=dict(**execution,obs_sha256=args.obs_sha256))
            print('R7 centered-target FINAL',final,flush=True)
    except Exception:
        try:
            io.write_json(output/f'execution-failure-{args.phase}.json',dict(phase=phase,traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise


if __name__=='__main__':
    main()
