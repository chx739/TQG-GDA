"""Freeze R18 pair relations and graph actions, then score the original E role."""
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
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs
from experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.scripts import paired_readout
from ..src import anchors, relations, screen, scoring

CODE = (*paired_readout.CODE,paired_readout,anchors,relations,screen,scoring)


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
        raise PermissionError('The registered R18 protocol differs')
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
                      label_phases={'freeze':['D','C'],'score':['E']},threads=1,
                      versions={m:importlib.metadata.version(m) for m in ('torch','torch-geometric','numpy','scipy')}))
    else:
        old = json.loads((output/'executed-config.json').read_bytes())
        if old['code_sha256']!=hashes or (output/'FINAL.json').exists():
            raise PermissionError('R18 scoring requires the frozen revision and a pending OBS')
    phase = 'authenticate_source_data'
    try:
        prepared = screen.prepare(repo,here,cfg)
        if args.phase=='freeze':
            phase = 'fit_D_and_freeze_actions_and_probabilities'
            obs = screen.freeze(output,cfg,prepared)
            print('R18 source screen OBS',obs,flush=True)
        else:
            phase = 'score_E_only'
            final = scoring.score(output,cfg,prepared,args.obs_sha256,execution=dict(**execution,obs_sha256=args.obs_sha256))
            print('R18 source screen FINAL',final,flush=True)
    except Exception:
        try:
            io.write_json(output/f'execution-failure-{args.phase}.json',dict(phase=phase,traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise


if __name__=='__main__':
    main()
