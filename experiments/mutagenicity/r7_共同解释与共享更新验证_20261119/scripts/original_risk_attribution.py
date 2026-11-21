"""Replay the reference R7 risk on ten original sealed P tasks."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.common import io
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import pilot
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from tsba.shared_update import explanations, solver
from ..src import (decisions, evaluate, fit, frozen_moment, frozen_moment_observation as obs,
                   frozen_moment_scoring as scoring, p_risk_replay, r5_inputs, restored_separation, seals)


CODE = (io, pilot, inputs, prior_run, roc_transfer, explanations, solver, decisions, evaluate, fit, frozen_moment,
        obs, scoring, p_risk_replay, r5_inputs, restored_separation, seals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--phase',required=True,choices=('freeze','score'))
    p.add_argument('--obs-sha256')
    a = p.parse_args()
    do_score = a.phase=='score'
    if do_score!=(a.obs_sha256 is not None):
        p.error('score requires --obs-sha256')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    cfg = json.loads(a.config.read_bytes())
    protocol = here/cfg['protocol']
    root = Path(cfg['archive'])

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('The registered original-risk protocol differs')
    head = git('rev-parse','HEAD').decode().strip()
    paths = (a.config,protocol,Path(__file__),*(Path(m.__file__) for m in CODE))
    hashes = {str(v.resolve().relative_to(repo)):io.digest(v) for v in paths}
    if any(inputs.sha(git('show',f'{head}:{k}'))!=v for k,v in hashes.items()):
        raise PermissionError('Original-risk execution does not match committed code')
    out = a.output.resolve()
    dest = out/'score' if do_score else out
    if dest.exists():
        raise FileExistsError(dest)
    exe = dict(code_commit=head,plan=cfg,code_sha256=hashes,cwd=str(Path.cwd()),
               command=[sys.executable,*sys.orig_argv[1:]],started_utc=datetime.now(timezone.utc).isoformat(),
               versions={n:version(n) for n in ('numpy','scipy','torch','torch-geometric')})
    if do_score:
        record = roc_transfer.stage(out,'OBS',a.obs_sha256)
        old = json.loads(prior_run.read_member(out,record['files'],'executed-config.json'))
        if any(old[k]!=exe[k] for k in ('plan','code_sha256','versions')):
            raise PermissionError('Original-risk scoring differs from sealed execution')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=f'.r7-original-{a.phase}-',dir=dest.parent) as temp:
        stage = Path(temp)/'stage'
        stage.mkdir()
        if do_score:
            sha = scoring.evaluate_frozen(out,stage,root,cfg,repo,a.obs_sha256,
                                         exe | dict(target_label_roles=['adapt','holdout'],obs_sha256=a.obs_sha256))
        else:
            io.write_json(stage/'executed-config.json',exe | dict(target_label_roles=[]))
            sha = obs.freeze(stage,root,cfg)
        stage.rename(dest)
    print(f'R7 original risk {"FINAL" if do_score else "OBS"} {sha}',flush=True)


if __name__=='__main__':
    main()
