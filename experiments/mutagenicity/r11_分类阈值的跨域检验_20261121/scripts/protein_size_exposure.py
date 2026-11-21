"""Freeze the registered protein size curves and score the original H roles."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.common import io
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from ..src import (inputs, prior_run, roc_rate, roc_score, roc_transfer, size_exposure,
                   size_exposure_observation as obs, size_exposure_scoring as scoring, thresholds)

CODE = (io,r5_inputs,seals,composition,r8_inputs,pilot,structural,structural_observation,historical,
        inputs,prior_run,roc_rate,roc_score,roc_transfer,size_exposure,obs,scoring,thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',required=True,choices=('freeze','score'))
    parser.add_argument('--obs-sha256')
    args = parser.parse_args()
    if (args.phase=='score')!=(args.obs_sha256 is not None):
        parser.error('--obs-sha256 is required only for score')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())
    protocol = here/cfg['protocol']
    roc_path = here/cfg['roc_config']
    roc_cfg = json.loads(roc_path.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('Registered protein exposure protocol differs')
    revision = git('rev-parse','HEAD').decode().strip()
    paths = [config,protocol,roc_path,here/roc_cfg['base_config'],Path(__file__),*(Path(m.__file__) for m in CODE)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in paths}
    if any(inputs.sha(git('show',f'{revision}:{p}'))!=sha for p,sha in hashes.items()):
        raise PermissionError('Protein exposure files differ from Git')
    out = args.output.resolve()
    execution = dict(code_commit=revision,cwd=str(Path.cwd()),command=[sys.executable,*sys.orig_argv[1:]],
                     started_utc=datetime.now(timezone.utc).isoformat())
    if args.phase=='score':
        actual = json.loads((out/'executed-config.json').read_bytes())
        if actual['code_sha256']!=hashes or actual['plan']!=cfg or actual['roc_plan']!=roc_cfg:
            raise PermissionError('Scoring provenance differs from the sealed fit')
    target = out if args.phase=='freeze' else out/'score'
    if target.exists():
        raise FileExistsError(target)
    out.parent.mkdir(parents=True,exist_ok=True)
    archive = obs.Archive(repo,here,cfg,roc_cfg)
    with TemporaryDirectory(prefix=f'.{out.name}-{args.phase}-',dir=out.parent) as tmp:
        stage = Path(tmp)/'stage'
        stage.mkdir()
        if args.phase=='freeze':
            io.write_json(stage/'executed-config.json',dict(**execution,code_sha256=hashes,plan=cfg,roc_plan=roc_cfg,
                          label_phases={'freeze':['source_fit'],'score':['holdout','source_validation_audit']},
                          versions={m:importlib.metadata.version(m) for m in ('torch','numpy','scipy','sympy')}))
            sha = obs.freeze(stage,archive,cfg,roc_cfg)
        else:
            sha = scoring.score(out,stage,archive,cfg,roc_cfg,args.obs_sha256,
                                dict(**execution,obs_sha256=args.obs_sha256))
        stage.rename(target)
    print(f'R11 protein size exposure {"OBS" if args.phase=="freeze" else "FINAL"}',sha,flush=True)


if __name__=='__main__':
    main()
