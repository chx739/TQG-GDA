"""Rebuild the R9 fixed learners and score their sealed source-internal supplement."""
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
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer, size_exposure
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from ..src import (feature_dropout_transfer as transfer, feature_dropout_transfer_scoring as scoring, geometry_models,
                   reference_geometry, residual_inputs, true_rate_replay, weighted_residuals, wl)

CODE = (io,r5_inputs,seals,composition,r8_inputs,pilot,structural,structural_observation,inputs,prior_run,roc_score,
        roc_transfer,size_exposure,source_inputs,transfer,scoring,geometry_models,reference_geometry,residual_inputs,
        true_rate_replay,weighted_residuals,wl)


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
    protocol,geometry_path=(here/cfg[k] for k in ('protocol','geometry_config'))
    geometry_cfg = json.loads(geometry_path.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('Original R9 feature dropout transfer protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    paths = [config,protocol,geometry_path,Path(__file__),*(Path(m.__file__) for m in CODE)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in paths}
    if any(inputs.sha(git('show',f'{revision}:{p}'))!=sha for p,sha in hashes.items()):
        raise PermissionError('Actual dropout replay code differs from the recorded version')
    out = args.output.resolve()
    execution = dict(code_commit=revision,cwd=str(Path.cwd()),command=[sys.executable,*sys.orig_argv[1:]],
                     started_utc=datetime.now(timezone.utc).isoformat())
    if args.phase=='score':
        record = json.loads((out/'executed-config.json').read_bytes())
        if record['code_sha256']!=hashes or record['plan']!=cfg:
            raise PermissionError('Scoring provenance differs from the sealed transfer')
    target = out if args.phase=='freeze' else out/'score'
    if target.exists():
        raise FileExistsError(target)
    out.parent.mkdir(parents=True,exist_ok=True)
    reference = reference_geometry.authenticate(here,geometry_cfg)
    screen = transfer.HistoricalScreen(repo,here,cfg,geometry_cfg,reference)
    with TemporaryDirectory(prefix=f'.{out.name}-{args.phase}-',dir=out.parent) as tmp:
        stage = Path(tmp)/'stage'
        stage.mkdir()
        if args.phase=='freeze':
            io.write_json(stage/'executed-config.json',dict(**execution,code_sha256=hashes,plan=cfg,
                label_phases={'freeze':['D','C'],'score':['supplement']},
                versions={m:importlib.metadata.version(m) for m in ('torch','torch-geometric','numpy','scipy','sympy')}))
            sha = transfer.freeze(stage,screen,cfg)
        else:
            sha = scoring.score(out,stage,screen,cfg,args.obs_sha256,dict(**execution,obs_sha256=args.obs_sha256))
        stage.rename(target)
    print(f'R9 feature dropout {"OBS" if args.phase=="freeze" else "FINAL"}',sha,flush=True)


if __name__=='__main__':
    main()
