"""Freeze all R13 WL-channel predictions before opening the original E labels."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.common import io
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    feature_dropout_transfer, geometry_models, reference_geometry, residual_inputs, true_rate_replay, weighted_residuals, wl)
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer, size_exposure
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.src import readout, screen
from ..src import wl_channel_scale as scale, wl_channel_scoring as scoring


CODE = (io, r5_inputs, seals, composition, r8_inputs, pilot, feature_dropout_transfer, geometry_models,
        reference_geometry, residual_inputs, true_rate_replay, weighted_residuals, wl, structural, structural_observation,
        inputs, prior_run, roc_score, roc_transfer, size_exposure, source_inputs, readout, screen, scale, scoring)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('freeze', 'score'))
    parser.add_argument('--obs-sha256')
    parser.add_argument('--blind-dir', type=Path)
    args = parser.parse_args()
    score = args.phase == 'score'
    if score != (args.obs_sha256 is not None) or score != (args.blind_dir is not None):
        parser.error('score requires --obs-sha256 and --blind-dir')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    cfg = json.loads(args.config.read_bytes())
    protocol = here / cfg['protocol']
    r9, r17 = (here.parent / name for name in (scale.R9, scale.R17))

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    if inputs.sha(git('show', f"{cfg['registration_commit']}:{protocol.relative_to(repo)}")) != io.digest(protocol):
        raise PermissionError('R13 WL channel registration differs')
    head = git('rev-parse', 'HEAD').decode().strip()
    paths = (args.config, protocol, Path(__file__), r9/scale.R9_PLAN, r9/scale.GEOMETRY, r17/scale.R17_PLAN,
             *(Path(m.__file__) for m in CODE))
    hashes = {str(p.resolve().relative_to(repo)): io.digest(p) for p in paths}
    if any(inputs.sha(git('show', f'{head}:{p}')) != h for p, h in hashes.items()):
        raise PermissionError('Current code and configuration must match their Git version')
    out = args.output.resolve()
    dest = out/'score' if score else out
    if dest.exists():
        raise FileExistsError(dest)
    execution = dict(code_commit=head, plan=cfg, code_sha256=hashes, cwd=str(Path.cwd()),
                     command=[sys.executable, *sys.orig_argv[1:]], started_utc=datetime.now(timezone.utc).isoformat(),
                     versions={n: version(n) for n in ('torch', 'torch-geometric', 'numpy', 'scipy')})
    if score:
        stage = roc_transfer.stage(out, 'OBS', args.obs_sha256)
        prev = json.loads(prior_run.read_member(out, stage['files'], 'executed-config.json'))
        if any(execution[k] != prev[k] for k in ('code_sha256', 'plan', 'versions')):
            raise PermissionError('Scoring differs from the frozen execution')
    frozen = scale.Frozen(repo, here, cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.r13-wl-{args.phase}-', dir=dest.parent) as tmp:
        stage = Path(tmp)/'stage'
        stage.mkdir()
        if score:
            sha = scoring.evaluate(out, stage, frozen, args.obs_sha256, args.blind_dir,
                                   execution | dict(label_roles=['E'], obs_sha256=args.obs_sha256))
        else:
            io.write_json(stage/'executed-config.json', execution | dict(label_roles=[]))
            sha = scale.freeze(stage, frozen)
        stage.rename(dest)
    print(f'R13 WL channels {"FINAL" if score else "OBS"} {sha}', flush=True)


if __name__ == '__main__':
    main()
