"""Run the registered Mutagenicity-edge R19 source-only screen."""
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
    feature_dropout_transfer, reference_geometry, residual_inputs, true_rate_replay, weighted_residuals, wl)
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs
from ..src import path_closure, screen_inputs, source_screen


CODE = (io, r5_inputs, seals, composition, r8_inputs, pilot, feature_dropout_transfer, reference_geometry,
        residual_inputs, true_rate_replay, weighted_residuals, wl, structural, structural_observation,
        inputs, prior_run, roc_score, roc_transfer, source_inputs, path_closure, screen_inputs, source_screen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('freeze', 'score'))
    parser.add_argument('--obs-sha256')
    args = parser.parse_args()
    if (args.phase == 'score') != (args.obs_sha256 is not None):
        parser.error('--obs-sha256 is required only for score')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())
    protocol = here / cfg['protocol']
    if (cfg['setting'], cfg['domains'], cfg['seed'], cfg['evaluation_count']) != ('mutagenicity_edge', [0, 1, 2, 3], 3405, 869):
        raise ValueError('Unexpected scope for R19')
    r9 = here.parent / screen_inputs.R9

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    if inputs.sha(git('show', f"{cfg['registration_commit']}:{protocol.relative_to(repo)}")) != io.digest(protocol):
        raise PermissionError('The registered R19 protocol changed')
    revision = git('rev-parse', 'HEAD').decode().strip()
    paths = [config, protocol, r9 / screen_inputs.GEOMETRY_CONFIG, r9 / screen_inputs.STATIC_CONFIG,
             Path(__file__), *(Path(m.__file__) for m in CODE)]
    hashes = {str(p.resolve().relative_to(repo)): io.digest(p) for p in paths}
    if any(inputs.sha(git('show', f'{revision}:{p}')) != sha for p, sha in hashes.items()):
        raise PermissionError('Actual R19 code or configuration differs from recorded version')
    out = args.output.resolve()
    landmark = 'OBS' if args.phase == 'freeze' else 'FINAL'
    target = out if args.phase == 'freeze' else out / 'score'
    if target.exists():
        raise FileExistsError(target)
    execution = dict(code_commit=revision, code_sha256=hashes, plan=cfg, cwd=str(Path.cwd()),
                     command=[sys.executable, *sys.orig_argv[1:]], started_utc=datetime.now(timezone.utc).isoformat(),
                     versions={m: version(m) for m in ('numpy', 'scipy', 'torch', 'torch-geometric')})
    if args.phase == 'score':
        obs = roc_transfer.stage(out, 'OBS', args.obs_sha256)
        freeze_execution = json.loads(prior_run.read_member(out, obs['files'], 'executed-config.json'))
        for key in ('code_sha256', 'plan', 'versions'):
            if freeze_execution[key] != execution[key]:
                raise PermissionError(f'Score differs from the sealed execution: {key}')
    archive = screen_inputs.Archive(repo, here, cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.r19-{args.phase}-', dir=target.parent) as tmp:
        stage = Path(tmp) / 'stage'
        stage.mkdir()
        if args.phase == 'freeze':
            io.write_json(stage / 'executed-config.json', execution | dict(label_roles=['D', 'C']))
            sha = source_screen.freeze(stage, archive)
        else:
            sha = source_screen.score(out, stage, archive, args.obs_sha256,
                                      execution | dict(label_roles=['E'], obs_sha256=args.obs_sha256))
        stage.rename(target)
    print(f'R19 {landmark} {sha}', flush=True)


if __name__ == '__main__':
    main()
