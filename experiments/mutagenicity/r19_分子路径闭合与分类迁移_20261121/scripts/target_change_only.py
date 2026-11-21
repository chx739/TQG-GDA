"""Freeze and independently check the R19 target-only correction before scoring."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.common import io
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from . import source_screen as original_entry
from ..src import screen_inputs, target_change, target_change_scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('freeze', 'score'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--obs-sha256')
    parser.add_argument('--blind-dir', type=Path)
    args = parser.parse_args()
    scoring = args.phase == 'score'
    if scoring != (args.obs_sha256 is not None) or scoring != (args.blind_dir is not None):
        parser.error('score requires both --obs-sha256 and --blind-dir')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())
    registration = here / cfg['protocol']
    source_cfg = here / cfg['source_config']
    r9 = here.parent / screen_inputs.R9

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    if inputs.sha(git('show', f"{cfg['registration_commit']}:{registration.relative_to(repo)}")) != io.digest(registration):
        raise PermissionError('R19 target-change protocol differs')
    commit = git('rev-parse', 'HEAD').decode().strip()
    sources = (config, registration, source_cfg, Path(__file__), Path(original_entry.__file__),
               r9 / screen_inputs.GEOMETRY_CONFIG, r9 / screen_inputs.STATIC_CONFIG,
               *(Path(m.__file__) for m in (*original_entry.CODE, target_change, target_change_scoring)))
    hashes = {str(p.resolve().relative_to(repo)): io.digest(p) for p in sources}
    if any(inputs.sha(git('show', f'{commit}:{p}')) != sha for p, sha in hashes.items()):
        raise PermissionError('Actual R19 target-only code or configuration differs from commit')
    run = args.output.resolve()
    target = run / 'score' if scoring else run
    if target.exists():
        raise FileExistsError(target)
    exe = dict(code_commit=commit, plan=cfg, code_sha256=hashes, command=[sys.executable, *sys.orig_argv[1:]],
               cwd=str(Path.cwd()), started_utc=datetime.now(timezone.utc).isoformat(),
               versions={m: version(m) for m in ('numpy', 'scipy', 'torch', 'torch-geometric')})
    if scoring:
        record = roc_transfer.stage(run, 'OBS', args.obs_sha256)
        prior_execution = json.loads(prior_run.read_member(run, record['files'], 'executed-config.json'))
        if any(exe[k] != prior_execution[k] for k in ('plan', 'code_sha256', 'versions')):
            raise PermissionError('R19 target-only scoring differs from the sealed code or plan')
    archive = screen_inputs.Archive(repo, here, json.loads(source_cfg.read_bytes()))
    frozen = target_change.Frozen(archive, here, cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.r19-delta-{args.phase}-', dir=target.parent) as tmp:
        stage = Path(tmp) / 'stage'
        stage.mkdir()
        if scoring:
            sha = target_change_scoring.evaluate(run, stage, frozen, args.obs_sha256, args.blind_dir,
                                                 exe | dict(label_roles=['supplement'], obs_sha256=args.obs_sha256))
        else:
            io.write_json(stage / 'executed-config.json', exe | dict(label_roles=[]))
            sha = target_change.freeze(stage, frozen)
        stage.rename(target)
    print(f'R19 target-only {"FINAL" if scoring else "OBS"} {sha}', flush=True)


if __name__ == '__main__':
    main()
