"""Freeze ATC choices and score them from already published H aggregates."""
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
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer, thresholds
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    conditional_votes, reverse_vote_observation, reverse_vote_scoring, reverse_votes, vote_mixture)
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from ..src import structural, structural_observation, target_confidence as atc, target_confidence_scoring as scoring


CODE = (io, r5_inputs, seals, composition, r8_inputs, pilot, wl, inputs, prior_run, roc_score, roc_transfer, thresholds,
        conditional_votes, reverse_vote_observation, reverse_vote_scoring, reverse_votes, vote_mixture, historical,
        observation, structural, structural_observation, atc, scoring)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--phase', required=True, choices=('freeze', 'score'))
    ap.add_argument('--obs-sha256')
    ap.add_argument('--blind-script', type=Path)
    args = ap.parse_args()
    score = args.phase == 'score'
    if score != (args.obs_sha256 is not None) or score != (args.blind_script is not None):
        ap.error('score requires both --obs-sha256 and --blind-script')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    cfg = json.loads(args.config.read_bytes())
    protocol = here / cfg['protocol']
    r12 = here.parent / atc.R12
    hist_cfg = r12 / cfg['historical_config']
    reference = json.loads(hist_cfg.read_bytes())
    old_cfg = r12 / reference['historical_config']
    original = json.loads(old_cfg.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    if inputs.sha(git('show', f"{cfg['registration_commit']}:{protocol.relative_to(repo)}")) != io.digest(protocol):
        raise PermissionError('ATC protocol differs from registration')
    revision = git('rev-parse', 'HEAD').decode().strip()
    paths = (args.config, protocol, hist_cfg, old_cfg, Path(__file__), *(Path(m.__file__) for m in CODE))
    hashes = {str(p.resolve().relative_to(repo)): io.digest(p) for p in paths}
    if any(inputs.sha(git('show', f'{revision}:{p}')) != v for p,v in hashes.items()):
        raise PermissionError('Current ATC configuration and code must match Git')
    out = args.output.resolve()
    target = out/'score' if score else out
    if target.exists():
        raise FileExistsError(target)
    execution = dict(code_commit=revision, code_sha256=hashes, plan=cfg, cwd=str(Path.cwd()),
                     started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable,*sys.orig_argv[1:]],
                     versions={m:version(m) for m in ('numpy','scipy','torch','torch-geometric')}, individual_label_reads=[])
    if score:
        obs = roc_transfer.stage(out, 'OBS', args.obs_sha256)
        record = json.loads(prior_run.read_member(out, obs['files'], 'executed-config.json'))
        if any(record[k] != execution[k] for k in ('plan','code_sha256','versions')):
            raise PermissionError('ATC scoring execution differs from the frozen selection')
    archive = atc.Archive(repo, here, cfg, reference, original)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.r10-atc-{args.phase}-', dir=target.parent) as temp:
        stage = Path(temp)/'stage'
        stage.mkdir()
        if score:
            result = scoring.evaluate(out, stage, archive, args.obs_sha256, args.blind_script.resolve(), execution)
        else:
            io.write_json(stage/'executed-config.json', execution)
            result = atc.freeze(stage, archive)
        stage.rename(target)
    print(f'R10 ATC {"FINAL" if score else "OBS"} {result}', flush=True)


if __name__ == '__main__':
    main()
