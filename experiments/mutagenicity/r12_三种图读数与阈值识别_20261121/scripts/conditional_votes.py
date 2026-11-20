"""Run and archive the registered R12 joint-vote natural-domain pilot."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import paired, r5_inputs, seals
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer, run, thresholds
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from ..src import conditional_votes, vote_mixture, vote_scoring


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
    if inputs.sha(git('show',f"{cfg['registration_commit']}:{protocol}")) != io.digest(repo/protocol):
        raise PermissionError('Registered R12 conditional-vote protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    modules = (io,paired,r5_inputs,seals,wl,structural,structural_observation,inputs,prior_run,roc_score,roc_transfer,
               run,thresholds,historical,observation,conditional_votes,vote_mixture,vote_scoring)
    checked = [config,Path(__file__),repo/protocol,*(Path(m.__file__) for m in modules)]
    hashes = {str(path.resolve().relative_to(repo)):io.digest(path) for path in checked}
    if any(inputs.sha(git('show',f'{revision}:{file}'))!=sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or configuration differs from the recorded revision')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    io.write_json(output/'executed-config.json',dict(code_commit=revision,code_sha256=hashes,plan=cfg,
        command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),torch_threads=1,
        git_status=git('status','--short').decode().splitlines(),started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')}))

    phase = 'historical_input_authentication'
    try:
        history = conditional_votes.Archive(repo,cfg)
        phase = 'freeze_source_and_adaptation'
        obs = conditional_votes.freeze(output,repo,cfg,history)
        print('R12 conditional votes OBS',obs,flush=True)
        phase = 'score_holdout'
        final = vote_scoring.score(output,repo,cfg,history,obs)
        print('R12 conditional votes FINAL',final,flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json',dict(phase=phase,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
