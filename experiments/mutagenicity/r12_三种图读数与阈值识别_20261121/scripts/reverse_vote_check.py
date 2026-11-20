"""Freeze R12 reverse-vote tables, or score them from published H aggregates."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from ..src import conditional_votes, reverse_vote_observation, reverse_vote_scoring, reverse_votes, vote_mixture

CODE = (io,r5_inputs,seals,composition,r8_inputs,pilot,wl,structural,structural_observation,
        inputs,prior_run,roc_score,roc_transfer,historical,observation,conditional_votes,
        reverse_vote_observation,reverse_vote_scoring,reverse_votes,vote_mixture)


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
    path = args.config.resolve()
    cfg = json.loads(path.read_bytes())
    protocol,supplement = (here/cfg[k] for k in ('protocol','supplement'))
    old_path = here/cfg['historical_config']
    old_cfg = json.loads(old_path.read_bytes())
    root = here/cfg['historical_attempt']

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    for doc,key in ((protocol,'registration_commit'),(supplement,'supplement_commit')):
        if inputs.sha(git('show',f"{cfg[key]}:{doc.relative_to(repo)}"))!=io.digest(doc):
            raise PermissionError(f'R12 reverse-check registration document differs: {doc.name}')
    revision = git('rev-parse','HEAD').decode().strip()
    files = [path,old_path,protocol,supplement,Path(__file__),*(Path(m.__file__) for m in CODE)]
    checksums = {str(p.resolve().relative_to(repo)):io.digest(p) for p in files}
    if any(inputs.sha(git('show',f'{revision}:{p}'))!=sha for p,sha in checksums.items()):
        raise PermissionError('Actual reverse-check code or configuration differs from the recorded version')
    reverse_vote_observation.plan(cfg,old_cfg)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = args.output.resolve()
    execution = dict(code_commit=revision,command=[sys.executable,*sys.orig_argv[1:]],
                     started_utc=datetime.now(timezone.utc).isoformat(),cwd=str(Path.cwd()))
    if args.phase=='freeze':
        out.mkdir(parents=True,exist_ok=False)
        io.write_json(out/'executed-config.json',dict(**execution,code_sha256=checksums,plan=cfg,
                      release=dict(allowed_graph_labels=['source_validation'],target_graph_labels=[],
                                   score_uses=['published_h-cells.csv','published_endpoints.csv']),
                      training_updates=0,threads=1,
                      versions={m:importlib.metadata.version(m) for m in ('torch','torch-geometric','numpy','scipy')}))
    else:
        record = json.loads((out/'executed-config.json').read_bytes())
        if record['code_sha256']!=checksums or (out/'FINAL.json').exists():
            raise PermissionError('Reverse-check scoring requires the frozen code and a pending OBS')
    try:
        if args.phase=='freeze':
            digest = reverse_vote_observation.freeze(out,repo,root,cfg,old_cfg)
            print('R12 reverse-vote OBS',digest,flush=True)
        else:
            digest = reverse_vote_scoring.score(out,root,cfg,old_cfg,args.obs_sha256,
                                                execution=dict(**execution,obs_sha256=args.obs_sha256))
            print('R12 reverse-vote FINAL',digest,flush=True)
    except Exception:
        name = out/f'execution-failure-{args.phase}.json'
        if not name.exists():
            io.write_json(name,dict(phase=args.phase,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
