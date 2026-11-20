"""Freeze and evaluate R7 decision transfers to WL and Size."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common import io
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from ..src import head_crossover, head_crossover_scoring, paired, r5_inputs, seals, transfer_flips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    plan = json.loads(config.read_bytes())
    crossover_config = here/plan['crossover_config']
    cfg = json.loads(crossover_config.read_bytes())
    protocol = here/plan['protocol']

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    registration = plan['registration_commit']
    protocol_path = protocol.relative_to(repo)
    if inputs.sha(git('show', f'{registration}:{protocol_path}')) != io.digest(protocol):
        raise PermissionError('The registered decision-transfer protocol changed')
    revision = git('rev-parse', 'HEAD').decode().strip()
    modules = (io,inputs,prior_run,roc_score,historical,observation,head_crossover,
               head_crossover_scoring,paired,r5_inputs,seals,transfer_flips)
    checked = [config,crossover_config,protocol,Path(__file__),*(Path(m.__file__) for m in modules)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in checked}
    if any(inputs.sha(git('show', f'{revision}:{file}')) != sha for file,sha in hashes.items()):
        raise PermissionError('Actual code or config differs from the recorded revision')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    io.write_json(output/'executed-config.json',
                  dict(code_commit=revision, code_sha256=hashes, plan=plan, crossover_config=cfg,
                       command=[sys.executable,*sys.orig_argv[1:]], cwd=str(Path.cwd()), training_updates=0,
                       label_phases={'freeze':[], 'score':['holdout']},
                       started_utc=datetime.now(timezone.utc).isoformat(),
                       versions={p:importlib.metadata.version(p) for p in ('numpy','scipy')}))

    parent = here/plan['crossover']
    parent_sha = plan['crossover_obs_sha256']
    phase = 'authenticate_and_freeze_transfers'
    try:
        c = historical.CArchive(cfg)
        frozen = transfer_flips.freeze(output, parent, parent_sha, c, cfg)
        print('R7 transfer of flips OBS', frozen, flush=True)
        phase = 'score_frozen_h'
        result = transfer_flips.score(output, parent, parent_sha, repo, c, cfg, plan, frozen)
        print('R7 transfer of flips FINAL', result, flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json', dict(phase=phase,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
