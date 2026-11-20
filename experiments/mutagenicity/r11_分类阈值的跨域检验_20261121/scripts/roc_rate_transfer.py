"""Run the registered R11 ROC transfer pilot without updating historical models."""
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
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from ..src import inputs, prior_estimation, prior_run, roc_rate, roc_score, roc_transfer, run, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    protocol = (here / cfg['protocol']).relative_to(repo)
    if inputs.sha(git('show', f"{cfg['registration_commit']}:{protocol}")) != io.digest(repo / protocol):
        raise PermissionError('Current ROC transfer protocol differs from registration')
    revision = git('rev-parse', 'HEAD').decode().strip()
    modules = (io, r5_inputs, seals, paired, composition, structural, structural_observation, historical, inputs,
               prior_estimation, prior_run, roc_rate, roc_score, roc_transfer, run, thresholds)
    files = [config, Path(__file__), repo / protocol, here / cfg['base_config'],
             *(Path(module.__file__) for module in modules)]
    hashes = {str(path.resolve().relative_to(repo)): io.digest(path) for path in files}
    for file, sha in hashes.items():
        if inputs.sha(git('show', f'{revision}:{file}')) != sha:
            raise PermissionError(f'Actual code or config differs from the recorded revision: {file}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    io.write_json(output / 'executed-config.json', dict(
        code_commit=revision, code_sha256=hashes, plan=cfg,
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()), torch_threads=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions={name: importlib.metadata.version(name) for name in ('torch', 'torch-geometric', 'numpy', 'scipy')}))

    phase = 'historical_inputs'
    try:
        bound = roc_transfer.historical_inputs(repo, here, cfg)
        parents = {k: cfg[k] for k in ('prior_unlabeled_sha256', 'structural_fit_sha256', 'structural_obs_sha256')}
        phase = 'freeze_deployment'
        unlabeled = roc_transfer.freeze(output, repo, cfg, bound, parents)
        print('R11 ROC UNLABELED', unlabeled, flush=True)
        phase = 'published_a_counts'
        observed = roc_transfer.diagnose(output, cfg, bound, parents, unlabeled)
        print('R11 ROC OBS', observed, flush=True)
        phase = 'h_scoring'
        final = roc_score.evaluate(output, repo, cfg, bound, parents, unlabeled, observed)
        print('R11 ROC FINAL', final, flush=True)
    except Exception:
        io.write_json(output / 'execution-failure.json', dict(phase=phase, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
