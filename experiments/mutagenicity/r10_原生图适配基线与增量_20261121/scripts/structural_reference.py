"""Run the registered R10 structural baseline with ordered FIT, OBS and FINAL seals."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from experiments.common import io
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs as r8_inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation as r13_observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import scoring as r13_scoring
from ..src import structural, structural_observation, structural_scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--output', type=Path)
    parser.add_argument('--reviewed-commit', help='Implementation revision approved for the formal run')
    args = parser.parse_args()
    if args.output and not args.reviewed_commit:
        parser.error('--output requires --reviewed-commit')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    plan = json.loads(config.read_bytes())
    structural.validate_plan(plan)

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    protocol = (here / plan['protocol']).relative_to(repo)
    if r8_inputs.sha(git('show', f"{plan['registration_commit']}:{protocol}")) != io.digest(repo / protocol):
        raise PermissionError('Protocol differs from the R10 registration')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.preflight:
        print(json.dumps(structural_observation.preflight(repo, plan), indent=2))
        return

    threads = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    if any(value != '1' for value in threads.values()):
        parser.error('Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 when launching')
    status = git('status', '--short', '--branch').decode()
    revision = git('rev-parse', args.reviewed_commit).decode().strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    git('merge-base', '--is-ancestor', revision, head)
    checked = [config, repo / protocol, Path(__file__)]
    modules = (io, evaluate, paired, r5_inputs, seals, composition, r8_inputs, pilot, wl, inputs, prior_run, run,
               historical, r13_observation, r13_scoring, structural, structural_observation, structural_scoring)
    checked.extend(Path(module.__file__) for module in modules)
    code = {str(path.resolve().relative_to(repo)): io.digest(path) for path in checked}
    for name, checksum in code.items():
        if r8_inputs.sha(git('show', f'{revision}:{name}')) != checksum:
            raise PermissionError(f'Working file differs from the reviewed implementation: {name}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    io.write_json(output / 'executed-config.json', dict(
        plan=plan, reviewed_code_commit=revision, checkout_commit=head, code_sha256=code, git_status=status,
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()), device='cpu',
        versions={p: importlib.metadata.version(p) for p in ('torch', 'torch-geometric', 'numpy', 'scipy')},
        threads=dict(torch=torch.get_num_threads(), interop=torch.get_num_interop_threads(), blas=1,
                     environment=threads),
        started_utc=datetime.now(timezone.utc).isoformat()))
    phase = 'source_fit'
    try:
        fit_sha = structural_observation.fit(output, repo, plan)
        print('R10 structural FIT sealed', fit_sha, flush=True)
        phase = 'h_inference'
        obs_sha = structural_observation.freeze(output, repo, plan, fit_sha)
        print('R10 structural OBS sealed', obs_sha, flush=True)
        phase = 'h_scoring'
        final = structural_scoring.score(output, repo, plan, fit_sha, obs_sha)
        print('R10 structural FINAL sealed', final, flush=True)
    except Exception:
        io.write_json(output / 'execution-failure.json', dict(phase=phase, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
