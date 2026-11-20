"""Authenticate, replay and score the registered frozen GRADE crossover."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs as r8_inputs, pilot
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as label_inputs, prior_run
from ..src import crossover, data, jobs, scoring, training, validation
from ..src.upstream import grade_base, reverse_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    plan = json.loads(config.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    protocol = here / plan['protocol']
    registered = git('show', f"{plan['registration_commit']}:{protocol.relative_to(repo)}")
    if r8_inputs.sha(registered) != io.digest(protocol):
        raise PermissionError('Crossover protocol differs from the registration')
    revision = git('rev-parse', 'HEAD').decode().strip()
    modules = (io, evaluate, paired, r5_inputs, seals, r8_inputs, pilot, label_inputs, prior_run, crossover,
               data, jobs, scoring, training, validation, grade_base, reverse_layer)
    checked = [config, protocol, Path(__file__), *(Path(module.__file__) for module in modules)]
    hashes = {str(path.resolve().relative_to(repo)): io.digest(path) for path in checked}
    for path, checksum in hashes.items():
        if r8_inputs.sha(git('show', f'{revision}:{path}')) != checksum:
            raise PermissionError(f'Working file differs from the recorded revision: {path}')

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    phase = 'input_authentication'
    try:
        authenticated = crossover.authenticate(repo, plan)
        native = authenticated[2]
        opts = native['execution']
        torch.set_num_threads(opts['torch_threads_per_worker'])
        torch.set_num_interop_threads(opts['interop_threads'])
        torch.use_deterministic_algorithms(True)
        io.write_json(output / 'executed-config.json', dict(
            plan=plan, code_commit=revision, code_sha256=hashes,
            command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()),
            python=sys.version, device=opts['device'], training_updates=0,
            versions={p: importlib.metadata.version(p) for p in ('torch', 'torch-geometric', 'numpy', 'scipy')},
            threads=dict(torch=torch.get_num_threads(), interop=torch.get_num_interop_threads()),
            git_status=git('status', '--short').decode().splitlines(),
            started_utc=datetime.now(timezone.utc).isoformat()))
        print('Authenticated 15 original R10 checkpoints and 10 held-out tasks', flush=True)
        phase = 'h_inference'
        obs_sha = crossover.freeze(output, plan, authenticated)
        print('R10 crossover OBS', obs_sha, flush=True)
        phase = 'h_scoring'
        final_sha = crossover.score(output, repo, plan, authenticated, obs_sha)
        print('R10 crossover FINAL', final_sha, flush=True)
    except Exception:
        io.write_json(output / 'execution-failure.json', dict(phase=phase, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
