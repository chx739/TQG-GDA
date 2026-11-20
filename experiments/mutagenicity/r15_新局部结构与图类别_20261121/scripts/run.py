"""Run the registered R15 screen once in a new output directory."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from experiments.common import io
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import inputs, roles, wl

from ..src import screen, source_inputs


THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    threads = {key: os.environ.get(key) for key in THREAD_VARIABLES}
    if set(threads.values()) != {'1'}:
        parser.error('Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS to 1 before running')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    repo = root.parents[2]
    config = root / 'configs/source-screen-plan.json'
    protocol = root / 'docs/source-screen.md'
    plan = json.loads(config.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    revision = git('rev-parse', 'HEAD').decode().strip()
    if git('show', f"{plan['registration_commit']}:{protocol.relative_to(repo)}") != protocol.read_bytes():
        raise PermissionError('R15 protocol differs from the registered commit')
    dependencies = [Path(m.__file__).resolve() for m in (io, composition, inputs, roles, wl)]
    paths = [config, protocol, *sorted(root.rglob('*.py')), *dependencies]
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(repo))
        actual = io.digest(path)
        if inputs.sha(git('show', f'{revision}:{relative}')) != actual:
            raise PermissionError(f'Commit the reviewed implementation before running: {relative}')
        hashes[relative] = actual
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    io.write_json(output / 'executed-config.json', dict(
        plan=plan, code_commit=revision, git_status=git('status', '--short').decode().strip(),
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        versions={name: version(name) for name in ('torch', 'numpy', 'scipy')},
        blas_thread_environment=threads, n_torch_threads=torch.get_num_threads(),
        code_and_config_sha256=hashes,
        label_scope='Reuse sealed R9 D fits. Parse E offsets only after both predictions and the dictionaries are sealed.'))
    try:
        original = source_inputs.Original(repo, plan)
        obs_sha = screen.freeze(output, plan, original)
        print(f'R15 OBS sealed {obs_sha}', flush=True)
        final_sha, verdict = screen.finish(output, obs_sha, plan, original)
        print(f"R15 FINAL sealed {final_sha}. Status={verdict['status']}. Passed={verdict['passed']}", flush=True)
    except Exception:
        with (output / 'failure.log').open('x') as log:
            traceback.print_exc(file=log)
        raise


if __name__ == '__main__':
    main()
