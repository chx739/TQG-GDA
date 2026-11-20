"""Freeze two source-only pseudo-directions, then apply the R9 action upper bound."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from ..src import inputs, screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[4]
    round_root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.config.read_bytes())
    output = args.output.resolve()

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo, text=True).strip()

    file_paths = [args.config.resolve(), round_root/plan['protocol'], Path(composition.__file__),
                  *(p for p in round_root.rglob('*.py') if p.is_file())]
    sha_records = {str(p.resolve().relative_to(repo)): digest(p) for p in file_paths}
    revision = git('rev-parse', 'HEAD')
    for relative, expected in sha_records.items():
        frozen = subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=repo)
        if inputs.sha(frozen) != expected:
            raise ValueError(f'Commit the reviewed implementation before running: {relative}')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'executed-config.json', dict(
        plan=plan, code_commit=revision, git_status=git('status', '--short'),
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy')},
        n_torch_threads=torch.get_num_threads(), code_and_config_sha256=sha_records))
    try:
        obs_sha = screen.freeze(output, plan, repo)
        print(f'R9 OBS sealed {obs_sha}', flush=True)
        verdict = screen.finish(output, obs_sha, plan, repo)
        print(f"R9 FINAL sealed. Passed={verdict['passed']}. Reason={verdict['reason']}. Output={output}", flush=True)
    except Exception:
        write_json(output/'execution-failure.json', dict(trace=traceback.format_exc(),
                   obs_created=(output/'OBS.json').exists()))
        raise


if __name__ == '__main__':
    main()
