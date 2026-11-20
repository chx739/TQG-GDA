"""Authenticate, fit, seal, and evaluate the registered R8 development pilot."""
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
from ..src import inputs, pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'run'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[4]
    round_root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.config.read_bytes())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.stage == 'preflight':
        record = inputs.preflight(plan, repo)
        write_json(output/'input-checks.json', record)
        print(f"Verified {len(record['blocks'])} blocks and {record['topology']['graph_count']} graphs without opening labels")
        return

    def git(*command):
        return subprocess.check_output(['git', *command], cwd=repo, text=True).strip()

    own_paths = [p for p in round_root.rglob('*.py') if p.is_file()]
    r7 = inputs.prior_donor(repo).parents[2] / 'src'
    shared = ['r5_inputs', 'seals', 'fit', 'evaluate', 'prior_control', 'decisions', 'paired', 'p_risk_replay']
    members = [*own_paths, *(r7/f'{name}.py' for name in shared), args.config.resolve(), round_root/plan['protocol']]
    sha_records = {str(path.relative_to(repo)): digest(path) for path in members}
    revision = git('rev-parse', 'HEAD')
    for relative, expected in sha_records.items():
        frozen = subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=repo)
        if inputs.sha(frozen) != expected:
            raise ValueError(f'Commit reviewed R8 code and configuration before execution: {relative}')
    write_json(output/'executed-config.json', dict(
        plan=plan, code_commit=revision, git_status=git('status', '--short'),
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy')},
        n_torch_threads=torch.get_num_threads(), code_and_config_sha256=sha_records))
    try:
        obs_sha, original = pilot.make_observation(output, plan, repo)
        print(f'R8 OBS sealed {obs_sha}', flush=True)
        result = pilot.evaluate_observation(output, obs_sha, original, plan, repo)
        print(f"R8 FINAL sealed. Gate passed={result['passed']}. Output {output}", flush=True)
    except Exception:
        write_json(output/'execution-failure.json', dict(trace=traceback.format_exc(),
                   obs_created=(output/'OBS.json').exists()))
        raise


if __name__ == '__main__':
    main()
