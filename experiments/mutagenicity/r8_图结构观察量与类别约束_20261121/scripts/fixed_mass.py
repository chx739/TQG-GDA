"""Run the one registered R8 fixed-mass operating-point diagnostic."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common.io import digest, write_json
from ..src import fixed_mass, inputs, pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    r8 = Path(__file__).resolve().parents[1]
    plan = json.loads(args.config.read_bytes())
    parent = (r8/plan['source_attempt']).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo, text=True).strip()

    revision = git('rev-parse', 'HEAD')
    files = [Path(__file__), Path(fixed_mass.__file__), Path(inputs.__file__), Path(pilot.__file__),
             args.config.resolve(), r8/'docs/fixed-mass-diagnostic.md']
    recorded = {str(path.resolve().relative_to(repo)): digest(path) for path in files}
    for relative, checksum in recorded.items():
        blob = subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=repo)
        if inputs.sha(blob) != checksum:
            raise ValueError(f'Commit the exact diagnostic code and configuration first: {relative}')
    write_json(output/'executed-config.json', dict(
        plan=plan, output=str(output), source_attempt=str(parent), code_commit=revision,
        git_status=git('status', '--short'), code_config_sha256=recorded, cwd=str(Path.cwd()),
        command=[sys.executable, *sys.orig_argv[1:]], started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch')}))
    try:
        obs_sha, r8_config = fixed_mass.freeze(parent, output, plan)
        print(f'R8 fixed-mass OBS sealed {obs_sha}', flush=True)
        fixed_mass.score(output, obs_sha, plan, r8_config, repo)
        print(f'R8 fixed-mass FINAL sealed in {output}', flush=True)
    except Exception:
        write_json(output/'execution-failure.json', dict(trace=traceback.format_exc(),
                   obs_created=(output/'OBS.json').exists()))
        raise


if __name__ == '__main__':
    main()
