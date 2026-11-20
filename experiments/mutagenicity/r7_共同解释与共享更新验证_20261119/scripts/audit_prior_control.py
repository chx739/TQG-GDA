"""Fit the preregistered source-calibrated MLLS control and score only after OBS."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from experiments.common.io import digest, write_json
from ..src import prior_control, prior_control_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Alternate location of the registered P archive')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    plan = json.loads(args.config.read_bytes())
    archive = (args.input or Path(plan['input'])).resolve()
    output = args.output.resolve()

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo, text=True).strip()

    code = [Path(__file__), Path(prior_control.__file__), Path(prior_control_run.__file__)]
    write_json(output/'executed-config.json', dict(
        plan=plan, source_archive=str(archive), output=str(output), cwd=str(Path.cwd()),
        commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
        code_hashes={str(p.resolve().relative_to(repo)): digest(p) for p in code}, config_sha256=digest(args.config),
        command=[sys.executable, *sys.orig_argv[1:]], started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch')}))
    obs, original = prior_control_run.freeze(archive, plan, repo, output)
    print(f'Prior control OBS sealed {obs}', flush=True)
    prior_control_run.score(archive, plan, repo, output, obs, original)
    print(f'Prior control FINAL sealed in {output}', flush=True)


if __name__ == '__main__':
    main()
