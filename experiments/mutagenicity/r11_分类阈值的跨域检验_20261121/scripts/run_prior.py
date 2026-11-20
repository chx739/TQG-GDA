"""Replay R11 with supervised and unsupervised controls for target label shift."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import paired, prior_control, r5_inputs, seals
from ..src import inputs, prior_estimation, prior_run, prior_score, run, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    output = args.output.resolve()
    registered = json.loads(config.read_bytes())
    prior_config = repo / registered['r7_prior_config']
    options = json.loads(prior_config.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    files = [config, here / registered['protocol'], prior_config, Path(__file__)]
    modules = (inputs, prior_estimation, prior_run, prior_score, run, thresholds, paired, prior_control, r5_inputs, seals)
    files.extend(Path(module.__file__) for module in modules)
    hashes = {str(path.resolve().relative_to(repo)): digest(path) for path in files}
    for rel, sha in hashes.items():
        committed = subprocess.check_output(['git', 'show', f'{revision}:{rel}'], cwd=repo)
        if inputs.sha(committed) != sha:
            raise ValueError(f'Commit and review R11 supplement before execution: {rel}')
    protocol = str((here / registered['protocol']).relative_to(repo))
    for rel in (protocol, registered['r7_prior_config']):
        registered_bytes = subprocess.check_output(['git', 'show', f"{registered['registration_commit']}:{rel}"], cwd=repo)
        if inputs.sha(registered_bytes) != hashes[rel]:
            raise ValueError(f'Registered source differs: {rel}')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'executed-config.json', dict(registration=registered,
        r7_options={k: options[k] for k in ('calibration', 'mlls')}, code_commit=revision, code_sha256=hashes,
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()), python=sys.version,
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy')}))
    phase = 'UNLABELED'
    try:
        checksum, plan, roots = prior_run.freeze(output, repo, registered, options)
        print('R11 prior UNLABELED sealed', checksum, flush=True)
        phase = 'OBS'
        obs_sha = prior_run.supervise(output, repo, registered, options, plan, roots, checksum)
        print('R11 prior OBS sealed', obs_sha, flush=True)
        phase = 'FINAL'
        result = prior_score.score(output, repo, registered, plan, roots, checksum, obs_sha)
        print('R11 prior FINAL sealed', result, flush=True)
    except Exception:
        write_json(output / 'execution-failure.json', dict(phase=phase, trace=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
