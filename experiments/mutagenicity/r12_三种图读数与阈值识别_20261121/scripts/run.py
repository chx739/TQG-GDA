"""Run the registered R12 source-only threshold identification screen."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from ..src import identification, readouts, screen, source_inputs, worlds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--authenticate-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    plan = json.loads(config_path.read_bytes())

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo)

    amendment = plan['construction_amendment_commit']
    revision = git('rev-parse', 'HEAD').decode().strip()
    protocol = here / plan['protocol']
    spec = source_inputs.ValidationArchive(plan).plan['provenance']
    registry = {key: repo / spec[key] for key in ('r5_archive_index', 'r5_actual_config')}
    prior_options = repo / plan['r9_source_options']
    files = [config_path, protocol, prior_options, *registry.values(), Path(__file__)]
    modules = (identification, readouts, screen, source_inputs, worlds, r5_inputs, seals, inputs, pilot, composition, wl)
    files.extend(Path(module.__file__) for module in modules)
    hashes = {str(p.resolve().relative_to(repo)): digest(p) for p in files}
    for p in files:
        rel = str(p.resolve().relative_to(repo))
        if p in (protocol, prior_options, *registry.values()):
            if inputs.sha(git('show', f'{amendment}:{rel}')) != hashes[rel]:
                raise PermissionError(f'Registered dependency differs: {rel}')
        if not args.authenticate_only and inputs.sha(git('show', f'{revision}:{rel}')) != hashes[rel]:
            raise PermissionError(f'Commit and review the R12 dependency before execution: {rel}')
    options = json.loads(prior_options.read_bytes())['source_model']
    index, actual = (json.loads(registry[key].read_bytes()) for key in ('r5_archive_index', 'r5_actual_config'))
    if args.authenticate_only:
        screen.run(None, repo, plan, index, actual, options, authenticate_only=True)
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'executed-config.json', dict(
        plan=plan, r9_source_model=options, code_commit=revision, file_sha256=hashes, cwd=str(Path.cwd()),
        command=[sys.executable, *sys.orig_argv[1:]], started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy')}))
    try:
        result = screen.run(output, repo, plan, index, actual, options)
        print('R12 FINAL', json.dumps(result, ensure_ascii=False), flush=True)
    except Exception:
        write_json(output / 'execution-failure.json', {'exception': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
