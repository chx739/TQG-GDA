"""Prepare and run the matched graph-level GRADE experiment."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from experiments.common.io import digest, write_json
from ..src import data, jobs


def prepare(root, config_path, repo):
    plan = json.loads(config_path.read_bytes())
    round_root = Path(__file__).resolve().parents[1]

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo, text=True).strip()

    revision = git('rev-parse','HEAD')
    files = [path for path in round_root.rglob('*') if path.is_file()
             and (path.suffix == '.py' or path.parent.name == 'upstream') and '__pycache__' not in path.parts]
    files += [config_path.resolve(),round_root/plan['protocol']]
    recorded = {}
    for path in files:
        relative = str(path.relative_to(repo))
        expected = git('rev-parse', f'{revision}:{relative}')
        if git('hash-object',relative) != expected:
            raise ValueError(f'Commit the reviewed implementation before preparing R10: {relative}')
        recorded[relative] = digest(path)
    source = json.loads((round_root/'src/upstream/SOURCE.json').read_bytes())
    for name, record in source['files'].items():
        path = round_root/'src/upstream'/name
        if git('hash-object', str(path)) != record['git_blob']:
            raise ValueError(f'Upstream Git blob differs: {name}')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'executed-config.json', dict(plan=plan, code_commit=revision, git_status=git('status','--short'),
        command=[sys.executable,*sys.orig_argv[1:]], cwd=str(Path.cwd()),
        prepared_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')},
        code_hashes=recorded, upstream=source))
    data.prepare(root, repo, plan)
    names = (root/'executed-config.json', root/'roles.json', *sorted((root/'data').iterdir()))
    write_json(root/'INPUT.json', dict(stage='R10_INPUT', files={str(p.relative_to(root)):digest(p) for p in names}))
    print(f'Prepared {root}', flush=True)


def main():
    this = 'experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','worker','train','evaluate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job')
    parser.add_argument('--first-seed-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[4]
    if args.stage == 'prepare':
        prepare(root,args.config,repo)
    elif args.stage == 'worker':
        jobs.model_job(root,args.job)
    elif args.stage == 'train':
        jobs.supervise(root,this,first_seed_only=args.first_seed_only)
    else:
        from ..src import scoring
        scoring.run(root,repo)


if __name__ == '__main__':
    main()
