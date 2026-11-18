"""Copy a sealed R7 phase's public evidence without changing the raw run."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.common.io import digest, manifest, write_json

from ..src.seals import STAGES, verify_seal


SUMMARY_FILES = ('executed-configs.json', 'executed-plan.json', 'engineering.json', 'engineering-regression.txt')
SCORED_FILES = {
    'endpoints.csv', 'gate.json', 'paired.csv', 'moments.csv', 'coverage.csv', 'separation.csv',
    'paired-audit.json', 'paired_new.csv', 'paired_audit.csv', 'block_contrasts.csv',
    'paired_new.json', 'paired_audit.json', 'paired.json',
}


def archive(source, destination, log):
    source, destination, log = (Path(path).resolve() for path in (source, destination, log))
    records = {stage: verify_seal(source, stage) for stage in STAGES}
    final_names = records['R7_FINAL']['files'].keys()
    public_names = set(final_names) & SCORED_FILES
    if not {'gate.json', 'endpoints.csv', 'paired.csv'} <= public_names:
        raise ValueError('The phase lacks public endpoints, paired measurements, or a verdict')
    sources = {name: source / name for name in public_names}
    for name in SUMMARY_FILES:
        if name in records['R7_INPUT']['files']:
            sources[name] = source / name
    sources.update({f'{stage}.json': source / 'seals' / f'{stage}.json' for stage in STAGES})
    provenance = source / 'provenance.json'
    if 'provenance.json' not in records['R7_INPUT']['files'] or not log.is_file():
        raise ValueError('Execution provenance or stdout log is missing')
    identity = json.loads(provenance.read_text())
    index = dict(root=str(source), execution_commit=identity['git_commit'],
                 provenance_sha256=digest(provenance), files=manifest(source),
                 seals={stage: spec['digest'] for stage, spec in records.items()},
                 stdout={'path': str(log), 'sha256': digest(log)},
                 omitted_final_artifacts=sorted(final_names - public_names))
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        shutil.copyfile(path, destination / name)
        if digest(destination / name) != digest(path):
            raise ValueError(f'Public copy does not match the sealed source: {name}')
    write_json(destination / 'archive-index.json', index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    archive(args.source, args.destination, args.log)


if __name__ == '__main__':
    main()
