"""Verify archived experiment bytes and evaluation-to-checkpoint links (stdlib only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {}
    for item in manifest['files']:
        name = item['path']
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in entries:
            raise ValueError('Unsafe or duplicate archive path: ' + name)
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('Missing file or changed size: ' + name)
        if sha256(path) != item['sha256']:
            raise ValueError('Changed checksum: ' + name)
        entries[name] = item
    with (root / 'evaluation_index.csv').open(newline='', encoding='utf-8-sig') as stream:
        evaluations = list(csv.DictReader(stream))
    for row in evaluations:
        item = entries.get(row['archived_checkpoint'])
        if item is None or item['sha256'] != row['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint mismatch: ' + str(row))
        if row['archived_history'] not in entries:
            raise ValueError('Evaluation has no training history: ' + row['archived_history'])
    with (root / 'job_index.csv').open(newline='', encoding='utf-8-sig') as stream:
        jobs = list(csv.DictReader(stream))
    for row in jobs:
        for key in ('history', 'last_checkpoint', 'selected_checkpoint', 'source_checkpoint'):
            if row.get(key) and row[key] not in entries:
                raise ValueError('Job link missing: ' + row[key])
    stats = manifest['statistics']
    histories = sum(item['category'] == 'history' for item in entries.values())
    checkpoints = sum(item['category'] == 'checkpoint' for item in entries.values())
    if (len(jobs), len(evaluations), histories, checkpoints) != (
        stats['indexed_jobs'], stats['evaluation_records'],
        stats['history_files'], stats['checkpoint_files']
    ):
        raise ValueError('Archive counts differ from the manifest')
    return dict(passed=True, verified_files=len(entries), jobs=len(jobs),
                evaluations=len(evaluations), history_files=histories,
                checkpoint_files=checkpoints,
                total_bytes=sum(item['bytes'] for item in entries.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == '__main__':
    main()
