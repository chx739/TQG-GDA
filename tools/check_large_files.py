#!/usr/bin/env python3
"""Check Git objects, not working-tree sizes; permit only pinned legacy evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

LIMIT = 5 * 1024 * 1024
POLICY = '.github/large-file-exceptions.json'


def git(*args, input=None):
    return subprocess.check_output(['git', *args], input=input)


def check(all_files=False):
    root = git('rev-parse', '--show-toplevel').decode().strip()
    import os
    os.chdir(root)
    entries = {}
    for entry in git('ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, oid, stage = meta.split()
        if stage != b'0':
            raise ValueError('Resolve unmerged files before checking.')
        if mode == b'160000':
            continue
        entries[name.decode('utf-8')] = oid.decode()
    # Read policy from the index too: an unstaged edit must not change the check.
    policy = json.loads(git('show', ':' + POLICY)) if POLICY in entries else {'exceptions': []}
    allowed = {(e['path'], e['blob']) for e in policy['exceptions'] if e.get('reason')}
    names = set(entries) if all_files else {
        p.decode('utf-8') for p in git('diff', '--cached', '--name-only', '--diff-filter=ACMRT', '-z').split(b'\0') if p
    }
    selected = {p: entries[p] for p in names if p in entries}
    ids = sorted(set(selected.values()))
    sizes = {}
    if ids:
        for line in git('cat-file', '--batch-check=%(objectname) %(objectsize)', input=('\n'.join(ids)+'\n').encode()).splitlines():
            oid, size = line.split()
            sizes[oid.decode()] = int(size)
    failures = [(p, sizes[oid]) for p, oid in selected.items() if sizes[oid] > LIMIT and (p, oid) not in allowed]
    for path, size in sorted(failures):
        print(f'{path}: {size / 1024**2:.2f} MiB exceeds 5 MiB. Store runtime artifacts outside Git.', file=sys.stderr)
    if not failures:
        print(f'Large-file check passed ({len(selected)} indexed files).')
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='check all indexed files (CI)')
    group.add_argument('--staged', action='store_true', help='check staged changes (default)')
    args = parser.parse_args()
    return check(args.all)


if __name__ == '__main__':
    sys.exit(main())
