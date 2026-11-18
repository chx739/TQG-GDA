"""Phase manifests for gated label access and replay of frozen outputs."""
import hashlib
import json
from pathlib import Path

from experiments.common.io import digest, write_json

STAGES = ('R7_INPUT', 'R7_OBS', 'R7_ORACLE', 'R7_FINAL')


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _file(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid sealed path: {name}')
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise FileNotFoundError(path)
    return path


def verify_seal(root, stage):
    root = Path(root)
    parent = None
    for name in STAGES[:STAGES.index(stage) + 1]:
        path = root / 'seals' / f'{name}.json'
        record = json.loads(path.read_text())
        checksum = record.pop('digest')
        if (record['stage'] != name or record['parent'] != parent or not record['files']
                or checksum != _signature(record)):
            raise ValueError(f'Invalid seal: {path}')
        for member, expected in record['files'].items():
            if digest(_file(root, member)) != expected:
                raise ValueError(f'Sealed file has changed: {member}')
        parent = checksum
    return dict(record, digest=checksum)


def seal(root, stage, paths, *, expected):
    """Seal exactly the artifacts declared for an entire phase."""
    root = Path(root)
    paths, expected = set(paths), set(expected)
    if not paths or paths != expected:
        raise ValueError(f'Incomplete {stage}: missing={expected - paths}; extra={paths - expected}')
    index = STAGES.index(stage)
    parent = verify_seal(root, STAGES[index - 1])['digest'] if index else None
    files = {name: digest(_file(root, name)) for name in sorted(paths)}
    record = dict(stage=stage, files=files, parent=parent)
    record['digest'] = _signature(record)
    (root / 'seals').mkdir(exist_ok=True)
    write_json(root / 'seals' / f'{stage}.json', record)
    return record
