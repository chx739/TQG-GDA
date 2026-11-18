"""Phase manifests for gated label access and replay of frozen outputs."""
from contextlib import contextmanager
import hashlib
import json
import os
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


def _verify_chain(root, stage):
    """Fully hash one chain and retain its authenticated seal metadata bytes."""
    root = Path(root).resolve()
    parent = None
    records, seal_hashes = {}, {}
    for name in STAGES[:STAGES.index(stage) + 1]:
        path = root / 'seals' / f'{name}.json'
        payload = path.read_bytes()
        record = json.loads(payload)
        checksum = record.pop('digest')
        if (record['stage'] != name or record['parent'] != parent or not record['files']
                or checksum != _signature(record)):
            raise ValueError(f'Invalid seal: {path}')
        for member, expected in record['files'].items():
            if digest(_file(root, member)) != expected:
                raise ValueError(f'Sealed file has changed: {member}')
        parent = checksum
        records[name] = dict(record, digest=checksum)
        seal_hashes[name] = hashlib.sha256(payload).hexdigest()
    return records, seal_hashes


def verify_seal(root, stage):
    records, _ = _verify_chain(root, stage)
    return records[stage]


class _VerifiedSealChain:
    """A live, process-local capability; never a reusable authorization cache."""

    def __init__(self, root, stage, records, seal_hashes):
        self._root = Path(root).resolve()
        self._stage = stage
        self._records = records
        self._seal_hashes = seal_hashes
        self._pid = os.getpid()
        self._active = True

    def __reduce__(self):
        raise TypeError('A verified stage capability cannot leave its evaluation scope')

    def _check_active(self):
        if not self._active or os.getpid() != self._pid:
            raise PermissionError('Verified stage capability is closed or belongs to another process')

    def _check_metadata(self):
        for stage, expected in self._seal_hashes.items():
            payload = (self._root / 'seals' / f'{stage}.json').read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise PermissionError(f'Verified stage metadata changed: {stage}')

    def read_bytes(self, relative, stage, *, required_stage=None):
        """Read an exact stage member under the required label authorization.

        Seal metadata and the returned member bytes are rehashed on EVERY read.
        A later stage appearing on disk cannot upgrade this capability.
        """
        self._check_active()
        if stage not in self._records or (required_stage or stage) not in self._records:
            raise PermissionError('The verified scope does not authorize the required stage')
        self._check_metadata()
        path = _file(self._root, relative)
        # Canonical names must match membership exactly, without traversal.
        name = str(path.relative_to(self._root))
        expected = self._records[stage]['files'].get(name)
        if expected is None:
            raise PermissionError(f'File is not a member of the required stage: {name}')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise PermissionError(f'Verified stage member changed: {name}')
        return payload

    def _close(self):
        self._check_active()
        self._active = False
        self._check_metadata()
        records, hashes = _verify_chain(self._root, self._stage)
        if records != self._records or hashes != self._seal_hashes:
            raise PermissionError('Verified stage chain changed during evaluation')


@contextmanager
def verified_seal_chain(root, stage):
    """Authenticate a batch scope, then recheck the whole chain before exit.

    Both entry and exit hash ALL files in INPUT through ``stage``. In between,
    ``read_bytes`` authenticates each requested member against the captured
    chain. This avoids rescanning unrelated score/checkpoint files per label
    request. Callers must keep evaluation results private until normal context
    exit; a changed unrelated artifact is rejected by the exit check. The
    surrounding phase must continue treating sealed files as immutable.
    """
    root = Path(root).resolve()
    records, hashes = _verify_chain(root, stage)
    capability = _VerifiedSealChain(root, stage, records, hashes)
    try:
        capability._check_metadata()
        yield capability
    finally:
        capability._close()


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
