"""Publish completed stage outputs; preserve an existing run on retry."""
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def digest(path):
    with Path(path).open('rb') as stream:
        h = sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
        return h.hexdigest()


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key {key!r}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_keys)


@contextmanager
def output_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(dir=path.parent, prefix='.cela-', delete=False) as tmp:
        name = Path(tmp.name)
    try:
        yield name
        os.link(name, path)
    finally:
        name.unlink(missing_ok=True)


def write_json(path, data):
    with output_file(path) as tmp:
        tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                                  indent=2, sort_keys=True) + '\n', encoding='utf-8')
