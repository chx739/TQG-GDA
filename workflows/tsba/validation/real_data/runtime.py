"""Append-only phase seals and bounded CPU experiment execution."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import signal
import time

from ..archive import digest, write_json


def seal(root, stage, paths):
    root = Path(root)
    files = {}
    for item in paths:
        item = root / item
        if not item.exists():
            raise FileNotFoundError(item)
        for path in sorted(item.rglob('*')) if item.is_dir() else [item]:
            if path.is_file():
                files[str(path.relative_to(root))] = digest(path)
    value = dict(stage=stage, files=dict(sorted(files.items())))
    value['digest'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (root / 'seals').mkdir(exist_ok=True)
    write_json(root / 'seals' / (stage + '.json'), value)
    return value


def rss_bytes():
    # Linux ru_maxrss is KiB; the experiment records the peak rather than VMS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class RunLog:
    def __init__(self, path, max_seconds=86400, max_rss=8 * 1024**3):
        self.stream = open(path, 'x', buffering=1)
        self.started = time.monotonic()
        self.max_seconds, self.max_rss = max_seconds, max_rss

    def __call__(self, message):
        elapsed = time.monotonic() - self.started
        peak = rss_bytes()
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} elapsed={elapsed:.1f}s peak_rss={peak} {message}'
        self.stream.write(line + '\n')
        print(line, flush=True)
        if elapsed > self.max_seconds or peak > self.max_rss:
            raise TimeoutError('predeclared whole-run time or peak RSS budget exceeded')

    def close(self):
        self.stream.close()


@contextmanager
def time_limit(seconds):
    def expired(signum, frame):
        raise TimeoutError(f'predeclared stage time limit {seconds}s exceeded')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
