"""Append-only run files, hashes and environment provenance."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import torch


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with open(path,'x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def save_tensor(path, value):
    with open(path,'xb') as f:torch.save(value,f)


def manifest(root, exclude=()):
    return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(Path(root).rglob('*')) if p.is_file() and str(p.relative_to(root)) not in exclude}


def provenance(repo):
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    paths=git('ls-files').splitlines()
    # Untracked new runner files included explicitly before the pre-run commit.
    paths=sorted(set(paths)|{str(p.relative_to(repo)) for folder in ('experiments', 'scripts') for p in (repo/folder).rglob('*.py')})
    hashes={p:digest(repo/p) for p in paths if p.endswith(('.py','.json','.md')) and (repo/p).is_file()}
    return dict(git_commit=git('rev-parse','HEAD'),git_status=git('status','--short'),file_hashes=hashes,
                python=sys.version,torch=torch.__version__,platform=platform.platform(),
                processor=platform.processor(),cpu_count=os.cpu_count(),torch_threads=torch.get_num_threads(),
                command=sys.argv,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
