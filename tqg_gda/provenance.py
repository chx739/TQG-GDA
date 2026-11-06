"""Content identities for SEGA-style caches and completed TQG tasks."""
from pathlib import Path
import platform

import torch
import torch_geometric

from .utils import sha256, fingerprint


def code_identity():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'tqg_gda').rglob('*.py')) + list((root/'workflows/tqg').rglob('*.py'))
    paths += [root/'pyproject.toml', root/'docs/v1/method/design.md', root/'docs/v2/method/design.md']
    hashes = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    return dict(sha256=fingerprint(hashes), files=hashes)


def runtime_identity(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else torch.device(config.device)
    if device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                pyg=torch_geometric.__version__, device=str(device), dtype=config.dtype,
                cuda=torch.version.cuda, num_threads=config.num_threads,
                deterministic=config.deterministic)


def source_identity(task):
    return {k: v for k, v in task.identity.items()
            if k not in ('target_ids', 'target_index_sha256', 'target_policy', 'fingerprint')}


def source_key(config, task, code, runtime):
    # v1/v2 and all target tasks share precisely the same source initialization.
    return fingerprint(dict(schema='tqg-sega-source-1', data=source_identity(task),
        encoder=config.to_dict()['encoder'], pretrain=config.to_dict()['pretrain'],
        seed=config.seed, runtime=runtime, code=code['sha256']))


def task_key(config, task, code, runtime):
    settings = config.to_dict()
    settings.pop('out_dir')
    return fingerprint(dict(schema='tqg-sega-task-1', config=settings,
                            data=task.identity, code=code['sha256'], runtime=runtime))


def verify_artifacts(root, records):
    root = Path(root).resolve()
    if not records:
        raise ValueError('Missing artifact manifest')
    for name, expected in records.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Artifact is missing or its bytes changed: ' + name)
