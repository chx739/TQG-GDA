"""Finite-value checks, reproducibility and content identities."""
import hashlib
import json
import random
import numpy as np
import torch


class NumericalError(RuntimeError):
    """An execution must stop without committing its candidate parameters."""


def finite(*values):
    for value in values:
        if not bool(torch.isfinite(torch.as_tensor(value)).all()):
            raise NumericalError("non-finite value")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(t.dtype), list(t.shape))).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
