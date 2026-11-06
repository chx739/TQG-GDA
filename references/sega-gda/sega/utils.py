from __future__ import annotations

import math
import os
import random
import sys
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_dtype(name: str) -> torch.dtype:
    return {"float64": torch.float64, "float32": torch.float32}[name]


def empirical_quantile(values, q: float) -> float:
    """Q_q(X): sort ascending (keep duplicates), take the ceil(q n)-th element (1-indexed)."""
    x = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(x)
    if n == 0:
        raise ValueError("quantile of an empty list")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0, 1]")
    k = int(math.ceil(q * n))
    return float(x[k - 1])


class Logger:
    def __init__(self, path: str | None = None, verbose: int = 1):
        self.verbose = verbose
        self.f = open(path, "a") if path else None
        self.t0 = time.time()

    def log(self, msg: str, level: int = 1) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        if self.verbose >= level:
            print(line, flush=True)
        if self.f is not None:
            self.f.write(line + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f is not None:
            self.f.close()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
