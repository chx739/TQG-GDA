"""Frozen-feature probes for the v2 assumption study; never updates the GNN."""
from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def ordered(ids, *salt):
    return sorted(map(int, ids), key=lambda i: hashlib.sha256(('|'.join(map(str, (*salt, i)))).encode()).hexdigest())


def split_folds(ids, count, *salt):
    sequence = ordered(ids, *salt)
    return [sequence[i::count] for i in range(count)]


def standardize(train, test):
    mean = train.mean(0)
    scale = train.std(0)
    scale = np.where(scale > 0, scale, 1.)
    return (train-mean)/scale, (test-mean)/scale, mean, scale


def metrics(y, raw):
    y = np.asarray(y, dtype=int)
    raw = np.asarray(raw, dtype=float)
    if not len(y):
        return dict(n=0, nll=None, brier=None, ba=None, outside=None, residual_mse=None, n0=0, n1=0)
    if not np.isfinite(raw).all() or not set(y.tolist()) <= {0, 1}:
        raise ValueError('Nonfinite predictions or invalid binary labels')
    p = np.clip(raw, 1e-6, 1-1e-6)
    recalls = [float(((raw >= .5)[y == k] == k).mean()) for k in [0, 1] if np.any(y == k)]
    return dict(n=len(y), nll=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
                brier=float(((p-y)**2).mean()), ba=float(np.mean(recalls)) if len(recalls) == 2 else None,
                outside=float(((raw < 0) | (raw > 1)).mean()),
                residual_mse=float(((raw-y)**2).mean()), n0=int((y == 0).sum()), n1=int((y == 1).sum()))


def logistic_objective(w, x, y, penalty):
    z = x @ w[:-1]+w[-1]
    loss = np.mean(np.logaddexp(0, z)-y*z)+.5*penalty*np.dot(w[:-1], w[:-1])
    error = (expit(z)-y)/len(y)
    grad = np.r_[x.T@error+penalty*w[:-1], error.sum()]
    return float(loss), grad


@dataclass(frozen=True)
class ProbeConfig:
    krr_grid: tuple = (.001, .01, .1, 1.)
    logistic_grid: tuple = (.01, .1, 1., 10.)
    mlp_grid: tuple = (.001, .01, .1)
    hidden: int = 32
    steps: int = 200
    lr: float = .001
    logistic_max_iterations: int = 500
    log_every: int = 20


def fit_grid(family, x, y_variants, train_p, eval_x, eval_p, k_train, k_eval,
             q, penalties, seed, config=ProbeConfig(), history=None, *, paired=False):
    """All variants and penalties are independent fits; batching only saves overhead.

    Returns probability predictions [variant, penalty, evaluation graph], states,
    convergence information. No evaluation labels accepted by this interface.
    """
    ys = np.asarray(y_variants, dtype=float)
    v, n = ys.shape
    penalties = np.asarray(penalties, dtype=float)
    width = 1 if paired else len(penalties)
    specifications = [(i, 0, penalties[i]) for i in range(v)] if paired else [(i, j, value) for i in range(v) for j, value in enumerate(penalties)]
    if paired and len(penalties) != v:
        raise ValueError("One refit penalty per label variant required")
    predictions = np.empty((v, width, len(eval_x)))
    states = {}
    info = []
    if family == 'KRR':
        eigenvalues, vectors = np.linalg.eigh((k_train+k_train.T)/2)
        if eigenvalues.min() < -1e-8:
            raise ValueError('Indefinite training kernel')
        eigenvalues = eigenvalues.clip(0)
        residual = (ys-train_p[None, :])*(q[1]-q[0])
        projected = vectors.T@residual.T
        for i, j, penalty in specifications:
            alpha = vectors@(projected[:, i]/(eigenvalues+penalty))
            predictions[i, j] = eval_p+q[1]*(k_eval@alpha)
            states[i, j] = dict(alpha=alpha)
            info.append(dict(variant=i, penalty=float(penalty), converged=True, iterations=1))
        return predictions, states, info
    x, eval_x, mean, scale = standardize(x, eval_x)
    if family == 'LR':
        for i, j, penalty in specifications:
            result = minimize(logistic_objective, np.zeros(x.shape[1]+1), args=(x, ys[i], penalty),
                              jac=True, method='L-BFGS-B', options=dict(maxiter=config.logistic_max_iterations,
                              ftol=1e-12, gtol=1e-8, maxls=40))
            if not np.isfinite(result.x).all():
                raise FloatingPointError('Nonfinite logistic fit')
            predictions[i, j] = expit(eval_x@result.x[:-1]+result.x[-1])
            states[i, j] = dict(weight=result.x, mean=mean, scale=scale)
            info.append(dict(variant=i, penalty=float(penalty), converged=bool(result.success),
                             iterations=int(result.nit), message=str(result.message)))
            if history:
                history(dict(family=family, variant=i, penalty=float(penalty), step=int(result.nit), loss=float(result.fun)))
        return predictions, states, info
    if family != 'MLP':
        raise ValueError(family)
    # Exactly the same initialization is used across penalties and label nulls.
    generator = torch.Generator().manual_seed(seed)
    count = len(specifications)
    d, hidden = x.shape[1], config.hidden
    def parameter(shape, fan):
        value = (torch.rand(shape, generator=generator, dtype=torch.float64)*2-1)/np.sqrt(fan)
        return value.unsqueeze(0).expand(count, *shape).clone().requires_grad_(True)
    w1, b1 = parameter((d, hidden), d), parameter((hidden,), d)
    w2, b2 = parameter((hidden, 1), hidden), parameter((1,), hidden)
    params = [w1, b1, w2, b2]
    tx = torch.from_numpy(x).unsqueeze(0).expand(count, -1, -1)
    targets = torch.from_numpy(np.stack([ys[i] for i, _, _ in specifications]))
    decay = torch.from_numpy(np.array([penalty for _, _, penalty in specifications]))
    optimizer = torch.optim.Adam(params, lr=config.lr)
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        output = (torch.relu(tx@w1+b1[:, None])@w2+b2[:, None]).squeeze(-1)
        per = torch.nn.functional.binary_cross_entropy_with_logits(output, targets, reduction='none').mean(1)
        regularizer = sum(p.square().flatten(1).sum(1) for p in params)*decay*.5
        loss = (per+regularizer).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite MLP training loss')
        loss.backward()
        optimizer.step()
        if history and ((step+1) % config.log_every == 0):
            for idx, value in enumerate(per.detach().tolist()):
                history(dict(family=family, variant=specifications[idx][0], penalty=float(specifications[idx][2]), step=step+1, loss=value))
    with torch.no_grad():
        ex = torch.from_numpy(eval_x).unsqueeze(0).expand(count, -1, -1)
        pred = torch.sigmoid((torch.relu(ex@w1+b1[:, None])@w2+b2[:, None]).squeeze(-1)).numpy()
    predictions[:] = pred.reshape(v, width, -1)
    for idx, (i, j, penalty) in enumerate(specifications):
        states[i, j] = dict(w1=w1[idx].detach().numpy(), b1=b1[idx].detach().numpy(),
                            w2=w2[idx].detach().numpy(), b2=b2[idx].detach().numpy(), mean=mean, scale=scale)
        info.append(dict(variant=i, penalty=float(penalty), converged=None, iterations=config.steps,
                         message='fixed step protocol; no convergence claim'))
    return predictions, states, info


def support_statistics(dist_ss, dist_sb, kernels_sb):
    loo = dist_ss.copy()
    np.fill_diagonal(loo, np.inf)
    threshold = float(np.quantile(loo.min(1), .9))
    nearest = dist_sb.min(0)
    maximum = kernels_sb.max(0)
    total = kernels_sb.sum(0)
    denominator = (kernels_sb**2).sum(0)
    ess = np.divide(total**2, denominator, out=np.full_like(total, np.nan), where=denominator > np.finfo(float).tiny)
    return dict(threshold=threshold, distance=nearest, near=nearest <= threshold,
                max_kernel=maximum, ess=ess)


class ProbeLabelBoundary:
    """A labels train diagnostic probes; B labels require a complete prediction seal."""
    def __init__(self, task, a, b, final, path):
        self.task, self.a, self.b, self.final = task, set(a), set(b), set(final)
        if self.a & self.b or (self.a | self.b) & self.final:
            raise ValueError('Overlapping diagnostic roles')
        self.path = Path(path)
        self.sealed = None

    def log(self, record):
        with self.path.open('a') as f:
            f.write(json.dumps(dict(time=time.time(), **record), allow_nan=False)+'\n')

    def seal(self, digest, models):
        self.sealed = digest
        self.log(dict(event='prediction_sealed', sha256=digest, models=models))

    def read(self, role, ids):
        ids = list(map(int, ids))
        allowed = self.a if role == 'A_train' else self.b if role == 'B_evaluation' else set()
        valid = bool(ids) and set(ids) <= allowed and not set(ids) & self.final
        valid = valid and (role != 'B_evaluation' or self.sealed is not None)
        self.log(dict(event='label_access' if valid else 'access_denied', role=role, ids=ids,
                      prediction_sha256=self.sealed))
        if not valid:
            raise PermissionError('Diagnostic target label boundary rejected request')
        from torch_geometric.data.separate import separate
        result = []
        for i in ids:
            graph = separate(self.task._raw.__class__, self.task._raw, i, self.task.slices, decrement=False)
            result.append(self.task.class_map[int(graph.y.item())])
        return np.asarray(result, dtype=int)
