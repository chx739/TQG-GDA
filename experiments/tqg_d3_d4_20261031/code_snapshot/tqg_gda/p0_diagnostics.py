"""Frozen-model P0 diagnostics. Target labels exist only in DevelopmentOracle.

The production training path is unchanged. Geometry has no label argument;
one-step arms use functional parameter copies and cannot advance a trajectory.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data.separate import separate

from .calibration_v2 import CalibrationConfig, solve_calibration
from .feedback_v2 import (ParameterSnapshot, V2Config, contrast_basis,
                          query_matrices, signature_kernels, target_space)
from .metrics import classification_metrics


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    header = f'{value.dtype}|{tuple(value.shape)}|'.encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def append_json(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + '\n')


class DevelopmentOracle:
    """Explicitly enabled, sealed, dev-only labels; logs denied requests too."""
    def __init__(self, task, development_ids, final_ids, log_path, *, enabled=False):
        self.task = task
        self.allowed = set(map(int, development_ids))
        self.final = set(map(int, final_ids))
        if self.allowed & self.final or not self.allowed <= set(task.target_ids):
            raise ValueError('Invalid Oracle development/final identity boundary')
        self.log_path = Path(log_path)
        self.enabled = enabled
        self.seals = set()
        self.access_count = 0

    def seal(self, token, a_ids, *, theta_hash, readout_hash):
        if not set(a_ids) <= self.allowed or set(a_ids) & self.final:
            raise PermissionError('Cannot seal IDs outside development')
        self.seals.add(token)
        append_json(self.log_path, dict(event='unlabeled_query_sealed', token=token,
                    ids=list(a_ids), theta_hash=theta_hash, readout_hash=readout_hash,
                    time=time.time()))

    def labels(self, ids, *, token, purpose):
        ids = list(map(int, ids))
        permitted = (self.enabled and token in self.seals and bool(ids)
                     and set(ids) <= self.allowed and not set(ids) & self.final)
        append_json(self.log_path, dict(event='label_access' if permitted else 'access_denied',
                    scope='development_oracle', token=token, purpose=purpose, ids=ids,
                    time=time.time()))
        if not permitted:
            raise PermissionError('Explicit enabled and sealed development-only Oracle required')
        values = []
        for index in ids:
            graph = separate(self.task._raw.__class__, self.task._raw, index,
                             self.task.slices, decrement=False)
            raw = int(graph.y.item())
            if raw not in self.task.class_map:
                raise ValueError('Oracle label outside source vocabulary')
            values.append(self.task.class_map[raw])
        self.access_count += len(ids)
        return torch.tensor(values, dtype=torch.long)


@dataclass
class Geometry:
    snapshot: ParameterSnapshot
    basis: torch.Tensor
    contrasts: torch.Tensor
    source_response: torch.Tensor
    target_response: torch.Tensor
    readout: torch.Tensor
    source_probabilities: torch.Tensor
    source_representation: torch.Tensor
    target_representation: torch.Tensor
    statistics: dict

    def gradient(self, labels):
        residual = ((F.one_hot(labels, self.source_probabilities.shape[1]).double()
                     - self.source_probabilities) @ self.contrasts).reshape(-1)
        coefficients = self.readout @ residual
        return -(self.basis @ coefficients), residual


def unlabeled_geometry(model, source_inputs, target_inputs, config=V2Config()):
    started = time.perf_counter()
    if getattr(source_inputs, 'y', None) is not None or getattr(target_inputs, 'y', None) is not None:
        raise PermissionError('Geometry accepts label-free inputs only')
    model.eval()
    snapshot = ParameterSnapshot(model)
    source_logits = snapshot.logits(source_inputs).detach()
    target_logits = snapshot.logits(target_inputs).detach()
    contrasts = contrast_basis(source_logits.shape[1], dtype=torch.float64, device='cpu')
    jacobian = snapshot.jacobian(target_inputs, contrasts)
    basis, stats = target_space(jacobian)
    sh, th = snapshot.representation(source_inputs).detach(), snapshot.representation(target_inputs).detach()
    if basis.shape[1] == 0:
        source_a = source_logits.new_zeros((len(source_logits), contrasts.shape[1], 0))
        target_a = target_logits.new_zeros((len(target_logits), contrasts.shape[1], 0))
        w = source_logits.new_zeros((0, len(source_logits) * contrasts.shape[1]))
        stats.update(status='zero_rank', calibration=dict(converged=True, termination='zero_rank', gap=0.))
    else:
        source_a = snapshot.response(source_inputs, contrasts, basis)
        target_a = (jacobian @ basis).reshape(len(target_logits), contrasts.shape[1], -1)
        kss, kst, ktt, kernel_stats = signature_kernels(sh, th, source_a, target_a, config)
        hst, htt = query_matrices(kst, ktt, target_a)
        solve_started = time.perf_counter()
        w, calibration = solve_calibration(kss, hst, htt, CalibrationConfig(**config.calibration))
        stats['solve_seconds'] = time.perf_counter() - solve_started
        stats.update(kernel_stats, calibration=calibration, status='gradient_feedback')
    stats['weight_norm'] = float(w.norm())
    stats['singular_values'] = torch.linalg.svdvals(jacobian).tolist()
    stats['geometry_seconds'] = time.perf_counter() - started
    return Geometry(snapshot, basis.detach(), contrasts.detach(), source_a.detach(),
                    target_a.detach(), w.detach(), source_logits.softmax(-1), sh, th, stats)


def label_free(batch):
    result = batch.clone()
    result.y = None
    return result


def gradient_of_ce(snapshot, batches, labels):
    """Graph-weighted average, including a possibly shorter last microbatch."""
    total = len(labels)
    result = torch.zeros_like(snapshot.flat)
    offset = 0
    for batch in batches:
        count = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
        variable = snapshot.flat.detach().requires_grad_(True)
        logits = snapshot.logits(batch, variable)
        loss = F.cross_entropy(logits, labels[offset:offset+count], reduction='sum') / total
        gradient, = torch.autograd.grad(loss, variable, allow_unused=True)
        if gradient is not None:
            result += gradient.detach()
        offset += count
    if offset != total or not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid gradient count or nonfinite gradient')
    return result


@torch.no_grad()
def evaluate_flat(snapshot, flat, batches, labels):
    logits = torch.cat([snapshot.logits(batch, flat) for batch in batches])
    return classification_metrics(logits, labels)


def normalized(vector, norm, epsilon):
    size = float(vector.norm())
    if size <= epsilon or norm <= epsilon:
        return torch.zeros_like(vector), False
    return vector * (norm / size), True


def cosine(left, right, epsilon):
    ln, rn = float(left.norm()), float(right.norm())
    return float(torch.dot(left, right) / (ln * rn)) if min(ln, rn) > epsilon else None


def gradient_metrics(estimate, g_a, g_b, g_source, basis, *, epsilon=1e-12, beta=.1):
    norms = [float(x.norm()) for x in (estimate, g_a, g_b, g_source)]
    e, a, b, s = norms
    b_inside = basis @ (basis.mT @ g_b)
    return dict(gradient_norm=e, oracle_A_norm=a, oracle_B_norm=b, source_norm=s,
                c_A=cosine(estimate, g_a, epsilon), c_B=cosine(estimate, g_b, epsilon),
                e_A=float((estimate-g_a).norm()) / a if a > epsilon else None,
                a_A=float(torch.dot(estimate, g_a)) / a**2 if a > epsilon else None,
                norm_ratio=e/a if a > epsilon else None,
                weighted_ratio=beta*e/s if s > epsilon else None,
                cosine_source=cosine(estimate, g_source, epsilon),
                B_space_outside_energy_fraction=float((g_b-b_inside).square().sum())/b**2 if b > epsilon else None,
                c_B_projected=cosine(estimate, b_inside, epsilon),
                valid_comparison=min(e, a) > epsilon,
                invalid_reason='zero_estimate' if e <= epsilon else ('zero_oracle_A' if a <= epsilon else ''),
                zero_source=s <= epsilon)


def sample_ids(pool, count, seed):
    generator = torch.Generator().manual_seed(seed)
    return tuple(pool[i] for i in torch.randperm(len(pool), generator=generator)[:count].tolist())


def development_pools(ids, domain):
    ordered = sorted(map(int, ids), key=lambda i: hashlib.sha256(
        f'tqg-v2-diag-20261031|domain={domain}|id={i}'.encode()).hexdigest())
    half = len(ordered) // 2
    if len(ordered) != 542 or half != 271:
        raise ValueError('P0 requires original 542-development roles')
    return tuple(ordered[:half]), tuple(ordered[half:])


def controls(geometry, source_labels, block_seed, permutation_offsets, random_offsets, epsilon):
    true, residual = geometry.gradient(source_labels)
    arms = {'TRUE': true}
    validity = {'TRUE': float(true.norm()) > epsilon}
    for rep, offset in enumerate(permutation_offsets, 1):
        generator = torch.Generator().manual_seed(block_seed + offset)
        labels = source_labels[torch.randperm(len(source_labels), generator=generator)]
        arms[f'PERM-{rep}'], _ = geometry.gradient(labels)
        validity[f'PERM-{rep}'] = float(arms[f'PERM-{rep}'].norm()) > epsilon
    for rep, offset in enumerate(random_offsets, 1):
        generator = torch.Generator().manual_seed(block_seed + offset)
        u = torch.randn(geometry.basis.shape[1], generator=generator, dtype=torch.float64)
        arms[f'RAND-{rep}'], validity[f'RAND-{rep}'] = normalized(
            geometry.basis @ u, float(true.norm()), epsilon)
    arms['ZERO'] = torch.zeros_like(true)
    validity['ZERO'] = False
    return arms, validity, residual


def one_step_arms(gradients, g_a, g_source, beta, epsilon):
    true = gradients['TRUE']
    norm = float(true.norm())
    oracle, oracle_valid = normalized(g_a, norm, epsilon)
    arms = {'S': g_source, 'Q': beta*true, 'O-M': beta*oracle, 'O-RAW': g_a,
            'S+Q': g_source+beta*true, 'S+O-M': g_source+beta*oracle}
    validity = {name: True for name in arms}
    validity['O-M'] = validity['S+O-M'] = oracle_valid
    for kind in ('PERM', 'RAND'):
        for rep in range(1, 4):
            vector, valid = normalized(gradients[f'{kind}-{rep}'], norm, epsilon)
            name = f'S+{"P" if kind == "PERM" else "R"}-{rep}'
            arms[name] = g_source + beta*vector
            validity[name] = valid
    return arms, validity


def fixed_isotropic(kss, hst, lambda_w):
    """Independent direct condition solve, Omega=I/r => D=2I."""
    q = hst.shape[0] // len(kss)
    gram = torch.kron(kss, torch.eye(q, dtype=kss.dtype))
    return torch.linalg.solve(2*gram + lambda_w*torch.eye(len(gram), dtype=kss.dtype), 2*hst).mT
