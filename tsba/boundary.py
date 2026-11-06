"""Finite-radius projection and detached normal displacement requests (§3.7–3.9)."""
from collections import Counter
from dataclasses import dataclass
import math
import torch
from .utils import finite


@dataclass
class BoundaryBatch:
    points: torch.Tensor
    pairs: torch.Tensor
    normal_norm: torch.Tensor
    old_margin: torch.Tensor
    zeta: torch.Tensor
    normals: torch.Tensor
    rejected: dict

    def __len__(self):
        return len(self.points)


def margin_and_normal(model, point, pair):
    with torch.enable_grad():
        point = point.detach().requires_grad_(True)
        out = model(point.unsqueeze(0))[0]
        margin = out[pair[0]] - out[pair[1]]
        normal, = torch.autograd.grad(margin, point)
    finite(margin, normal, out)
    return margin.detach(), normal.detach(), out.detach()


def project(model, x, pair, radius, tolerance, cfg):
    point = x.detach().clone()
    for _ in range(cfg.projection_steps):
        margin, gradient, _ = margin_and_normal(model, point, pair)
        norm = gradient.norm()
        finite(norm)
        if float(norm) <= cfg.min_normal:
            return None, "small_normal"
        residual = margin.abs() / norm
        finite(residual)
        if float(residual) <= tolerance:
            break
        direction = -margin * gradient / (norm.square() + cfg.projection_eps)
        finite(direction)
        accepted = False
        for back in range(cfg.projection_backtracks):
            candidate = point + (0.5 ** back) * direction
            offset = candidate - x
            offset_norm = offset.norm()
            finite(offset_norm)
            candidate = x + offset * torch.clamp(radius / offset_norm.clamp_min(1e-30), max=1)
            cm, cg, _ = margin_and_normal(model, candidate, pair)
            candidate_norm = cg.norm()
            finite(candidate_norm)
            if float(candidate_norm) > cfg.min_normal:
                candidate_residual = cm.abs() / candidate_norm
                finite(candidate_residual)
            else:
                continue
            if float(candidate_residual) < float(residual):
                point, accepted = candidate.detach(), True
                break
        if not accepted:
            return None, "projection_stalled"
    margin, gradient, out = margin_and_normal(model, point, pair)
    norm = gradient.norm()
    finite(norm)
    if float(norm) <= cfg.min_normal:
        return None, "small_normal"
    residual = margin.abs() / norm
    finite(residual)
    if float(residual) > tolerance:
        return None, "root_tolerance"
    if float((point - x).norm()) > radius * (1 + 1e-6):
        return None, "radius"
    if out.numel() > 2:
        mask = torch.ones_like(out, dtype=torch.bool)
        mask[pair] = False
        if bool((out[mask] > out[pair].min() + tolerance * norm).any()):
            return None, "inactive_class_pair"
    normal = gradient / norm
    finite(normal)
    return (point, margin, normal, norm), None


def build_boundaries(model, target, score, sigma, cfg):
    radius = cfg.radius_multiplier * sigma * math.sqrt(target.shape[1])
    tolerance = cfg.root_tolerance * sigma
    rows, rejected = [], Counter()
    for x in target.detach():
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
            finite(logits)
            # Fix binary orientation; multi-class ties resolve by class index.
            pair = (torch.tensor([0, 1], device=x.device) if len(logits) == 2 else
                    torch.argsort(logits, descending=True, stable=True)[:2])
        projected, reason = project(model, x, pair, radius, tolerance, cfg)
        if projected is None:
            rejected[reason] += 1
            continue
        point, old_margin, normal, norm = projected
        with torch.no_grad():
            value = score(point.unsqueeze(0))[0]
            finite(value)
            gamma = normal @ value
            # zeta = -nu = clip(tau sigma^2 gamma), with frozen reference quantities.
            zeta = (cfg.tau * sigma ** 2 * gamma).clamp(-cfg.request_multiplier * sigma,
                                                      cfg.request_multiplier * sigma)
            finite(gamma, zeta)
        rows.append((point, pair, norm, old_margin, zeta, normal))
    if rows:
        columns = [torch.stack(column).detach() for column in zip(*rows)]
        return BoundaryBatch(*columns, dict(rejected))
    d = target.shape[1]
    return BoundaryBatch(target.new_empty((0, d)), torch.empty((0, 2), device=target.device, dtype=torch.long),
                         target.new_empty(0), target.new_empty(0), target.new_empty(0),
                         target.new_empty((0, d)), dict(rejected))
