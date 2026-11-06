"""Damped GGN proposals: direct or matrix-free JVP/VJP CG (§3.11)."""
import torch
from torch.func import grad, jacrev, jvp, vjp
from .utils import finite


class GGNSystem:
    def __init__(self, problem):
        self.problem = problem
        self.base = problem.base
        self.p = problem.source_logits(self.base).softmax(-1).detach()
        self.gs = grad(problem.source_loss)(self.base).detach()
        self.coefficient = problem.cfg.lambda_boundary / len(problem.boundaries)
        _, self.source_pullback = vjp(problem.source_logits, self.base)
        _, self.margin_pullback = vjp(problem.normalized_margin, self.base)
        self.rhs = (-self.gs + self.coefficient * self.margin_pullback(problem.boundaries.zeta)[0]).detach()
        finite(self.p, self.gs, self.rhs)

    def categorical_product(self, value):
        return self.p * (value - (self.p * value).sum(-1, keepdim=True))

    def matvec(self, direction):
        source_jv = jvp(self.problem.source_logits, (self.base,), (direction,))[1]
        margin_jv = jvp(self.problem.normalized_margin, (self.base,), (direction,))[1]
        fs = self.source_pullback(self.categorical_product(source_jv) / len(self.p))[0]
        db = self.margin_pullback(margin_jv)[0]
        result = (fs + self.coefficient * db + self.problem.cfg.damping * direction).detach()
        finite(result)
        return result

    def dense(self):
        source_j = jacrev(self.problem.source_logits)(self.base).detach()
        d = jacrev(self.problem.normalized_margin)(self.base).detach()
        hj = self.p[..., None] * (source_j - (self.p[..., None] * source_j).sum(1, keepdim=True))
        fs = torch.einsum("bcp,bcq->pq", source_j, hj) / len(self.p)
        matrix = fs + self.coefficient * d.T @ d
        matrix = (matrix + matrix.T) / 2 + self.problem.cfg.damping * torch.eye(
            self.base.numel(), dtype=self.base.dtype, device=self.base.device)
        finite(matrix)
        return matrix


def solve(problem):
    cfg = problem.cfg
    system = GGNSystem(problem)
    rhs = system.rhs
    rhs_norm = float(rhs.norm())
    method = cfg.solver
    if method == "auto":
        method = "direct" if rhs.numel() <= cfg.direct_max_params else "cg"
    if rhs_norm == 0:
        return torch.zeros_like(rhs), dict(valid=False, reason="stationary", solver=method)
    if method == "direct":
        try:
            delta = torch.linalg.solve(system.dense(), rhs)
        except torch.linalg.LinAlgError as exc:
            return None, dict(valid=False, reason=str(exc), solver=method)
        iterations = 1
    else:
        delta = torch.zeros_like(rhs)
        residual, direction = rhs.clone(), rhs.clone()
        rr = residual @ residual
        iterations = 0
        for iterations in range(1, cfg.cg_steps + 1):
            product = system.matvec(direction)
            denom = direction @ product
            finite(denom, rr)
            if float(denom) <= 0:
                return None, dict(valid=False, reason="cg_nonpositive_curvature", solver=method)
            step = rr / denom
            delta = delta + step * direction
            residual = residual - step * product
            new_rr = residual @ residual
            finite(delta, new_rr)
            if float(new_rr.sqrt()) <= cfg.cg_tolerance * rhs_norm:
                break
            direction = residual + (new_rr / rr) * direction
            rr = new_rr
    finite(delta)
    relative = float((system.matvec(delta) - rhs).norm()) / rhs_norm
    directional = float((-rhs) @ delta)
    valid = relative <= cfg.cg_tolerance and directional < 0
    return delta.detach(), dict(valid=valid, reason="solved" if valid else "inaccurate_or_non_descent",
                               solver=method, iterations=iterations, relative_residual=relative,
                               directional_derivative=directional)
