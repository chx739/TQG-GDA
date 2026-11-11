"""Convex local constraints and true-forward acceptance for the R3 ablations."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F

from tsba.adapt import backtrack
from tsba.subproblem import GGNSystem, solve
from tsba.utils import finite


def margin_jacobian(params, vector, x):
    """Exact derivative of class-0 minus class-1 logits in ParameterMap order."""
    weights = params.mapping(vector)
    hidden = F.linear(x, weights["U1.weight"], weights["U1.bias"])
    activation = F.silu(hidden)
    sigmoid = hidden.sigmoid()
    derivative = sigmoid * (1 + hidden * (1 - sigmoid))
    output_difference = weights["U2.weight"][0] - weights["U2.weight"][1]
    sensitivity = derivative * output_difference
    ones = torch.ones_like(x[:, 0])
    blocks = {
        "linear.weight": torch.cat((x, -x), dim=1),
        "linear.bias": torch.stack((ones, -ones), dim=1),
        "U1.weight": (sensitivity[:, :, None] * x[:, None, :]).flatten(1),
        "U1.bias": sensitivity,
        "U2.weight": torch.cat((activation, -activation), dim=1),
    }
    return torch.cat([blocks[name] for name, _, _ in params.entries], dim=1)


class Budgets:
    def __init__(self, model, source, labels, target, job, plan):
        self.source = source
        self.labels = labels
        self.target = target
        self.protect = job.protect
        self.epsilon = job.epsilon
        self.tolerance = plan["constraint_tolerance"]
        self.points = torch.cat((source, target)) if self.epsilon is not None else source
        self.groups = [labels == c for c in (0, 1)]
        with torch.no_grad():
            initial = model(self.points)
            self.margin_limit = (initial[:, 0] - initial[:, 1]).abs() + (self.epsilon or 0.)
            self.risk_limit = self.risks(initial) + plan["risk_budget"]
        if not all(bool(mask.any()) for mask in self.groups):
            raise ValueError("Both source classes required for risk constraints")

    def risks(self, logits):
        ce = F.cross_entropy(logits[:len(self.source)], self.labels, reduction="none")
        return torch.stack([ce[mask].mean() for mask in self.groups])

    @torch.no_grad()
    def check(self, params, vector):
        logits = params.logits(vector, self.points)
        margin = (logits[:, 0] - logits[:, 1]).abs()
        risks = self.risks(logits)
        finite(margin, risks)
        risk_gap = risks - self.risk_limit
        margin_gap = margin - self.margin_limit
        return dict(
            risk_ok=not self.protect or bool((risk_gap <= self.tolerance).all()),
            confidence_ok=self.epsilon is None or bool((margin_gap <= self.tolerance).all()),
            class_ce=risks.tolist(),
            max_risk_excess=float(risk_gap.max()) if self.protect else None,
            max_margin_excess=float(margin_gap.max()) if self.epsilon is not None else None,
            fraction_margin_over_cap=float((margin_gap > self.tolerance).double().mean())
            if self.epsilon is not None else None,
            fraction_margin_at_cap=float((margin_gap.abs() <= self.tolerance).double().mean())
            if self.epsilon is not None else None,
        )


class AffineBudgets:
    def __init__(self, problem, budgets):
        self.budgets = budgets
        logits = problem.params.logits(problem.base, budgets.points).detach()
        self.margin = (logits[:, 0] - logits[:, 1]).numpy()
        self.jac = margin_jacobian(problem.params, problem.base, budgets.points).numpy()
        self.limit = budgets.margin_limit.numpy()
        # Binary cross entropy equals softplus of the negative signed margin.
        self.sign = (2 * budgets.labels.numpy() - 1)
        self.groups = [mask.numpy() for mask in budgets.groups]

    def risks(self, margin):
        losses = np.logaddexp(0., self.sign * margin[:len(self.sign)])
        return np.array([losses[mask].mean() for mask in self.groups])

    def violations(self, delta):
        margin = self.margin + self.jac @ delta
        if not np.isfinite(margin).all():
            raise FloatingPointError("Nonfinite affine margins")
        b = self.budgets
        return (
            int(np.sum(self.risks(margin) > b.risk_limit.numpy() + b.tolerance)) if b.protect else 0,
            int(np.sum(np.abs(margin) > self.limit + b.tolerance)) if b.epsilon is not None else 0,
        )

    def feasible(self, delta):
        return not any(self.violations(delta))


def constrained_minimum(matrix, rhs, affine, options):
    """Solve the convex subproblem, adding violated pointwise rows as needed.

    A relaxed optimum feasible for the complete constraint set also minimizes
    the complete problem. Both source CE constraints use all training examples.
    """
    b = affine.budgets
    factor = cholesky(matrix, lower=True, check_finite=True)
    center = solve_triangular(factor, rhs, lower=True)
    delta = solve_triangular(factor.T, center, lower=False)
    risk_trigger, margin_trigger = affine.violations(delta)
    triggers = dict(unconstrained_risk_violations=risk_trigger,
                    unconstrained_margin_violations=margin_trigger if b.epsilon is not None else None,
                    unconstrained_margin_violation_fraction=margin_trigger / len(affine.margin)
                    if b.epsilon is not None else None)
    if not risk_trigger and not margin_trigger:
        return delta, dict(solver="direct_feasible", iterations=0, added_margin_rows=0, **triggers)

    jac = solve_triangular(factor, affine.jac.T, lower=True).T
    margin = affine.margin
    limit = affine.limit
    chosen = np.zeros(len(margin), dtype=bool)
    constraints = []
    if b.protect:
        signs = affine.sign
        source_jac = jac[:len(signs)] * signs[:, None]
        risk_limit = b.risk_limit.numpy()

        def risk_slack(z):
            return risk_limit - affine.risks(margin + jac @ z)

        def risk_gradient(z):
            risk_logit = signs * (margin[:len(signs)] + jac[:len(signs)] @ z)
            weights = expit(risk_logit)
            return -np.stack([weights[mask] @ source_jac[mask] / mask.sum()
                              for mask in affine.groups])

        constraints.append(dict(type="ineq", fun=risk_slack, jac=risk_gradient))

    point = center
    iterations = 0
    for _ in range(options["constraint_generation_limit"]):
        excess = np.abs(margin + jac @ point) - limit
        if b.epsilon is not None:
            violated = np.flatnonzero((excess > b.tolerance) & ~chosen)
            ranked = violated[np.argsort(-excess[violated], kind="stable")]
            chosen[ranked[:options["add_rows"]]] = True
        local = list(constraints)
        if chosen.any():
            selected = jac[chosen]
            bounds = np.concatenate((limit[chosen] - margin[chosen], limit[chosen] + margin[chosen]))
            rows = np.concatenate((selected, -selected))
            local.append(dict(type="ineq", fun=lambda z: bounds - rows @ z,
                              jac=lambda z: -rows))
        result = minimize(
            lambda z: (.5 * np.dot(z - center, z - center), z - center),
            np.zeros_like(center) if iterations == 0 else point, jac=True,
            method="SLSQP", constraints=local,
            options=dict(ftol=options["ftol"], maxiter=options["iterations"]),
        )
        point = result.x
        iterations += result.nit
        delta = solve_triangular(factor.T, point, lower=False)
        info = dict(solver="SLSQP", iterations=iterations, added_margin_rows=int(chosen.sum()),
                    solver_status=int(result.status), solver_message=str(result.message), **triggers)
        if not result.success:
            return None, info
        if affine.feasible(delta):
            return delta, info
        if b.epsilon is None or not np.any(
            (np.abs(margin + jac @ point) - limit > b.tolerance) & ~chosen
        ):
            return None, dict(info, solver_message="Constraint residual exceeds tolerance")
    return None, dict(info, solver_message="Constraint generation limit reached")


def propose(problem, budgets, options):
    if not budgets.protect and budgets.epsilon is None:
        return solve(problem)
    system = GGNSystem(problem)
    rhs = system.rhs
    # Form the protected RHS directly so a small density request is not lost
    # through cancellation with the source CE gradient.
    if budgets.protect:
        rhs = system.coefficient * system.margin_pullback(problem.boundaries.zeta)[0].detach()
    if not bool(rhs.any()):
        return torch.zeros_like(rhs), dict(valid=False, reason="stationary")
    matrix = system.dense()
    affine = AffineBudgets(problem, budgets)
    delta, info = constrained_minimum(matrix.numpy(), rhs.numpy(), affine, options)
    if delta is None:
        return None, dict(info, valid=False, reason="constrained_solve_failed")
    delta = torch.from_numpy(delta)
    directional = float(-rhs @ delta)
    finite(delta, directional)
    return delta, dict(info, valid=directional < 0, directional_derivative=directional,
                       reason="solved" if directional < 0 else "non_descent")


def actual_objective(problem, vector, budgets):
    values = problem.evaluate(vector)
    if budgets.protect:
        values["objective"] = problem.cfg.lambda_boundary * values["move_loss"]
    values.update(budgets.check(problem.params, vector))
    return values


def assess(problem, delta, directional, alpha, sigma, budgets, initial):
    candidate = problem.base + alpha * delta
    result = actual_objective(problem, candidate, budgets)
    cfg = problem.cfg
    result["armijo"] = result["objective"] <= initial + cfg.armijo * alpha * directional
    result["bounded"] = result["max_movement"] <= cfg.execution_multiplier * sigma
    result["alpha"] = alpha
    valid = all(result[k] for k in ("armijo", "bounded", "risk_ok", "confidence_ok"))
    return candidate, result, valid


def accept(problem, delta, directional, sigma, budgets):
    if not budgets.protect and budgets.epsilon is None:
        return backtrack(problem, delta, directional, sigma)
    initial = actual_objective(problem, problem.base, budgets)["objective"]
    trials = []
    for step in range(problem.cfg.backtracks):
        candidate, result, valid = assess(problem, delta, directional, .5 ** step, sigma, budgets, initial)
        trials.append(result)
        if valid:
            return candidate.detach(), trials
    return None, trials
