"""Target estimation from frozen graph/order traces (design §§3.6–3.12).

The structural information matrix is the Hessian of the graph-averaged, order-
averaged *sequence sum*. The optimizer additionally scales by mean edge count
and adds ridge. Neither operation enters the reported empirical information.
Fitting uses detached CPU data and analytic derivatives. Ill-conditioned but
resolved multiclass moments use additional arithmetic precision before returning float64.
"""

import math
from itertools import chain
from typing import NamedTuple

import numpy as np
import torch
from mpmath import mp
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp, softmax


N_TOKENS = 10


def _array(value, dtype=np.float64):
    return value.detach().to(device="cpu").numpy().astype(dtype, copy=False)


def _validate_solver(ridge, max_iter, tol):
    if not np.isfinite(ridge) or ridge < 0 or max_iter < 1 or not (0 < tol < 1):
        raise ValueError("Expected finite nonnegative ridge, positive max_iter and 0 < tol < 1")


def _checked(pool):
    found = False
    for orders in pool:
        if not orders:
            raise ValueError("Each adaptation graph needs at least one order")
        if any(len(tr.true_j) != len(orders[0].true_j) for tr in orders):
            raise ValueError("All permutations of one graph must have the same edge count")
        found = True
        yield orders
    if not found:
        raise ValueError("The adaptation pool needs at least one graph")


def _class_count_and_orders(pool):
    iterator = _checked(pool)
    first = next(iterator)
    classes = len(first[0].prior_logits)
    if classes < 2:
        raise ValueError("Prior calibration needs at least two classes")
    return classes, chain((first,), iterator)


def _helmert(n):
    basis = np.zeros((n, n - 1))
    for i in range(1, n):
        scale = math.sqrt(i * (i + 1))
        basis[:i, i - 1] = -1 / scale
        basis[i, i - 1] = i / scale
    return basis


def _components(support):
    # All ten tokens, including unseen tokens, get deterministic, contiguous ids.
    ids = np.full(N_TOKENS, -1, dtype=np.int64)
    members = []
    for start in range(N_TOKENS):
        if ids[start] >= 0:
            continue
        frontier = [start]
        component = []
        ids[start] = len(members)
        while frontier:
            j = frontier.pop()
            component.append(int(j))
            for neighbor in np.flatnonzero((support[j] > 0) & (ids < 0)):
                ids[neighbor] = ids[j]
                frontier.append(neighbor)
        members.append(sorted(component))
    basis = np.zeros((N_TOKENS, N_TOKENS - len(members)))
    offset = 0
    for group in members:
        dim = len(group) - 1
        basis[group, offset:offset + dim] = _helmert(len(group))
        offset += dim
    return ids, members, basis


def _structural_graph(orders):
    # Access only the shared masses and actual group/token, never class quantities.
    masses = []
    choices = []
    for tr in orders:
        k, j = _array(tr.true_k, np.int64), _array(tr.true_j, np.int64)
        lw = _array(tr.log_w)
        if lw.ndim != 3 or lw.shape[2] != N_TOKENS or k.shape != (len(lw),) or j.shape != k.shape:
            raise ValueError("Invalid structural trace shapes")
        if (k < 0).any() or (k >= lw.shape[1]).any() or (j < 0).any() or (j >= N_TOKENS).any():
            raise ValueError("Observed state index out of range")
        row = lw[np.arange(len(k)), k]
        if np.isnan(row).any() or np.isposinf(row).any() or not np.isfinite(row[np.arange(len(j)), j]).all():
            raise ValueError("Invalid log-mass in the observed group")
        masses.append(row)
        choices.append(j)
    mass = np.concatenate(masses)
    observed = np.concatenate(choices)
    present = np.isfinite(mass).astype(np.int64)
    pairs = present.T @ present
    np.fill_diagonal(pairs, 0)
    wins = np.zeros_like(pairs)
    np.add.at(wins, observed, present)
    return mass, observed, pairs, wins, len(orders)


def _structural_risk(mass, observed, weight, eta, information=False):
    logits = mass + eta
    # Shift before taking the log-sum so that large common offsets cancel exactly.
    logits = logits - np.max(logits, axis=1, keepdims=True)
    normalizer = logsumexp(logits, axis=1)
    p = np.exp(logits - normalizer[:, None])
    loss = weight @ (normalizer - logits[np.arange(len(observed)), observed])
    grad = weight @ p - np.bincount(observed, weights=weight, minlength=len(eta))
    if not information:
        return loss, grad
    # Sum p_j p_k (e_j-e_k)(e_j-e_k)^T, avoiding subtraction of nearly equal
    # values on the diagonal for very concentrated probabilities.
    laplacian = -p.T @ (p * weight[:, None])
    np.fill_diagonal(laplacian, 0.0)
    np.fill_diagonal(laplacian, -laplacian.sum(axis=1))
    return loss, grad, laplacian


class _InformationBlock(NamedTuple):
    tokens: list
    directions: np.ndarray  # columns are orthonormal, component-centered
    eigenvalues: np.ndarray  # eigenvalues of I_component / scale
    scale: float
    threshold: float  # backward-error threshold on the scaled matrix

    @property
    def resolved(self):
        return self.eigenvalues > self.threshold


def _information_blocks(information, groups):
    blocks = []
    for tokens in groups:
        if len(tokens) < 2:
            continue  # A singleton has no identifiable directions.
        block = information[np.ix_(tokens, tokens)]
        scale = float(np.max(np.abs(block)))
        basis = _helmert(len(tokens))
        normed = block / scale if scale else block
        values, vectors = np.linalg.eigh(basis.T @ normed @ basis)
        threshold = np.finfo(float).eps * len(tokens) * np.max(np.abs(values))
        blocks.append(_InformationBlock(tokens, basis @ vectors, values, scale, threshold))
    return blocks


def _local_sensitivity(blocks, response):
    """Orderwise b'I†b. NaN marks an unresolved or unrepresentable quadratic form."""
    # A component can be much weaker than another while being well conditioned
    # on its own contrast space. Use its own scale and never invert gauge modes.
    value = np.zeros(len(response), dtype=np.longdouble)
    unavailable = np.zeros(len(response), dtype=bool)
    for block in blocks:
        b = response[:, block.tokens]
        projection = b @ block.directions
        keep = block.resolved
        if not np.all(keep):
            # Allow only backward error from the orthogonal multiplication. In
            # particular, a response supported solely in a lost mode is flagged
            # even when its absolute magnitude is small.
            roundoff = len(block.tokens) * np.finfo(float).eps * np.max(np.abs(b), axis=1)
            unavailable |= np.any(np.abs(projection[:, ~keep]) > roundoff[:, None], axis=1)
        if np.any(keep):
            eig = block.eigenvalues[keep].astype(np.longdouble) * block.scale
            value += np.sum(projection[:, keep].astype(np.longdouble)**2 / eig, axis=1)
    unavailable |= ~np.isfinite(value) | (value > np.finfo(float).max)
    value[unavailable] = np.nan
    return value.astype(float)


def _lbfgs(objective, dim, max_iter, tol):
    result = minimize(objective, np.zeros(dim), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol, "ftol": np.finfo(float).eps})
    norm = float(np.linalg.norm(result.jac, ord=np.inf))
    status = ("gradient_converged" if norm <= tol else
              "relative_objective_converged" if result.success else "solver_stopped")
    return result.x, {"status": status, "converged": bool(result.success or norm <= tol),
                      "iterations": int(result.nit), "gradient_inf_norm": norm,
                      "solver_message": str(result.message)}


def _fit_structure(graphs, ridge, max_iter, tol):
    pair_steps = sum(g[2] for g in graphs)
    pair_graphs = sum((g[2] > 0).astype(np.int64) for g in graphs)
    ids, groups, basis = _components(pair_graphs)
    dim = basis.shape[1]
    mass = np.concatenate([g[0] for g in graphs])
    chosen = np.concatenate([g[1] for g in graphs])
    weight = np.concatenate([np.full(len(g[1]), 1.0 / (len(graphs) * g[4])) for g in graphs])
    mean_edges = float(weight.sum())
    scale = max(1.0, mean_edges)

    def objective(z):
        eta = basis @ z
        nll, grad = _structural_risk(mass, chosen, weight, eta)
        return nll / scale + ridge * (z @ z) / 2, basis.T @ grad / scale + ridge * z

    if dim:
        z, solver = _lbfgs(objective, dim, max_iter, tol)
    else:
        z = np.empty(0)
        solver = {"status": "no_comparisons", "converged": True, "iterations": 0,
                  "gradient_inf_norm": 0.0}
    eta = basis @ z
    nll, _, info = _structural_risk(mass, chosen, weight, eta, information=True)
    source_nll, _ = _structural_risk(mass, chosen, weight, np.zeros_like(eta))
    # A finite unregularized optimum needs directed observed-choice paths in
    # both directions throughout each undirected comparison component.
    reachable = sum(g[3] for g in graphs) > 0
    np.fill_diagonal(reachable, True)
    for j in range(N_TOKENS):
        reachable |= reachable[:, j, None] & reachable[None, j, :]
    finite_mle = all(reachable[np.ix_(group, group)].all() for group in groups)
    if ridge == 0 and not finite_mle:
        solver.update(status="no_finite_unregularized_optimum", converged=False)

    blocks = _information_blocks(info, groups)
    eig = np.sort([v * block.scale for block in blocks for v in block.eigenvalues])
    positive = np.sort([v * block.scale for block in blocks
                        for v in block.eigenvalues[block.resolved]])
    radius = max(0.0, float(eig[-1])) if len(eig) else 0.0
    diag = dict(solver, graphs=len(graphs), orders=sum(g[4] for g in graphs),
                steps=len(chosen), mean_edges=mean_edges, components=groups,
                effective_dimension=dim, pair_graph_support=pair_graphs.tolist(),
                pair_step_support=pair_steps.tolist(), information_matrix=info.tolist(),
                identifiable_eigenvalues=eig.tolist(), positive_eigenvalues=positive.tolist(),
                numerical_rank=len(positive), component_information=[
                    dict(tokens=block.tokens, eigenvalues=(block.eigenvalues * block.scale).tolist(),
                         scale=block.scale, scaled_rank_tolerance=float(block.threshold),
                         numerical_rank=int(block.resolved.sum())) for block in blocks],
                relative_eigenvalues=(eig / radius).tolist() if radius else [],
                finite_unregularized_optimum=bool(finite_mle), conditional_nll_zero=float(source_nll),
                conditional_nll_fitted=float(nll), objective=float(nll / scale + ridge * (z @ z) / 2),
                ridge=float(ridge))
    return torch.from_numpy(eta), torch.from_numpy(ids), diag


@torch.no_grad()
def fit_structure(traces_by_graph, ridge=.01, max_iter=100, tol=1e-6, *,
                  bootstrap_samples=0, seed=0):
    """Estimate within-component token contrasts. Resampling units are entire graphs."""
    _validate_solver(ridge, max_iter, tol)
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    graphs = [_structural_graph(orders) for orders in _checked(traces_by_graph)]
    eta, ids, diagnostics = _fit_structure(graphs, ridge, max_iter, tol)
    if bootstrap_samples:
        rng = np.random.default_rng(seed)
        estimates, partitions, statuses = [], [], []
        for _ in range(bootstrap_samples):
            sample = [graphs[i] for i in rng.integers(len(graphs), size=len(graphs))]
            fitted, partition, diag = _fit_structure(sample, ridge, max_iter, tol)
            estimates.append(fitted.tolist())
            partitions.append(partition.tolist())
            statuses.append(diag["status"])
        diagnostics["graph_bootstrap"] = dict(
            seed=int(seed), samples=bootstrap_samples, eta=estimates, component_ids=partitions,
            solver_status=statuses, eta_std=np.std(estimates, axis=0).tolist())
    return eta, ids, diagnostics


def _state_moments(trace, eta, components):
    """Return d[t,c], M[t,k*L+l,c], observed state and log P(true edge | c).

    Inactive states have exact zero mass. Each component retains its source
    probability. The within-component group probability uses tilted shared masses.
    """
    lw, scores = _array(trace.log_w), _array(trace.a)
    k, j = _array(trace.true_k, np.int64), _array(trace.true_j, np.int64)
    prior, u = _array(trace.prior_logits), _array(trace.true_u)
    m, groups, _ = lw.shape
    ncomp = int(components.max()) + 1
    classes = len(prior)
    if scores.shape != (m, groups, classes):
        raise ValueError("Class/group scores disagree with the shared mass array")
    if not np.isfinite(prior).all():
        raise ValueError("Nonfinite prior logits")
    # Subtract a common group score in each class to eliminate arbitrary class
    # constants before normalization. Only scores of occupied groups are needed.
    time = np.arange(m)
    scores = scores - scores[time, k, None, :]
    tilted = np.full((m, groups, ncomp), -np.inf)
    source = np.full_like(tilted, -np.inf)
    for l in range(ncomp):
        members = components == l
        source[:, :, l] = logsumexp(lw[:, :, members], axis=2)
        tilted[:, :, l] = logsumexp(lw[:, :, members] + eta[members], axis=2)
    log_source = source[..., None] + scores[:, :, None, :]
    log_tilt = tilted[..., None] + scores[:, :, None, :]
    log_z0 = logsumexp(log_source, axis=1, keepdims=True)
    log_z = logsumexp(log_tilt, axis=1, keepdims=True)
    total = logsumexp(log_z0, axis=2, keepdims=True)
    # Missing components contribute zero probability and have no normalizer.
    offset = np.zeros_like(log_z)
    np.subtract(log_z0, log_z, where=np.isfinite(log_z), out=offset)
    log_state = log_tilt + (offset - total)
    h = components[j]
    edge_logp = (log_state[time, k, h] +
                 (u + eta[j] - tilted[time, k, h])[:, None])
    prefix = np.broadcast_to(prior, (m, classes)).copy()
    if m > 1:
        prefix[1:] += np.cumsum(edge_logp[:-1], axis=0)
    return prefix, np.exp(log_state).reshape(m, groups * ncomp, classes), k * ncomp + h, edge_logp


def _fixed_parameters(eta, components):
    eta_np, ids = _array(eta), _array(components, np.int64)
    if (eta_np.shape != (N_TOKENS,) or not np.isfinite(eta_np).all() or ids.shape != eta_np.shape
            or not np.array_equal(np.unique(ids), np.arange(len(np.unique(ids))))):
        raise ValueError("Expected ten finite token offsets and contiguous component ids starting at zero")
    return eta_np, ids


@torch.no_grad()
def prefix_moments(trace, eta, components):
    """Expose prefix logits, group–component probability matrix and observed rows."""
    d, matrix, observed, _ = _state_moments(trace, *_fixed_parameters(eta, components))
    return torch.from_numpy(d), torch.from_numpy(matrix), torch.from_numpy(observed)


def _binary_fit(d, w, y, ridge, max_iter, tol):
    target = float(y.sum() / w.sum())
    weight = w / w.sum()
    diff = d[:, 1] - d[:, 0]

    def residual(gamma):
        return target - weight @ expit(diff + gamma) - ridge * gamma / 2

    if ridge:
        # With p in [0,1], gamma = 2(target-p)/ridge. Round the enclosing
        # bounds outwards. Their unrounded separation can be smaller than one ulp.
        ratio = np.longdouble(2) / ridge
        bounds = [float(ratio * (np.longdouble(target) - p)) for p in (1, 0)]
        if not np.isfinite(bounds).all():
            return np.zeros(2), dict(status="root_outside_float64", converged=False,
                                     iterations=0, moment_target=target)
        lo, hi = np.nextafter(bounds[0], -np.inf), np.nextafter(bounds[1], np.inf)
    elif 0 < target < 1:
        lo, hi = logit(target) - diff.max() - 1, logit(target) - diff.min() + 1
        bounds = [lo, hi]
    else:
        return np.zeros(2), dict(status="no_finite_unregularized_root", converged=False,
                                 iterations=0, moment_target=target)
    endpoint = next((x for x in (0.0, *bounds, lo, hi) if residual(x) == 0.0), None)
    status = "root_converged"
    if endpoint is not None:
        gamma, iterations, converged = endpoint, 0, True
    elif ridge and bounds[0] == bounds[1]:
        # All possible p produce the same representable gamma. The remaining
        # raw residual is rounding in ridge*gamma/2, not an unbracketed root.
        gamma, iterations, converged = bounds[0], 0, True
        status = "root_at_float_resolution"
    else:
        gamma, result = brentq(residual, lo, hi, xtol=tol, maxiter=max_iter, disp=False, full_output=True)
        converged, iterations = result.converged, result.iterations
    return np.array([-gamma / 2, gamma / 2]), dict(
        status=status if converged else "solver_stopped", converged=bool(converged),
        iterations=int(iterations), moment_residual=float(residual(gamma)), moment_target=target,
        relative_moment_residual=float(abs(residual(gamma)) / max(1., abs(target))))


def _multiclass_fit(d, w, y, basis, ridge, max_iter, tol):
    target = y.sum(axis=0) / w.sum()
    weight = w / w.sum()
    classes = basis.shape[0]
    if ridge == 0 and np.min(1 / classes + basis @ target) <= 0:
        return np.zeros(classes), dict(status="no_finite_unregularized_root", converged=False,
                                       iterations=0)
    centered = d - d.max(axis=1, keepdims=True)

    def objective(beta):
        scores = centered + basis @ beta
        value = weight @ logsumexp(scores, axis=1) - beta @ target + ridge * (beta @ beta) / 2
        grad = (weight @ softmax(scores, axis=1)) @ basis - target + ridge * beta
        return value, grad

    beta, diagnostics = _lbfgs(objective, classes - 1, max_iter, tol)
    diagnostics.update(moment_residual=objective(beta)[1].tolist())
    return basis @ beta, diagnostics


def _close_contrast(contrast, anchor):
    """Recover the largest-probability state's difference from the other states."""
    t = np.arange(len(contrast))
    contrast[t, anchor] = 0.
    contrast[t, anchor] = -contrast.sum(axis=1, dtype=np.longdouble)
    return contrast


def _precise_moment(matrix, observed, basis, anchor, relative_smin):
    """Evaluate a full-rank moment when disparate state masses exhaust float64."""
    rows = np.flatnonzero(matrix.max(axis=1) > 0)
    masses = matrix[rows]
    # The non-anchor row-scaled design has singular value >= relative_smin.
    # Undoing row scaling costs at most 1/min_mass, forming F squares that
    # condition number. Budget precision for both operations and float64 output.
    min_mass = masses.max(axis=1).min()
    bits = math.ceil(math.log2(masses.size) - math.log2(min_mass) - math.log2(relative_smin))
    ctx = mp.clone()
    ctx.prec = max(128, 2 * bits + 128)
    classes = matrix.shape[1]
    U = ctx.matrix(basis.tolist())
    design = ctx.matrix(len(rows), classes - 1)
    x = ctx.matrix(len(rows), 1)
    anchor_i = int(np.flatnonzero(rows == anchor)[0])
    for i, row in enumerate(rows):
        if i == anchor_i:
            continue
        p = [ctx.mpf(float(z)) for z in matrix[row]]
        contrast = ctx.matrix([[value - p[0] for value in p]]) * U
        for k in range(classes - 1):
            design[i, k] = contrast[k]
        x[i] = int(row == observed) - ctx.fsum(p) / classes
    for k in range(classes - 1):
        design[anchor_i, k] = -ctx.fsum(design[:, k])
    x[anchor_i] = -ctx.fsum(x)
    gram = design.T * design
    omega = ctx.eigsy(gram, eigvals_only=True)[0]
    if omega <= 0:
        raise ArithmeticError("Full-rank row-scaled moment has nonpositive high-precision information")
    result = ctx.lu_solve(gram, omega * (design.T * x))
    return float(omega), [float(z) for z in result]


def _moment_terms(matrix, observed, basis):
    steps, _, classes = matrix.shape
    dim = classes - 1
    weight = np.zeros(steps)
    y = np.zeros(steps if classes == 2 else (steps, dim))
    if not steps:
        return weight, y
    # Columns of M sum to one. With V occupied states, rank(M U) <= V-1.
    state_mass = matrix.max(axis=2)
    n_states = np.count_nonzero(state_mass > 0., axis=1)
    eligible = np.flatnonzero(n_states >= classes)
    if not len(eligible):
        return weight, y
    m = matrix[eligible]
    t = np.arange(len(m))
    anchor = m.mean(axis=2).argmax(axis=1)
    response = -(m[:, :, 0].copy() if classes == 2 else m.mean(axis=2))
    response[t, observed[eligible]] += 1.
    _close_contrast(response, anchor)
    contrasts = _close_contrast(m - m[:, :, :1], anchor)
    if classes == 2:
        diff = contrasts[:, :, 1]
        weight[eligible] = np.einsum("tv,tv->t", diff, diff)
        y[eligible] = np.einsum("tv,tv->t", diff, response)
        return weight, y

    design = contrasts @ basis
    # The anchor row is redundant. Test numerical rank on the other rows in
    # units of their own probability mass, since a rare state's relative odds
    # can be well resolved even when its probability is much smaller than eps.
    mass = state_mass[eligible, :, None]
    relative = np.divide(design, mass, out=np.zeros_like(design), where=mass > 0)
    relative[t, anchor] = 0.
    singular = np.linalg.svd(relative, compute_uv=False)
    eps = np.finfo(float).eps
    independent = singular[:, -1] > eps * np.maximum(classes, n_states[eligible]) * np.maximum(1., singular[:, 0])
    if not independent.any():
        return weight, y
    # If A=L diag(s) R^T, omega F^-1 A^T=R diag(s_min^2/s) L^T.
    # Direct SVD avoids squaring the condition number in the rank test/solve.
    left, s, right_t = np.linalg.svd(design[independent], full_matrices=False)
    idx = eligible[independent]
    # The second test selects arithmetic precision, never structural rank.
    # When even sqrt(eps) relative accuracy cannot be assured, reconstruct A
    # from its measured rows and enforce its column-sum constraint in high precision.
    ordinary = s[:, -1] > math.sqrt(eps) * n_states[idx] * s[:, 0]
    for local in np.flatnonzero(~ordinary):
        index = idx[local]
        weight[index], y[index] = _precise_moment(
            matrix[index], observed[index], basis, anchor[independent][local],
            singular[independent][local, -1])
    if ordinary.any():
        idx = idx[ordinary]
        s = s[ordinary]
        w = s[:, -1]**2
        projected = np.einsum("tvd,tv->td", left[ordinary], response[independent][ordinary])
        scaled = projected * ((s[:, -1, None] / s) * s[:, -1, None])
        weight[idx] = w
        y[idx] = np.einsum("tdc,td->tc", right_t[ordinary], scaled)
    return weight, y


def _prior_graph(orders, fixed, basis):
    """Retain only per-step C-dimensional moment data from one graph."""
    ds, weights, responses = [], [], []
    steps, raw_information = 0, 0.0
    for tr in orders:
        if len(tr.prior_logits) != len(basis):
            raise ValueError("Inconsistent number of classes across traces")
        d, matrix, observed, _ = _state_moments(tr, *fixed)
        w, y = _moment_terms(matrix, observed, basis)
        keep = w > 0
        ds.append(d[keep])
        weights.append(w[keep] / len(orders))
        responses.append(y[keep] / len(orders))
        steps += len(d)
        raw_information += float(w.sum())
    return np.concatenate(ds), np.concatenate(weights), np.concatenate(responses), steps, raw_information


def _solve_prior(graphs, basis, ridge, max_iter, tol):
    ds, weights, responses, counts, info = zip(*graphs)
    d, w, y = map(np.concatenate, (ds, weights, responses))
    classes, steps, raw_information = len(basis), sum(counts), sum(info)
    key = "S_B" if classes == 2 else "S_omega"
    diagnostics = {"method": "binary" if classes == 2 else "multiclass", key: raw_information,
                   "order_averaged_information": float(w.sum()), "steps": steps,
                   "informative_steps": len(w), "estimable": bool(len(w)), "ridge": float(ridge)}
    if not len(w):
        nu = np.zeros(classes)
        solver = {"status": "no_calibration_information", "iterations": 0, "converged": False}
    elif classes == 2:
        nu, solver = _binary_fit(d, w, y, ridge, max_iter, tol)
    else:
        nu, solver = _multiclass_fit(d, w, y, basis, ridge, max_iter, tol)
    diagnostics.update(solver)
    return torch.from_numpy(nu), diagnostics


@torch.no_grad()
def fit_prior(traces_by_graph, eta, components, ridge=.01, max_iter=100, tol=1e-6):
    """Calibrate a centered class bias from prefix moments with fixed structure."""
    _validate_solver(ridge, max_iter, tol)
    classes, pool = _class_count_and_orders(traces_by_graph)
    basis = _helmert(classes)
    fixed = _fixed_parameters(eta, components)
    graphs = [_prior_graph(orders, fixed, basis) for orders in pool]
    return _solve_prior(graphs, basis, ridge, max_iter, tol)


def _order_response(trace, eta, ids):
    """Source/adapted log-posteriors and summed responses for class pairs c < r."""
    k, j = _array(trace.true_k, np.int64), _array(trace.true_j, np.int64)
    scores, prior = _array(trace.a), _array(trace.prior_logits)
    lw, u = _array(trace.log_w), _array(trace.true_u)
    time = np.arange(len(j))
    token = logsumexp(lw[..., None] + scores[:, :, None, :], axis=1)
    observed = u[:, None] + scores[time, k]
    source = prior + (observed - logsumexp(token, axis=1)).sum(axis=0)
    allowed = ids[None, :] == ids[j, None]
    part = np.where(allowed[..., None], token, -np.inf)
    tilted = part + eta[None, :, None]
    norm = logsumexp(tilted, axis=1)
    adapted = source + (logsumexp(part, axis=1) - norm).sum(axis=0)
    rho = np.exp(tilted - norm[:, None, :])
    responses = []
    for c in range(len(prior)):
        for r in range(c + 1, len(prior)):
            # Use a common dominant token inside the observed component of each
            # step. Its true contrast can be e.g. -1e-20 when both rounded class
            # probabilities equal 1. Sum the paired contrasts only after closure.
            anchor = (rho[:, :, c] + rho[:, :, r]).argmax(axis=1)
            delta = _close_contrast(rho[:, :, r] - rho[:, :, c], anchor)
            responses.append(delta.sum(axis=0, dtype=np.longdouble).astype(float))
    return source - logsumexp(source), adapted - logsumexp(adapted), np.stack(responses)


def _quantiles(values):
    quantiles = np.quantile(values, [0, .25, .5, .75, 1])
    return dict(zip(("min", "q25", "median", "q75", "max"), quantiles.tolist()))


def _response_diagnostics(pool, eta, ids, structural, classes):
    information = np.asarray(structural["information_matrix"])
    groups = [np.flatnonzero(ids == l).tolist() for l in np.unique(ids)]
    blocks = _information_blocks(information, groups)
    pairs = [(c, r) for c in range(classes) for r in range(c + 1, classes)]
    corrections = {pair: [] for pair in pairs}
    sensitivities = {pair: [] for pair in pairs}
    unresolved_orders = dict.fromkeys(pairs, 0)
    for orders in pool:
        traces = [_order_response(tr, eta, ids) for tr in orders]
        source, adapted, response = (np.stack(arrays) for arrays in zip(*traces))
        # Both means are over posteriors with class-independent uniform order weights.
        correction = logsumexp(adapted, axis=0) - logsumexp(source, axis=0)
        for i, (c, r) in enumerate(pairs):
            local = _local_sensitivity(blocks, response[:, i])
            corrections[c, r].append(correction[c] - correction[r])
            unresolved_orders[c, r] += int(np.isnan(local).sum())
            if np.isfinite(local).all():
                sensitivities[c, r].append(float(np.mean(local, dtype=np.longdouble)))
    return {f"{c}/{r}": {
        "graph_log_odds_change": _quantiles(corrections[c, r]),
        "mean_order_local_information_sensitivity": (
            None if unresolved_orders[c, r] else _quantiles(sensitivities[c, r])),
        "sensitivity_status": "unresolved" if unresolved_orders[c, r] else "resolved",
        "sensitivity_unresolved_orders": unresolved_orders[c, r],
        "sensitivity_unresolved_graphs": len(corrections[c, r]) - len(sensitivities[c, r]),
        "sensitivity_resolved_graphs": len(sensitivities[c, r])}
        for c, r in pairs}


def _require_converged(mode, diagnostics):
    status = diagnostics["status"]
    if not diagnostics["converged"] and status not in {"no_comparisons", "no_calibration_information"}:
        raise RuntimeError(f"Cannot construct {mode} predictions: fit status {status}")


@torch.no_grad()
def fit_modes(traces_by_graph, ridge=.01, max_iter=100, tol=1e-6, *,
              calibrate_prior=True, bootstrap_samples=0, seed=0):
    """Fit in two passes. Supply a replayable sequence or a fixed-order trace factory."""
    replay = traces_by_graph if callable(traces_by_graph) else lambda: iter(traces_by_graph)
    eta, components, structural = fit_structure(replay(), ridge, max_iter, tol,
                                                bootstrap_samples=bootstrap_samples, seed=seed)
    _require_converged('structure', structural)
    classes, pool = _class_count_and_orders(replay())
    zero = torch.zeros_like(eta)
    no_bias = torch.zeros(classes, dtype=torch.float64)
    modes = {"zero": (zero, components, no_bias),
             "structural_only": (eta, components, no_bias),
             "v2_global": (eta, torch.zeros_like(components), no_bias)}
    offsets = {"prior_only": zero, "prior_calibrated": eta} if calibrate_prior else {}
    fixed = {name: _fixed_parameters(value, components) for name, value in offsets.items()}
    summaries = {name: [] for name in offsets}
    basis = _helmert(classes)

    def collect_moments():
        for orders in pool:
            for name, parameters in fixed.items():
                summaries[name].append(_prior_graph(orders, parameters, basis))
            yield orders

    diagnostics = {"structure": structural, "prediction_response": _response_diagnostics(
        collect_moments(), eta.numpy(), components.numpy(), structural, classes)}
    for name, graphs in summaries.items():
        nu, diagnostics[name] = _solve_prior(graphs, basis, ridge, max_iter, tol)
        _require_converged(name, diagnostics[name])
        modes[name] = offsets[name], components, nu
    return modes, diagnostics
