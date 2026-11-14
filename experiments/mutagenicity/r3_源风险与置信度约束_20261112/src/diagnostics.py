"""Density support and intersections along each frozen old normal."""
import torch
from torch.nn import functional as F

from tsba.utils import finite
from experiments.mutagenicity.r1b_密度与单步适配诊断_20261111.src.steps import displacement_request
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.score import comparison


@torch.no_grad()
def density_audit(bounds, target, origins, bandwidth, sigma, learned=None):
    points, normals = bounds.points, bounds.normals
    differences = target[None] - points[:, None]
    log_kernel = -differences.square().sum(-1) / (2 * bandwidth ** 2)
    weights = log_kernel.softmax(-1)
    empirical = (weights[..., None] * differences).sum(1) / bandwidth ** 2
    row = torch.arange(len(points))
    origin_weight = weights[row, origins]
    log_kernel[row, origins] = -torch.inf
    loo = (log_kernel.softmax(-1)[..., None] * differences).sum(1) / bandwidth ** 2
    gamma, request = displacement_request(empirical, normals, sigma)
    loo_gamma, loo_request = displacement_request(loo, normals, sigma)
    result = dict(ess=1 / weights.square().sum(-1), origin_weight=origin_weight,
                  maximum_weight=weights.max(-1).values, kde_gamma=gamma, kde_request=request,
                  loo_gamma=loo_gamma, loo_request=loo_request)
    if learned is not None:
        result["learned_vs_kde"] = comparison(empirical, learned(points), normals, sigma)
    return result


@torch.no_grad()
def relocate(params, vector, bounds, sigma, cfg, plan):
    """Scan a finite normal segment, retaining crossings and near-tangencies."""
    weights = params.mapping(vector)
    points, normals = bounds.points, bounds.normals
    linear = weights["linear.weight"][0] - weights["linear.weight"][1]
    hidden_difference = weights["U2.weight"][0] - weights["U2.weight"][1]
    m0 = points @ linear + weights["linear.bias"][0] - weights["linear.bias"][1]
    slope = normals @ linear
    hidden = F.linear(points, weights["U1.weight"], weights["U1.bias"])
    hidden_slope = normals @ weights["U1.weight"].T

    def evaluate(idx, t):
        pre = hidden[idx] + t[..., None] * hidden_slope[idx]
        sigmoid = pre.sigmoid()
        margin = m0[idx] + t * slope[idx] + F.silu(pre) @ hidden_difference
        derivative = slope[idx] + ((sigmoid * (1 + pre * (1 - sigmoid))) *
                                   hidden_slope[idx]) @ hidden_difference
        return margin, derivative

    def bisect(idx, left, right, derivative):
        f_left = evaluate(idx, left)[derivative]
        for _ in range(plan["relocation_bisections"]):
            middle = (left + right) / 2
            value = evaluate(idx, middle)[derivative]
            advance = (value.sign() == f_left.sign()) & (value != 0)
            left = torch.where(advance, middle, left)
            right = torch.where(advance, right, middle)
        return (left + right) / 2

    radius = plan["relocation_radius_sigma"] * sigma
    grid = torch.linspace(-radius, radius, plan["relocation_scan_points"], dtype=points.dtype)
    index = torch.arange(len(points))
    margin, derivative = evaluate(index[:, None], grid[None])
    finite(margin, derivative)
    turn_rows, turn_cols = torch.where(derivative[:, :-1] * derivative[:, 1:] < 0)
    turns = bisect(turn_rows, grid[turn_cols], grid[turn_cols + 1], 1)
    turn_margin, _ = evaluate(turn_rows, turns)
    tolerance = cfg.root_tolerance * sigma * bounds.normal_norm

    # Splitting scanned intervals at detected extrema can reveal two crossings
    # even when both original grid endpoints have the same sign.
    left_m, right_m = margin[:, :-1], margin[:, 1:]
    crossing = (left_m * right_m < 0)
    crossing[turn_rows, turn_cols] = False
    rows, cols = torch.where(crossing)
    split_left = left_m[turn_rows, turn_cols] * turn_margin < 0
    split_right = turn_margin * right_m[turn_rows, turn_cols] < 0
    root_rows = torch.cat((rows, turn_rows[split_left], turn_rows[split_right]))
    left = torch.cat((grid[cols], grid[turn_cols[split_left]], turns[split_right]))
    right = torch.cat((grid[cols + 1], turns[split_left], grid[turn_cols[split_right] + 1]))
    roots = bisect(root_rows, left, right, 0)
    exact_rows, exact_cols = torch.where(margin == 0)
    tangent_rows, tangent_cols = torch.where((margin.abs() <= tolerance[:, None]) &
                                             (derivative.abs() <= cfg.root_tolerance * bounds.normal_norm[:, None]))
    near = turn_margin.abs() <= tolerance[turn_rows]
    candidates = [[] for _ in points]
    tangencies = [[] for _ in points]
    for row, value in zip(torch.cat((root_rows, exact_rows)).tolist(),
                          torch.cat((roots, grid[exact_cols])).tolist()):
        candidates[row].append(value)
    for row, value in zip(torch.cat((turn_rows[near], tangent_rows)).tolist(),
                          torch.cat((turns[near], grid[tangent_cols])).tolist()):
        tangencies[row].append(value)

    result = []
    for row, values in enumerate(candidates):
        distinct = []
        for value in sorted(values):
            if not distinct or value - distinct[-1] > cfg.root_tolerance * sigma:
                distinct.append(value)
        info = dict(roots=distinct, near_touches=tangencies[row])
        if not distinct:
            info["status"] = "near_touch" if tangencies[row] else "no_root"
        elif len(distinct) > 1:
            info["status"] = "multiple_roots"
        else:
            t = distinct[0]
            at = points[row] + t * normals[row]
            pre = F.linear(at, weights["U1.weight"], weights["U1.bias"])
            sigmoid = pre.sigmoid()
            gradient = linear + (hidden_difference * sigmoid *
                                 (1 + pre * (1 - sigmoid))) @ weights["U1.weight"]
            magnitude = gradient.norm()
            info.update(displacement=t, slope_ratio=float(magnitude / bounds.normal_norm[row]),
                        normal_cosine=float((gradient @ normals[row]) / magnitude)
                        if magnitude > cfg.min_normal else None,
                        status="near_touch" if tangencies[row] else "single_root")
        result.append(info)
    return dict(grid=grid, margin=margin, directional_derivative=derivative, intersections=result)
