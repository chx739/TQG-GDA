"""Functional same-state request decomposition and geometric diagnostics."""
import torch
from torch.func import grad, vmap, jacrev
from tsba.classifier import ParameterMap
from tsba.boundary import BoundaryBatch
from tsba.problem import TsbaProblem
from tsba.subproblem import GGNSystem
from tsba.utils import finite
from .shared import d3


def normalized_margin(params, vector, x, threshold):
    def margin(point):
        logits = params.logits(vector, point[None])[0]
        return logits[0] - logits[1]
    values = vmap(margin)(x)
    norm = vmap(grad(margin))(x).norm(dim=1)
    finite(values, norm)
    valid = norm >= threshold
    # Store validity separately, keeping missing values out of arithmetic.
    return dict(value=torch.where(valid, values / norm.clamp_min(threshold), torch.zeros_like(values)).detach(),
                valid=valid.detach())


def side_diagnostics(model, pools, y, record, sigma, cfg, plan):
    side = record["side"]
    if side["status"] != "available":
        return dict(status=side["status"])
    params = ParameterMap(model)
    committed = params.vector()
    rng = torch.random.get_rng_state()
    try:
        params.commit(record["before"])
        bounds = BoundaryBatch(**record["bounds"])
        ids = record["batch"]["source_positions"]
        problem = TsbaProblem(model, pools["source_fit"][ids], y[ids], bounds, cfg)
        system = GGNSystem(problem)
        matrix = system.dense()
        response = jacrev(problem.normalized_margin)(problem.base).detach()
        candidates = side["candidates"]
        zero = candidates[0]["delta"]
        origins = torch.tensor(record["batch"]["target_positions"])[bounds.input_indices].unique()
        result = dict(status="available", origins=origins, request=bounds.zeta,
                      request_rms_sigma=float(bounds.zeta.square().mean().sqrt()/sigma),
                      baseline=normalized_margin(params, record["before"], pools["target"], cfg.min_normal),
                      candidates={})
        for sign in (-1, 0, 1):
            item = candidates[sign]
            increment = item["delta"] - zero
            rhs = system.coefficient * system.margin_pullback(sign * bounds.zeta)[0].detach()
            residual = matrix @ increment - rhs
            tolerance = 10 * cfg.cg_tolerance * max(float(system.rhs.norm()), float(rhs.norm()), 1e-12)
            if float(residual.norm()) > tolerance:
                raise ValueError("Request increment does not satisfy the registered linear system")
            vector = item["vector"]
            relocated = d3.relocate(params, vector, bounds, sigma, cfg, plan)
            located = [(i, r["displacement"]) for i, r in enumerate(relocated["intersections"]) if r["status"] == "single_root"]
            root_error = None
            if located:
                indices, displacements = zip(*located)
                actual = bounds.zeta.new_tensor(displacements)
                proxy = -problem.movement(vector).detach()[list(indices)]
                root_error = float(((actual - proxy)/sigma).square().mean().sqrt())
            result["candidates"][sign] = dict(
                increment=increment, increment_residual=float(residual.norm()),
                increment_motion_rms_sigma=float((side["alpha"] * response @ increment).square().mean().sqrt()/sigma),
                total_motion_rms_sigma=float((side["alpha"] * response @ item["delta"]).square().mean().sqrt()/sigma),
                actual_vs_zero_rms_sigma=float((problem.movement(vector) -
                    problem.movement(candidates[0]["vector"])).square().mean().sqrt()/sigma),
                normalized_margin=normalized_margin(params, vector, pools["target"], cfg.min_normal),
                relocation=relocated, single_root_count=len(located), root_rmse_vs_proxy_sigma=root_error)
        return result
    finally:
        params.commit(committed)
        if not torch.equal(rng, torch.random.get_rng_state()):
            raise ValueError("Functional side diagnostic changed RNG")
