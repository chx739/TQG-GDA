"""R1b research-only single proposals on the old source boundary bank.

Only source-fit labels are accepted. Score fitting uses target inputs and fixed
noise; empirical scores refer to the finite target pool, not a population oracle.
No encoder, label broker, probe, or closed-loop adaptation is used here.
"""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import time
import traceback

import torch
from torch.func import grad, jacrev

from tsba.adapt import backtrack
from tsba.boundary import BoundaryBatch
from tsba.classifier import ResidualClassifier
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.score import ScoreNetwork, denoising_loss
from tsba.subproblem import GGNSystem, solve
from tsba.utils import finite
from experiments.common.synthetic.archive import digest, save_tensor, write_json


ARMS = ("normal", "reverse", "zero", "matched_source")
ESTIMATORS = ("empirical", "learned")
STEP_DEFAULTS = dict(bandwidth_multipliers=[1., 2.], small_step_start=1 / 16,
                     small_step_floor=1 / 32768, lambda_boundary=1., damping=.01,
                     tau=.1, request_multiplier=.1, execution_multiplier=.1,
                     source_fit_count=779, anchor_count=64)
SCORE_DEFAULTS = dict(hidden=64, lr=.001, batch=128, epochs=100, validation=True)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _log(log, message):
    if callable(log):
        log(message)
    elif log is not None:
        log.write(message + "\n")
        log.flush()


def _open_observation(root):
    if (Path(root) / "seals/OBS.json").exists():
        raise PermissionError("step_and_score_fitting_closed_after_OBS")


def _limits(started, resources):
    if time.monotonic() - started > min(3600, resources.get("max_score_seconds", 3600)):
        raise TimeoutError("score_fit_time_limit")
    if _rss() > min(8 * 1024**3, resources.get("max_rss_bytes", 8 * 1024**3)):
        raise MemoryError("step_or_score_rss_limit")


class EmpiricalScore:
    """Exact Gaussian-KDE score using stable softmax weights over all centers."""
    def __init__(self, centers, bandwidth, chunk=256):
        self.centers = centers.detach().double()
        self.bandwidth = float(bandwidth)
        self.chunk = int(chunk)
        if self.centers.ndim != 2 or not len(self.centers) or not self.centers.shape[1]:
            raise ValueError("empty_or_invalid_KDE_centers")
        if not math.isfinite(self.bandwidth) or self.bandwidth <= 0 or self.chunk <= 0:
            raise ValueError("invalid_KDE_bandwidth_or_chunk")
        finite(self.centers)

    def _weights(self, query):
        difference = self.centers[None] - query[:, None]
        log_kernel = -difference.square().sum(-1) / (2 * self.bandwidth**2)
        finite(log_kernel)
        return difference, log_kernel.softmax(-1), log_kernel

    def __call__(self, query):
        query = query.double()
        finite(query)
        if query.ndim != 2 or query.shape[1] != self.centers.shape[1]:
            raise ValueError("invalid_KDE_query_shape")
        values = []
        for block in query.split(self.chunk):
            difference, weights, _ = self._weights(block)
            values.append((weights[..., None] * difference).sum(1) / self.bandwidth**2)
        result = torch.cat(values) if values else query.new_empty(query.shape)
        finite(result)
        return result

    def log_density(self, query):
        values = []
        for block in query.double().split(self.chunk):
            _, _, log_kernel = self._weights(block)
            constant = math.log(len(self.centers)) + self.centers.shape[1] * math.log(
                self.bandwidth * math.sqrt(2 * math.pi))
            values.append(torch.logsumexp(log_kernel, -1) - constant)
        result = torch.cat(values)
        finite(result)
        return result


def displacement_request(score, normals, sigma0):
    """The bandwidth changes the score only; every movement unit stays sigma0."""
    finite(score, normals, sigma0)
    if score.shape != normals.shape or sigma0 <= 0:
        raise ValueError("invalid_request_inputs")
    gamma = (score * normals).sum(-1)
    request = (.1 * sigma0**2 * gamma).clamp(-.1 * sigma0, .1 * sigma0)
    finite(gamma, request)
    return gamma.detach(), request.detach()


def fit_score(directory, x, sigma, cfg, resources=None, log=None):
    """Production train_score arithmetic/RNG, plus every-epoch resumable states.

    This helper accepts smaller Config budgets for synthetic parity checks. The
    run_steps entry point enforces the predeclared 100-epoch, 64-hidden grid.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "epochs").mkdir()
    resources = resources or {}
    started = time.monotonic()
    x = x.detach()
    finite(x, sigma)
    metadata = dict(status="failed", epochs_requested=cfg.score_epochs,
                    updates_completed=0, epochs_completed=0, selected_epoch=None,
                    seed=cfg.seed, bandwidth=float(sigma), hidden=cfg.score_hidden,
                    lr=cfg.score_lr, batch=cfg.score_batch,
                    initialization_seed=cfg.seed + 101, training_rng_seed=cfg.seed + 102,
                    validation_rng_seed=cfg.seed + 103, checkpoints={},
                    validation_scope="independent fixed noise on the same unlabeled target pool")
    history, model, opt, rng, val_rng, val_noise = [], None, None, None, None, None
    best_state, best_epoch, best = None, None, math.inf
    epoch = -1

    def checkpoint():
        return dict(epoch=epoch, updates_completed=metadata["updates_completed"],
                    model_state=copy.deepcopy(model.state_dict()),
                    optimizer_state=copy.deepcopy(opt.state_dict()),
                    training_rng_state=rng.get_state(), validation_rng_state=val_rng.get_state(),
                    torch_rng_state=torch.random.get_rng_state(), validation_noise=val_noise.clone(),
                    history=copy.deepcopy(history), selected_epoch=best_epoch,
                    best_validation_loss=best if math.isfinite(best) else None,
                    best_model_state=copy.deepcopy(best_state), sigma=float(sigma),
                    config=cfg.to_dict())

    try:
        _limits(started, resources)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 101)
            model = ScoreNetwork(x.shape[1], cfg.score_hidden).to(dtype=x.dtype)
        model = model.to(x.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.score_lr)
        rng = torch.Generator(device=x.device).manual_seed(cfg.seed + 102)
        val_rng = torch.Generator(device=x.device).manual_seed(cfg.seed + 103)
        val_noise = torch.randn(x.shape, generator=val_rng, device=x.device, dtype=x.dtype)
        save_tensor(directory / "initial.pt", checkpoint())
        for epoch in range(cfg.score_epochs):
            order = torch.randperm(len(x), generator=rng, device=x.device)
            total = 0.0
            for ids in order.split(cfg.score_batch):
                _limits(started, resources)
                noise = torch.randn(x[ids].shape, generator=rng, device=x.device, dtype=x.dtype)
                loss = denoising_loss(model, x[ids], sigma, noise)
                finite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                finite(*(p.grad for p in model.parameters()))
                opt.step()
                finite(*(p for p in model.parameters()))
                total += float(loss.detach()) * len(ids)
                metadata["updates_completed"] += 1
            with torch.no_grad():
                validation = sum(float(denoising_loss(model, xb, sigma, eb)) * len(xb)
                                 for xb, eb in zip(x.split(cfg.score_batch), val_noise.split(cfg.score_batch))) / len(x)
            finite(validation)
            history.append(dict(epoch=epoch, train_loss=total / len(x), noise_validation_loss=validation))
            if not cfg.score_validation or validation < best:
                best, best_epoch = validation, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            metadata["epochs_completed"] = epoch + 1
            path = f"epochs/epoch{epoch:03d}.pt"
            save_tensor(directory / path, checkpoint())
            metadata["checkpoints"][str(epoch)] = path
            if epoch % 20 == 0 or epoch == cfg.score_epochs - 1:
                _log(log, f"score seed={cfg.seed} h={sigma:.8g} epoch={epoch} validation={validation:.7g}")
        # The final epoch archive retains its Adam/RNG; selection is a frozen copy.
        model.load_state_dict(best_state)
        model.requires_grad_(False).eval()
        save_tensor(directory / "selected.pt", dict(model_state=best_state, epoch=best_epoch,
                    sigma=float(sigma), config=cfg.to_dict(), noise_validation_loss=best))
        metadata.update(status="success", selected_epoch=best_epoch,
                        noise_validation_loss=best, selected_checkpoint="selected.pt")
    except Exception as exc:
        metadata.update(failure=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if model is not None and opt is not None and val_noise is not None:
            save_tensor(directory / "partial.pt", checkpoint())
        model = None
    finally:
        metadata.update(elapsed_seconds=time.monotonic() - started, peak_rss_bytes=_rss())
        write_json(directory / "history.json", history)
        write_json(directory / "metadata.json", metadata)
    return model, metadata


def old_boundaries(bank, model, sigma0):
    """Use valid old points/normals only, independent of density/support fields."""
    points, normals, norms, margins, kept, rejected = [], [], [], [], [], []
    dim = model.linear.in_features
    for position, row in enumerate(bank["anchors"]):
        reason = None
        if not row.get("success"):
            reason = row.get("failure") or "old_projection_failed"
        else:
            point, normal = row.get("point"), row.get("normal")
            norm = row.get("normal_norm")
            if not isinstance(point, torch.Tensor) or not isinstance(normal, torch.Tensor):
                reason = "missing_old_point_or_normal"
            elif point.shape != (dim,) or normal.shape != (dim,):
                reason = "invalid_old_point_or_normal_shape"
            elif not bool(torch.isfinite(point).all() and torch.isfinite(normal).all()):
                reason = "nonfinite_old_point_or_normal"
            elif norm is None or not math.isfinite(float(norm)) or float(norm) <= 1e-8:
                reason = "invalid_old_normal_norm"
            elif not math.isclose(float(normal.norm()), 1., rel_tol=1e-8, abs_tol=1e-10):
                reason = "invalid_old_unit_normal"
        if reason is not None:
            rejected.append(dict(origin_id=int(row["origin_id"]), position=position, reason=reason))
            continue
        points.append(point.double().detach())
        normals.append(normal.double().detach())
        norms.append(float(norm))
        kept.append(position)
    if not points:
        empty = torch.empty(0, dtype=torch.float64)
        result = BoundaryBatch(torch.empty((0, dim), dtype=torch.float64), torch.empty((0, 2), dtype=torch.long),
                               empty, empty, empty, torch.empty((0, dim), dtype=torch.float64), {},
                               torch.empty(0, dtype=torch.long))
        return result, rejected
    points, normals = torch.stack(points), torch.stack(normals)
    with torch.no_grad():
        logits = model(points)
        margins = logits[:, 0] - logits[:, 1]
    # U2 is zero; the exact affine normal supplies a dependency-integrity check.
    affine = model.linear.weight[0] - model.linear.weight[1]
    expected_norm = affine.norm()
    torch.testing.assert_close(normals, (affine / expected_norm).expand_as(normals), rtol=1e-8, atol=1e-10)
    norms = torch.tensor(norms, dtype=torch.float64)
    torch.testing.assert_close(norms, expected_norm.expand_as(norms), rtol=1e-8, atol=1e-10)
    finite(points, normals, norms, margins)
    return BoundaryBatch(points, torch.tensor([[0, 1]] * len(points)), norms, margins.detach(),
                         torch.zeros_like(norms), normals, {}, torch.tensor(kept)), rejected


def propose_arms(model, source, labels, boundaries, zeta, cfg, sigma0=1.):
    """Solve three RHSs with the production solver and identical D curvature.

    Production zero requests are an exact no-op. matched_source deliberately
    solves the zero-request problem, retaining D.T D, as a research control.
    """
    proposals = {}
    for arm, request in (("normal", zeta), ("reverse", -zeta),
                         ("zero", torch.zeros_like(zeta)), ("matched_source", torch.zeros_like(zeta))):
        problem = TsbaProblem(model, source, labels, replace(boundaries, zeta=request.detach().clone()), cfg)
        if arm == "zero":
            delta, info = torch.zeros_like(problem.base), dict(valid=True, reason="zero_requests", solver="none", no_op=True)
        elif not len(boundaries):
            delta, info = torch.zeros_like(problem.base), dict(valid=False, reason="no_valid_boundary", solver="none")
        elif arm != "matched_source" and bool((request.abs() <= cfg.movement_zero * sigma0).all()):
            delta, info = torch.zeros_like(problem.base), dict(valid=True, reason="zero_requests", solver="none", no_op=True)
        else:
            delta, info = solve(problem)
        proposals[arm] = dict(problem=problem, delta=delta, solver=info)
    matched = proposals["matched_source"]["delta"]
    plus, minus = proposals["normal"]["delta"], proposals["reverse"]["delta"]
    paired = None
    if all(v is not None for v in (matched, plus, minus)) and len(boundaries):
        # If normal/reverse invoke the production all-zero guard, they are both
        # exact no-ops rather than solves; the algebraic identity does not apply.
        guarded = any(proposals[a]["solver"].get("no_op") for a in ("normal", "reverse"))
        if not guarded:
            torch.testing.assert_close(plus + minus, 2 * matched, rtol=1e-7, atol=1e-9)
            paired = float((plus + minus - 2 * matched).abs().max())
    return proposals, paired


def shared_small_alpha(proposals, d, sigma0, start=1 / 16, floor=1 / 32768):
    """One predeclared alpha across every estimator, bandwidth and nonzero arm."""
    if not 0 < floor <= start or sigma0 <= 0:
        raise ValueError("invalid_small_step_limits")
    values = []
    for group in proposals:
        for arm in ("normal", "reverse", "matched_source"):
            item = group[arm]
            if item["delta"] is None or not item["solver"].get("valid"):
                return None, dict(reason="unavailable_proposal", halvings=0)
            movement = d @ item["delta"]
            finite(movement)
            values.append(float(movement.abs().max()) if movement.numel() else 0.)
    largest = max(values, default=0.)
    alpha, halvings = start, 0
    while alpha * largest > .1 * sigma0 and alpha > floor:
        alpha /= 2
        halvings += 1
    if alpha * largest > .1 * sigma0:
        return None, dict(reason="small_step_floor_exhausted", halvings=halvings,
                          max_full_predicted_movement_sigma=largest / sigma0, attempted_alpha=alpha)
    return alpha, dict(reason="motion_guard_halving" if halvings else "predeclared_start",
                       halvings=halvings, initial_alpha=start, selected_alpha=alpha,
                       max_full_predicted_movement_sigma=largest / sigma0,
                       max_predicted_movement_sigma=alpha * largest / sigma0,
                       criterion="max absolute D delta across both estimators, both bandwidths and all three solved arms",
                       classification_labels_used=False)


def score_comparison(empirical, learned, normals, sigma0):
    empirical_gamma, empirical_request = displacement_request(empirical, normals, sigma0)
    learned_gamma, learned_request = displacement_request(learned, normals, sigma0)
    finite(empirical_gamma, learned_gamma)
    identifiable = (sigma0 * empirical_gamma).abs() >= 1e-3
    identifiable_n = int(identifiable.sum())
    def rmse(value):
        return float(value.square().mean().sqrt()) if value.numel() else None
    def agreement(a, b):
        return float((a.sign() == b.sign()).double().mean()) if a.numel() else None
    return dict(reference="exact finite-pool Gaussian KDE, not a true-score oracle", n=len(normals),
                vector_rmse=rmse(learned - empirical),
                vector_rmse_sigma=rmse(sigma0 * (learned - empirical)),
                normal_score_rmse=rmse(learned_gamma - empirical_gamma),
                normal_score_rmse_sigma=rmse(sigma0 * (learned_gamma - empirical_gamma)),
                normal_sign_agreement=agreement(learned_gamma, empirical_gamma),
                normal_sign_threshold_sigma=1e-3, normal_sign_threshold_denominator=identifiable_n,
                normal_sign_agreement_above_threshold=(
                    float((learned_gamma[identifiable].sign() == empirical_gamma[identifiable].sign()).double().mean())
                    if identifiable_n else None),
                request_rmse_sigma=rmse((learned_request - empirical_request) / sigma0),
                request_sign_agreement=agreement(learned_request, empirical_request),
                empirical_gamma=empirical_gamma, learned_gamma=learned_gamma,
                empirical_request=empirical_request, learned_request=learned_request)


def _predictions(problem, candidate, features, pools, archived=None):
    result = {}
    with torch.no_grad():
        for name, ids in pools.items():
            logits = problem.params.logits(candidate, features[ids])
            finite(logits)
            probability, log_probability = logits.softmax(-1), logits.log_softmax(-1)
            if archived is not None:
                expected = archived["probabilities"][ids].double()
                expected_log = archived["log_probabilities"][ids].double()
                torch.testing.assert_close(probability, expected, rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(log_probability, expected_log, rtol=1e-10, atol=1e-12)
                probability, log_probability = expected.clone(), expected_log.clone()
            result[name] = dict(ids=list(map(int, ids)), probabilities=probability.detach(),
                                log_probabilities=log_probability.detach())
    return result


def _candidate_view(item, view, alpha, sigma0, d, features, pools, archived):
    problem, delta, solver = item["problem"], item["delta"], item["solver"]
    noop = bool(solver.get("no_op"))
    trials, candidate, accepted = [], problem.base, False
    reason, used_alpha = solver["reason"], 0.
    if noop:
        reason = "zero_requests_exact_noop"
    elif solver.get("valid") and delta is not None:
        if view == "small_step":
            if alpha is not None:
                candidate, accepted, used_alpha = problem.base + alpha * delta, True, alpha
                reason = "common_small_step"
            else:
                reason = "common_alpha_unavailable"
        else:
            proposal, trials = backtrack(problem, delta, solver["directional_derivative"], sigma0)
            if proposal is not None:
                candidate, accepted, used_alpha = proposal, True, trials[-1]["alpha"]
                reason = "accepted"
            else:
                reason = "backtracking_exhausted"
    finite(candidate)
    identical = torch.equal(candidate, problem.base)
    predictions = _predictions(problem, candidate, features, pools, archived if identical else None)
    with torch.no_grad():
        actual = problem.movement(candidate)
        predicted = d @ (candidate - problem.base)
    finite(actual, predicted)
    evaluation = problem.evaluate(candidate) if len(problem.boundaries) else None
    available = view == "production_backtrack" or noop or accepted
    metadata = dict(status="success" if available else "failed",
                    accepted=accepted, no_op=identical, reason=reason, alpha=float(used_alpha),
                    before=problem.evaluate(problem.base) if len(problem.boundaries) else None,
                    after=evaluation, trials=trials,
                    predicted_max_movement_sigma=float(predicted.abs().max()) / sigma0 if len(predicted) else None,
                    actual_max_movement_sigma=float(actual.abs().max()) / sigma0 if len(actual) else None,
                    movement_definition="true-forward normalized margin change at fixed old boundary points; physical first-order boundary displacement is its negative; not a relocated root")
    payload = dict(**metadata, vector=candidate.detach().clone(),
                   model_state={k: v.detach().clone() for k, v in problem.params.mapping(candidate).items()},
                   probabilities=predictions, predicted_margin_movement=predicted.detach(),
                   actual_margin_movement=actual.detach(),
                   predicted_boundary_displacement=-predicted.detach(),
                   actual_boundary_displacement_proxy=-actual.detach())
    return payload, metadata


def _validate_plan(plan):
    if list(plan["seeds"]) != list(range(3405, 3410)) or list(plan.get("domains", [1, 3])) != [1, 3]:
        raise ValueError("step_grid_must_be_the_five_declared_seeds_and_domains_1_3")
    for section, defaults in (("steps", STEP_DEFAULTS), ("scores", SCORE_DEFAULTS)):
        for key, expected in defaults.items():
            if plan.get(section, {}).get(key, expected) != expected:
                raise ValueError(f"undeclared_{section}_setting:{key}")
    return {**STEP_DEFAULTS, **plan.get("steps", {})}, {**SCORE_DEFAULTS, **plan.get("scores", {})}


def _cell_inputs(root, seed, domain, labels, cfg):
    deps = root / "deps"
    roles = json.loads((deps / "roles.json").read_text())
    seed_roles = json.loads((deps / "roles" / f"seed{seed}.json").read_text())
    fit_ids = list(map(int, seed_roles["fit_ids"]))
    if len(fit_ids) != 779 or len(set(fit_ids)) != 779 or set(labels) != set(fit_ids):
        raise PermissionError("steps_accept_exact_source_fit779_labels_only")
    if any(type(v) is not int or v not in (0, 1) for v in labels.values()):
        raise ValueError("invalid_source_fit_class")
    frozen_path = deps / "source" / f"seed{seed}" / "frozen.pt"
    frozen = _load(frozen_path)
    x = frozen["x"].double()
    finite(x)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ResidualClassifier(x.shape[1], 2, 16, dtype=torch.float64)
    model.load_state_dict(frozen["initial_head"])
    model.eval().requires_grad_(False)
    if bool((model.U2.weight != 0).any()):
        raise ValueError("step_initial_head_must_have_U2_zero")
    archived = _load(deps / "source" / f"seed{seed}" / "predictions.pt")
    if list(map(int, archived["ids"])) != list(range(len(x))):
        raise ValueError("source_prediction_id_order_mismatch")
    geometry = deps / "geometry" / f"seed{seed}_domain{domain}"
    bank = _load(geometry / "banks/source.pt")
    bootstrap = _load(geometry / "bootstrap.pt")
    target_ids = list(map(int, roles["targets"][str(domain)]["ids"]))
    if bank["target_ids"] != target_ids or bootstrap["target_ids"] != target_ids:
        raise ValueError("old_bank_or_bootstrap_target_pool_mismatch")
    if len(bank["anchors"]) != 64 or len(bank["origin_ids"]) != 64:
        raise ValueError("old_source_bank_must_have_64_predeclared_anchors")
    if [int(r["origin_id"]) for r in bank["anchors"]] != list(map(int, bank["origin_ids"])):
        raise ValueError("old_anchor_id_order_mismatch")
    if len(set(bank["origin_ids"])) != 64 or not set(bank["origin_ids"]).issubset(roles["targets"][str(domain)]["A"]):
        raise ValueError("old_anchors_not_distinct_A_queries")
    sigma0 = float(bank["sigma"])
    if not math.isfinite(sigma0) or sigma0 <= 0 or bootstrap["sigma"] != sigma0:
        raise ValueError("invalid_fixed_sigma0")
    boundaries, rejected = old_boundaries(bank, model, sigma0)
    pools = {f"B{domain}": roles["targets"][str(domain)]["B"],
             **{f"source_{k}": seed_roles[f"{k}_ids"] for k in ("fit", "validation", "diagnostic")}}
    pools["source_all"] = sorted(set().union(*(set(v) for k, v in pools.items() if k.startswith("source_"))))
    y = torch.tensor([labels[i] for i in fit_ids], dtype=torch.long)
    problem = TsbaProblem(model, x[fit_ids], y, boundaries, cfg)
    _predictions(problem, problem.base, x, pools, archived)
    if len(boundaries):
        system = GGNSystem(problem)
        matrix = system.dense()
        d = jacrev(problem.normalized_margin)(problem.base).detach()
        source_gradient = system.gs
    else:
        matrix = None
        d = problem.base.new_empty((0, len(problem.base)))
        source_gradient = grad(problem.source_loss)(problem.base).detach()
    return dict(seed=seed, domain=domain, x=x, model=model, source=x[fit_ids], labels=y,
                fit_ids=fit_ids, target_ids=target_ids, bank=bank, boundaries=boundaries, rejected=rejected,
                sigma0=sigma0, pools=pools, archived=archived, cfg=cfg, matrix=matrix, d=d,
                source_gradient=source_gradient, source_rhs=-source_gradient,
                dependency_hashes={"source_frozen": digest(frozen_path),
                    "source_bank": digest(geometry / "banks/source.pt"), "bootstrap": digest(geometry / "bootstrap.pt")},
                combinations={})


def _combo(cell, estimator, multiplier, score):
    gamma, zeta = displacement_request(score, cell["boundaries"].normals, cell["sigma0"])
    proposals, paired = propose_arms(cell["model"], cell["source"], cell["labels"],
                                    cell["boundaries"], zeta, cell["cfg"], cell["sigma0"])
    d = cell["d"]
    score_rhs = (cell["cfg"].lambda_boundary / len(d) * d.T @ zeta
                 if len(d) else torch.zeros_like(cell["source_rhs"]))
    finite(score_rhs)
    for arm, item in proposals.items():
        if item["solver"].get("no_op") or not len(d):
            continue
        # Verify RHS independence and the shared A, without replacing the solver.
        system = GGNSystem(item["problem"])
        expected = cell["source_rhs"] + (score_rhs if arm == "normal" else -score_rhs if arm == "reverse" else 0)
        torch.testing.assert_close(system.rhs, expected, rtol=1e-10, atol=1e-12)
        if item["delta"] is not None:
            torch.testing.assert_close(system.matvec(item["delta"]), cell["matrix"] @ item["delta"],
                                       rtol=1e-8, atol=1e-10)
    value = dict(estimator=estimator, bandwidth_multiplier=multiplier, h=cell["sigma0"] * multiplier,
                 score=score.detach(), gamma=gamma, zeta=zeta, proposals=proposals,
                 score_rhs=score_rhs.detach(), paired_max_abs_error=paired)
    cell["combinations"][(estimator, multiplier)] = value
    return value


def _proposal_archive(value):
    proposals = value["proposals"]
    source_delta = proposals["matched_source"]["delta"]
    return {k: v for k, v in value.items() if k != "proposals"} | dict(
        arms={name: dict(delta=item["delta"], solver=item["solver"],
                        request=item["problem"].boundaries.zeta) for name, item in proposals.items()},
        source_delta=source_delta,
        score_delta=(proposals["normal"]["delta"] - source_delta
                     if source_delta is not None and proposals["normal"]["delta"] is not None else None))


def run_steps(root, plan, source_labels, log=None):
    """Execute empirical proposals, then learned fits, then paired one-step views.

    source_labels is {integer seed: {integer source-fit graph ID: class}}.
    All 160 arms start from the same frozen head within a seed. Index entries
    point to .pt files with views[view]['model_state'/'probabilities']; each pool
    contains ids, probabilities and log_probabilities. OBS closes all writes.
    """
    root = Path(root)
    _open_observation(root)
    step_spec, score_spec = _validate_plan(plan)
    if set(source_labels) != set(plan["seeds"]):
        raise PermissionError("source_labels_must_match_declared_seeds_exactly")
    # Validate every label map before any fitting or writing, including extra IDs.
    for seed in plan["seeds"]:
        sr = json.loads((root / "deps/roles" / f"seed{seed}.json").read_text())
        if len(sr["fit_ids"]) != 779 or set(source_labels[seed]) != set(sr["fit_ids"]):
            raise PermissionError("steps_accept_exact_source_fit779_labels_only")
    (root / "steps").mkdir(exist_ok=False)
    (root / "scores").mkdir(exist_ok=False)
    cells, score_index, step_index = [], [], []
    # Every empirical proposal precedes every learned score fit.
    for seed in plan["seeds"]:
        cfg = Config(seed=seed, source_adapt_batch=779, lambda_boundary=1., damping=.01,
                     tau=.1, request_multiplier=.1, execution_multiplier=.1,
                     score_hidden=score_spec["hidden"], score_epochs=score_spec["epochs"],
                     score_batch=score_spec["batch"], score_lr=score_spec["lr"],
                     score_validation=score_spec["validation"], solver="direct")
        for domain in plan.get("domains", [1, 3]):
            _open_observation(root)
            _limits(time.monotonic(), plan["resources"])
            cell = _cell_inputs(root, seed, domain, source_labels[seed], cfg)
            path = root / "steps" / f"seed{seed}_domain{domain}"
            path.mkdir()
            cell["path"] = path
            source_ggn = None
            if cell["matrix"] is not None:
                source_ggn = (cell["matrix"] - cfg.lambda_boundary / len(cell["d"]) * cell["d"].T @ cell["d"]
                              - cfg.damping * torch.eye(len(cell["source_rhs"]), dtype=torch.float64))
                finite(source_ggn)
            save_tensor(path / "system.pt", dict(seed=seed, domain=domain, sigma0=cell["sigma0"],
                        source_fit_ids=cell["fit_ids"], target_ids=cell["target_ids"],
                        origin_ids=cell["bank"]["origin_ids"],
                        valid_origin_ids=[cell["bank"]["origin_ids"][i] for i in cell["boundaries"].input_indices.tolist()],
                        rejected=cell["rejected"], boundaries=cell["boundaries"], matrix=cell["matrix"],
                        source_ggn=source_ggn,
                        source_ggn_definition="production A minus lambda/M D.T D minus damping I; full source-fit GGN",
                        d=cell["d"], source_gradient=cell["source_gradient"], source_rhs=cell["source_rhs"],
                        base_state=cell["model"].state_dict(), config=cfg.to_dict(),
                        dependencies=cell["dependency_hashes"], anchor_filter="old valid point and normal only; no ESS or support count filter"))
            for multiplier in step_spec["bandwidth_multipliers"]:
                kde = EmpiricalScore(cell["x"][cell["target_ids"]], cell["sigma0"] * multiplier)
                score = kde(cell["boundaries"].points)
                value = _combo(cell, "empirical", multiplier, score)
                combo_path = path / f"empirical_b{multiplier:g}"
                combo_path.mkdir()
                save_tensor(combo_path / "proposal.pt", _proposal_archive(value))
            cells.append(cell)
            _log(log, f"empirical proposals seed={seed} domain={domain} anchors={len(cell['boundaries'])}/64")
    # Independent fits reuse production seed offsets and fixed validation noise.
    for cell in cells:
        for multiplier in step_spec["bandwidth_multipliers"]:
            _open_observation(root)
            directory = root / "scores" / f"seed{cell['seed']}_domain{cell['domain']}" / f"b{multiplier:g}"
            model, metadata = fit_score(directory, cell["x"][cell["target_ids"]],
                                        cell["sigma0"] * multiplier, cell["cfg"], plan["resources"], log)
            info = dict(metadata, seed=cell["seed"], domain=cell["domain"], bandwidth_multiplier=multiplier,
                        sigma0=cell["sigma0"], h=cell["sigma0"] * multiplier,
                        path=str(directory.relative_to(root)))
            score_index.append(info)
            if model is None:
                _log(log, f"score failed seed={cell['seed']} domain={cell['domain']} b={multiplier}: {metadata.get('failure')}")
                continue
            expected_updates = math.ceil(len(cell["target_ids"]) / 128) * 100
            if expected_updates != 900 or metadata["updates_completed"] != expected_updates:
                raise ValueError("learned_score_must_complete_the_declared_900_updates")
            with torch.no_grad():
                score = model(cell["boundaries"].points)
            finite(score)
            comparison = score_comparison(cell["combinations"][("empirical", multiplier)]["score"],
                                          score, cell["boundaries"].normals, cell["sigma0"])
            comparison.update(seed=cell["seed"], domain=cell["domain"], bandwidth_multiplier=multiplier,
                              sigma0=cell["sigma0"], h=cell["sigma0"] * multiplier)
            save_tensor(directory / "static.pt", comparison)
            info["static_comparison"] = {k: v for k, v in comparison.items() if not isinstance(v, torch.Tensor)}
            value = _combo(cell, "learned", multiplier, score)
            combo_path = cell["path"] / f"learned_b{multiplier:g}"
            combo_path.mkdir()
            save_tensor(combo_path / "proposal.pt", _proposal_archive(value))
    write_json(root / "scores/index.json", score_index)
    for cell in cells:
        _open_observation(root)
        complete = len(cell["combinations"]) == 4
        alpha, alpha_info = shared_small_alpha([v["proposals"] for v in cell["combinations"].values()],
                                               cell["d"], cell["sigma0"], step_spec["small_step_start"],
                                               step_spec["small_step_floor"])
        if not complete:
            alpha, alpha_info = None, dict(reason="missing_estimator_bandwidth_combination", halvings=0)
        write_json(cell["path"] / "common_alpha.json", alpha_info)
        for estimator in ESTIMATORS:
            for multiplier in step_spec["bandwidth_multipliers"]:
                value = cell["combinations"].get((estimator, multiplier))
                for arm in ARMS:
                    record = dict(seed=cell["seed"], domain=cell["domain"], estimator=estimator,
                                  bandwidth_multiplier=multiplier, arm=arm, sigma0=cell["sigma0"],
                                  h=cell["sigma0"] * multiplier, common_alpha=alpha,
                                  common_alpha_reason=alpha_info["reason"], views={}, status="failed")
                    if value is None:
                        if arm not in ("zero", "matched_source"):
                            record["failure"] = "missing_learned_score_fit"
                            step_index.append(record)
                            continue
                        effective_value = cell["combinations"][("empirical", multiplier)]
                        record["score_estimator_available"] = False
                    else:
                        effective_value = value
                        record["score_estimator_available"] = True
                    item = effective_value["proposals"][arm]
                    matched_delta = effective_value["proposals"]["matched_source"]["delta"]
                    source_delta = torch.zeros_like(item["problem"].base) if item["solver"].get("no_op") else matched_delta
                    score_delta = item["delta"] - source_delta if item["delta"] is not None and source_delta is not None else None
                    payload = dict(record, base_state=copy.deepcopy(cell["model"].state_dict()),
                        base_vector=item["problem"].base, proposal_delta=item["delta"], solver=item["solver"],
                        source_proposal_delta=source_delta, score_proposal_delta=score_delta,
                        matched_source_proposal_delta=matched_delta,
                        proposal_source_movement=cell["d"] @ source_delta if source_delta is not None else None,
                        proposal_score_movement=cell["d"] @ score_delta if score_delta is not None else None,
                        proposal_predicted_margin_movement=cell["d"] @ item["delta"] if item["delta"] is not None else None,
                        gamma=value["gamma"] if value is not None else None,
                        score=value["score"] if value is not None else None, boundaries=item["problem"].boundaries,
                        source_rhs=cell["source_rhs"],
                        score_rhs=(effective_value["score_rhs"] if arm == "normal" else
                                   -effective_value["score_rhs"] if arm == "reverse" else torch.zeros_like(effective_value["score_rhs"])),
                        paired_max_abs_error=value["paired_max_abs_error"] if value is not None else None,
                        common_alpha_details=alpha_info,
                        views={})
                    for view in ("small_step", "production_backtrack"):
                        payload["views"][view], record["views"][view] = _candidate_view(
                            item, view, alpha, cell["sigma0"], cell["d"], cell["x"], cell["pools"], cell["archived"])
                    if arm == "zero":
                        for view in payload["views"].values():
                            torch.testing.assert_close(view["vector"], item["problem"].base, rtol=0, atol=0)
                    status = "success"
                    record.update(status=status, solver=item["solver"],
                                  paired_max_abs_error=value["paired_max_abs_error"] if value is not None else None,
                                  valid_anchors=len(cell["boundaries"]))
                    payload["status"] = status
                    path = cell["path"] / f"{estimator}_b{multiplier:g}" / f"{arm}.pt"
                    path.parent.mkdir(exist_ok=True)
                    record["path"] = str(path.relative_to(root))
                    save_tensor(path, payload)
                    step_index.append(record)
        _log(log, f"paired one-step views seed={cell['seed']} domain={cell['domain']} common_alpha={alpha} reason={alpha_info['reason']}")
    if len(step_index) != 160 or len(score_index) != 20:
        raise AssertionError("incomplete_predeclared_step_or_score_grid")
    write_json(root / "steps/index.json", step_index)
    return dict(steps=step_index, scores=score_index)
