"""Original R5 M00 A0/B1 on R7's adaptation inputs only.

The source encoder is never trained here. Score fitting stops at an exact
optimizer-update count, including short minibatches and a partial final epoch.
The M00 update uses R5's propose/accept functions: in particular B1's zero
request does NOT skip its source-loss update. Passive density/relocation and
three-way side diagnostics are omitted; an engineering fixture checks the
resulting decisions against R5's actual ``one_attempt`` implementation.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from tsba.boundary import build_boundaries
from tsba.classifier import Coordinates, ParameterMap, ResidualClassifier
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.score import ScoreNetwork, bandwidth, denoising_loss
from tsba.utils import finite, file_hash, identity, tensor_hash
from experiments.common.io import write_json
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.adapt import ZeroScore
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import atomic_tensor, batches
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.score import validation
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src.constraints import Budgets, accept, propose
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src.protocol import Job


_BLIND_KEYS = {
    "task", "source_h", "source_x", "source_y", "source_ids", "adapt_h", "adapt_x", "adapt_z0", "adapt_ids",
    "source_head_weight", "source_head_bias", "tsba_mean", "tsba_scale", "tsba_initial_head", "r5_plan",
    "r5_model_config", "r5_adam_config", "provenance",
}
_OFFSETS = dict(score_initialization=101, score_training=102, score_diagnostic=103, adaptation_batches=201)
_DEPENDENCIES = (
    "tsba/classifier.py", "tsba/boundary.py", "tsba/problem.py", "tsba/score.py", "tsba/subproblem.py", "tsba/adapt.py",
    "experiments/mutagenicity/r2_分数训练预算与多轮适配_20261111/src/common.py",
    "experiments/mutagenicity/r2_分数训练预算与多轮适配_20261111/src/score.py",
    "experiments/mutagenicity/r3_源风险与置信度约束_20261112/src/constraints.py",
    "experiments/mutagenicity/r3_源风险与置信度约束_20261112/src/trajectory.py",
)


def blind_inputs(loaded):
    """Copy an authenticated include_tsba=True loader result to a fit whitelist.

    This coordinator helper does not inspect H arrays. The returned dict has no
    H inputs, validation inputs, target labels or complete R7 input manifest.
    """
    manifest = loaded["manifest"]
    result = {key: loaded[key] for key in _BLIND_KEYS - {"task", "r5_adam_config"}}
    result["r5_adam_config"] = deepcopy(loaded.get("r5_adam_config", loaded["r5_plan"]["adam"]))
    result["task"] = {name: manifest[name] for name in ("setting", "source", "target", "seed")}
    return result


def _tensor(value, *, integer=False):
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
    else:
        result = torch.from_numpy(np.array(value, copy=True))
    if integer and result.dtype != torch.int64:
        raise ValueError("Expected int64 source labels")
    result = result.to(torch.int64 if integer else torch.float64)
    finite(result)
    return result


def _state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _state_hash(state):
    return identity({name: tensor_hash(value) for name, value in state.items()})


def _head(state):
    state = {name: _tensor(value) for name, value in state.items()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = ResidualClassifier(state["linear.weight"].shape[1], state["linear.weight"].shape[0],
                                   state["U1.weight"].shape[0], dtype=torch.float64)
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)


def _adam(model, learning_rate, arguments):
    options = deepcopy(arguments)
    options["betas"] = tuple(options["betas"])
    return torch.optim.Adam(model.parameters(), lr=learning_rate, **options)


def _train_exact_score(x, sigma, cfg, adam_arguments, updates):
    """R5 DSM RNG/Adam semantics with an exact minibatch-update stopping rule."""
    if type(updates) is not int or updates < 1:
        raise ValueError("A positive exact score update count is required")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("R5 initializes ScoreNetwork in the original float32 default dtype before conversion")
    diagnostic_rng = torch.Generator().manual_seed(cfg.seed + 103)
    diagnostic_noise = torch.randn(x.shape, generator=diagnostic_rng, dtype=x.dtype)
    rng = torch.Generator().manual_seed(cfg.seed + 102)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + 101)
        model = ScoreNetwork(x.shape[1], cfg.score_hidden).double()
    initial_hash = _state_hash(_state(model))
    optimizer = _adam(model, cfg.score_lr, adam_arguments)
    completed, epoch, history, batch_sizes = 0, 0, [], []
    trace = hashlib.sha256()
    while completed < updates:
        epoch += 1
        order = torch.randperm(len(x), generator=rng)
        weighted_loss, samples_seen, batches_seen = 0., 0, 0
        for indices in order.split(cfg.score_batch):
            noise = torch.randn(x[indices].shape, generator=rng, dtype=x.dtype)
            loss = denoising_loss(model, x[indices], sigma, noise)
            finite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(parameter.grad for parameter in model.parameters()))
            optimizer.step()
            finite(*model.parameters())
            completed += 1
            batches_seen += 1
            samples_seen += len(indices)
            weighted_loss += float(loss.detach()) * len(indices)
            batch_sizes.append(len(indices))
            trace.update(indices.numpy().astype("<i8", copy=False).tobytes())
            if completed == updates:
                break
        # Same fixed independent noise on A as R5; it never selects a checkpoint.
        diagnostic = validation(model, x, sigma, diagnostic_noise)
        history.append(dict(epoch=epoch, updates=completed, batches=batches_seen,
                            samples_seen=samples_seen, full_epoch=samples_seen == len(x),
                            train=weighted_loss / samples_seen, validation=diagnostic))
    model.eval().requires_grad_(False)
    record = dict(updates=completed, selected_update=completed, checkpoint_selection="exact_final_update",
                  epochs_started=epoch, complete_epochs=sum(row["full_epoch"] for row in history),
                  final_epoch_complete=history[-1]["full_epoch"], batch_sizes=batch_sizes,
                  minibatch_position_trace_sha256=trace.hexdigest(),
                  model_state=_state(model), optimizer_state=deepcopy(optimizer.state_dict()),
                  training_rng_state=rng.get_state().clone(), last_epoch_order=order.clone(),
                  batches_consumed_in_last_epoch=batches_seen,
                  diagnostic_noise=diagnostic_noise, diagnostic_rng_state=diagnostic_rng.get_state().clone(),
                  initialization_state_sha256=initial_hash, history=history, bandwidth=sigma,
                  validation_scope="fixed_independent_noise_on_adaptation_inputs_only",
                  validation_used_for_checkpoint_selection=False)
    return model, record


def _m00_attempt(model, batch, source, labels, target, estimator, sigma, budgets, job, cfg, old_plan):
    """R5 one_attempt's decision path, without its passive observers."""
    if job.protect or job.epsilon is not None or job.group != "M00":
        raise ValueError("Matched baseline requires the original unconstrained M00")
    params = ParameterMap(model)
    base = params.vector()
    bounds = build_boundaries(model, target[batch["target_positions"]], estimator, sigma, cfg)
    bounds = replace(bounds, zeta=bounds.zeta * job.direction)
    record = dict(before=base, after=base, batch=deepcopy(batch), bounds=asdict(bounds),
                  status="no_valid_boundary", accepted=False, trials=[])
    if not len(bounds):
        return record
    selected = batch["source_positions"]
    problem = TsbaProblem(model, source[selected], labels[selected], bounds, cfg)
    # This condition deliberately does not apply to B1 (direction == 0).
    if job.direction and not bool((bounds.zeta.abs() > cfg.movement_zero * sigma).any()):
        delta = torch.zeros_like(base)
        info = dict(valid=False, reason="zero_requests")
    else:
        delta, info = propose(problem, budgets, old_plan["solver"])
    record.update(solver=info, delta=delta, status=info["reason"])
    if info["valid"]:
        vector, trials = accept(problem, delta, info["directional_derivative"], sigma, budgets)
        record["trials"] = trials
        record["status"] = "backtracking_exhausted" if vector is None else "accepted"
        if vector is not None:
            record.update(after=vector, accepted=True)
    record["margin_proxy"] = problem.movement(record["after"]).detach()
    record["true_budgets"] = budgets.check(params, record["after"])
    if record["accepted"]:
        params.commit(record["after"])
    return record


def _prepare(plan, inputs, seed):
    if set(inputs) != _BLIND_KEYS:
        raise ValueError(f"Matched fit requires the blind whitelist; missing={_BLIND_KEYS - set(inputs)}, "
                         f"extra={set(inputs) - _BLIND_KEYS}")
    registration = plan["phases"]["C"]["matched_tsba"]
    task = inputs["task"]
    if (set(task) != {"setting", "source", "target", "seed"} or task["seed"] != seed
            or type(seed) is not int or seed not in plan["real_model_seeds"]
            or task["setting"] not in plan["phases"]["C"]["settings"]
            or task["source"] not in plan["phases"]["C"]["domains"]
            or task["target"] not in plan["phases"]["C"]["domains"] or task["source"] == task["target"]):
        raise ValueError("Unregistered matched baseline task")
    if (registration["compared_groups"] != ["B1", "A0"] or not registration["use_R5_M00_implementation"]
            or registration["seed_offsets"] != _OFFSETS or registration["score_exact_updates"] != 7200
            or registration["attempts_per_group"] != 30):
        raise ValueError("Matched baseline registration differs from the R7 contract")
    old_plan = deepcopy(inputs["r5_plan"])
    if (old_plan["model"] != inputs["r5_model_config"] or old_plan["adam"] != inputs["r5_adam_config"]
            or old_plan["setting"]["id"] != task["setting"]
            or seed not in old_plan["seeds"] or old_plan["solver"] != {}):
        raise ValueError("Matched configuration differs from authenticated R5 configuration")
    for method, density in (("B1", "zero"), ("A0", "learned")):
        if old_plan["methods"][method]["group"] != "M00" or old_plan["methods"][method]["density"] != density:
            raise ValueError("Archived A0/B1 definitions differ from original M00")
    cfg = Config(**{**inputs["r5_model_config"], "seed": seed, "lambda_boundary": old_plan["weight"]})
    checks = {"source_adapt_batch": registration["source_and_target_batch"],
              "target_batch": registration["source_and_target_batch"], "score_batch": registration["score_batch"],
              "score_lr": registration["score_lr"], "adapt_rounds": registration["attempts_per_group"],
              "lambda_boundary": registration["lambda_boundary"], "tau": registration["boundary_tau"],
              "damping": registration["damping"], "kappa_sigma": registration["kappa_sigma"],
              "device": "cpu", "adaptation_dtype": "float64"}
    if any(getattr(cfg, key) != value for key, value in checks.items()):
        raise ValueError("Original R5 configuration does not match registered C baseline parameters")
    h_source, h_adapt = _tensor(inputs["source_h"]), _tensor(inputs["adapt_h"])
    source, target, labels = (_tensor(inputs[key], integer=key == "source_y")
                              for key in ("source_x", "adapt_x", "source_y"))
    if (h_source.shape != source.shape or h_adapt.shape != target.shape or source.ndim != 2 or target.ndim != 2
            or not len(source) or not len(target)
            or source.shape[1] != cfg.emb_dim or target.shape[1] != cfg.emb_dim
            or labels.shape != (len(source),) or labels.unique().tolist() != [0, 1]):
        raise ValueError("Invalid frozen source/adaptation pools")
    ids = [list(inputs[key]) for key in ("source_ids", "adapt_ids")]
    if (any(len(group) != count or len(set(group)) != count for group, count in zip(ids, (len(source), len(target))))
            or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or i < 0
                   for group in ids for i in group) or set(ids[0]) & set(ids[1])):
        raise ValueError("Source and A IDs must be ordered, unique and disjoint")
    ids = [[int(i) for i in group] for group in ids]
    coordinates = Coordinates.fit(h_source, cfg.coordinate_eps)
    torch.testing.assert_close(coordinates.mean, _tensor(inputs["tsba_mean"]), rtol=0, atol=0)
    torch.testing.assert_close(coordinates.scale, _tensor(inputs["tsba_scale"]), rtol=0, atol=0)
    torch.testing.assert_close(coordinates(h_source), source, rtol=0, atol=0)
    torch.testing.assert_close(coordinates(h_adapt), target, rtol=0, atol=0)
    initial = _head(inputs["tsba_initial_head"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        source_head = torch.nn.Linear(cfg.emb_dim, 2, dtype=torch.float64)
    with torch.no_grad():
        source_head.weight.copy_(_tensor(inputs["source_head_weight"]))
        source_head.bias.copy_(_tensor(inputs["source_head_bias"]))
    recreated = ResidualClassifier.from_source(source_head, coordinates, cfg.adapter_width, seed)
    for key, value in recreated.state_dict().items():
        torch.testing.assert_close(value, initial.state_dict()[key], rtol=0, atol=0)
    baseline = _tensor(inputs["adapt_z0"])
    if baseline.shape != (len(target), 2):
        raise ValueError("Invalid adaptation z0 shape")
    with torch.no_grad():
        initial_logits = initial(target)
        torch.testing.assert_close(initial_logits.softmax(-1), baseline.softmax(-1), rtol=1e-10, atol=1e-12)
        if not torch.equal(initial_logits.argmax(1), baseline.argmax(1)):
            raise ValueError("Initial M00 classifier changes B0 predictions")
    return cfg, old_plan, initial, coordinates, source, labels, target, ids


def fit_matched_tsba(plan, inputs, seed, *, output_directory=None):
    """Fit registered A0/B1 using ONLY a ``blind_inputs`` whitelist.

    Return {metadata, coordinates, initial_head, schedule, score, arms}. Each arm
    contains frozen ``state``, ``after``, ``adapt_logits`` (CPU tensors), history,
    attempted/accepted counts and validity. No H array or target label is accepted.
    An optional fresh output directory receives append-only metadata, score and
    per-attempt artifacts. Numerical exceptions fail the run; no zero fallback
    or best-target-checkpoint selection is applied.
    """
    cfg, old_plan, initial, coordinates, source, labels, target, ids = _prepare(plan, inputs, seed)
    out = Path(output_directory) if output_directory is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    budget = plan["phases"]["C"]["matched_tsba"]
    metadata = dict(task=deepcopy(inputs["task"]), method="R7_matched_R5_M00", seed=seed,
                    r5_model_config=deepcopy(inputs["r5_model_config"]), effective_adaptation_config=asdict(cfg),
                    adam=deepcopy(inputs["r5_adam_config"]), score_exact_updates=budget["score_exact_updates"],
                    attempts_per_group=cfg.adapt_rounds, seed_offsets=deepcopy(_OFFSETS),
                    source_count=len(source), adaptation_count=len(target), target_labels_used=False,
                    holdout_inputs_received=False, score_uses_adaptation_pool_only=True,
                    original_global_RMS_coordinates=True, initial_head_sha256=_state_hash(_state(initial)),
                    r5_provenance=deepcopy(inputs["provenance"]),
                    torch_threads=torch.get_num_threads(), score_initialization_dtype="float32_then_float64",
                    omitted_passive_observers=["density_audit", "relocate", "R5_three_way_side_diagnostics"],
                    update_kernel="R5 M00 build_boundaries / R3 propose / R3 accept / ParameterMap.commit",
                    implementation_sha256=file_hash(Path(__file__)),
                    implementation_dependencies_sha256={name: file_hash(Path(__file__).resolve().parents[4] / name)
                                                        for name in _DEPENDENCIES},
                    archived_score_epochs_not_used=cfg.score_epochs,
                    score_steps_per_full_epoch=math.ceil(len(target) / cfg.score_batch),
                    derived_max_score_epochs=math.ceil(budget["score_exact_updates"] /
                                                      math.ceil(len(target) / cfg.score_batch)))
    if out is not None:
        write_json(out / "metadata.json", metadata)
    active_stage = "bandwidth"
    try:
        sigma = bandwidth(target, cfg.kappa_sigma)
        if sigma is None:
            raise ValueError("Degenerate A bandwidth: fewer than two distinct target representations")
        schedule = batches(ids[0], ids[1], seed, rounds=cfg.adapt_rounds, size=cfg.target_batch)
        if out is not None:
            atomic_tensor(out / "schedule.pt", schedule)
        active_stage = "score"
        estimator, score = _train_exact_score(target, sigma, cfg, inputs["r5_adam_config"],
                                               budget["score_exact_updates"])
        if score["updates"] != budget["score_exact_updates"] or len(score["batch_sizes"]) != score["updates"]:
            raise ValueError("Score optimizer update budget was not completed exactly")
        if out is not None:
            atomic_tensor(out / "score.pt", score)
        arms = {}
        for method in ("B1", "A0"):
            active_stage = method
            model = _head(_state(initial))
            params = ParameterMap(model)
            job = Job(seed, inputs["task"]["target"], "M00", "learned" if method == "A0" else "zero",
                      int(method == "A0"), int(method == "A0"), cfg.lambda_boundary, None)
            budgets = Budgets(model, source, labels, target, job, old_plan)
            history = []
            if out is not None:
                atomic_tensor(out / method / "round00.pt", dict(after=params.vector(), adapt_logits=model(target).detach()))
            for batch in schedule:
                active_stage = f"{method}/attempt{batch['attempt']:02d}"
                before = params.vector()
                try:
                    record = _m00_attempt(model, batch, source, labels, target,
                                          estimator if method == "A0" else ZeroScore(), sigma,
                                          budgets, job, cfg, old_plan)
                except Exception:
                    params.commit(before)
                    raise
                record["attempt"] = batch["attempt"]
                if "true_budgets" not in record:
                    record["true_budgets"] = budgets.check(params, record["after"])
                with torch.no_grad():
                    record["adapt_logits"] = model(target).detach().clone()
                finite(record["adapt_logits"])
                history.append(record)
                if out is not None:
                    atomic_tensor(out / method / f"round{batch['attempt']:02d}.pt", record)
            arms[method] = dict(valid=True, status="completed", stop_reason="attempt_budget_completed",
                                attempts=len(history), accepted=sum(row["accepted"] for row in history),
                                state=_state(model), after=params.vector(),
                                adapt_logits=model(target).detach().clone(), history=history)
        result = dict(metadata=metadata, coordinates=dict(mean=coordinates.mean.clone(), scale=coordinates.scale.clone()),
                      initial_head=_state(initial), bandwidth=sigma, schedule=schedule, score=score, arms=arms)
        if out is not None:
            artifact = {key: value for key, value in result.items() if key not in ("score", "schedule", "arms")}
            artifact["arms"] = {name: {key: value for key, value in arm.items() if key != "history"}
                                for name, arm in arms.items()}
            atomic_tensor(out / "predictor.pt", artifact)
            write_json(out / "complete.json", dict(valid=True, updates=score["updates"], bandwidth=sigma,
                                                      arms={name: dict(attempts=arm["attempts"], accepted=arm["accepted"])
                                                            for name, arm in arms.items()}))
        return result
    except Exception as error:
        if out is not None:
            write_json(out / "failure.json", dict(stage=active_stage, exception=type(error).__name__, error=str(error)))
        raise


@torch.no_grad()
def predict_matched(result, embeddings):
    """Evaluate frozen A0/B1 on H only after fitting has returned successfully."""
    h = _tensor(embeddings)
    coordinates = Coordinates(_tensor(result["coordinates"]["mean"]), _tensor(result["coordinates"]["scale"]))
    if h.ndim != 2 or h.shape[1:] != coordinates.mean.shape:
        raise ValueError("Prediction embeddings have incompatible dimensions")
    x = coordinates(h)
    output = {}
    for method in ("B1", "A0"):
        if not result["arms"][method]["valid"] or result["arms"][method]["attempts"] != 30:
            raise ValueError("Only completed registered matched baselines may predict")
        value = _head(result["arms"][method]["state"])(x)
        finite(value)
        output[method] = value.detach().numpy().copy()
    return output


def check_matched_tsba():
    """Small engineering fixture; no actual dataset, labels or full budget run."""
    from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.score import assert_nested_equal, epoch_update
    from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src.trajectory import one_attempt

    cfg = Config(seed=17, emb_dim=2, adapter_width=2, score_hidden=4, score_batch=2,
                 source_adapt_batch=4, target_batch=4, adapt_rounds=2, radius_multiplier=4.)
    adam = dict(betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False,
                maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
    x = torch.tensor([[-.4, -.2], [-.1, .3], [.1, -.3], [.2, .1], [.5, .4]], dtype=torch.float64)
    learned, five = _train_exact_score(x, .4, cfg, adam, 5)
    assert five["updates"] == 5 and five["batch_sizes"] == [2, 2, 1, 2, 2]
    assert five["complete_epochs"] == 1 and not five["final_epoch_complete"]
    # At a whole-epoch endpoint, weights, Adam and the training RNG match R5.
    _, six = _train_exact_score(x, .4, cfg, adam, 6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + 101)
        reference_score = ScoreNetwork(2, 4).double()
    optimizer = _adam(reference_score, cfg.score_lr, adam)
    rng = torch.Generator().manual_seed(cfg.seed + 102)
    epoch_update(reference_score, optimizer, rng, x, .4, batch=2)
    epoch_update(reference_score, optimizer, rng, x, .4, batch=2)
    assert_nested_equal(six["model_state"], reference_score.state_dict())
    assert_nested_equal(six["optimizer_state"], optimizer.state_dict())
    assert_nested_equal(six["training_rng_state"], rng.get_state())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        initial = ResidualClassifier(2, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        initial.linear.weight.copy_(torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64))
        initial.linear.bias.zero_()
    source = torch.tensor([[-1., -.2], [-.5, .3], [.5, -.3], [1., .2]], dtype=torch.float64)
    labels = torch.tensor([1, 1, 0, 0])
    schedule = batches([0, 1, 2, 3], [4, 5, 6, 7, 8], cfg.seed, rounds=2, size=4)
    old_plan = dict(constraint_tolerance=1e-9, risk_budget=0., solver={},
                    relocation_radius_sigma=1., relocation_scan_points=11, relocation_bisections=8)
    accepted_zero = 0
    for method, score in (("B1", ZeroScore()), ("A0", learned)):
        actual, reference = _head(_state(initial)), _head(_state(initial))
        job = Job(cfg.seed, 1, "M00", "learned" if method == "A0" else "zero",
                  int(method == "A0"), int(method == "A0"), cfg.lambda_boundary, None)
        budgets = Budgets(actual, source, labels, x, job, old_plan)
        other_budgets = Budgets(reference, source, labels, x, job, old_plan)
        for batch in schedule:
            row = _m00_attempt(actual, batch, source, labels, x, score, .4, budgets, job, cfg, old_plan)
            original = one_attempt(reference, batch, source, labels, x, score, .4, other_budgets,
                                   job, cfg, old_plan, dict(source_fit=source, target=x), side=False)
            for key in ("before", "after", "bounds", "solver", "status", "accepted", "trials"):
                assert_nested_equal(row[key], original[key])
            assert_nested_equal(actual.state_dict(), reference.state_dict())
            if method == "B1":
                accepted_zero += int(row["accepted"])
    assert accepted_zero > 0, "The B1 fixture must exercise a nonzero source-driven update"
    try:
        _prepare({}, {"holdout_h": x}, 17)
    except ValueError:
        pass
    else:
        raise AssertionError("H data was accepted into the fit interface")
    return dict(passed=True, score_updates_tested=[5, 6], partial_epoch_and_short_batch=True,
                exact_R5_score_state_adam_rng=True, original_M00_attempts_compared=4,
                zero_request_source_update_exercised=True, holdout_input_rejected=True)
