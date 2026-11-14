"""Research-only source-affine warm starts; this module accepts A labels only.

All inputs live in the copied, label-free ``deps`` tree. No encoder is loaded or
trained, and no label broker is constructed here. B and source pools are used
only as input IDs until the caller seals these predictions for later evaluation.
"""
import copy
import json
import math
from pathlib import Path
import resource
import time
import traceback

import torch
from torch.nn import functional as F

from experiments.common.io import digest, save_tensor, write_json
from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.probes import Probe, model_directory


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _limits(started, plan):
    resources = plan["resources"]
    if time.monotonic() - started > min(120, resources["max_probe_seconds"]):
        raise RuntimeError("warm_time_limit")
    if _rss() > min(8 * 1024**3, resources["max_rss_bytes"]):
        raise RuntimeError("warm_rss_limit")


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite_" + name)


def _source_logits(features, state):
    """The frozen ResidualClassifier has a zero residual at initialization."""
    if "U2.weight" not in state or torch.count_nonzero(state["U2.weight"]).item():
        raise ValueError("source_initial_head_is_not_affine")
    weight, bias = state["linear.weight"].double(), state["linear.bias"].double()
    _finite(weight, "source_weight")
    _finite(bias, "source_bias")
    logits = F.linear(features.double(), weight, bias)
    return logits - logits.mean(-1, keepdim=True)


def source_initialized_probe(features, source_state, old_initial):
    """Reuse the old scaler and express the source affine map in its coordinates.

    For z=(x-m)/s, Wz=Wx*s and bz=bx+Wx@m. Equality is checked in
    float64; the coordinate change can introduce floating-point roundoff.
    """
    mean, scale = old_initial["mean"].double(), old_initial["scale"].double()
    _finite(mean, "fold_mean")
    _finite(scale, "fold_scale")
    if mean.ndim != 1 or scale.shape != mean.shape or not bool((scale > 0).all()):
        raise ValueError("invalid_old_fold_scaler")
    for name, value in (("mean", mean), ("scale", scale)):
        torch.testing.assert_close(value, old_initial["model_state"][name].double(), rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Probe(mean, scale, "linear")
    with torch.no_grad():
        weight, bias = source_state["linear.weight"].double(), source_state["linear.bias"].double()
        model.network.weight.copy_(weight * scale)
        model.network.bias.copy_(bias + weight @ mean)
        expected = _source_logits(features, source_state).softmax(-1)
        actual = model(features.double()).softmax(-1)
        _finite(actual, "initial_probabilities")
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        error = float((actual - expected).abs().max())
    return model, error


def _validate_split(features, labels, split, old_initial):
    budget, train, validation = (list(map(int, split[k])) for k in
                                 ("budget_ids", "train_ids", "validation_ids"))
    if any(len(v) != len(set(v)) for v in (budget, train, validation)):
        raise ValueError("duplicate_fold_ids")
    if set(train) & set(validation) or set(train) | set(validation) != set(budget):
        raise ValueError("invalid_fold_partition")
    if not train or not validation or not set(budget).issubset(labels):
        raise ValueError("missing_A_training_or_validation_labels")
    ytrain = torch.tensor([int(labels[i]) for i in train], dtype=torch.long)
    yval = torch.tensor([int(labels[i]) for i in validation], dtype=torch.long)
    if set(ytrain.tolist()) != {0, 1} or set(yval.tolist()) != {0, 1}:
        raise ValueError("unestimable_missing_class_in_train_or_inner_validation")
    xtrain, xval = features[train], features[validation]
    _finite(xtrain, "training_features")
    _finite(xval, "validation_features")
    # Validation only: the actual model uses the archived scaler, unchanged.
    torch.testing.assert_close(old_initial["mean"].double(), xtrain.mean(0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(old_initial["scale"].double(), xtrain.std(0, unbiased=False).clamp_min(1e-8),
                               rtol=1e-12, atol=1e-12)
    return xtrain, xval, ytrain, yval


def _predict(model, features, evaluation_ids):
    model.eval()
    result = {}
    with torch.no_grad():
        for name, ids in evaluation_ids.items():
            ids = list(map(int, ids))
            logits = model(features[ids])
            _finite(logits, "evaluation_logits")
            result[name] = dict(ids=ids, probabilities=logits.softmax(-1).clone(),
                                log_probabilities=logits.log_softmax(-1).clone())
    return result


def fit_warm(directory, features, labels, split, source_state, old_initial,
             model_info, plan, evaluation_ids, source_probabilities=None):
    """One fresh Adam fit. Small budgets are supported for synthetic tests only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    started = time.monotonic()
    model, optimizer, final_training_state = None, None, None
    training, validation = [], []
    metadata = dict(model_info, **split, head="linear", feature="H", condition="true",
                    initialization_kind="source_affine", initialization_seed=0,
                    status="failed", updates_completed=0, selected_step=None, checkpoints={})

    def checkpoint(step):
        return dict(step=step, model_state=copy.deepcopy(model.state_dict()),
                    optimizer_state=copy.deepcopy(optimizer.state_dict()),
                    mean=model.mean.clone(), scale=model.scale.clone(), head="linear",
                    initialization_kind="source_affine", initialization_seed=0,
                    training_history=list(training), validation_history=list(validation))

    try:
        spec = plan["warm"]
        updates, cadence = int(spec["updates"]), int(spec["validation_cadence"])
        if not 0 < updates <= 800 or cadence != 20 or updates % cadence:
            raise ValueError("invalid_fixed_warm_budget")
        if spec["weight_decay"] != .001 or model_info["lr"] not in (.001, .0001):
            raise ValueError("undeclared_warm_optimizer")
        _limits(started, plan)
        features = features.double()
        xtrain, xval, ytrain, yval = _validate_split(features, labels, split, old_initial)
        model, error = source_initialized_probe(features, source_state, old_initial)
        metadata["source_probability_max_abs_error"] = error
        if source_probabilities is not None:
            with torch.no_grad():
                actual = model(features).softmax(-1)
                expected = source_probabilities.double()
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                metadata["archived_source_probability_max_abs_error"] = float((actual - expected).abs().max())
        metadata.update(train_class_counts=torch.bincount(ytrain, minlength=2).tolist(),
                        validation_class_counts=torch.bincount(yval, minlength=2).tolist(),
                        weight_decay=.001, updates_requested=updates,
                        validation_cadence=cadence, optimizer="Adam", batch="full")
        optimizer = torch.optim.Adam(model.parameters(), lr=model_info["lr"], weight_decay=.001)
        best_ce, selected_step = float("inf"), None
        for step in range(updates + 1):
            _limits(started, plan)
            if step:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xtrain), ytrain)
                _finite(loss, "training_loss")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        _finite(parameter.grad, "gradient")
                optimizer.step()
                metadata["updates_completed"] = step
            model.eval()
            with torch.no_grad():
                train_ce = float(F.cross_entropy(model(xtrain), ytrain))
                if not math.isfinite(train_ce):
                    raise ValueError("nonfinite_post_update_training_loss")
                training.append(dict(step=step, train_ce=train_ce))
                if step % cadence == 0:
                    val_ce = float(F.cross_entropy(model(xval), yval))
                    if not math.isfinite(val_ce):
                        raise ValueError("nonfinite_validation_loss")
                    validation.append(dict(step=step, validation_ce=val_ce))
                    path = f"checkpoints/step{step:04d}.pt"
                    save_tensor(directory / path, checkpoint(step))
                    metadata["checkpoints"][str(step)] = path
                    if val_ce < best_ce:  # Includes step zero and chooses the earliest exact tie.
                        best_ce, selected_step = val_ce, step
        metadata.update(selected_step=selected_step, selected_validation_ce=best_ce)
        metadata["checkpoints"]["selected"] = metadata["checkpoints"][str(selected_step)]
        final_training_state = copy.deepcopy(model.state_dict())
        endpoints = ["0", "selected"] + [str(i) for i in spec["supplementary_endpoints"]]
        predictions = {}
        for endpoint in dict.fromkeys(endpoints):
            if endpoint not in metadata["checkpoints"]:
                raise ValueError("supplementary_endpoint_has_no_checkpoint")
            state = _load(directory / metadata["checkpoints"][endpoint])
            model.load_state_dict(state["model_state"])
            predictions[endpoint] = _predict(model, features, evaluation_ids)
            _limits(started, plan)
        save_tensor(directory / "probabilities.pt", predictions)
        metadata["status"] = "success"
    except Exception as exc:
        metadata.update(failure=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if model is not None and optimizer is not None:
            if final_training_state is not None:
                model.load_state_dict(final_training_state)
            save_tensor(directory / "partial.pt", checkpoint(metadata["updates_completed"]))
    finally:
        metadata.update(elapsed_seconds=time.monotonic() - started, peak_rss_bytes=_rss())
        write_json(directory / "history.json", dict(training=training, validation=validation))
        write_json(directory / "metadata.json", metadata)
    return metadata


def rescore_random(directory, old_directory, features, labels, split, model_info,
                   evaluation_ids):
    """Select old random initialization or old selected state using only A29 CE.

    There is no optimizer or training in this function. Both original and
    corrected baseline predictions remain available in the new run directory.
    """
    directory, old_directory = Path(directory), Path(old_directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    metadata = dict(model_info, **split, status="failed", checkpoints={},
                    updates_completed=0, new_training=False, selection_includes_step0=True)
    try:
        old_meta = json.loads((old_directory / "metadata.json").read_text())
        if old_meta["status"] != "success":
            raise ValueError("old_random_probe_failed")
        for key in ("budget_ids", "train_ids", "validation_ids"):
            if old_meta[key] != split[key]:
                raise ValueError("old_random_fold_mismatch:" + key)
        initial = _load(old_directory / "initial.pt")
        selected_path = old_directory / old_meta["checkpoints"]["selected"]
        selected = _load(selected_path)
        features = features.double()
        _, xval, _, yval = _validate_split(features, labels, split, initial)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = Probe(initial["mean"], initial["scale"], "linear")
        ce, predictions = {}, {}
        for endpoint, state in (("0", initial), ("original_selected", selected)):
            model.load_state_dict(state["model_state"])
            with torch.no_grad():
                value = float(F.cross_entropy(model(xval), yval))
            if not math.isfinite(value):
                raise ValueError("nonfinite_old_random_validation_loss")
            ce[endpoint] = value
        archived_ce = old_meta["selected_validation_ce"]
        if not math.isclose(ce["original_selected"], archived_ce, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("old_random_selected_validation_ce_mismatch")
        chosen = "0" if ce["0"] <= archived_ce else "original_selected"
        # Freeze the A29 decision before even evaluating B inputs.
        for endpoint, state in (("0", initial), ("original_selected", selected)):
            model.load_state_dict(state["model_state"])
            predictions[endpoint] = _predict(model, features, evaluation_ids)
        predictions["selected"] = predictions[chosen]
        save_tensor(directory / "probabilities.pt", predictions)
        for endpoint, state in (("0", initial), ("original_selected", selected)):
            path = f"checkpoints/{endpoint}.pt"
            save_tensor(directory / path, state)
            metadata["checkpoints"][endpoint] = path
        metadata["checkpoints"]["selected"] = metadata["checkpoints"][chosen]
        metadata.update(status="success", selected_endpoint=chosen,
                        selected_step=0 if chosen == "0" else old_meta["selected_step"],
                        original_selected_step=old_meta["selected_step"],
                        initial_validation_ce=ce["0"], original_selected_validation_ce=archived_ce,
                        rescored_original_selected_validation_ce=ce["original_selected"],
                        selected_validation_ce=ce[chosen],
                        old_initial_sha256=digest(old_directory / "initial.pt"),
                        old_selected_sha256=digest(selected_path))
    except Exception as exc:
        metadata.update(failure=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    write_json(directory / "metadata.json", metadata)
    return metadata


def _failed_dependency(directory, info, error):
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(info, status="failed", updates_completed=0, checkpoints={},
                  failure=f"dependency_failure:{type(error).__name__}: {error}")
    write_json(directory / "metadata.json", record)
    write_json(directory / "history.json", dict(training=[], validation=[]))
    return record


def run_warm(root, plan, labels, log=None):
    """Execute the declared warm grid and old-random step-zero correction.

    ``labels`` maps integer target domain to {integer A graph ID: class}; extra
    IDs are refused before any fitting. No source or held-out labels are read.
    """
    root = Path(root)
    if (root / "seals/OBS.json").exists():
        raise PermissionError("warm_fitting_closed_after_OBS")
    deps = root / "deps"
    roles = json.loads((deps / "roles.json").read_text())
    spec = plan["warm"]
    domains = [int(d) for d in plan.get("domains", (1, 3))]
    if len(plan["seeds"]) * len(domains) * spec["outer_folds"] * len(spec["learning_rates"]) > 80:
        raise ValueError("warm_fit_budget_exceeds_80")
    if len(set(spec["learning_rates"])) != len(spec["learning_rates"]):
        raise ValueError("duplicate_warm_learning_rates")
    for domain, values in labels.items():
        if int(domain) not in domains or not set(values).issubset(roles["targets"][str(domain)]["A"]):
            raise PermissionError("warm_accepts_only_declared_target_A_labels")
    for domain in domains:
        if domain not in labels:
            raise ValueError("missing_target_A_labels")
    summaries, corrections = [], []
    for seed in plan["seeds"]:
        seed_roles = json.loads((deps / "roles" / f"seed{seed}.json").read_text())
        source_path = deps / "source" / f"seed{seed}"
        source_error = None
        try:
            frozen = _load(source_path / "frozen.pt")
            features = frozen["x"].double()
            source_state = frozen["initial_head"]
            source_predictions = _load(source_path / "predictions.pt")
            if list(map(int, source_predictions["ids"])) != list(range(len(features))):
                raise ValueError("source_prediction_id_order_mismatch")
        except Exception as exc:
            source_error = exc
        source_ids = {f"source_{name}": seed_roles[f"{name}_ids"]
                      for name in ("fit", "validation", "diagnostic")}
        source_ids["source_all"] = sorted(set().union(*(set(ids) for ids in source_ids.values())))
        for domain in domains:
            for fold in range(spec["outer_folds"]):
                split = seed_roles["folds"]["targets"][str(domain)][str(fold)]
                evaluations = {f"B{domain}": roles["targets"][str(domain)]["B"], **source_ids}
                old_dir = model_directory(deps, "target", seed, domain, fold, "H", "linear", "true")
                info = dict(seed=seed, domain=domain, fold=fold, origin="target",
                            old_random_directory=str(old_dir.relative_to(root)))
                dependency_error = source_error
                if dependency_error is None:
                    try:
                        old_meta = json.loads((old_dir / "metadata.json").read_text())
                        if old_meta["status"] != "success":
                            raise ValueError("old_random_probe_failed")
                        for key in ("budget_ids", "train_ids", "validation_ids"):
                            if old_meta[key] != split[key]:
                                raise ValueError("old_fold_mismatch:" + key)
                        initial = _load(old_dir / "initial.pt")
                        info["old_initial_sha256"] = digest(old_dir / "initial.pt")
                    except Exception as exc:
                        dependency_error = exc
                corrected_dir = root / "warm/random_step0_corrected" / f"seed{seed}" / f"domain{domain}" / f"fold{fold}"
                if dependency_error is None:
                    corrected = rescore_random(corrected_dir, old_dir, features, labels[domain], split, info, evaluations)
                else:
                    corrected = _failed_dependency(corrected_dir, info, dependency_error)
                corrections.append(dict(path=str(corrected_dir.relative_to(root)), **corrected))
                for lr in spec["learning_rates"]:
                    directory = root / "warm" / f"seed{seed}" / f"domain{domain}" / f"fold{fold}" / f"lr{lr:.0e}"
                    arm = dict(info, lr=lr)
                    if dependency_error is None:
                        record = fit_warm(directory, features, labels[domain], split, source_state,
                                          initial, arm, plan, evaluations, source_predictions["probabilities"])
                    else:
                        record = _failed_dependency(directory, arm, dependency_error)
                    summaries.append(dict(path=str(directory.relative_to(root)), **record))
                    if log is not None:
                        log(dict(stage="warm_fit", seed=seed, domain=domain, fold=fold, lr=lr,
                                 status=record["status"], fits=len(summaries),
                                 updates_completed=record["updates_completed"]))
    (root / "warm/random_step0_corrected").mkdir(parents=True, exist_ok=True)
    write_json(root / "warm/index.json", summaries)
    write_json(root / "warm/random_step0_corrected/index.json", corrections)
    return summaries
