"""Fixed-budget R1 diagnostic probes; no B labels are accepted here."""
import copy
import json
from pathlib import Path
import resource
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F

from ..archive import save_tensor, write_json
from .data import LabelBroker, rng_seed


class Probe(nn.Module):
    """The input and derivatives remain in the common TSBA coordinate system."""
    def __init__(self, mean, scale, head="linear"):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone().double())
        self.register_buffer("scale", scale.detach().clone().double())
        self.head_kind = head
        width = mean.numel()
        if head == "linear":
            self.network = nn.Linear(width, 2).double()
        elif head == "mlp":
            self.network = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 2)).double()
        else:
            raise ValueError(f"Unknown probe head {head}")

    def forward(self, x):
        return self.network((x - self.mean) / self.scale)

    @property
    def linear_head(self):
        return self.network if self.head_kind == "linear" else None

    def affine(self):
        if self.head_kind != "linear":
            raise ValueError("MLP is not globally affine")
        weight = self.network.weight / self.scale
        bias = self.network.bias - weight @ self.mean
        return weight, bias


def _new_probe(mean, scale, head, seed):
    # Do not let model creation perturb any other experiment RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Probe(mean, scale, head)


def load_probe(path, endpoint="selected"):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["status"] != "success":
        return None
    state = torch.load(path / metadata["checkpoints"][str(endpoint)], map_location="cpu", weights_only=False)
    model = _new_probe(state["mean"], state["scale"], metadata["head"], metadata["initialization_seed"])
    model.load_state_dict(state["model_state"])
    model.eval()
    model.requires_grad_(False)
    return model


def model_directory(root, origin, seed, domain, fold, feature, head, condition):
    family = "source" if origin == "source" else f"target{domain}"
    return Path(root) / "probes" / family / f"seed{seed}" / f"fold{fold}" / feature / head / condition


def load_reference(root, seed, domain, fold, head):
    head = {"hidden32_silu_mlp": "mlp", "LR": "linear", "MLP": "mlp"}.get(head, head)
    path = model_directory(root, "target", seed, domain, fold, "H", head, "true")
    if not (path / "metadata.json").exists():
        return None
    return load_probe(path)


def _rss():
    # Linux ru_maxrss is KiB; the execution protocol targets this Linux host.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def fit_probe(directory, features, labels_by_id, split, model_info, plan, evaluation_ids):
    """Fit one predeclared arm. evaluation_ids contain inputs only, never labels.

    Histories use CE of each state after its numbered optimizer update. All
    every-20 snapshots include Adam so predictions and the selected state can be
    reproduced without another fit. A failure retains the actual partial state.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    started = time.monotonic()
    history, validation, model, optimizer = [], [], None, None
    metadata = dict(model_info, **split, status="failed", checkpoints={}, prior_probabilities=None,
                    budget_sensitive=None, selected_step=None, updates_completed=0)
    mean, scale = None, None
    step = 0
    initialization_seed = None
    protocol = plan["protocol_id"]
    budget = list(split["budget_ids"])
    # Actual names from the frozen machine protocol are the RNG coordinates.
    feature_key = "H16" if model_info["feature"] == "H" else "log1p_nodes_log1p_undirected_edges"
    head_key = "linear" if model_info["head"] == "linear" else "hidden32_silu_mlp"
    initialization_seed = rng_seed(protocol, "probe_init", model_info["seed"], 0,
                                   model_info["fold"], feature_key, head_key)
    permutation_seed = rng_seed(protocol, "label_permutation", 0, model_info["domain"],
                                model_info["fold"], "*", "*")
    metadata.update(initialization_seed=initialization_seed, permutation_seed=permutation_seed,
                    rng_feature_key=feature_key, rng_head_key=head_key)
    def checkpoint(update):
        return dict(step=update, model_state=copy.deepcopy(model.state_dict()),
                    optimizer_state=copy.deepcopy(optimizer.state_dict()), mean=mean, scale=scale,
                    head=model_info["head"], initialization_seed=initialization_seed,
                    torch_rng_state=torch.random.get_rng_state(), training_history=list(history),
                    validation_history=list(validation))
    try:
        features = features.double()
        true_labels = torch.tensor([int(labels_by_id[i]) for i in budget], dtype=torch.long)
        permutation = torch.randperm(len(budget), generator=torch.Generator().manual_seed(permutation_seed))
        used_labels = true_labels[permutation] if model_info["condition"] == "permuted" else true_labels.clone()
        lookup = {i: p for p, i in enumerate(budget)}
        train_positions = torch.tensor([lookup[i] for i in split["train_ids"]])
        val_positions = torch.tensor([lookup[i] for i in split["validation_ids"]])
        ytrain, yval = used_labels[train_positions], used_labels[val_positions]
        metadata["train_class_counts"] = torch.bincount(ytrain, minlength=2).tolist()
        metadata["validation_class_counts"] = torch.bincount(yval, minlength=2).tolist()
        save_tensor(directory / "restricted-labels.pt", dict(budget_ids=budget, true_labels=true_labels,
                    used_labels=used_labels, permutation=permutation, train_positions=train_positions,
                    validation_positions=val_positions))
        if set(ytrain.tolist()) != {0,1} or set(yval.tolist()) != {0,1}:
            raise ValueError("unestimable_missing_class_in_train_or_inner_validation")
        xtrain, xval = features[split["train_ids"]], features[split["validation_ids"]]
        if not torch.isfinite(xtrain).all() or not torch.isfinite(xval).all():
            raise ValueError("nonfinite_features")
        mean = xtrain.mean(0)
        # Population standard deviation, fitted only to the optimization118.
        scale = xtrain.std(0, unbiased=False).clamp_min(1e-8)
        model = _new_probe(mean, scale, model_info["head"], initialization_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=plan["probes"]["lr"],
                                     weight_decay=plan["probes"]["weight_decay"])
        prior = torch.bincount(ytrain, minlength=2).double() / len(ytrain)
        metadata["prior_probabilities"] = prior.clamp(1e-6, 1-1e-6).tolist()
        save_tensor(directory / "initial.pt", checkpoint(0))
        best_ce, selected_step = float("inf"), None
        updates = plan["probes"]["updates"]
        cadence = plan["probes"]["validation_cadence"]
        for step in range(1, updates + 1):
            if time.monotonic() - started > plan["resources"]["max_probe_seconds"]:
                raise RuntimeError("probe_time_limit")
            if _rss() > plan["resources"]["max_rss_bytes"]:
                raise RuntimeError("probe_rss_limit")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xtrain), ytrain)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite_gradient")
            optimizer.step()
            metadata["updates_completed"] = step
            model.eval()
            with torch.no_grad():
                train_ce = float(F.cross_entropy(model(xtrain), ytrain))
                if not math_isfinite(train_ce):
                    raise ValueError("nonfinite_post_update_loss")
                history.append(dict(step=step, train_ce=train_ce))
                if step % cadence == 0:
                    val_ce = float(F.cross_entropy(model(xval), yval))
                    if not math_isfinite(val_ce):
                        raise ValueError("nonfinite_validation_loss")
                    validation.append(dict(step=step, validation_ce=val_ce))
                    path = f"checkpoints/step{step:04d}.pt"
                    save_tensor(directory / path, checkpoint(step))
                    if val_ce < best_ce:  # strict comparison implements earliest tie
                        best_ce, selected_step = val_ce, step
        if selected_step is None:
            raise ValueError("No scheduled validation checkpoint")
        metadata["selected_step"] = selected_step
        metadata["selected_validation_ce"] = best_ce
        metadata["checkpoints"]["selected"] = f"checkpoints/step{selected_step:04d}.pt"
        for endpoint in plan["probes"]["supplementary_endpoints"]:
            if endpoint > updates or endpoint % cadence:
                raise ValueError("Supplementary endpoint does not have a saved checkpoint")
            metadata["checkpoints"][str(endpoint)] = f"checkpoints/step{endpoint:04d}.pt"
        windows = plan["probes"].get("budget_sensitive_windows")
        if windows and len(history) >= windows["last_inclusive"][1]:
            a,b = windows["previous_inclusive"]
            c,d = windows["last_inclusive"]
            previous = sum(v["train_ce"] for v in history[a-1:b]) / (b-a+1)
            last = sum(v["train_ce"] for v in history[c-1:d]) / (d-c+1)
            relative = (previous-last) / max(abs(previous), 1e-12)
            metadata.update(budget_sensitive=relative > plan["probes"]["budget_sensitive_relative_ce_drop"],
                            previous_window_ce=previous, last_window_ce=last, relative_ce_drop=relative)
        predictions = {}
        for endpoint, checkpoint_path in metadata["checkpoints"].items():
            state = torch.load(directory / checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state"])
            model.eval()
            predictions[endpoint] = {}
            with torch.no_grad():
                for name, ids in evaluation_ids.items():
                    logits = model(features[ids])
                    probabilities = logits.softmax(-1)
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("nonfinite_evaluation_probabilities")
                    predictions[endpoint][name] = dict(ids=list(ids), probabilities=probabilities.clone(),
                                                       log_probabilities=logits.log_softmax(-1).clone())
        save_tensor(directory / "probabilities.pt", predictions)
        metadata["status"] = "success"
    except Exception as exc:
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
        if model is not None and optimizer is not None:
            save_tensor(directory / "partial.pt", checkpoint(metadata["updates_completed"]))
    finally:
        metadata.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=_rss())
        write_json(directory / "history.json", dict(training=history, validation=validation))
        write_json(directory / "metadata.json", metadata)
    return metadata


def math_isfinite(value):
    # Kept scalar to avoid introducing a float32 conversion in a float64 fit.
    return value != float("inf") and value != -float("inf") and value == value


def run_probes(root, plan, seal=None, log=None):
    root = Path(root)
    if (root / "seals/S2.json").exists():
        raise PermissionError("Probe fitting is closed after S2")
    broker = LabelBroker(root, plan)
    roles = broker.roles
    if seal is None:
        seal = json.loads((root / "seals/S1.json").read_text())["digest"]
    # One broker request per role, never labels of B or final.
    target_labels = {}
    for domain, target in roles["targets"].items():
        values = broker.read(target["A"], "target_A", "R1 diagnostic probes", seal=seal, domain=int(domain))
        target_labels[int(domain)] = dict(zip(target["A"], values.tolist()))
    size = torch.load(root / "inputs/size.pt", map_location="cpu", weights_only=False)
    summaries = []
    for seed in plan["seeds"]:
        seed_roles = json.loads((root / "roles" / f"seed{seed}.json").read_text())
        diagnostic = seed_roles["diagnostic_ids"]
        values = broker.read(diagnostic, "source_diagnostic", "R1 diagnostic probes", seal=seal, seed=seed)
        source_labels = dict(zip(diagnostic, values.tolist()))
        frozen_path = root / "source" / f"seed{seed}" / "frozen.pt"
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False) if frozen_path.exists() else None
        for domain in [0] + [int(d) for d in roles["targets"]]:
            origin = "source" if domain == 0 else "target"
            folds = seed_roles["folds"]["source"] if domain == 0 else seed_roles["folds"]["targets"][str(domain)]
            labels = source_labels if domain == 0 else target_labels[domain]
            for fold in range(plan["probes"]["outer_folds"]):
                split = folds[str(fold)]
                evaluation_ids = {}
                if domain == 0:
                    evaluation_ids["audit"] = split["audit_ids"]
                    for d, target in roles["targets"].items():
                        evaluation_ids[f"B{d}"] = target["B"]
                else:
                    evaluation_ids[f"B{domain}"] = roles["targets"][str(domain)]["B"]
                for feature in ("H", "SIZE"):
                    for head in ("linear", "mlp"):
                        for condition in ("true", "permuted"):
                            info = dict(seed=seed, domain=domain, origin=origin, fold=fold,
                                        feature=feature, head=head, condition=condition)
                            directory = model_directory(root, origin, seed, domain, fold, feature, head, condition)
                            if frozen is None:
                                directory.mkdir(parents=True, exist_ok=False)
                                record = dict(info, **split, status="failed", failure="dependency_missing:source_model",
                                              checkpoints={}, selected_step=None, prior_probabilities=None,
                                              budget_sensitive=None, updates_completed=0)
                                write_json(directory / "metadata.json", record)
                            else:
                                features = frozen["x"] if feature == "H" else size
                                record = fit_probe(directory, features, labels, split, info, plan, evaluation_ids)
                            summaries.append(dict(path=str(directory.relative_to(root)), **record))
                progress = dict(stage="probe_fold", seed=seed, domain=domain, fold=fold,
                                successes=sum(s["status"] == "success" for s in summaries), fits=len(summaries))
                print(json.dumps(progress), flush=True)
                if log is not None:
                    log(progress)
    write_json(root / "probes/index.json", summaries)
    return summaries
