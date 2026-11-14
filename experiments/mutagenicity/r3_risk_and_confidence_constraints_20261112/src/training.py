"""Five fresh source encoders and twenty exact-budget score fits."""
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch
from torch_geometric.data import Data

from tsba.boundary import build_boundaries
from tsba.classifier import Coordinates, ResidualClassifier
from tsba.config import Config
from tsba.data import GraphPool, TaskData
from tsba.pretrain import encode
from tsba.score import ScoreNetwork, bandwidth
from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.steps import EmpiricalScore
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.adapt import ZeroScore
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.common import atomic_tensor, load, write_json
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.score import comparison, epoch_update, restore, validation
from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.source import train_source
from experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.protocol import domain_ids, roles_for


def source_dir(root, seed):
    return root / "source" / f"seed{seed}"


def score_dir(root, seed, domain, multiplier):
    return root / "scores" / f"seed{seed}_d{domain}_h{multiplier}"


def source_fit(root, seed, plan):
    out = source_dir(root, seed)
    if out.exists():
        return
    root = Path(root)
    roles = roles_for(root, seed)
    graphs = [Data(**r) for r in load(root / "inputs" / "graphs.pt")]
    fit, val = roles["fit"], roles["validation"]
    target = domain_ids(root, plan["domains"][0])

    def pool(ids, role=None):
        labels = load(root / "source_labels" / f"seed{seed}_{role}.pt") if role else None
        return GraphPool([graphs[i] for i in ids], [str(i) for i in ids], labels)

    data = TaskData(pool(fit, "fit"), pool(val, "validation"), pool(target), "transductive")
    cfg = Config(seed=seed, **plan["source"])
    staging = root / "interrupted_source" / f"seed{seed}-{uuid4().hex}"
    model, info = train_source(cfg, data, staging)
    h = encode(model.encoder, graphs, cfg.source_batch, cfg.device).double()
    coordinates = Coordinates.fit(h[fit], cfg.coordinate_eps)
    x = coordinates(h)
    head = ResidualClassifier.from_source(model.head, coordinates, 16, seed).eval()
    with torch.no_grad():
        prob = head(x).softmax(-1)
        original = (h @ model.head.weight.double().T + model.head.bias.double()).softmax(-1)
        original_float32 = model.head(h.float()).softmax(-1).double()
    torch.testing.assert_close(prob, original, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(prob, original_float32, rtol=1e-5, atol=1e-6)
    if not torch.equal(prob.argmax(-1), original_float32.argmax(-1)):
        raise ValueError("Residual initialization changed source predictions")
    atomic_tensor(staging / "frozen.pt", dict(x=x, h=h, mean=coordinates.mean,
                  scale=coordinates.scale, initial_head=head.state_dict()))
    for domain in plan["domains"]:
        ids = domain_ids(root, domain)
        sigma = bandwidth(x[ids], cfg.kappa_sigma)
        if sigma is None:
            raise ValueError("Degenerate target geometry")
        batch = load(root / "batches" / f"seed{seed}_d{domain}.pt")[0]
        bounds = build_boundaries(head, x[batch["target_ids"]], ZeroScore(), sigma, cfg)
        atomic_tensor(staging / f"geometry_d{domain}.pt", dict(
            sigma=sigma, queries=batch["target_ids"], boundaries=asdict(bounds)))
    write_json(staging / "checks.json", dict(
        epochs=len(info["history"]), selected=info["selected_epoch"], coordinate_fit_ids=fit,
        source_probability_max_error=float((prob - original_float32).abs().max())))
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out)


def score_fit(root, spec, plan):
    seed, domain, multiplier = spec
    out = score_dir(root, seed, domain, multiplier)
    if (out / "complete.json").exists():
        return
    settings = plan["score"]
    frozen = load(source_dir(root, seed) / "frozen.pt")
    x = frozen["x"][domain_ids(root, domain)]
    geometry = load(source_dir(root, seed) / f"geometry_d{domain}.pt")
    h = multiplier * geometry["sigma"]
    endpoints = settings["endpoints"]
    updates_per_epoch = (len(x) + settings["batch"] - 1) // settings["batch"]
    if any(step % updates_per_epoch for step in endpoints):
        raise ValueError("This registration requires score endpoints at epoch boundaries")
    noise_rng = torch.Generator().manual_seed(seed + 103)
    noise = torch.randn(x.shape, generator=noise_rng, dtype=x.dtype)
    checkpoints = sorted(out.glob("epoch*.pt"))
    if checkpoints:
        state = load(checkpoints[-1])
        model, optimizer, rng = restore(state)
        history = state["history"]
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            model = ScoreNetwork(x.shape[1], settings["hidden"]).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
        rng = torch.Generator().manual_seed(seed + 102)
        history = []
    for epoch in range(len(history) + 1, endpoints[-1] // updates_per_epoch + 1):
        train = epoch_update(model, optimizer, rng, x, h, batch=settings["batch"])
        val = validation(model, x, h, noise)
        updates = epoch * updates_per_epoch
        history.append(dict(epoch=epoch, updates=updates, train=train, validation=val))
        if updates in endpoints or epoch % settings["recovery_epochs"] == 0:
            atomic_tensor(out / f"epoch{epoch:04d}.pt", dict(
                model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                training_rng_state=rng.get_state(), torch_rng_state=torch.random.get_rng_state(),
                validation_noise=noise, validation_rng_state=noise_rng.get_state(),
                history=history, updates=updates, bandwidth=h))
    diagnostics = []
    b = geometry["boundaries"]
    kde = EmpiricalScore(x, h)(b["points"])
    for step in endpoints:
        state = load(out / f"epoch{step // updates_per_epoch:04d}.pt")
        if state["updates"] != step:
            raise ValueError("Unexpected score optimizer update count")
        model.load_state_dict(state["model_state"])
        with torch.no_grad():
            learned = model(b["points"])
        values = comparison(kde, learned, b["normals"], geometry["sigma"])
        diagnostics.append(dict(step=step, validation=state["history"][-1]["validation"], **values))
    if not (out / "diagnostics.pt").exists():
        atomic_tensor(out / "diagnostics.pt", diagnostics)
    write_json(out / "complete.json", dict(endpoints=endpoints, epochs=len(history), bandwidth=h))


def score_model(root, job, plan, dim):
    directory = score_dir(root, job.seed, job.domain, job.bandwidth)
    final_step = plan["score"]["endpoints"][-1]
    size = len(domain_ids(root, job.domain))
    batch = plan["score"]["batch"]
    state = load(directory / f"epoch{final_step // ((size + batch - 1) // batch):04d}.pt")
    if state["updates"] != final_step:
        raise ValueError("Invalid primary score checkpoint")
    with torch.random.fork_rng(devices=[]):
        model = ScoreNetwork(dim, plan["score"]["hidden"]).double()
    model.load_state_dict(state["model_state"])
    return model.eval().requires_grad_(False)
