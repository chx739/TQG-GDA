"""Resumable source and DSM fits under the fully specified R4 plan."""
from dataclasses import asdict

import torch
from torch.nn.functional import cross_entropy
from torch_geometric.data import Batch, Data

from tsba.boundary import build_boundaries
from tsba.classifier import Coordinates, ResidualClassifier
from tsba.data import GraphPool, TaskData
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel, encode, graph_batches, source_metrics
from tsba.score import ScoreNetwork, bandwidth
from tsba.utils import finite, set_seed
from experiments.common.real_data.next_round.steps import EmpiricalScore
from experiments.common.real_data.r2.adapt import ZeroScore
from experiments.common.real_data.r2.common import atomic_tensor, load, write_json
from experiments.common.real_data.r2.score import comparison, epoch_update, validation
from experiments.mutagenicity.r4_20261112.protocol import batch_path, config, domain, labels_path, role, score_dir, source_dir, tasks


def source_model(cfg, dim, edge_dim, classes):
    return SourceModel(GraphEncoder(dim, edge_dim, cfg.hidden, cfg.num_layers, cfg.emb_dim,
                                    cfg.dropout, cfg.encoder), classes).to(cfg.device)


def adam(model, lr, plan):
    options = dict(plan["adam"])
    options["betas"] = tuple(options["betas"])
    return torch.optim.Adam(model.parameters(), lr=lr, **options)


def consecutive_files(directory, count):
    files = sorted(directory.glob("epoch*.pt"))
    expected = [f"epoch{i:03d}.pt" for i in range(len(files))]
    if len(files) > count or [p.name for p in files] != expected:
        raise ValueError(f"Incomplete or excessive source checkpoint sequence in {directory}")
    return files


def put_tensor(path, record):
    # A downstream completion marker is published only after all stage files.
    if not path.exists():
        atomic_tensor(path, record)


def train_source(root, plan, item):
    source, seed = item
    out = source_dir(root, plan, source, seed)
    if (out / "complete.json").is_file():
        return
    cfg = config(plan, seed)
    selected = role(root, source, seed)
    graph_data = load(root / "inputs/graphs.pt")
    graphs = [Data(**record) for record in graph_data]

    def pool(split):
        ids = selected[split]
        return GraphPool([graphs[i] for i in ids], [str(i) for i in ids],
                         load(labels_path(root, source, seed, split)))

    fit, val = pool("fit"), pool("validation")
    first_target = next(d for d in plan["domains"] if d != source)
    target_ids = domain(root, first_target)
    task_data = TaskData(fit, val, GraphPool([graphs[i] for i in target_ids],
                                           [str(i) for i in target_ids]), "transductive")
    dim, edge_dim, classes = task_data.validate()
    set_seed(seed)
    model = source_model(cfg, dim, edge_dim, classes)
    optimizer = adam(model, cfg.source_lr, plan)
    rng = torch.Generator().manual_seed(seed + 1)
    files = consecutive_files(out / "epochs", cfg.source_epochs)
    history = [load(path)["metrics"] for path in files]
    if files:
        state = load(files[-1])
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["permutation_rng"])
        torch.random.set_rng_state(state["torch_rng"])

    for epoch in range(len(files), cfg.source_epochs):
        model.train()
        order = torch.randperm(len(fit.graphs), generator=rng).tolist()
        total = 0.
        for ids in graph_batches(fit.graphs, cfg.source_batch, order):
            batch = Batch.from_data_list([fit.graphs[i] for i in ids]).to(cfg.device)
            loss = cross_entropy(model(batch), fit.labels[ids].to(cfg.device))
            finite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters() if p.grad is not None))
            optimizer.step()
            finite(*(p for p in model.parameters()))
            total += float(loss.detach()) * len(ids)
        metrics = dict(epoch=epoch, train_ce=total / len(order),
                       source_validation=source_metrics(model, val, cfg.source_batch, cfg.device))
        history.append(metrics)
        atomic_tensor(out / "epochs" / f"epoch{epoch:03d}.pt", dict(
            epoch=epoch, state=model.state_dict(), optimizer=optimizer.state_dict(),
            order=[fit.ids[i] for i in order], metrics=metrics,
            torch_rng=torch.random.get_rng_state(), permutation_rng=rng.get_state()))
        if epoch % 25 == 0 or epoch == cfg.source_epochs - 1:
            print(f"M{source} seed={seed} epoch={epoch} val_ce={metrics['source_validation']['ce']:.6g}", flush=True)

    best = min(history, key=lambda row: row["source_validation"]["ce"])["epoch"]
    model.load_state_dict(load(out / "epochs" / f"epoch{best:03d}.pt")["state"])
    model.encoder.freeze()
    model.eval()
    h = encode(model.encoder, graphs, cfg.source_batch, cfg.device).double()
    coords = Coordinates.fit(h[selected["fit"]], cfg.coordinate_eps)
    x = coords(h)
    head = ResidualClassifier.from_source(model.head, coords, cfg.adapter_width, seed).eval()
    with torch.no_grad():
        probability = head(x).softmax(-1)
        original = (h @ model.head.weight.double().T + model.head.bias.double()).softmax(-1)
        float32_probability = model.head(h.float()).softmax(-1).double()
    torch.testing.assert_close(probability, original, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(probability, float32_probability, rtol=1e-5, atol=1e-6)
    if not torch.equal(probability.argmax(1), float32_probability.argmax(1)):
        raise ValueError("R4 residual initialization changed source predictions")
    put_tensor(out / "source.pt", dict(encoder_spec=model.encoder.spec, state=model.state_dict(),
                                      selected_epoch=best, history=history))
    put_tensor(out / "frozen.pt", dict(x=x, h=h, mean=coords.mean, scale=coords.scale,
                                      initial_head=head.state_dict()))
    for task in tasks(plan):
        if (task.source, task.seed) != (source, seed):
            continue
        ids = domain(root, task.target)
        sigma = bandwidth(x[ids], cfg.kappa_sigma)
        if sigma is None:
            raise ValueError(f"Degenerate bandwidth for {task.key}")
        first = load(batch_path(root, task))[0]
        bounds = build_boundaries(head, x[first["target_ids"]], ZeroScore(), sigma, cfg)
        put_tensor(out / f"geometry_M{task.target}.pt",
                   dict(sigma=sigma, queries=first["target_ids"], boundaries=asdict(bounds)))
    write_json(out / "complete.json", dict(source=source, seed=seed, epochs=len(history),
               selected_epoch=best, validation_ce=history[best]["source_validation"]["ce"],
               initial_probability_max_error=float((probability - float32_probability).abs().max())))


def geometry(root, plan, task):
    return load(source_dir(root, plan, task.source, task.seed) / f"geometry_M{task.target}.pt")


def train_score(root, plan, task):
    out = score_dir(root, plan, task)
    if (out / "complete.json").is_file():
        return
    cfg = config(plan, task.seed)
    data = load(source_dir(root, plan, task.source, task.seed) / "frozen.pt")
    x = data["x"][domain(root, task.target)]
    spec = geometry(root, plan, task)
    h = spec["sigma"]
    steps_per_epoch = (len(x) + cfg.score_batch - 1) // cfg.score_batch
    endpoints = plan["score"]["endpoints"]
    if any(step % steps_per_epoch for step in endpoints) or cfg.score_epochs * steps_per_epoch != endpoints[-1]:
        raise ValueError("The registered score budget must equal whole epochs for this domain")
    noise_rng = torch.Generator().manual_seed(task.seed + 103)
    noise = torch.randn(x.shape, generator=noise_rng, dtype=x.dtype)
    rng = torch.Generator().manual_seed(task.seed + 102)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(task.seed + 101)
        model = ScoreNetwork(x.shape[1], cfg.score_hidden).double()
    optimizer = adam(model, cfg.score_lr, plan)
    checkpoints = sorted(out.glob("epoch*.pt"))
    history = []
    if checkpoints:
        state = load(checkpoints[-1])
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        rng.set_state(state["training_rng_state"])
        torch.random.set_rng_state(state["torch_rng_state"])
        torch.testing.assert_close(noise, state["validation_noise"], rtol=0, atol=0)
        torch.testing.assert_close(noise_rng.get_state(), state["validation_rng_state"], rtol=0, atol=0)
        history = state["history"]
        if state["updates"] != len(history) * steps_per_epoch or state["bandwidth"] != h:
            raise ValueError("Score checkpoint does not match this task")

    for epoch in range(len(history) + 1, cfg.score_epochs + 1):
        loss = epoch_update(model, optimizer, rng, x, h, batch=cfg.score_batch)
        val = validation(model, x, h, noise)
        updates = epoch * steps_per_epoch
        history.append(dict(epoch=epoch, updates=updates, train=loss, validation=val))
        if updates in endpoints or epoch % plan["score"]["recovery_epochs"] == 0:
            atomic_tensor(out / f"epoch{epoch:04d}.pt", dict(
                model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                training_rng_state=rng.get_state(), torch_rng_state=torch.random.get_rng_state(),
                validation_noise=noise, validation_rng_state=noise_rng.get_state(),
                history=history, updates=updates, bandwidth=h))

    b = spec["boundaries"]
    kde = EmpiricalScore(x, h)(b["points"])
    diagnostics = []
    for step in endpoints:
        state = load(out / f"epoch{step // steps_per_epoch:04d}.pt")
        if state["updates"] != step:
            raise ValueError("Unexpected score optimizer update count")
        model.load_state_dict(state["model_state"])
        with torch.no_grad():
            learned = model(b["points"])
        diagnostics.append(dict(step=step, validation=state["history"][-1]["validation"],
                                **comparison(kde, learned, b["normals"], h)))
    put_tensor(out / "diagnostics.pt", diagnostics)
    write_json(out / "complete.json", dict(asdict(task), bandwidth=h, endpoints=endpoints,
                                          epochs=len(history), updates=history[-1]["updates"]))


def learned_score(root, plan, task):
    cfg = config(plan, task.seed)
    checkpoint = load(score_dir(root, plan, task) / f"epoch{cfg.score_epochs:04d}.pt")
    if checkpoint["updates"] != plan["score"]["endpoints"][-1]:
        raise ValueError("Adaptation requires the exact registered score checkpoint")
    with torch.random.fork_rng(devices=[]):
        model = ScoreNetwork(cfg.emb_dim, cfg.score_hidden).double()
    model.load_state_dict(checkpoint["model_state"])
    return model.eval().requires_grad_(False)
