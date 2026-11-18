"""Fresh, paired sum/mean source fits using authenticated R5 graph/role inputs.

``train_source`` prepares, resumes and completes one source artifact. Its key is
``setting/M{source}/seed{seed}/{pooling}``, independent of the target direction
and P/C phase. Only the current source's archived fit/validation labels are
opened. Target graphs are first forwarded after source checkpoint selection and
encoder freezing; no target density, bandwidth or label is used here.

``load_inputs`` has the R5 loader's result contract, including optional original
TSBA coordinates/head. Its manifest deliberately retains the R5 *data* format
for the existing gated label broker, while ``provenance.source_refit`` explicitly
identifies the new representation. The coordinator must seal the returned
manifest and this artifact's files in R7_INPUT, or authenticate their unchanged
digest against a prior sealed P archive before reusing them in C.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import io
import json
from pathlib import Path
import platform

import numpy as np
import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool

from tsba.classifier import Coordinates, ResidualClassifier
from tsba.config import Config
from tsba.data import GraphPool
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel, encode, graph_batches, source_metrics
from tsba.utils import file_hash, finite, set_seed, tensor_hash
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import atomic_tensor, write_json
from . import r5_inputs


_RECIPE = "r5_frozen_role_source_refit_v1"
_PREPARED = "r7-source-refit-prepared-v1"
_COMPLETE = "r7-source-refit-complete-v1"
_DEPENDENCIES = (
    "tsba/encoder.py", "tsba/pretrain.py", "tsba/classifier.py", "tsba/config.py", "tsba/data.py", "tsba/utils.py",
    "experiments/mutagenicity/r2_分数训练预算与多轮适配_20261111/src/common.py",
    "experiments/mutagenicity/r7_共同解释与共享更新验证_20261119/src/r5_inputs.py",
    "experiments/mutagenicity/r7_共同解释与共享更新验证_20261119/src/source_refit_inputs.py",
)


class PoolingGraphEncoder(GraphEncoder):
    """Change only each layer's graph pooling; parameter names/init stay exact.

    ``spec`` retains the stock constructor fields. Checkpoints must store
    ``pooling`` separately and pass it explicitly when reconstructing this class.
    Mean pooling is node averaging before concatenation and the readout MLP.
    """

    def __init__(self, *args, pooling="sum", **kwargs):
        if pooling not in ("sum", "mean"):
            raise ValueError("Pooling must be 'sum' or 'mean'")
        super().__init__(*args, **kwargs)
        self.pooling = pooling

    def forward(self, data):
        if self.pooling == "sum":
            return super().forward(data)
        dtype = next(self.parameters()).dtype
        x = data.x.to(dtype)
        ea = data.edge_attr.to(dtype) if data.edge_attr is not None else None
        counts = torch.bincount(data.batch).to(x).unsqueeze(1)
        if not len(counts) or bool((counts <= 0).any()):
            raise ValueError("Mean pooling requires nonempty graphs")
        pooled = []
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, ea) if self.spec["conv"] == "gine" else conv(x, data.edge_index)
            x = torch.relu(bn(x))
            x = F.dropout(x, self.spec["dropout"], self.training)
            pooled.append(global_add_pool(x, data.batch) / counts)
        return self.readout(torch.cat(pooled, dim=-1))


def _state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_identity(state):
    return r5_inputs._identity({name: tensor_hash(value) for name, value in state.items()}, compact=True)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _put_tensor(path, value):
    if path.exists():
        if not _same(_load(path), value):
            raise PermissionError(f"Existing source artifact differs: {path}")
    else:
        atomic_tensor(path, value)


def _signed(value):
    return {**value, "digest": r5_inputs._identity(value, compact=True)}


def _verify_signature(record):
    if record.get("digest") != r5_inputs._identity({k: v for k, v in record.items() if k != "digest"}, compact=True):
        raise PermissionError("Source artifact manifest identity differs")


def _assert_mutable(out):
    for ancestor in (out, *out.parents):
        if (ancestor / "seals/R7_INPUT.json").exists():
            raise PermissionError("A source artifact inside a sealed R7 input archive cannot be extended")


@contextmanager
def _locked(out):
    if not (out / ".build.lock").exists():
        _assert_mutable(out)
    out.mkdir(parents=True, exist_ok=True)
    # The empty lock remains a stable file for the enclosing stage's file seal.
    with (out / ".build.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _pooling(plan, pooling):
    registration = plan.get("source_refit")
    if (not isinstance(registration, dict) or set(registration) != {"pooling", "recipe"}
            or registration["recipe"] != _RECIPE or registration["pooling"] not in ("sum", "mean")):
        raise ValueError("Unrecognized registered source-refit recipe")
    if pooling is not None and pooling != registration["pooling"]:
        raise ValueError("Requested pooling differs from the registered plan")
    return registration["pooling"]


def _directory(output_root, setting, source, seed, pooling):
    root = Path(output_root).resolve()
    if not isinstance(setting, str) or not setting or Path(setting).name != setting or setting in (".", ".."):
        raise ValueError("Setting must be a single safe path component")
    result = root / setting / f"M{source}" / f"seed{seed}" / pooling
    if not result.resolve().is_relative_to(root):
        raise PermissionError("Source artifact path escapes output root")
    return result


def _declared_inputs(context, source, seed):
    """Authenticate the original INPUT seal; do not reread graphs at load time."""
    blob = r5_inputs._inside(context["root"], "seals/R5_INPUT.json").read_bytes()
    if r5_inputs._sha(blob) != context["provenance"]["r5_files"]["seals/R5_INPUT.json"]:
        raise PermissionError("R5 input seal changed during source preparation")
    files = json.loads(blob)["files"]
    names = ["inputs/graphs.pt", f"roles/M{source}/seed{seed}.json",
             f"source_labels/M{source}/seed{seed}_fit.pt",
             f"source_labels/M{source}/seed{seed}_validation.pt"]
    if any(name not in files for name in names):
        raise PermissionError("R5 INPUT lacks source-refit dependencies")
    return {name: files[name] for name in names}


def _registration(plan, context, source, seed, pooling, repository):
    old = context["old_plan"]
    cfg = Config(**{**old["model"], "seed": seed, "lambda_boundary": old["weight"]})
    fixed = dict(encoder="gin", hidden=64, num_layers=3, emb_dim=16, dropout=0.,
                 source_epochs=100, source_batch=128, source_lr=0.001, device="cpu")
    if any(getattr(cfg, key) != value for key, value in fixed.items()):
        raise ValueError("Archived source recipe differs from the fixed R7/R5 GIN recipe")
    if (plan["features"]["output_dimension_real"] != cfg.emb_dim
            or plan["features"]["raw_cache_key"] != "h"
            or plan["features"]["source_logits_dtype"] != "float64"
            or plan["features"]["source_model_state_key"] != "state"
            or plan["features"]["source_head_state_keys"] != ["head.weight", "head.bias"]
            or plan["execution"]["threads_per_worker"] != 1):
        raise ValueError("Unexpected source-refit representation/runtime contract")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Source initialization and training require the original float32 default dtype")
    torch.set_num_threads(1)
    provenance = context["provenance"]
    value = dict(
        format=_PREPARED, recipe=_RECIPE, setting=context["spec"]["id"], source=source, seed=seed,
        pooling=pooling, model_config=cfg.to_dict(), adam=deepcopy(old["adam"]),
        source_config_hash=r5_inputs._identity({k: v for k, v in old.items() if k != "workers"}),
        ids={key: list(context["ids"][key]) for key in ("source", "validation")},
        graph_ids=list(range(context["total"])), source_roles=deepcopy(context["roles"]),
        r5_data={key: deepcopy(provenance[key]) for key in
                 ("archive_root", "execution_commit", "r5_seal_digests", "repository_files")},
        r5_input_files=_declared_inputs(context, source, seed),
        code={relative: file_hash(r5_inputs._inside(repository, relative)) for relative in _DEPENDENCIES},
        environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                         torch_geometric=torch_geometric.__version__, device="cpu", source_dtype="float32",
                         frozen_dtype="float64", torch_threads=torch.get_num_threads()),
        selection="minimum_source_validation_ce_earliest_tie", target_labels_loaded=False,
        target_forward_pass_before_freezing=False,
    )
    return cfg, _signed(value)


def _source_context(plan, setting, source, seed, repository):
    setting_id = setting["id"] if isinstance(setting, dict) else setting
    specs = [item for item in plan["real_settings"] if item["id"] == setting_id]
    if len(specs) != 1:
        raise ValueError("Unknown real setting")
    # _context only authenticates label metadata; no target label values are read.
    target = next((i for i in range(len(specs[0]["domain_counts"])) if i != source), None)
    return r5_inputs._context(plan, setting, source, target, seed, repository)


def _prepare_record(out, registration):
    path = out / "prepared.json"
    if path.exists():
        if json.loads(path.read_text()) != registration:
            raise PermissionError("Existing source artifact has another recipe/data/code/runtime identity")
    else:
        _assert_mutable(out)
        write_json(path, registration)


def prepare_source(plan, setting, source, seed, *, pooling=None, output_root, repository=None):
    """Register source-only dependencies without opening any graph label values.

    Graph tensor/label bytes are authenticated when training starts; preparation
    authenticates their archived INPUT hashes. No model is trained here.
    """
    repository = r5_inputs._repository(repository)
    pooling = _pooling(plan, pooling)
    context = _source_context(plan, setting, source, seed, repository)
    _, registration = _registration(plan, context, source, seed, pooling, repository)
    out = _directory(output_root, context["spec"]["id"], source, seed, pooling)
    with _locked(out):
        _prepare_record(out, registration)
    return {**deepcopy(registration), "artifact_root": str(out.resolve())}


def _graphs(records, total):
    if not isinstance(records, list) or len(records) != total:
        raise ValueError("Archived graph count differs")
    graphs, dims, edge_dims = [], set(), set()
    for record in records:
        if (not isinstance(record, dict) or not {"x", "edge_index"} <= set(record)
                or set(record) - {"x", "edge_index", "edge_attr"}):
            raise ValueError("Archived graph records must be label-free tensor dictionaries")
        x, edge, ea = record["x"], record["edge_index"], record.get("edge_attr")
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1 or not x.is_floating_point()
                or not isinstance(edge, torch.Tensor) or edge.dtype != torch.int64
                or edge.ndim != 2 or edge.shape[0] != 2):
            raise ValueError("Invalid archived node/edge tensors")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= len(x)):
            raise ValueError("Archived edge endpoint lies outside its graph")
        finite(x)
        dims.add(x.shape[1])
        if ea is not None:
            if (not isinstance(ea, torch.Tensor) or not ea.is_floating_point() or ea.ndim != 2
                    or ea.shape[0] != edge.shape[1] or ea.shape[1] < 1):
                raise ValueError("Invalid archived edge features")
            finite(ea)
        edge_dims.add(0 if ea is None else ea.shape[1])
        graphs.append(Data(x=x, edge_index=edge, edge_attr=ea))
    if len(dims) != 1 or len(edge_dims) != 1:
        raise ValueError("Incompatible archived graph feature interfaces")
    return graphs, next(iter(dims)), next(iter(edge_dims))


def _source_labels(context, source, seed, role):
    name = "source" if role == "fit" else "validation"
    selected = context["ids"][name]
    path = f"source_labels/M{source}/seed{seed}_{role}.pt"
    value = torch.load(io.BytesIO(context["read"](path, "INPUT")), map_location="cpu", weights_only=True)
    labels = r5_inputs._tensor(value, (len(selected),), f"source {role} labels", integer=True)
    if (labels.unique().tolist() != [0, 1]
            or torch.bincount(labels, minlength=2).tolist() != context["roles"]["class_counts"][role]):
        raise ValueError("Source labels differ from sealed role counts/classes")
    return labels


def _fit_source(cfg, fit, validation, in_dim, edge_dim, pooling, adam_arguments, out, identity, logger=None):
    """Source-only training kernel; tiny synthetic configs are allowed here.

    Public builders enforce the exact real-data recipe. Checkpoint state, Adam,
    permutation RNG and torch RNG are retained after every completed epoch.
    """
    if cfg.device != "cpu" or torch.get_default_dtype() != torch.float32:
        raise ValueError("The paired source kernel requires CPU/float32")
    if (not fit.graphs or not validation.graphs or fit.labels is None or validation.labels is None
            or fit.labels.dtype != torch.int64 or validation.labels.dtype != torch.int64
            or set(fit.ids) & set(validation.ids)
            or any(graph.y is not None for graph in fit.graphs + validation.graphs)):
        raise ValueError("Source training requires disjoint, label-free graph objects and separate source labels")
    out = Path(out)
    set_seed(cfg.seed)
    model = SourceModel(PoolingGraphEncoder(in_dim, edge_dim, cfg.hidden, cfg.num_layers, cfg.emb_dim,
                                            cfg.dropout, cfg.encoder, pooling=pooling), 2).to(cfg.device)
    options = deepcopy(adam_arguments)
    if "betas" in options:
        options["betas"] = tuple(options["betas"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.source_lr, **options)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    initial = dict(preparation_digest=identity, state=_state(model), torch_rng=torch.random.get_rng_state(),
                   permutation_rng=rng.get_state())
    _put_tensor(out / "initialization.pt", initial)
    files = sorted((out / "epochs").glob("epoch*.pt"))
    if (len(files) > cfg.source_epochs
            or [p.name for p in files] != [f"epoch{i:03d}.pt" for i in range(len(files))]):
        raise ValueError("Incomplete/excessive source epoch sequence")
    history = []
    replay = torch.Generator().manual_seed(cfg.seed + 1)
    last = None
    for epoch, path in enumerate(files):
        last = _load(path)
        expected_order = [fit.ids[i] for i in torch.randperm(len(fit.graphs), generator=replay).tolist()]
        if (last["preparation_digest"] != identity or last["epoch"] != epoch
                or last["metrics"]["epoch"] != epoch or last["order"] != expected_order
                or not torch.equal(last["permutation_rng"], replay.get_state())):
            raise PermissionError("Source epoch identity/order/RNG differs")
        finite(last["metrics"]["train_ce"], last["metrics"]["source_validation"]["ce"])
        history.append(last["metrics"])
    if last is not None:
        model.load_state_dict(last["state"])
        optimizer.load_state_dict(last["optimizer"])
        rng.set_state(last["permutation_rng"])
        torch.random.set_rng_state(last["torch_rng"])
    for epoch in range(len(files), cfg.source_epochs):
        model.train()
        order = torch.randperm(len(fit.graphs), generator=rng).tolist()
        total = 0.
        for positions in graph_batches(fit.graphs, cfg.source_batch, order):
            batch = Batch.from_data_list([fit.graphs[i] for i in positions]).to(cfg.device)
            loss = F.cross_entropy(model(batch), fit.labels[positions].to(cfg.device))
            finite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters() if p.grad is not None))
            optimizer.step()
            finite(*model.parameters())
            total += float(loss.detach()) * len(positions)
        metrics = dict(epoch=epoch, train_ce=total / len(order),
                       source_validation=source_metrics(model, validation, cfg.source_batch, cfg.device))
        history.append(metrics)
        atomic_tensor(out / "epochs" / f"epoch{epoch:03d}.pt", dict(
            preparation_digest=identity, epoch=epoch, state=_state(model), optimizer=optimizer.state_dict(),
            order=[fit.ids[i] for i in order], metrics=metrics,
            torch_rng=torch.random.get_rng_state(), permutation_rng=rng.get_state()))
        if logger is not None:
            logger(dict(epoch=epoch, source_validation_ce=metrics["source_validation"]["ce"], pooling=pooling))
    selected = min(history, key=lambda row: row["source_validation"]["ce"])["epoch"]
    model.load_state_dict(_load(out / "epochs" / f"epoch{selected:03d}.pt")["state"])
    model.encoder.freeze()
    model.eval().zero_grad(set_to_none=True)
    return model, dict(selected_epoch=selected, history=history,
                       initialization_state_sha256=_state_identity(initial["state"]))


def _complete(out, registration, *, verify_files=True):
    record = json.loads(r5_inputs._inside(out, "complete.json").read_text())
    _verify_signature(record)
    if (record.get("format") != _COMPLETE or record.get("preparation_digest") != registration["digest"]
            or record.get("pooling") != registration["pooling"]):
        raise PermissionError("Completed source artifact identity differs")
    _verify_signature(registration)
    if json.loads(r5_inputs._inside(out, "prepared.json").read_text()) != registration:
        raise PermissionError("Completed source preparation differs")
    expected = {"prepared.json", "initialization.pt", "source.pt", "frozen.pt"}
    expected.update(f"epochs/epoch{i:03d}.pt" for i in range(registration["model_config"]["source_epochs"]))
    if set(record["files"]) != expected:
        raise PermissionError("Completed source file membership differs")
    actual = {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()
              and path.name not in (".build.lock", "complete.json") and ".partial-" not in path.name}
    if actual != expected:
        raise PermissionError("Unexpected/missing source artifact files")
    if verify_files:
        for relative, digest in record["files"].items():
            if file_hash(r5_inputs._inside(out, relative)) != digest:
                raise PermissionError(f"Completed source artifact changed: {relative}")
    return record


def train_source(plan, setting, source, seed, *, pooling=None, output_root, repository=None, logger=None):
    """Prepare and fit one source artifact; completed artifacts are verified/reused.

    This function does not authorize a study run. The phase coordinator is
    responsible for the registered S/P/C gates and permitted source domains.
    """
    repository = r5_inputs._repository(repository)
    pooling = _pooling(plan, pooling)
    context = _source_context(plan, setting, source, seed, repository)
    cfg, registration = _registration(plan, context, source, seed, pooling, repository)
    out = _directory(output_root, context["spec"]["id"], source, seed, pooling)
    with _locked(out):
        _prepare_record(out, registration)
        if (out / "complete.json").exists():
            return {**_complete(out, registration), "artifact_root": str(out.resolve())}
        _assert_mutable(out)
        records = torch.load(io.BytesIO(context["read"]("inputs/graphs.pt", "INPUT")),
                             map_location="cpu", weights_only=True)
        graphs, in_dim, edge_dim = _graphs(records, context["total"])
        fit_y = _source_labels(context, source, seed, "fit")
        val_y = _source_labels(context, source, seed, "validation")
        pools = {}
        for name, labels in (("source", fit_y), ("validation", val_y)):
            ids = context["ids"][name]
            pools[name] = GraphPool([graphs[i] for i in ids], [str(i) for i in ids], labels)
        model, info = _fit_source(cfg, pools["source"], pools["validation"], in_dim, edge_dim, pooling,
                                  registration["adam"], out, registration["digest"], logger)
        checkpoint = dict(encoder_spec=deepcopy(model.encoder.spec), pooling=pooling, state=_state(model),
                          preparation_digest=registration["digest"], **info)
        _put_tensor(out / "source.pt", checkpoint)
        if not model.encoder.frozen or model.encoder.training or any(p.requires_grad for p in model.encoder.parameters()):
            raise RuntimeError("All graph encoding must follow permanent source encoder freezing")
        h = encode(model.encoder, graphs, cfg.source_batch, cfg.device).double()
        coords = Coordinates.fit(h[context["ids"]["source"]], cfg.coordinate_eps)
        x = coords(h)
        head = ResidualClassifier.from_source(model.head, coords, cfg.adapter_width, seed).eval()
        with torch.no_grad():
            z0 = h @ model.head.weight.double().T + model.head.bias.double()
            logp = head(x).log_softmax(-1)
            native = model.head(h.float()).softmax(-1).double()
        torch.testing.assert_close(logp.exp(), z0.softmax(-1), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(logp.exp(), native, rtol=1e-5, atol=1e-6)
        if not torch.equal(logp.argmax(1), native.argmax(1)):
            raise ValueError("Fresh TSBA initialization changes source model predictions")
        _put_tensor(out / "frozen.pt", dict(h=h, x=x, z0=z0, baseline_logp=logp, mean=coords.mean,
                                            scale=coords.scale, initial_head=_state(head),
                                            graph_ids=registration["graph_ids"],
                                            preparation_digest=registration["digest"]))
        files = ["prepared.json", "initialization.pt", "source.pt", "frozen.pt"]
        files += [f"epochs/epoch{i:03d}.pt" for i in range(cfg.source_epochs)]
        complete = _signed(dict(
            format=_COMPLETE, preparation_digest=registration["digest"], pooling=pooling,
            files={relative: file_hash(out / relative) for relative in sorted(files)},
            selected_epoch=info["selected_epoch"], epochs=cfg.source_epochs,
            initialization_state_sha256=info["initialization_state_sha256"],
            source_validation_ce=info["history"][info["selected_epoch"]]["source_validation"]["ce"],
            residual_probability_max_error=float((logp.exp() - z0.softmax(-1)).abs().max()),
            float32_probability_max_error=float((logp.exp() - native).abs().max()),
            target_labels_loaded=False, target_forward_pass_before_freezing=False))
        write_json(out / "complete.json", complete)
        _complete(out, registration)
    return {**complete, "artifact_root": str(out.resolve())}


def load_inputs(plan, setting, source, target, seed, *, pooling=None, output_root, repository=None,
                include_tsba=False):
    """Read completed new embeddings while retaining the old label-broker contract.

    Only source-fit labels are opened by this loader. No historical R5 baseline
    logits are treated as the new B0; parity is checked against this source
    artifact's own raw head and probability-preserving TSBA initialization.
    """
    repository = r5_inputs._repository(repository)
    pooling = _pooling(plan, pooling)
    context = r5_inputs._context(plan, setting, source, target, seed, repository)
    cfg, registration = _registration(plan, context, source, seed, pooling, repository)
    out = _directory(output_root, context["spec"]["id"], source, seed, pooling)
    complete = _complete(out, registration)
    checkpoint, frozen = _load(out / "source.pt"), _load(out / "frozen.pt")
    if (checkpoint["preparation_digest"] != registration["digest"]
            or frozen["preparation_digest"] != registration["digest"]
            or checkpoint["pooling"] != pooling or frozen["graph_ids"] != list(range(context["total"]))):
        raise PermissionError("New checkpoint/cache identity differs")
    history = checkpoint["history"]
    if (len(history) != cfg.source_epochs or [row["epoch"] for row in history] != list(range(cfg.source_epochs))
            or any(not np.isfinite(row["source_validation"]["ce"]) for row in history)
            or checkpoint["selected_epoch"] != min(history, key=lambda row: row["source_validation"]["ce"])["epoch"]
            or checkpoint["selected_epoch"] != complete["selected_epoch"]):
        raise ValueError("Fresh source selection differs from the registered minimum CE rule")
    dimension, ids = cfg.emb_dim, context["ids"]
    h = r5_inputs._tensor(frozen["h"], (context["total"], dimension), "new frozen embeddings")
    weight = r5_inputs._tensor(checkpoint["state"]["head.weight"], (2, dimension), "new source weight")
    bias = r5_inputs._tensor(checkpoint["state"]["head.bias"], (2,), "new source bias")
    z0 = h @ weight.T + bias
    torch.testing.assert_close(z0, r5_inputs._tensor(frozen["z0"], z0.shape, "new cached z0"), rtol=0, atol=0)
    logp = r5_inputs._tensor(frozen["baseline_logp"], z0.shape, "new B0 log probabilities")
    torch.testing.assert_close(z0.softmax(-1), logp.exp(), rtol=1e-10, atol=1e-12)
    if not torch.equal(z0.argmax(1), logp.argmax(1)):
        raise ValueError("Reconstructed fresh B0 classes differ")
    labels = _source_labels(context, source, seed, "fit")
    provenance = context["provenance"]
    provenance.update(
        source_config_hash=registration["source_config_hash"], selected_source_epoch=checkpoint["selected_epoch"],
        id_hashes={name: r5_inputs._identity(value, compact=True) for name, value in ids.items()},
        target_labels_loaded=False,
        source_refit=dict(recipe=_RECIPE, pooling=pooling, artifact_root=str(out.resolve()),
                          artifact_digest=complete["digest"], preparation_digest=registration["digest"],
                          files={**complete["files"], "complete.json": file_hash(out / "complete.json")},
                          code=deepcopy(registration["code"]), environment=deepcopy(registration["environment"]),
                          source_roles=deepcopy(registration["source_roles"]),
                          r5_input_files=deepcopy(registration["r5_input_files"]),
                          b0_origin="fresh_source_refit", b0_rtol=1e-10, b0_atol=1e-12,
                          b0_probability_max_error=float((z0.softmax(-1) - logp.exp()).abs().max()),
                          initialization_state_sha256=complete["initialization_state_sha256"]))
    manifest = dict(format=r5_inputs._FORMAT, r7_plan=deepcopy(plan), setting=context["spec"]["id"],
                    source=source, target=target, seed=seed, ids=deepcopy(ids), provenance=deepcopy(provenance),
                    label_records=dict(relative_path=context["label_relative"],
                                       sha256=provenance["r5_files"][context["label_relative"]]))
    result = dict(source_y=r5_inputs._numpy(labels), source_head_weight=r5_inputs._numpy(weight),
                  source_head_bias=r5_inputs._numpy(bias), provenance=deepcopy(provenance), manifest=manifest)
    for name, selected in ids.items():
        result[name + "_ids"] = list(selected)
        result[name + "_h"] = r5_inputs._numpy(h[selected])
        result[name + "_z0"] = r5_inputs._numpy(z0[selected])
    if include_tsba:
        coords = Coordinates.fit(h[ids["source"]], cfg.coordinate_eps)
        mean = r5_inputs._tensor(frozen["mean"], (dimension,), "new TSBA source mean")
        scale = r5_inputs._tensor(frozen["scale"], (), "new TSBA RMS scale")
        x = r5_inputs._tensor(frozen["x"], h.shape, "new TSBA RMS coordinates")
        torch.testing.assert_close(coords.mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(coords.scale, scale, rtol=0, atol=0)
        torch.testing.assert_close(coords(h), x, rtol=0, atol=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            raw_head = torch.nn.Linear(dimension, 2, dtype=torch.float64)
        with torch.no_grad():
            raw_head.weight.copy_(weight)
            raw_head.bias.copy_(bias)
        initial = ResidualClassifier.from_source(raw_head, coords, cfg.adapter_width, seed).eval()
        if not _same(_state(initial), frozen["initial_head"]):
            raise ValueError("Fresh TSBA initialization differs from the original R5 construction")
        old = context["old_plan"]
        result.update(tsba_mean=r5_inputs._numpy(mean), tsba_scale=float(scale), tsba_initial_head=_state(initial),
                      r5_plan=deepcopy(old), r5_model_config=deepcopy(old["model"]),
                      r5_adam_config=deepcopy(old["adam"]))
        for name, selected in ids.items():
            result[name + "_x"] = r5_inputs._numpy(x[selected])
    return result
