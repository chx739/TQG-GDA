"""Future R7 D input preparation, M0 source training, and gated label reads.

No download, import-time data access, or target-label-dependent feature fitting
occurs here. ``prepare_inputs`` requires an authenticated passing C decision
and the caller's frozen D file hashes before opening the external raw files.
Acquisition must separately record the downloaded ZIP SHA256; this module
authenticates the four extracted files against the supplied raw checksums.

TU node/graph-indicator IDs are one-based. Repository domain indices and
exported graph IDs are zero-based. Class integers are sorted using M0 labels
only; no assumption about whether the TU binary labels are 0/1 or -1/1 is
made. Target classes outside this source mapping fail only at gated evaluation.
"""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch

from experiments.common.io import digest, save_tensor, write_json
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src.source_roles import split_source_roles
from tsba.data import GraphPool
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel, encode, graph_batches, source_metrics
from tsba.utils import finite, set_seed

from .fit import assert_density_execution, assert_fitter_sources, config
from .r5_inputs import _identity, _inside, _numpy, _repository, split_target_ids
from .seals import verify_seal


_FORMAT = "r7-nci1-inputs-v1"
_PREPARED = "r7-nci1-prepared-v1"
_FILES = ("NCI1_A.txt", "NCI1_graph_indicator.txt", "NCI1_node_labels.txt", "NCI1_graph_labels.txt")
_SOURCE = dict(encoder="gin", num_layers=3, hidden=64, emb_dim=16, dropout=0,
               source_epochs=100, source_lr=.001, source_batch=128, optimizer="Adam",
               role_split="R5_class_stratified_source_roles", selection="minimum_source_validation_CE",
               features="source_fit_sorted_node_label_onehot_plus_unknown", edge_attr=None)
_REFIT_RECIPE = "r5_frozen_role_source_refit_v1"
SOURCE_FIT_FILES = (
    "tsba/data.py", "tsba/encoder.py", "tsba/pretrain.py", "tsba/utils.py",
    "experiments/common/io.py",
    "experiments/mutagenicity/r3_源风险与置信度约束_20261112/src/source_roles.py",
)
_ADAM = dict(betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False,
             maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)


def _json(path):
    return json.loads(Path(path).read_text())


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _source_recipe(plan):
    spec = plan["phases"]["D"]["source_model"]
    refit = plan.get("source_refit")
    if refit is None:
        if spec not in (_SOURCE, _SOURCE | {"pooling": "sum"}):
            raise ValueError("D requires the registered R5 source recipe with sum pooling")
    elif (not isinstance(refit, dict) or set(refit) != {"pooling", "recipe"}
          or refit["recipe"] != _REFIT_RECIPE or refit["pooling"] not in ("mean", "sum")
          or spec != _SOURCE | {"pooling": refit["pooling"]}):
        raise ValueError("D source pooling must match the registered source refit used in C")
    return spec


def _registered(plan):
    external, phase = plan["external_input"], plan["phases"]["D"]
    _source_recipe(plan)
    if (external["id"] != "nci1_edge" or external["tu_dataset"] != "NCI1"
            or external["expected_raw_files"] != list(_FILES)
            or external["domain_index_pattern"] != "data/idx_NCI1_{domain}.npy"
            or external["raw_directory"] != "data/NCI1/raw"
            or external["domain_counts"] != [1027, 1027, 1027, 1029]
            or phase["setting"] != external["id"] or phase["source_domain"] != 0
            or phase["directions"] != [[0, 1], [0, 2], [0, 3]]
            or len(plan["real_model_seeds"]) != 5
            or len(set(plan["real_model_seeds"])) != 5
            or phase["source_checkpoints_new"] != 5 or phase["expected_blocks"] != 15
            or plan["execution"]["threads_per_worker"] != 1
            or not phase["freeze_before_target_access"]
            or not phase["enable_only_if_C_point_estimate_gate_and_complete_primary_pairs"]):
        raise ValueError("External input/source recipe differs from registered R7 D")
    return external


def _verify_access(plan, c_output, repository, locked_files):
    """Check the caller's frozen D dependencies before any raw-file access."""
    _registered(plan)
    c_output, repository = Path(c_output).resolve(), _repository(repository)
    final = verify_seal(c_output, "R7_FINAL")
    inputs = verify_seal(c_output, "R7_INPUT")
    for filename, record in (("gate.json", final), ("executed-plan.json", inputs),
                              ("executed-configs.json", inputs), ("provenance.json", inputs)):
        if record["files"].get(filename) != digest(_inside(c_output, filename)):
            raise PermissionError(f"C authorization file is not authenticated: {filename}")
    gate = _json(c_output / "gate.json")
    execution = _json(c_output / "executed-configs.json")
    tau = plan["phases"]["D"]["primary_tau"]
    refit = plan.get("source_refit")
    assert_density_execution(plan, execution)
    if (gate.get("passed") is not True or gate.get("next_phase") != "D"
            or gate.get("contrasts", {}).get("J-B0", {}).get("point_estimate_pass") is not True
            or gate["contrasts"]["J-B0"].get("complete_primary_pairs") is not True
            or execution.get("phase") != "C" or execution.get("primary_tau") != tau
            or tau != plan["shared_update"]["primary_tau"]
            or execution.get("separation_variant") != plan.get("separation_variant", "stock")
            or any(config(plan, seed, tau).to_dict() not in execution.get("common", [])
                   for seed in plan["real_model_seeds"])
            or _json(c_output / "executed-plan.json") != plan):
        raise PermissionError("C did not authorize this frozen plan's external confirmation")
    frozen = _json(c_output / "provenance.json")["file_hashes"]
    assert_fitter_sources(plan, repository, frozen, additional=["evaluate.py", "stage_p.py", "paired.py"])
    if refit is not None and any(frozen.get(name) != digest(_inside(repository, name)) for name in SOURCE_FIT_FILES):
        raise PermissionError("Core source fitting dependency differs from the sealed C implementation")
    own_path = str(Path(__file__).resolve().relative_to(repository))
    if not isinstance(locked_files, dict) or own_path not in locked_files:
        raise PermissionError("Caller must freeze the D input implementation and its dependency hashes")
    for relative, expected in locked_files.items():
        if not _checksum(expected) or digest(_inside(repository, relative)) != expected:
            raise PermissionError(f"Frozen D dependency changed: {relative}")
    return dict(c_output=str(c_output), c_final_digest=final["digest"],
                c_plan_sha256=inputs["files"]["executed-plan.json"],
                locked_files=deepcopy(locked_files), locked_files_digest=_identity(locked_files, compact=True))


def _domains(plan, repository):
    spec = plan["external_input"]
    domains, hashes = {}, {}
    total = sum(spec["domain_counts"])
    for number, count in enumerate(spec["domain_counts"]):
        relative = spec["domain_index_pattern"].format(domain=number)
        blob = _inside(repository, relative).read_bytes()
        checksum = hashlib.sha256(blob).hexdigest()
        ids = np.load(io.BytesIO(blob), allow_pickle=False)
        if (checksum != spec["index_sha256"][number] or ids.shape != (count,)
                or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= total)):
            raise PermissionError("NCI1 domain indices differ in values, order, or checksum")
        domains[str(number)] = ids.tolist()
        hashes[relative] = checksum
        a, h = split_target_ids(plan, spec["tu_dataset"], ids.tolist())
        if [len(a), len(h)] != spec["planned_adaptation_holdout_counts"][number]:
            raise ValueError("NCI1 hash-fold counts differ from registration")
    if sorted(i for ids in domains.values() for i in ids) != list(range(total)):
        raise ValueError("NCI1 domains must partition all zero-based TU graph IDs")
    return domains, hashes


def _integers(path, columns):
    # This parser is called only for topology or node-category files.
    rows = []
    with Path(path).open("r", encoding="ascii") as stream:
        for line in stream:
            fields = line.strip().split(",")
            if len(fields) != columns or any(not field.strip() for field in fields):
                raise ValueError(f"Expected {columns} integer column(s) in {Path(path).name}")
            rows.append([int(field) for field in fields])
    return np.asarray(rows, dtype=np.int64).reshape(-1, columns)


def parse_tu_topology(raw_directory, expected_hashes, expected_graphs):
    """Authenticate and parse three label-free files; never open graph labels."""
    raw = Path(raw_directory)
    for filename in _FILES[:3]:
        if not _checksum(expected_hashes.get(filename)) or digest(raw / filename) != expected_hashes[filename]:
            raise PermissionError(f"TU structural input changed: {filename}")
    indicator = _integers(raw / _FILES[1], 1)[:, 0]
    node_labels = _integers(raw / _FILES[2], 1)[:, 0]
    if (not len(indicator) or len(indicator) != len(node_labels)
            or not np.array_equal(np.unique(indicator), np.arange(1, expected_graphs + 1))
            or np.any(np.diff(indicator) < 0)):
        raise ValueError("TU graph indicators must be contiguous, ordered, one-based IDs")
    edge = _integers(raw / _FILES[0], 2) - 1
    if len(edge) and (edge.min() < 0 or edge.max() >= len(indicator)
                      or np.any(indicator[edge[:, 0]] != indicator[edge[:, 1]])):
        raise ValueError("Invalid TU node index or cross-graph edge")
    counts = np.bincount(indicator - 1, minlength=expected_graphs)
    offsets = np.concatenate(([0], counts.cumsum()))
    edge_domains = indicator[edge[:, 0]] - 1
    edge = edge[np.argsort(edge_domains, kind="stable")]
    edge_offsets = np.concatenate(([0], np.bincount(edge_domains, minlength=expected_graphs).cumsum()))
    records, sizes = [], []
    for graph_id in range(expected_graphs):
        start, stop = map(int, offsets[graph_id:graph_id + 2])
        first, last = edge_offsets[graph_id:graph_id + 2]
        local = edge[first:last] - start
        pairs = {tuple(pair) for pair in local.tolist()}
        if len(pairs) != len(local) or any((b, a) not in pairs for a, b in pairs):
            raise ValueError("TU edges must be unique and bidirectional; no silent coalescing")
        records.append(dict(node_labels=torch.from_numpy(node_labels[start:stop].copy()),
                            edge_index=torch.from_numpy(local.T.copy())))
        sizes.append(dict(id=graph_id, nodes=stop - start,
                          undirected_edges=len({(min(a, b), max(a, b)) for a, b in pairs})))
    return records, sizes


def _file_identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def _label_metadata(path, expected, count):
    """Commit the opaque label container without decoding class values."""
    offsets, state = [], hashlib.sha256()
    path = Path(path).resolve()
    with path.open("rb", buffering=0) as stream:
        before = _file_identity(os.fstat(stream.fileno()))
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            offsets.append([start, len(line)])
            state.update(line)
        after = _file_identity(os.fstat(stream.fileno()))
    if before != after or after != _file_identity(path.stat()) or state.hexdigest() != expected or len(offsets) != count:
        raise PermissionError("TU label-byte identity or count changed during acquisition indexing")
    return dict(path=str(path), sha256=expected, offsets=offsets, stat=after)


def _index_labels(path, expected, count):
    # The process exits before training and returns only identity and offsets.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as broker:
        return broker.submit(_label_metadata, path, expected, count).result()


def _selected_raw_labels(metadata, ids, *, authenticate_all=False):
    path = Path(metadata["path"])
    if authenticate_all:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise PermissionError("TU graph-label bytes differ from the frozen external snapshot")
        payloads = [raw[start:start + length] for start, length in (metadata["offsets"][i] for i in ids)]
    else:
        payloads = []
        with path.open("rb", buffering=0) as stream:
            identity = metadata["stat"]
            if _file_identity(os.fstat(stream.fileno())) != identity:
                raise PermissionError("Raw label file changed before permitted whole-file authentication")
            for graph_id in ids:
                start, length = metadata["offsets"][graph_id]
                stream.seek(start)
                payloads.append(stream.read(length))
            if _file_identity(os.fstat(stream.fileno())) != identity:
                raise PermissionError("Raw label file changed during role-scoped reads")
    if any(len(payload) != metadata["offsets"][graph_id][1] for payload, graph_id in zip(payloads, ids)):
        raise ValueError("Truncated permitted label record")
    return [int(payload.strip()) for payload in payloads]


def _roles(source_ids, raw_values, total, seed):
    classes = sorted(set(raw_values))
    if len(classes) != 2:
        raise ValueError("M0 must contain exactly two source classes")
    mapping = {str(value): index for index, value in enumerate(classes)}
    scoped = np.full(total, -1, dtype=np.int64)
    scoped[source_ids] = [mapping[str(value)] for value in raw_values]
    allocated = asdict(split_source_roles(np.asarray(source_ids), scoped, .4, .4, 2, seed))
    fit = sorted(np.concatenate((allocated["ref"], allocated["fit"])).tolist())
    validation = sorted(allocated["cal"].tolist())
    counts = {key: np.bincount(scoped[ids], minlength=2).tolist()
              for key, ids in (("fit", fit), ("validation", validation))}
    if (sorted(fit + validation) != sorted(source_ids)
            or any(not all(values) for values in counts.values())):
        raise ValueError("R5 source roles must partition M0 and retain both classes")
    roles = dict(fit=fit, validation=validation, class_counts=counts,
                 sega={key: value.tolist() for key, value in allocated.items()}, seed=seed)
    return roles, torch.tensor(scoped[fit]), torch.tensor(scoped[validation]), mapping


def _vocabulary(records, fit_ids):
    return sorted({int(value) for i in fit_ids for value in records[i]["node_labels"].tolist()})


def _graphs(records, vocabulary):
    from torch_geometric.data import Data

    mapping, unknown = {value: i for i, value in enumerate(vocabulary)}, len(vocabulary)
    if not vocabulary or vocabulary != sorted(set(vocabulary)):
        raise ValueError("Source-fit node vocabulary must be sorted, distinct, and nonempty")
    graphs = []
    for record in records:
        if set(record) != {"node_labels", "edge_index"}:
            raise ValueError("Structural records cannot contain graph classification labels")
        indices = torch.tensor([mapping.get(int(value), unknown) for value in record["node_labels"]])
        x = torch.nn.functional.one_hot(indices, unknown + 1).float()
        graphs.append(Data(x=x, edge_index=record["edge_index"].clone(), edge_attr=None))
    return graphs


def prepare_inputs(plan, directory, *, repository, raw_hashes, zip_sha256, c_output, locked_files):
    """Prepare immutable inputs after C passes and D code/parameters are locked.

    Only M0 graph-class records are parsed. The ZIP hash is acquisition evidence
    supplied by the caller, not independently verified against a ZIP here.
    Nothing is downloaded; absent/mismatched raw files stop D preparation.
    """
    repository = _repository(repository)
    authorization = _verify_access(plan, c_output, repository, locked_files)
    if set(raw_hashes) != set(_FILES) or not all(_checksum(v) for v in raw_hashes.values()) or not _checksum(zip_sha256):
        raise ValueError("All extracted raw hashes and the separately recorded ZIP SHA256 are required")
    domains, indices = _domains(plan, repository)
    raw = (repository / plan["external_input"]["raw_directory"]).resolve()
    records, sizes = parse_tu_topology(raw, raw_hashes, sum(plan["external_input"]["domain_counts"]))
    ranges = [[min(sizes[i]["undirected_edges"] for i in ids), max(sizes[i]["undirected_edges"] for i in ids)]
              for ids in domains.values()]
    ordered = list(zip(ranges, ranges[1:]))
    ascending = all(left[1] <= right[0] for left, right in ordered)
    descending = all(left[0] >= right[1] for left, right in ordered)
    if not ascending and not descending:
        raise ValueError("Fixed domain IDs do not form monotone edge-count domains")
    edge_order = 'tied' if ascending and descending else 'ascending' if ascending else 'descending'
    label_meta = _index_labels(raw / _FILES[3], raw_hashes[_FILES[3]], len(records))
    values = _selected_raw_labels(label_meta, domains["0"])
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "roles").mkdir()
    (directory / "source_labels").mkdir()
    (directory / "vocabulary").mkdir()
    save_tensor(directory / "topology.pt", records)
    write_json(directory / "domains.json", domains)
    write_json(directory / "graph_sizes.json", sizes)
    members = ["topology.pt", "domains.json", "graph_sizes.json", "label_records.json"]
    for seed in plan["real_model_seeds"]:
        roles, fit_y, validation_y, mapping = _roles(domains["0"], values, len(records), seed)
        label_meta["mapping"] = mapping
        filenames = (f"roles/seed{seed}.json", f"source_labels/seed{seed}_fit.pt",
                     f"source_labels/seed{seed}_validation.pt", f"vocabulary/seed{seed}.json")
        write_json(directory / filenames[0], roles)
        save_tensor(directory / filenames[1], fit_y)
        save_tensor(directory / filenames[2], validation_y)
        write_json(directory / filenames[3], dict(values=_vocabulary(records, roles["fit"]),
                                                  fit_ids=roles["fit"], unknown_slot="last"))
        members.extend(filenames)
    write_json(directory / "label_records.json", label_meta)
    prepared = dict(format=_PREPARED, r7_plan=deepcopy(plan), repository=str(repository),
                    authorization=authorization, raw_hashes=deepcopy(raw_hashes),
                    zip_sha256_supplied_by_acquisition=zip_sha256, zip_independently_verified_here=False,
                    domain_index_hashes=indices, edge_count_ranges=ranges, edge_count_domain_order=edge_order,
                    parsed_graph_label_ids=domains["0"], target_graph_labels_parsed=False,
                    label_acquisition="opaque hash and byte offsets in a separate process; no class values returned",
                    node_vocabulary_scope="source_fit_per_seed", source_model=deepcopy(_source_recipe(plan)),
                    source_adam=deepcopy(_ADAM), source_role_fractions=[.4, .4],
                    files={name: digest(directory / name) for name in members})
    write_json(directory / "prepared.json", prepared)
    return prepared


def _prepared(plan, directory, repository=None):
    directory = Path(directory).resolve()
    blob = _inside(directory, "prepared.json").read_bytes()
    prepared = json.loads(blob)
    if prepared.get("format") != _PREPARED or prepared["r7_plan"] != plan:
        raise PermissionError("External preparation belongs to a different plan")
    _registered(plan)
    repository = _repository(repository or prepared["repository"])
    domains, hashes = _domains(plan, repository)
    if (hashes != prepared["domain_index_hashes"] or prepared["source_model"] != _source_recipe(plan)
            or prepared["source_adam"] != json.loads(json.dumps(_ADAM))
            or prepared["source_role_fractions"] != [.4, .4]
            or prepared["parsed_graph_label_ids"] != domains["0"]
            or prepared["target_graph_labels_parsed"] is not False):
        raise PermissionError("External source recipe, permitted label scope, or domain identity changed")
    for relative, expected in prepared["files"].items():
        if digest(_inside(directory, relative)) != expected:
            raise PermissionError(f"External prepared file changed: {relative}")
    for relative, expected in prepared["authorization"]["locked_files"].items():
        if digest(_inside(repository, relative)) != expected:
            raise PermissionError(f"Frozen D implementation or plan changed: {relative}")
    if _json(directory / "domains.json") != domains:
        raise PermissionError("Prepared graph order differs from frozen domains")
    return directory, prepared, domains, hashlib.sha256(blob).hexdigest()


def _source_encoder(spec, in_dim):
    options = dict(in_dim=in_dim, edge_dim=0, hidden=spec["hidden"], num_layers=spec["num_layers"],
                   out_dim=spec["emb_dim"], dropout=spec["dropout"], conv=spec["encoder"])
    if "pooling" in spec:
        from .source_refit_inputs import PoolingGraphEncoder

        return PoolingGraphEncoder(**options, pooling=spec["pooling"])
    return GraphEncoder(**options)


def check_source_pooling():
    """Check inherited sum and independent per-layer means on unequal graph sizes."""
    from torch_geometric.data import Batch, Data

    spec = _SOURCE | {"hidden": 5, "num_layers": 2, "emb_dim": 3}
    batch = Batch.from_data_list([
        Data(x=torch.tensor([[1., 0.], [0., 1.]]), edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(x=torch.tensor([[0., 2.], [1., 1.], [1., 2.]]), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])),
    ])
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(17)
        legacy = _source_encoder(spec, 2).eval()
        refit = {name: _source_encoder(spec | {"pooling": name}, 2).eval() for name in ("sum", "mean")}
        for encoder in refit.values():
            encoder.load_state_dict(legacy.state_dict())
        torch.testing.assert_close(refit["sum"](batch), legacy(batch), rtol=0, atol=0)
        x = batch.x
        pooled = []
        for conv, norm in zip(legacy.convs, legacy.norms):
            x = torch.relu(norm(conv(x, batch.edge_index)))
            pooled.append(torch.stack([x[batch.batch == graph].mean(0) for graph in range(batch.num_graphs)]))
        expected = legacy.readout(torch.cat(pooled, dim=-1))
        torch.testing.assert_close(refit["mean"](batch), expected, rtol=1e-6, atol=1e-7)
        assert not torch.allclose(refit["mean"](batch), refit["sum"](batch))


def _fit_source(graphs, roles, fit_y, validation_y, spec, seed, directory):
    """R5 CPU float32 training mathematics; generated fixtures may use fewer epochs."""
    from torch_geometric.data import Batch

    if torch.get_default_dtype() != torch.float32:
        raise ValueError("R5 source initialization requires the float32 default dtype")
    fit_ids, validation_ids = roles["fit"], roles["validation"]
    fit_graphs, validation_graphs = ([graphs[i] for i in ids] for ids in (fit_ids, validation_ids))
    if any(graph.y is not None or graph.edge_attr is not None for graph in graphs):
        raise ValueError("D graphs must have no graph labels or edge attributes")
    validation = GraphPool(validation_graphs, list(map(str, validation_ids)), validation_y)
    set_seed(seed)
    model = SourceModel(_source_encoder(spec, graphs[0].x.shape[1]), 2).cpu()
    pooling = spec.get("pooling", "sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["source_lr"], **_ADAM)
    rng = torch.Generator().manual_seed(seed + 1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "epochs").mkdir()
    history, best, best_state, selected = [], float("inf"), None, None
    for epoch in range(spec["source_epochs"]):
        model.train()
        order = torch.randperm(len(fit_ids), generator=rng).tolist()
        total = 0.
        for ids in graph_batches(fit_graphs, spec["source_batch"], order):
            batch = Batch.from_data_list([fit_graphs[i] for i in ids])
            loss = torch.nn.functional.cross_entropy(model(batch), fit_y[ids])
            finite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters() if p.grad is not None))
            optimizer.step()
            finite(*model.parameters())
            total += float(loss.detach()) * len(ids)
        metrics = dict(epoch=epoch, train_ce=total / len(order),
                       source_validation=source_metrics(model, validation, spec["source_batch"], "cpu"))
        history.append(metrics)
        if metrics["source_validation"]["ce"] < best:
            best, selected = metrics["source_validation"]["ce"], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        save_tensor(directory / "epochs" / f"epoch{epoch:03d}.pt", dict(
            epoch=epoch, state=model.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics, pooling=pooling,
            order=[fit_ids[i] for i in order], torch_rng=torch.random.get_rng_state(), permutation_rng=rng.get_state()))
    model.load_state_dict(best_state)
    model.encoder.freeze()
    model.eval().zero_grad(set_to_none=True)
    h = encode(model.encoder, graphs, spec["source_batch"], "cpu").double()
    with torch.no_grad():
        weight, bias = model.head.weight.double(), model.head.bias.double()
        z0 = h @ weight.T + bias
        native_probability = model.head(h.float()).softmax(-1).double()
        probability = z0.softmax(-1)
    torch.testing.assert_close(probability, native_probability, rtol=1e-5, atol=1e-6)
    if not torch.equal(probability.argmax(1), native_probability.argmax(1)):
        raise ValueError("Promoted raw source logits changed native B0 classes")
    save_tensor(directory / "source.pt", dict(encoder_spec=model.encoder.spec, state=model.state_dict(),
                                               selected_epoch=selected, history=history, source_model=spec, pooling=pooling))
    save_tensor(directory / "frozen.pt", dict(h=h, z0=z0, ids=torch.arange(len(graphs)),
                                               source_head_weight=weight, source_head_bias=bias))
    return dict(selected_epoch=selected, source_validation_ce=best, source_epochs=spec["source_epochs"], pooling=pooling,
                native_probability_max_error=float((probability - native_probability).abs().max()),
                encoder_frozen=not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters()),
                source_labels_used_only_for_training_and_selection=True, target_labels_loaded=False)


def train_source(plan, prepared_root, seed):
    """Train one registered M0 seed; output is append-only and not resumable."""
    directory, prepared, domains, checksum = _prepared(plan, prepared_root)
    torch.set_num_threads(plan["execution"]["threads_per_worker"])
    if type(seed) is not int or seed not in plan["real_model_seeds"]:
        raise ValueError("Unregistered D source seed")
    roles = _json(directory / f"roles/seed{seed}.json")
    if (sorted(roles["fit"] + roles["validation"]) != sorted(domains["0"])
            or roles["fit"] != sorted(roles["fit"]) or roles["validation"] != sorted(roles["validation"])):
        raise PermissionError("D source roles must partition M0, with the original sorted role order")
    vocabulary = _json(directory / f"vocabulary/seed{seed}.json")
    if vocabulary["fit_ids"] != roles["fit"] or vocabulary["unknown_slot"] != "last":
        raise PermissionError("Node vocabulary role differs from source fit")
    records = torch.load(directory / "topology.pt", map_location="cpu", weights_only=True)
    if vocabulary["values"] != _vocabulary(records, roles["fit"]):
        raise PermissionError("Node vocabulary contains categories outside source fit")
    graphs = _graphs(records, vocabulary["values"])
    fit_y = torch.load(directory / f"source_labels/seed{seed}_fit.pt", weights_only=True)
    validation_y = torch.load(directory / f"source_labels/seed{seed}_validation.pt", weights_only=True)
    for name, labels in (("fit", fit_y), ("validation", validation_y)):
        if (labels.dtype != torch.int64 or labels.shape != (len(roles[name]),)
                or labels.unique().tolist() != [0, 1]
                or torch.bincount(labels, minlength=2).tolist() != roles["class_counts"][name]):
            raise PermissionError("D source labels differ from the permitted M0 role")
    out = directory / "source" / f"seed{seed}"
    spec = prepared["source_model"]
    record = _fit_source(graphs, roles, fit_y, validation_y, spec, seed, out)
    files = ["source.pt", "frozen.pt"] + [f"epochs/epoch{i:03d}.pt" for i in range(spec["source_epochs"])]
    complete = dict(format="r7-nci1-source-v1", seed=seed, source=0, prepared_sha256=checksum,
                    info=record, files={name: digest(out / name) for name in files})
    write_json(out / "complete.json", complete)
    return complete


def _context(plan, setting, source, target, seed, prepared_root, repository=None):
    if (setting != "nci1_edge" or source != 0 or type(target) is not int or target not in (1, 2, 3)
            or type(seed) is not int or seed not in plan["real_model_seeds"]):
        raise ValueError("D only permits the registered M0 to M1/M2/M3 tasks")
    directory, prepared, domains, checksum = _prepared(plan, prepared_root, repository)
    roles = _json(directory / f"roles/seed{seed}.json")
    if sorted(roles["fit"] + roles["validation"]) != sorted(domains["0"]):
        raise PermissionError("External source roles do not partition M0")
    a, h = split_target_ids(plan, "NCI1", domains[str(target)])
    ids = dict(source=roles["fit"], validation=roles["validation"], adapt=a, holdout=h)
    out = directory / "source" / f"seed{seed}"
    complete_blob = _inside(out, "complete.json").read_bytes()
    complete = json.loads(complete_blob)
    spec = prepared["source_model"]
    pooling = spec.get("pooling", "sum")
    if (complete.get("format") != "r7-nci1-source-v1" or complete["source"] != 0
            or complete["seed"] != seed or complete["prepared_sha256"] != checksum
            or complete["info"]["source_epochs"] != spec["source_epochs"]
            or complete["info"].get("pooling", "sum") != pooling):
        raise PermissionError("Source checkpoint has different external inputs, training budget, or pooling")
    for name in ("source.pt", "frozen.pt"):
        if digest(_inside(out, name)) != complete["files"].get(name):
            raise PermissionError(f"External source checkpoint changed: {name}")
    provenance = dict(prepared_root=str(directory), prepared_sha256=checksum,
                      source_complete_sha256=hashlib.sha256(complete_blob).hexdigest(),
                      source_files={name: complete["files"][name] for name in ("source.pt", "frozen.pt")},
                      source_domain=0, source_seed=seed, source_pooling=pooling,
                      domain_index_hashes=prepared["domain_index_hashes"],
                      id_hashes={key: _identity(value, compact=True) for key, value in ids.items()},
                      target_labels_loaded=False)
    return directory, out, prepared, roles, ids, provenance


def load_inputs(plan, setting, source, target, seed, *, prepared_root, repository=None):
    """Return R5-compatible raw representations, with source-fit labels only."""
    directory, out, prepared, roles, ids, provenance = _context(
        plan, setting, source, target, seed, prepared_root, repository)
    checkpoint = torch.load(out / "source.pt", map_location="cpu", weights_only=True)
    frozen = torch.load(out / "frozen.pt", map_location="cpu", weights_only=True)
    history = checkpoint["history"]
    spec = prepared["source_model"]
    if (checkpoint["source_model"] != spec or checkpoint.get("pooling", "sum") != spec.get("pooling", "sum")
            or len(history) != spec["source_epochs"]
            or [row["epoch"] for row in history] != list(range(len(history)))
            or any(not np.isfinite(row["source_validation"]["ce"]) for row in history)
            or checkpoint["selected_epoch"] != min(history, key=lambda row: row["source_validation"]["ce"])["epoch"]):
        raise PermissionError("External source checkpoint pooling or selection differs from the registered recipe")
    total = sum(plan["external_input"]["domain_counts"])
    h, z0 = frozen["h"], frozen["z0"]
    if (h.shape != (total, spec["emb_dim"]) or z0.shape != (total, 2) or h.dtype != torch.float64
            or z0.dtype != torch.float64 or not torch.equal(frozen["ids"], torch.arange(total))):
        raise ValueError("Malformed external raw representation cache")
    weight = checkpoint["state"]["head.weight"].double()
    bias = checkpoint["state"]["head.bias"].double()
    finite(h, z0, weight, bias)
    torch.testing.assert_close(z0, h @ weight.T + bias, rtol=0, atol=0)
    torch.testing.assert_close(weight, frozen["source_head_weight"], rtol=0, atol=0)
    torch.testing.assert_close(bias, frozen["source_head_bias"], rtol=0, atol=0)
    labels = torch.load(directory / f"source_labels/seed{seed}_fit.pt", map_location="cpu", weights_only=True)
    if (labels.shape != (len(ids["source"]),) or labels.dtype != torch.int64
            or torch.bincount(labels, minlength=2).tolist() != roles["class_counts"]["fit"]):
        raise ValueError("Source-fit labels disagree with frozen roles")
    provenance.update(selected_source_epoch=checkpoint["selected_epoch"], b0_raw_logit_replay_exact=True)
    manifest = dict(format=_FORMAT, r7_plan=deepcopy(plan), setting=setting, source=source, target=target,
                    seed=seed, ids=deepcopy(ids), provenance=deepcopy(provenance),
                    label_records=dict(relative_path="label_records.json", sha256=prepared["files"]["label_records.json"]))
    result = dict(source_y=_numpy(labels), source_head_weight=_numpy(weight), source_head_bias=_numpy(bias),
                  provenance=provenance, manifest=manifest)
    for pool, selected in ids.items():
        result[f"{pool}_ids"] = list(selected)
        result[f"{pool}_h"] = _numpy(h[selected])
        result[f"{pool}_z0"] = _numpy(z0[selected])
    return result


def load_labels(root, manifest_relative_path, pool, *, repository=None, authenticated_snapshot=False):
    """Read A only after OBS, and H/source validation only after ORACLE.

    Before ORACLE only selected A offsets are opened; target-label bytes are
    not scanned. After ORACLE the full snapshot hash is authenticated, and
    ``authenticated_snapshot=True`` also reauthenticates previously read A.
    """
    if pool not in ("adapt", "holdout", "validation"):
        raise ValueError("Unsupported D label pool")
    root = Path(root).resolve()
    inputs = verify_seal(root, "R7_INPUT")
    verify_seal(root, "R7_OBS")
    if pool in ("holdout", "validation") or authenticated_snapshot:
        verify_seal(root, "R7_ORACLE")
    path = _inside(root, manifest_relative_path)
    blob = path.read_bytes()
    if inputs["files"].get(str(path.relative_to(root))) != hashlib.sha256(blob).hexdigest():
        raise PermissionError("D label-request manifest is outside the sealed inputs")
    manifest = json.loads(blob)
    if manifest.get("format") != _FORMAT:
        raise ValueError("Unsupported external input manifest")
    directory, _, prepared, roles, ids, provenance = _context(
        manifest["r7_plan"], manifest["setting"], manifest["source"], manifest["target"], manifest["seed"],
        manifest["provenance"]["prepared_root"], repository)
    if manifest["ids"] != ids or any(manifest["provenance"][key] != value for key, value in provenance.items()):
        raise PermissionError("External label-request provenance or graph IDs changed")
    if manifest["label_records"] != dict(relative_path="label_records.json", sha256=prepared["files"]["label_records.json"]):
        raise PermissionError("External label-offset metadata differs from sealed inputs")
    if pool == "validation":
        labels = torch.load(directory / f"source_labels/seed{manifest['seed']}_validation.pt",
                            map_location="cpu", weights_only=True)
        if labels.shape != (len(ids[pool]),) or torch.bincount(labels, minlength=2).tolist() != roles["class_counts"][pool]:
            raise ValueError("Source validation labels disagree with frozen roles")
        return _numpy(labels)
    metadata = _json(directory / "label_records.json")
    expected_path = Path(prepared["repository"]) / manifest["r7_plan"]["external_input"]["raw_directory"] / _FILES[3]
    if Path(metadata["path"]).resolve() != expected_path.resolve() or metadata["sha256"] != prepared["raw_hashes"][_FILES[3]]:
        raise PermissionError("External raw label path or checksum changed")
    values = _selected_raw_labels(metadata, ids[pool], authenticate_all=pool == "holdout" or authenticated_snapshot)
    if any(str(value) not in metadata["mapping"] for value in values):
        raise ValueError("Target class is outside the mapping defined by M0 source labels")
    labels = np.array([metadata["mapping"][str(value)] for value in values], dtype=np.int64)
    labels.setflags(write=False)
    return labels


def _check_sealed_reader(parent):
    """Exercise the real seal/offset reader with a generated, mocked context."""
    import sys
    from unittest.mock import patch

    from .seals import seal

    folder = Path(parent) / "sealed-label-toy"
    folder.mkdir()
    labels_path = folder / _FILES[3]
    labels_path.write_text("-1\n1\n-1\n1\n-1\n1\n")
    metadata = _label_metadata(labels_path, digest(labels_path), 6)
    metadata["mapping"] = {"-1": 0, "1": 1}
    write_json(folder / "label_records.json", metadata)
    plan = dict(external_input=dict(raw_directory=""))
    prepared = dict(repository=str(folder), raw_hashes={_FILES[3]: metadata["sha256"]},
                    files={"label_records.json": digest(folder / "label_records.json")})
    ids = dict(source=[0], validation=[1], adapt=[2, 3], holdout=[4, 5])
    provenance = dict(prepared_root=str(folder))
    manifest = dict(format=_FORMAT, r7_plan=plan, setting="nci1_edge", source=0, target=1, seed=3405,
                    ids=ids, provenance=provenance,
                    label_records=dict(relative_path="label_records.json", sha256=prepared["files"]["label_records.json"]))
    write_json(folder / "manifest.json", manifest)
    seal(folder, "R7_INPUT", ["manifest.json"], expected=["manifest.json"])
    context = (folder, folder, prepared, {}, ids, provenance)
    module = sys.modules[__name__]
    with patch.object(module, "_context", return_value=context), patch.object(
            module, "_selected_raw_labels", wraps=_selected_raw_labels) as reader:
        for pool in ("adapt", "holdout"):
            try:
                load_labels(folder, "manifest.json", pool)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Target labels were readable before OBS")
        assert not reader.called
        write_json(folder / "observed.json", {"toy_predictions_frozen": True})
        seal(folder, "R7_OBS", ["observed.json"], expected=["observed.json"])
        np.testing.assert_array_equal(load_labels(folder, "manifest.json", "adapt"), [0, 1])
        assert reader.call_args.args[1] == [2, 3] and reader.call_args.kwargs == {"authenticate_all": False}
        try:
            load_labels(folder, "manifest.json", "holdout")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("H labels were readable before ORACLE")
        assert reader.call_count == 1
        write_json(folder / "oracle.json", {"toy_oracle_frozen": True})
        seal(folder, "R7_ORACLE", ["oracle.json"], expected=["oracle.json"])
        np.testing.assert_array_equal(load_labels(folder, "manifest.json", "holdout"), [0, 1])
        assert reader.call_args.args[1] == [4, 5] and reader.call_args.kwargs == {"authenticate_all": True}
        np.testing.assert_array_equal(load_labels(folder, "manifest.json", "adapt", authenticated_snapshot=True), [0, 1])
        assert reader.call_args.args[1] == [2, 3] and reader.call_args.kwargs == {"authenticate_all": True}
        labels_path.write_text(" 1\n1\n-1\n1\n-1\n1\n")
        try:
            load_labels(folder, "manifest.json", "holdout")
        except PermissionError:
            pass
        else:
            raise AssertionError("Whole-file authentication missed changed label bytes")


def check_nci1_inputs():
    """Generated TU fixtures only; never accesses repository NCI1 raw inputs."""
    from tempfile import TemporaryDirectory

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with TemporaryDirectory(prefix="r7-nci1-engineering-") as temp:
            raw = Path(temp)
            graphs, source_count = 24, 20
            indicator, edges, node_labels = [], [], []
            for graph_id in range(graphs):
                first = 2 * graph_id + 1
                indicator += [str(graph_id + 1)] * 2
                edges += [f"{first},{first + 1}", f"{first + 1},{first}"]
                node_labels += ["1", "2" if graph_id < source_count else "99"]
            for name, lines in ((_FILES[0], edges), (_FILES[1], indicator), (_FILES[2], node_labels)):
                (raw / name).write_text("\n".join(lines) + "\n")
            hashes = {name: digest(raw / name) for name in _FILES[:3]}
            records, sizes = parse_tu_topology(raw, hashes, graphs)
            assert not (raw / _FILES[3]).exists() and len(records) == graphs
            # Invalid target class tokens prove allocation never parses them.
            (raw / _FILES[3]).write_text("\n".join([str(i % 2 * 2 - 1) for i in range(source_count)]
                                                 + ["TARGET_NOT_PERMITTED"] * (graphs - source_count)) + "\n")
            metadata = _index_labels(raw / _FILES[3], digest(raw / _FILES[3]), graphs)
            assert set(metadata) == {"path", "sha256", "offsets", "stat"}
            source_ids = list(range(source_count))
            values = _selected_raw_labels(metadata, source_ids)
            roles, fit_y, val_y, mapping = _roles(source_ids, values, graphs, 3405)
            assert mapping == {"-1": 0, "1": 1} and roles["class_counts"] == {"fit": [8, 8], "validation": [2, 2]}
            vocabulary = _vocabulary(records, roles["fit"])
            assert vocabulary == [1, 2]
            prepared_graphs = _graphs(records, vocabulary)
            assert prepared_graphs[-1].x[:, -1].sum() == 1 and all(g.y is None for g in prepared_graphs)
            toy_spec = _SOURCE | dict(source_epochs=2, source_batch=7, hidden=8, num_layers=2, emb_dim=4)
            first = _fit_source(prepared_graphs, roles, fit_y, val_y, toy_spec, 3405, raw / "toy-source")
            second = _fit_source(prepared_graphs, roles, fit_y, val_y, toy_spec, 3405, raw / "toy-repeat")
            a = torch.load(raw / "toy-source/frozen.pt", weights_only=True)
            b = torch.load(raw / "toy-repeat/frozen.pt", weights_only=True)
            assert first == second and first["encoder_frozen"] and not first["target_labels_loaded"]
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            checkpoint = torch.load(raw / "toy-source/source.pt", weights_only=True)
            assert checkpoint["selected_epoch"] == min(checkpoint["history"], key=lambda r: r["source_validation"]["ce"])["epoch"]
            # An unchanged-length edit is rejected before any permitted parse.
            (raw / _FILES[3]).write_text((raw / _FILES[3]).read_text().replace("-1", " 1", 1))
            try:
                _selected_raw_labels(metadata, [0])
            except PermissionError:
                pass
            else:
                raise AssertionError("Changed pre-oracle label bytes were accepted")
            _check_sealed_reader(raw)
            return dict(status="passed", toy_graphs=graphs, source_graphs=source_count,
                        target_class_tokens_never_parsed=True, source_vocabulary_only=True,
                        deterministic_source_training=True, source_role_counts=roles["class_counts"],
                        raw_logit_replay=True, source_epochs_in_toy=2,
                        label_seals_and_precise_adapt_offsets=True, post_oracle_full_hash=True)
    finally:
        torch.set_num_threads(previous_threads)
