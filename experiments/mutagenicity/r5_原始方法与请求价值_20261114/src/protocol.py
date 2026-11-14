"""R5 setting identities, TU inputs and immutable dependency registration."""
import copy
from pathlib import Path
import shutil

import numpy as np
import torch
from torch_geometric.utils import coalesce, remove_self_loops

from .shared import data, io, p3, p4, digest, split_source_roles


def resolve(request, setting):
    selected = next(s for s in request["settings"] if s["id"] == setting)
    plan = copy.deepcopy(request)
    plan.update(setting=selected, workers=request["execution"]["max_workers"],
                weight=request["adaptation"]["weight"], epsilon=None, risk_budget=0.,
                constraint_tolerance=1e-9, solver={}, historical_directions=[],
                side_attempts=request["side"]["attempts"], side_alpha_start=request["side"]["alpha_start"],
                relocation_radius_sigma=request["relocation"]["radius_sigma"],
                relocation_scan_points=request["relocation"]["scan_points"],
                relocation_bisections=request["relocation"]["bisections"],
                endpoints=request["evaluation"]["reported_endpoints"],
                pairs=[request["evaluation"]["primary_contrast"], *request["evaluation"]["secondary_contrasts"]],
                counts=dict(sources=20, scores=60, trajectories=120, attempts=3600))
    plan["model"]["score_epochs"] = selected["score_epochs"]
    if any((s["count"] + plan["model"]["score_batch"] - 1) // plan["model"]["score_batch"]
           * selected["score_epochs"] != request["score"]["optimizer_updates"]
           for s in selected["domain_indices"]):
        raise ValueError("Exact score update count mismatch")
    p4.tasks(plan)
    return plan


def graphs(plan):
    spec = plan["input"]
    raw = Path(spec["raw_directory"])
    prefix = plan["dataset"]
    count = plan["setting"]["total_graphs"]
    if prefix == "Mutagenicity":
        return data.reconstruct_graphs(raw, prefix, count, (14, 3))[::2]
    # PROTEINS has categorical nodes and no edge-label input.
    def read(name, ndmin=1):
        return torch.from_numpy(np.loadtxt(raw / f"{prefix}_{name}.txt", delimiter=",",
                                          dtype=np.int64, ndmin=ndmin))
    batch = read("graph_indicator") - 1
    edges = read("A", 2).t().contiguous() - 1
    labels = read("node_labels")
    if sorted(labels.unique().tolist()) != spec["node_label_values"]:
        raise ValueError("Unexpected node feature vocabulary")
    if len(labels) != len(batch) or batch.min() != 0 or batch.max() != count - 1 or (batch[1:] < batch[:-1]).any():
        raise ValueError("Invalid graph membership")
    sizes = torch.bincount(batch, minlength=count)
    if (sizes == 0).any() or edges.min() < 0 or edges.max() >= len(batch):
        raise ValueError("Invalid node or edge count")
    if not torch.equal(batch[edges[0]], batch[edges[1]]):
        raise ValueError("Cross-graph edge")
    edges, _ = remove_self_loops(edges)
    edges = coalesce(edges, num_nodes=len(batch))
    x = torch.nn.functional.one_hot(labels, spec["node_feature_dim"]).float()
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
    records, hashes = [], []
    for number in range(count):
        start, stop = map(int, offsets[number:number+2])
        local = edges[:, batch[edges[0]] == number] - start
        pairs = set(map(tuple, local.t().tolist()))
        if any((b, a) not in pairs for a, b in pairs):
            raise ValueError("Missing reverse edge")
        record = dict(x=x[start:stop].clone(), edge_index=local.clone(), edge_attr=None)
        records.append(record)
        hashes.append(dict(id=number, input_sha256={
            key: io.tensor_sha(value) if value is not None else None for key, value in record.items()}))
    return records, hashes


def prepare(root, repo, plan):
    spec, selected = plan["input"], plan["setting"]
    raw = Path(spec["raw_directory"])
    for name, expected in spec["files"].items():
        data._verify(raw / name, expected)
    domains = {}
    for entry in selected["domain_indices"]:
        path = repo / entry["path"]
        data._verify(path, entry)
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or len(values) != entry["count"]:
            raise ValueError("Invalid domain IDs")
        domains[str(entry["domain"])] = values.tolist()
    if sorted(i for ids in domains.values() for i in ids) != list(range(selected["total_graphs"])):
        raise ValueError("Domain partition is not disjoint and exhaustive")
    io.write_json(root / "inputs/domains.json", domains)
    records, hashes = graphs(plan)
    io.atomic_tensor(root / "inputs/graphs.pt", records)
    io.write_json(root / "inputs/graph_hashes.json", hashes)
    label_path = raw / f"{plan['dataset']}_graph_labels.txt"
    offsets = data.label_offsets(label_path)
    if len(offsets) != selected["total_graphs"]:
        raise ValueError("Graph label record count differs")
    sha = digest(label_path)
    mapping = None
    for source in plan["domains"]:
        ids = domains[str(source)]
        values = p3.read_values(label_path, offsets, ids, sha)
        current = {str(v): i for i, v in enumerate(sorted(set(values)))}
        if len(current) != 2 or (mapping is not None and current != mapping):
            raise ValueError("Source class mapping mismatch")
        mapping = current
        for seed in plan["seeds"]:
            roles, fit, val = p4.split_domain(ids, values, mapping, split_source_roles, seed,
                        selected["total_graphs"], (plan["source"]["ref_fraction"], plan["source"]["fit_fraction"]))
            io.write_json(root / "roles" / f"M{source}" / f"seed{seed}.json", roles)
            io.atomic_tensor(p4.labels_path(root, source, seed, "fit"), fit)
            io.atomic_tensor(p4.labels_path(root, source, seed, "validation"), val)
    io.write_json(root / "inputs/label_records.json", dict(path=str(label_path), offsets=offsets,
                                                        sha256=sha, mapping=mapping))
    for task in p4.tasks(plan):
        fit = p4.role(root, task.source, task.seed)["fit"]
        target = domains[str(task.target)]
        io.atomic_tensor(p4.batch_path(root, task),
                         io.batches(fit, target, task.seed, plan["model"]["adapt_rounds"],
                                    plan["model"]["target_batch"]))


def reuse(root, plan):
    ref = plan["setting"]["reference"]
    main, a0 = Path(ref["main_root"]), Path(ref["a0_root"])
    seals = {}
    for base, prefix, name in ((main, "R4", "main"), (a0, "A0", "a0")):
        for stage in ("INPUT", "OBS", "FINAL"):
            value = io.verify(base, f"{prefix}_{stage}")
            if value["digest"] != ref[f"{name}_seals"][stage]:
                raise ValueError("Historical seal mismatch")
            seals[f"{name}_{stage}"] = value["digest"]
        if p3.read_json(base / "protocol/provenance.json")["git_commit"] != ref[f"{name}_execution_commit"]:
            raise ValueError("Historical execution version mismatch")
    old_plan = p3.read_json(main / "protocol/r4-plan.json")
    a0_plan = p3.read_json(a0 / "protocol/plan.json")
    compatibility = ("model", "adam", "weight", "side_attempts", "side_alpha_start",
                     "relocation_scan_points", "relocation_radius_sigma", "relocation_bisections",
                     "domains", "seeds", "directions")
    for previous in (old_plan, a0_plan):
        for key in compatibility:
            if plan[key] != previous[key]:
                raise ValueError(f"Historical scientific configuration differs: {key}")
        if previous["score"]["endpoints"] != plan["score"]["endpoints"]:
            raise ValueError("Historical score endpoints differ")
    for previous, method in ((old_plan, "B1"), (a0_plan, "A0")):
        for key in ("group", "density"):
            if previous["methods"][method][key] != plan["methods"][method][key]:
                raise ValueError("Historical method definition differs")
    copied = []
    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(source) != digest(destination):
            raise ValueError("Historical copy mismatch")
        copied.append(dict(source=str(source), destination=str(destination.relative_to(root)),
                           sha256=digest(source)))
    if p3.read_json(main / "inputs/domains.json") != p3.read_json(root / "inputs/domains.json"):
        raise ValueError("Historical ordered domains differ")
    if p3.read_json(main / "inputs/graph_hashes.json") != p3.read_json(root / "inputs/graph_hashes.json"):
        raise ValueError("Historical graph hashes differ")
    for source in plan["domains"]:
        for seed in plan["seeds"]:
            if p4.role(root, source, seed) != p4.role(main, source, seed):
                raise ValueError("Historical roles differ")
            from .shared import assert_nested_equal
            for split in ("fit", "validation"):
                assert_nested_equal(io.load(p4.labels_path(root, source, seed, split)),
                                    io.load(p4.labels_path(main, source, seed, split)))
            before = p4.source_dir(main, old_plan, source, seed)
            after = p4.source_dir(root, plan, source, seed)
            for path in before.rglob("*"):
                if path.is_file():
                    copy(path, after / path.relative_to(before))
    for task in p4.tasks(plan):
        if digest(p4.batch_path(root, task)) != digest(p4.batch_path(main, task)):
            # Torch archives may encode a different filename; compare tensor contents.
            from .shared import assert_nested_equal
            assert_nested_equal(io.load(p4.batch_path(root, task)), io.load(p4.batch_path(main, task)))
        for path in p4.score_dir(main, old_plan, task).glob("*"):
            if path.is_file():
                copy(path, p4.score_dir(root, plan, task) / path.name)
        for method, base, previous in (("B1", main, old_plan), ("A0", a0, a0_plan)):
            for path in p4.job_dir(base, previous, task, method).glob("*"):
                if path.is_file():
                    copy(path, p4.job_dir(root, plan, task, method) / path.name)
    io.write_json(root / "inputs/reuse.json", dict(seals=seals, files=copied))
