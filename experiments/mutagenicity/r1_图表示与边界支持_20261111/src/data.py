"""R1 label-free TU reconstruction and narrowly scoped, audited label access.

The broker is an API boundary, not operating-system isolation. Scanning/hash of
raw label bytes is allowed; parsing values is restricted by ID and sealed phase.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch_geometric.utils import coalesce, one_hot, remove_self_loops

from experiments.common.io import digest, save_tensor, write_json


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rng_seed(protocol, purpose, model_seed, domain, fold, feature, head, rep=-1):
    key = [protocol, purpose, model_seed, domain, fold, feature, head, rep]
    raw = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def hash_order(ids, prefix):
    return sorted(map(int, ids), key=lambda i: (hashlib.sha256(f"{prefix}|{i}".encode()).hexdigest(), i))


def _verify(path, expected):
    if "bytes" in expected and Path(path).stat().st_size != expected["bytes"]:
        raise ValueError(f"Byte count mismatch: {path}")
    if digest(path) != expected["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def _export(repo, revision, name):
    return subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=repo)


def historical_inputs(repo, spec, historical_commit=None):
    revision = spec["historical_commit"] if historical_commit is None else historical_commit
    # The large-file history cleanup changed this commit ID. The manifest and
    # every role file are still authenticated against the original hashes.
    if historical_commit is None and revision == "a672fefe10aba413283a6fb184704f6fae75e1fa":
        if subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=repo,
                          stderr=subprocess.DEVNULL).returncode:
            revision = "822b0926515489491516d21035aa1c40d0bb1bc4"
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=repo, text=True).strip()
    content = _export(repo, revision, spec["manifest_path"])
    if hashlib.sha256(content).hexdigest() != spec["manifest_sha256"]:
        raise ValueError("Historical manifest SHA mismatch")
    return revision, content


def source_split(ids, raw_labels, seed):
    """Independent TSBA two-stage split; no SEGA algorithm roles."""
    if len(ids) != len(raw_labels) or len(set(ids)) != len(ids):
        raise ValueError("Invalid source IDs/labels")
    classes = sorted(set(map(int, raw_labels)))
    if len(classes) != 2:
        raise ValueError("Source requires exactly two classes")
    py_rng = random.Random(seed)
    torch_rng = torch.Generator(device="cpu").manual_seed(seed)
    result = {"seed": seed, "fit_ids": [], "validation_ids": [], "diagnostic_ids": [], "classes": {}}
    for label in classes:
        group = sorted(int(i) for i, y in zip(ids, raw_labels) if int(y) == label)
        if len(group) < 3:
            raise ValueError("At least three source graphs per class required")
        shuffled = list(group)
        py_rng.shuffle(shuffled)
        nval = max(1, min(len(group) - 2, round(0.10 * len(group))))
        validation, remaining = shuffled[:nval], sorted(shuffled[nval:])
        permutation = torch.randperm(len(remaining), generator=torch_rng).tolist()
        ordered = [remaining[i] for i in permutation]
        nfit = max(1, min(len(remaining) - 1, math.floor(.80 * len(remaining))))
        fit, diagnostic = ordered[:nfit], ordered[nfit:]
        record = dict(count=len(group), validation_shuffle=shuffled, remaining_ids=remaining,
                      fit_permutation=permutation, fit_ids=sorted(fit), validation_ids=sorted(validation),
                      diagnostic_ids=sorted(diagnostic))
        result["classes"][str(label)] = record
        for role, values in (("fit", fit), ("validation", validation), ("diagnostic", diagnostic)):
            result[f"{role}_ids"].extend(values)
    for role in ("fit", "validation", "diagnostic"):
        result[f"{role}_ids"].sort()
    groups = [set(result[f"{role}_ids"]) for role in ("fit", "validation", "diagnostic")]
    if set.union(*groups) != set(ids) or sum(map(len, groups)) != len(ids):
        raise ValueError("Source partition must be disjoint and exhaustive")
    result["roles_sha256"] = canonical_sha({r: result[r] for r in ("fit_ids", "validation_ids", "diagnostic_ids")})
    return result


def probe_folds(diagnostic, targets, protocol):
    def inner(budget, role, domain, fold, audit=()):
        # A sorted budget order gives the shared 147-position permutation a
        # stable meaning. Inner membership is independently hash ordered.
        budget = sorted(budget)
        ordered = hash_order(budget, f"{protocol}|inner|{role}|{domain}|{fold}")
        return dict(budget_ids=budget, train_ids=ordered[:118], validation_ids=ordered[118:], audit_ids=sorted(audit))
    ordered = hash_order(diagnostic, "tsba-validation-v1|source_partition|0|all")
    out = {"source": {}, "targets": {}}
    for fold in range(4):
        audit = ordered[fold::4]
        out["source"][str(fold)] = inner(set(diagnostic) - set(audit), "source", 0, fold, audit)
    for domain, target in targets.items():
        out["targets"][str(domain)] = {}
        for fold in range(4):
            budget = hash_order(target["A"], f"tsba-validation-v1|target_probe_train|{domain}|{fold}")[:147]
            out["targets"][str(domain)][str(fold)] = inner(budget, "target", int(domain), fold)
    return out


def reconstruct_graphs(raw, prefix="Mutagenicity", expected_count=None, expected_dims=None):
    """Match PyG 2.6.1 input transforms, without ever opening graph labels."""
    raw = Path(raw)
    def integer_file(suffix, ndmin=1):
        return torch.from_numpy(np.loadtxt(raw / f"{prefix}_{suffix}.txt", delimiter=",", dtype=np.int64, ndmin=ndmin))
    edges = integer_file("A", 2).t().contiguous() - 1
    batch = integer_file("graph_indicator") - 1
    if batch.numel() == 0 or batch.min().item() != 0 or not bool((batch[1:] >= batch[:-1]).all()):
        raise ValueError("Graph indicators must be contiguous ordered nonempty graph blocks")
    ngraphs = int(batch.max()) + 1
    if expected_count is not None and ngraphs != expected_count:
        raise ValueError("Wrong graph count")
    counts = torch.bincount(batch, minlength=ngraphs)
    if not bool((counts > 0).all()):
        raise ValueError("Empty graph")
    def categorical(suffix):
        values = integer_file(suffix, 2)
        values = values - values.min(dim=0).values
        return torch.cat([one_hot(column) for column in values.unbind(-1)], dim=-1)
    x, edge_attr = categorical("node_labels"), categorical("edge_labels")
    if x.shape[0] != batch.numel() or edge_attr.shape[0] != edges.shape[1]:
        raise ValueError("Node/edge feature count mismatch")
    if edges.numel() and (edges.min() < 0 or edges.max() >= batch.numel()):
        raise ValueError("Edge endpoint out of bounds")
    if not torch.equal(batch[edges[0]], batch[edges[1]]):
        raise ValueError("Cross-graph edge")
    edges, edge_attr = remove_self_loops(edges, edge_attr)
    edges, edge_attr = coalesce(edges, edge_attr, batch.numel())
    if expected_dims is not None and (x.shape[1], edge_attr.shape[1]) != tuple(expected_dims):
        raise ValueError("Unexpected node/edge dimensions")
    node_offset = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    graphs, size, records = [], [], []
    edge_graph = batch[edges[0]]
    for graph_id in range(ngraphs):
        start, stop = map(int, node_offset[graph_id:graph_id + 2])
        selected = edge_graph == graph_id
        local_edges = edges[:, selected] - start
        pairs = {tuple(pair) for pair in local_edges.t().tolist()}
        if any((b, a) not in pairs for a, b in pairs):
            raise ValueError(f"Non-bidirectional graph {graph_id}")
        undirected = len({(min(a, b), max(a, b)) for a, b in pairs})
        graph = dict(x=x[start:stop].clone(), edge_index=local_edges.clone(), edge_attr=edge_attr[selected].clone())
        graphs.append(graph)
        size.append([math.log1p(stop - start), math.log1p(undirected)])
        hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in graph.items()}
        records.append(dict(id=graph_id, nodes=stop-start, undirected_edges=undirected, input_sha256=hashes))
    return graphs, torch.tensor(size, dtype=torch.float64), records


def label_offsets(path):
    """Scan byte boundaries only. No decoding and no parsing of class values."""
    offsets = []
    with open(path, "rb", buffering=0) as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            offsets.append([start, len(line)])
    return offsets


class LabelBroker:
    def __init__(self, root, plan=None, *, raw_directory=None):
        self.root = Path(root)
        self.plan = plan
        self.roles = json.loads((self.root / "roles.json").read_text())
        self.info = json.loads((self.root / "inputs/broker.json").read_text())
        if raw_directory is not None:
            self.info["path"] = str(Path(raw_directory) / Path(self.info["path"]).name)
        if digest(self.info["path"]) != self.info["sha256"]:
            raise ValueError("Raw label bytes changed since offset indexing")
        self._verified = {}

    def _seal(self, minimum, token):
        if not isinstance(token, str) or len(token) != 64:
            raise PermissionError("Actual seal digest required")
        for stage in (("S1", "S2") if minimum == "S1" else ("S2",)):
            path = self.root / "seals" / f"{stage}.json"
            if not path.exists():
                continue
            seal = json.loads(path.read_text())
            content = {"stage": seal.get("stage"), "files": seal.get("files")}
            if seal.get("digest") != token or canonical_sha(content) != token or content["stage"] != stage:
                continue
            if not isinstance(content["files"], dict) or not content["files"]:
                raise PermissionError("Empty or malformed seal")
            if stage == "S2" and "seals/S1.json" not in content["files"]:
                raise PermissionError("S2 must seal S1 dependency")
            signature = []
            for rel, expected in sorted(content["files"].items()):
                file = (self.root / rel).resolve()
                if not file.is_relative_to(self.root.resolve()) or not file.is_file():
                    raise PermissionError("Invalid sealed file")
                stat = file.stat()
                signature.append((rel, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            signature = tuple(signature)
            if self._verified.get(token) != signature:
                for rel, expected in content["files"].items():
                    if digest(self.root / rel) != expected:
                        raise PermissionError(f"Sealed file modified: {rel}")
                self._verified[token] = signature
            return stage
        raise PermissionError("Missing, invalid, or incorrect-phase seal token")

    def read(self, ids, role, purpose, seal=None, seed=None, domain=None):
        ids = list(map(int, ids))
        event = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), role=role,
                     ids=ids, purpose=purpose, token=seal, seed=seed, domain=domain, permitted=False)
        try:
            if not purpose or len(set(ids)) != len(ids):
                raise PermissionError("Purpose and distinct IDs required")
            # Explicit final refusal applies before any stage or role interpretation.
            final = set().union(*(set(t["final"]) for t in self.roles["targets"].values()))
            if set(ids) & final or role == "final":
                raise PermissionError("Final class values are never permitted")
            if role == "source_allocator":
                if (self.root / "inputs/class_mapping.json").exists() or (self.root / "seals/S1.json").exists():
                    raise PermissionError("R0 allocator is closed after source roles are allocated")
                allowed = self.roles["source"]["domain_ids"]
            elif role in ("source_fit", "source_validation", "source_diagnostic"):
                roles = json.loads((self.root / "roles" / f"seed{int(seed)}.json").read_text())
                allowed = roles[role.removeprefix("source_") + "_ids"]
                if role == "source_diagnostic":
                    self._seal("S1", seal)
            elif role in ("target_A", "target_B"):
                self._seal("S1" if role == "target_A" else "S2", seal)
                allowed = self.roles["targets"][str(domain)][role[-1]]
            else:
                raise PermissionError(f"Unknown label role {role}")
            if not set(ids).issubset(set(allowed)):
                raise PermissionError("Requested IDs outside allowed role")
            values = []
            with open(self.info["path"], "rb", buffering=0) as handle:
                for graph_id in ids:
                    offset, length = self.info["offsets"][graph_id]
                    handle.seek(offset)
                    values.append(int(handle.read(length).strip()))
            if role != "source_allocator":
                mapping = json.loads((self.root / "inputs/class_mapping.json").read_text())["raw_to_class"]
                if any(str(y) not in mapping for y in values):
                    raise ValueError("Unknown class outside source mapping")
                values = [mapping[str(y)] for y in values]
            event["permitted"] = True
            return torch.tensor(values, dtype=torch.long)
        except Exception as exc:
            event["reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            log = self.root / "logs/label-access.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def domain_index_path(repo, recorded_path):
    """Resolve the archived SEGA split location to TSBA's identical data copy."""
    path = Path(recorded_path)
    if path.parent == Path("references/sega-gda/splits"):
        path = Path("data") / path.name
    return Path(repo) / path


def prepare_inputs(root, repo, plan, *, raw_directory=None, cache=None,
                   historical_repo=None, historical_commit=None):
    root, repo = Path(root), Path(repo)
    for directory in ("inputs", "roles", "logs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    spec = json.loads((repo / "experiments/mutagenicity/stage1-inputs.json").read_text())
    history = Path(historical_repo or repo).resolve()
    revision, manifest_bytes = historical_inputs(history, spec, historical_commit)
    with (root / "inputs/historical-manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
    old = json.loads(manifest_bytes)
    raw = Path(raw_directory or spec["raw_directory"]).resolve()
    for name, expected in spec["files"].items():
        _verify(raw / name, expected)
    cache_path = Path(cache or spec["cache"]["path"]).resolve()
    _verify(cache_path, spec["cache"])  # bytes only, never deserialize y
    domains = {}
    for domain, item in spec["sega_domain_files"].items():
        path = domain_index_path(repo, item["path"])
        _verify(path, item)
        ids = np.load(path, allow_pickle=False).astype(np.int64).reshape(-1).tolist()
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate domain IDs")
        domains[domain] = ids
    all_ids = [i for ids in domains.values() for i in ids]
    if sorted(all_ids) != list(range(4337)) or [len(domains[str(d)]) for d in range(4)] != [1084,1084,1084,1085]:
        raise ValueError("Domains must partition all 4337 graphs with expected counts")
    roles = dict(protocol_id=plan["protocol_id"], source=dict(domain_ids=domains["0"]), targets={}, domains=domains)
    for domain in ("1", "3"):
        prior = old["targets"][domain]
        exported = {}
        for role in ("target", "development", "final"):
            item = prior[role]
            content = _export(history, revision, item["path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Historical {domain}/{role} SHA mismatch")
            exported[role] = np.load(io.BytesIO(content), allow_pickle=False).astype(np.int64).reshape(-1).tolist()
            if len(exported[role]) != item["count"]:
                raise ValueError("Historical role count mismatch")
        development = exported["development"]
        ordered = sorted(development, key=lambda i: (hashlib.sha256(f"tqg-v2-diag-20261031|domain={domain}|id={i}".encode()).hexdigest(), i))
        if ordered[:271] != prior["diagnostic_A_ids"] or ordered[271:] != prior["diagnostic_B_ids"]:
            raise ValueError("Historical A/B content/order mismatch")
        if sorted(exported["target"]) != sorted(domains[domain]) or sorted(development + exported["final"]) != sorted(domains[domain]):
            raise ValueError("Target roles fail domain coverage/disjointness")
        target = dict(ids=domains[domain], A=ordered[:271], B=ordered[271:], final=exported["final"])
        if len(set(development + exported["final"])) != len(domains[domain]):
            raise ValueError("Overlapping target roles")
        target["query_ids"] = hash_order(target["A"], f"{plan['protocol_id']}|geometry|{domain}")[:64]
        roles["targets"][domain] = target
    write_json(root / "roles.json", roles)
    label_file = raw / "Mutagenicity_graph_labels.txt"
    offsets = label_offsets(label_file)
    if len(offsets) != 4337:
        raise ValueError("Wrong number of label byte offsets")
    write_json(root / "inputs/broker.json", dict(path=str(label_file), offsets=offsets,
               sha256=spec["files"][label_file.name]["sha256"], values_parsed=False))
    broker = LabelBroker(root, plan)
    source_ids = domains["0"]
    labels = broker.read(source_ids, "source_allocator", "R0 stratified role allocation only").tolist()
    mapping = {str(y): i for i, y in enumerate(sorted(set(labels)))}
    if len(mapping) != 2:
        raise ValueError("Source requires exactly two classes")
    for seed in plan["seeds"]:
        split = source_split(source_ids, labels, seed)
        actual = {r: len(split[f"{r}_ids"]) for r in ("fit", "validation", "diagnostic")}
        if actual != plan["source_split"]["expected_counts"]:
            raise ValueError(f"Source role counts disagree: {actual}")
        split["folds"] = probe_folds(split["diagnostic_ids"], roles["targets"], plan["protocol_id"])
        write_json(root / "roles" / f"seed{seed}.json", split)
    write_json(root / "inputs/class_mapping.json", dict(raw_to_class=mapping, source_ids_sha256=canonical_sha(source_ids)))
    # Full raw class vector is never cached or passed to the source trainer.
    del labels
    graphs, sizes, records = reconstruct_graphs(raw, expected_count=4337, expected_dims=(14,3))
    save_tensor(root / "inputs/graphs.pt", graphs)
    save_tensor(root / "inputs/size.pt", sizes)
    write_json(root / "inputs/graphs.json", records)
    write_json(root / "inputs/verification.json", dict(source_role_counts=plan["source_split"]["expected_counts"],
               graph_count=len(graphs), raw_hashes_verified=True, cache_deserialized=False,
               final_class_values_parsed=False, torch_version=torch.__version__,
               raw_directory=str(raw), cache_path=str(cache_path),
               historical_repo=str(history), historical_commit=revision))
    return roles
