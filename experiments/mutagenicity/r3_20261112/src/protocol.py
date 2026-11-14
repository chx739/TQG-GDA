"""R3 grid, exact input roles and explicit target label gate."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from tsba.config import Config
from experiments.common.io import digest
from experiments.mutagenicity.r1_20261111.src.data import domain_index_path, label_offsets, reconstruct_graphs
from experiments.mutagenicity.r2_20261111.src.common import atomic_tensor, batches, seal, verify, write_json

from .source_roles import split_source_roles


@dataclass(frozen=True)
class Job:
    seed: int
    domain: int
    group: str
    density: str
    bandwidth: int
    direction: int
    weight: float
    epsilon: float | None

    @property
    def protect(self):
        return self.group[1] == "1"

    @property
    def key(self):
        return (f"seed{self.seed}_d{self.domain}/{self.group}_{self.density}"
                f"_h{self.bandwidth}_s{self.direction}_l{self.weight:g}_e{self.epsilon}")

    def config(self, plan):
        return Config(seed=self.seed, lambda_boundary=self.weight, **plan["adapt"])


def grid(plan):
    jobs = []
    for seed in plan["seeds"]:
        for domain in plan["domains"]:
            for group in plan["groups"]:
                epsilon = plan["main_epsilon"] if group[2] == "1" else None
                jobs.append(Job(seed, domain, group, "zero", 0, 0, plan["main_lambda"], epsilon))
                for density in ("learned", "kde"):
                    for bandwidth in plan["score"]["bandwidths"]:
                        for direction in (1, -1):
                            jobs.append(Job(seed, domain, group, density, bandwidth, direction,
                                            plan["main_lambda"], epsilon))
            for weight in plan["sensitivity_lambda"]:
                for epsilon in plan["sensitivity_epsilon"]:
                    if (weight, epsilon) == (plan["main_lambda"], plan["main_epsilon"]):
                        continue
                    for direction in (1, -1):
                        jobs.append(Job(seed, domain, "M11", "learned", 1, direction, weight, epsilon))
    counts = plan["counts"]
    if len({j.key for j in jobs}) != counts["main"] + counts["sensitivity"]:
        raise ValueError("R3 grid is incomplete or contains duplicate configurations")
    if len(jobs) * plan["adapt"]["adapt_rounds"] != counts["attempts"]:
        raise ValueError("Attempt count differs from registration")
    return jobs


def read_json(path):
    return json.loads(Path(path).read_text())


def roles_for(root, seed):
    return read_json(root / "roles" / f"seed{seed}.json")


def domain_ids(root, domain):
    return read_json(root / "inputs" / "domains.json")[str(domain)]


def read_values(path, offsets, ids, sha256):
    """Authenticate the same byte snapshot that supplies the requested records."""
    import hashlib
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Raw label file hash mismatch")
    return [int(data[start:start + length]) for start, length in (offsets[i] for i in ids)]


def prepare(root, repo, plan, inputs):
    raw = Path(inputs["raw_directory"])
    for name, spec in inputs["files"].items():
        if digest(raw / name) != spec["sha256"]:
            raise ValueError(f"Raw input hash mismatch: {name}")
    domains = {}
    for domain, spec in inputs["sega_domain_files"].items():
        path = domain_index_path(repo, spec["path"])
        if digest(path) != spec["sha256"]:
            raise ValueError(f"SEGA index hash mismatch: {domain}")
        domains[domain] = np.load(path, allow_pickle=False).astype(np.int64).tolist()
    joined = sorted(i for ids in domains.values() for i in ids)
    if joined != list(range(sum(plan["domain_counts"]))) or [
        len(domains[str(d)]) for d in range(4)
    ] != plan["domain_counts"]:
        raise ValueError("SEGA domain partition mismatch")

    records, _, graph_hashes = reconstruct_graphs(raw, expected_count=len(joined), expected_dims=(14, 3))
    label_path = raw / "Mutagenicity_graph_labels.txt"
    offsets = label_offsets(label_path)
    if len(offsets) != len(joined):
        raise ValueError("Label record count mismatch")
    label_sha = inputs["files"][label_path.name]["sha256"]
    source_ids = np.array(domains["0"], dtype=np.int64)
    source_values = read_values(label_path, offsets, source_ids, label_sha)
    mapping = {label: index for index, label in enumerate(sorted(set(source_values)))}
    if len(mapping) != 2:
        raise ValueError("Binary source labels required")
    # The reference allocator indexes by graph ID. All other slots are sentinels.
    source_labels = np.full(len(joined), -1, dtype=np.int64)
    source_labels[source_ids] = [mapping[label] for label in source_values]
    for seed in plan["seeds"]:
        reference = asdict(split_source_roles(source_ids, source_labels, plan["sega_ref_fraction"],
                                 plan["sega_fit_fraction"], 2, seed))
        fit = sorted(np.concatenate([reference["ref"], reference["fit"]]).tolist())
        validation = sorted(reference["cal"].tolist())
        if [len(fit), len(validation)] != plan["source_counts"] or sorted(fit + validation) != sorted(source_ids):
            raise ValueError(f"Unexpected SEGA role membership for seed {seed}")
        class_counts = {
            name: np.bincount(source_labels[ids], minlength=2).tolist()
            for name, ids in (("fit", fit), ("validation", validation))
        }
        write_json(root / "roles" / f"seed{seed}.json", dict(
            seed=seed, fit=fit, validation=validation, class_counts=class_counts,
            sega={name: ids.tolist() for name, ids in reference.items()}))
        for role, ids in (("fit", fit), ("validation", validation)):
            atomic_tensor(root / "source_labels" / f"seed{seed}_{role}.pt",
                          torch.tensor(source_labels[ids], dtype=torch.long))
        for domain in plan["domains"]:
            atomic_tensor(root / "batches" / f"seed{seed}_d{domain}.pt",
                          batches(fit, domains[str(domain)], seed,
                                  plan["adapt"]["adapt_rounds"], plan["adapt"]["target_batch"]))

    atomic_tensor(root / "inputs" / "graphs.pt", records)
    write_json(root / "inputs" / "graph_hashes.json", graph_hashes)
    write_json(root / "inputs" / "domains.json", domains)
    write_json(root / "inputs" / "label_records.json", dict(
        path=str(label_path), offsets=offsets, sha256=label_sha, mapping=mapping))
    write_json(root / "inputs" / "label_access.json", dict(role="source_all", ids=source_ids.tolist()))
    write_json(root / "inputs" / "jobs.json", [asdict(j) for j in grid(plan)])


def target_values(root, domains):
    """The only target-label reader in R3. Verify frozen evidence before access."""
    if tuple(domains) != (1, 3):
        raise PermissionError("R3 evaluation covers exactly domains 1 and 3")
    observation = verify(root, "R3_OBS")
    files = observation["files"]
    plan_path = "protocol/r3-plan.json"
    meta_path = "inputs/label_records.json"
    required = {plan_path, meta_path, "inputs/domains.json", "replay.json", "seals/R3_INPUT.json"}
    if not required <= files.keys():
        raise PermissionError("R3 plan, label metadata and replay must be sealed")
    plan = read_json(root / plan_path)
    jobs = grid(plan)
    attempts = plan["adapt"]["adapt_rounds"]
    replay = read_json(root / "replay.json")
    if (len(jobs) != 520 or attempts != 30 or replay.get("trajectories") != 520
            or replay.get("endpoints_replayed") != len(jobs) * len(plan["endpoints"])):
        raise PermissionError("R3 full-grid replay is incomplete")
    for job in jobs:
        prefix = f"trajectories/{job.key}/"
        required.add(prefix + "complete.json")
        required.update(prefix + f"round{i:02d}.pt" for i in range(attempts + 1))
    for seed in plan["seeds"]:
        required.update(f"source/seed{seed}/{name}" for name in ("source.pt", "frozen.pt", "checks.json"))
        for domain in domains:
            required.add(f"baselines/seed{seed}_d{domain}.pt")
            for h in plan["score"]["bandwidths"]:
                prefix = f"scores/seed{seed}_d{domain}_h{h}/"
                required.update(prefix + name for name in ("complete.json", "epoch0800.pt", "diagnostics.pt"))
    if not required <= files.keys():
        raise PermissionError("R3 observations are missing registered models, rounds or predictions")
    meta = read_json(root / meta_path)
    result = {}
    for domain in domains:
        ids = domain_ids(root, domain)
        raw = read_values(meta["path"], meta["offsets"], ids, meta["sha256"])
        result[domain] = torch.tensor([meta["mapping"][str(label)] for label in raw], dtype=torch.long)
    return observation["digest"], result
