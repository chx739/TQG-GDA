"""R4 registration and task-scoped inputs, authenticated against R3."""
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from tsba.config import Config
from tsba.utils import identity
from ...archive import digest
from ..data import label_offsets, reconstruct_graphs
from ..r2.common import atomic_tensor, batches, verify, write_json
from ..r3.protocol import Job, read_json, read_values, reference_split


@dataclass(frozen=True)
class Task:
    source: int
    target: int
    seed: int

    @property
    def key(self):
        return f"M{self.source}_to_M{self.target}/seed{self.seed}"

    @property
    def source_key(self):
        return f"M{self.source}/seed{self.seed}"


def tasks(plan):
    if sorted(tuple(pair) for pair in plan["directions"]) != list(permutations(plan["domains"], 2)):
        raise ValueError("R4 requires each directed cross-domain task exactly once")
    return [Task(s, t, seed) for s, t in plan["directions"] for seed in plan["seeds"]]


def config_hash(plan):
    return identity({k: v for k, v in plan.items() if k != "workers"})


def config(plan, seed):
    return Config(**{**plan["model"], "seed": seed, "lambda_boundary": plan["weight"]})


def source_dir(root, plan, source, seed):
    return root / "source" / plan["dataset"] / f"M{source}" / f"seed{seed}" / config_hash(plan)


def job_dir(root, plan, task, method):
    return root / "trajectories" / plan["dataset"] / task.key / method / config_hash(plan)


def score_dir(root, plan, task):
    return root / "scores" / plan["dataset"] / task.key / "learned" / config_hash(plan)


def role(root, source, seed):
    return read_json(root / "roles" / f"M{source}" / f"seed{seed}.json")


def labels_path(root, source, seed, split):
    return root / "source_labels" / f"M{source}" / f"seed{seed}_{split}.pt"


def domain(root, number):
    return read_json(root / "inputs/domains.json")[str(number)]


def batch_path(root, task):
    return root / "batches" / (task.key + ".pt")


def r3_job(plan, task, method):
    spec = plan["methods"][method]
    group, density = spec["group"], spec["density"]
    return Job(task.seed, task.target, group, density, int(density != "zero"),
               int(density != "zero"), plan["weight"], plan["epsilon"] if group[2] == "1" else None)


def split_domain(ids, values, mapping, splitter, seed, total, fractions):
    scoped = np.full(total, -1, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    scoped[ids] = [mapping[str(value)] for value in values]
    allocated = asdict(splitter(ids, scoped, *fractions, 2, seed))
    fit = sorted(np.concatenate((allocated["ref"], allocated["fit"])).tolist())
    validation = sorted(allocated["cal"].tolist())
    if sorted(fit + validation) != sorted(ids.tolist()):
        raise ValueError("Source roles must partition the current source")
    counts = {key: np.bincount(scoped[indices], minlength=2).tolist()
              for key, indices in (("fit", fit), ("validation", validation))}
    if not all(all(group) for group in counts.values()):
        raise ValueError("Both classes must be represented in each source role")
    metadata = dict(fit=fit, validation=validation, class_counts=counts,
                    sega={key: ids.tolist() for key, ids in allocated.items()})
    return metadata, torch.tensor(scoped[fit]), torch.tensor(scoped[validation])


def prepare(root, repo, plan, reference):
    inputs_seal = verify(reference, "R3_INPUT")
    if inputs_seal["digest"] != plan["r3_inputs_digest"]:
        raise ValueError("Unexpected historical R3 input identity")
    inputs = read_json(reference / "protocol/stage1-inputs.json")
    raw = Path(inputs["raw_directory"])
    for filename, expected in inputs["files"].items():
        if digest(raw / filename) != expected["sha256"]:
            raise ValueError(f"R3 raw file hash mismatch: {filename}")
    domains = {}
    for key, item in inputs["sega_domain_files"].items():
        path = repo / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"R3 domain index hash mismatch: {key}")
        domains[key] = np.load(path, allow_pickle=False).astype(np.int64).tolist()
    total = sum(plan["domain_counts"])
    if sorted(i for ids in domains.values() for i in ids) != list(range(total)):
        raise ValueError("Domains must partition every original graph ID")
    if [len(domains[str(d)]) for d in plan["domains"]] != plan["domain_counts"]:
        raise ValueError("Domain sizes differ from registration")
    if domains != read_json(reference / "inputs/domains.json"):
        raise ValueError("R3 domain membership or order changed")
    records, _, hashes = reconstruct_graphs(raw, expected_count=total, expected_dims=(14, 3))
    if hashes != read_json(reference / "inputs/graph_hashes.json"):
        raise ValueError("R3 graph structure or feature tensors changed")
    atomic_tensor(root / "inputs/graphs.pt", records)
    write_json(root / "inputs/graph_hashes.json", hashes)
    write_json(root / "inputs/domains.json", domains)
    write_json(root / "inputs/r3-inputs.json", inputs)
    label_path = raw / "Mutagenicity_graph_labels.txt"
    offsets = label_offsets(label_path)
    r3_meta = read_json(reference / "inputs/label_records.json")
    if len(offsets) != total or offsets != r3_meta["offsets"]:
        raise ValueError("R3 label offsets changed")
    mapping = r3_meta["mapping"]
    label_sha = inputs["files"][label_path.name]["sha256"]
    write_json(root / "inputs/label_records.json", dict(
        path=str(label_path), offsets=offsets, sha256=label_sha, mapping=mapping))
    splitter = reference_split(repo)
    for source in plan["domains"]:
        ids = domains[str(source)]
        # Each allocator handles one source role. Workers receive split files only.
        values = read_values(label_path, offsets, ids, label_sha)
        for seed in plan["seeds"]:
            row, fit_y, val_y = split_domain(ids, values, mapping, splitter, seed, total,
                                            (plan["sega_ref_fraction"], plan["sega_fit_fraction"]))
            write_json(root / "roles" / f"M{source}" / f"seed{seed}.json", row)
            atomic_tensor(labels_path(root, source, seed, "fit"), fit_y)
            atomic_tensor(labels_path(root, source, seed, "validation"), val_y)
    registry = []
    for task in tasks(plan):
        fit = role(root, task.source, task.seed)["fit"]
        target = domains[str(task.target)]
        cfg = config(plan, task.seed)
        if cfg.source_adapt_batch != cfg.target_batch:
            raise ValueError("Registered batch schedules require equal pool batch sizes")
        atomic_tensor(batch_path(root, task),
                      batches(fit, target, task.seed, cfg.adapt_rounds, cfg.target_batch))
        registry.append(dict(asdict(task), dataset=plan["dataset"], config_hash=config_hash(plan),
                             source_model=str(source_dir(root, plan, task.source, task.seed).relative_to(root)),
                             score=str(score_dir(root, plan, task).relative_to(root)),
                             jobs={m: str(job_dir(root, plan, task, m).relative_to(root))
                                   for m in plan["methods"] if m != "B0"}))
    write_json(root / "inputs/registry.json", registry)
    write_json(root / "inputs/source_label_access.json",
               dict(purpose="SEGA source-role allocation", source_domains=plan["domains"]))
    write_json(root / "inputs/reuse.json", dict(
        r3_input_seal=inputs_seal["digest"], r3_path=str(reference),
        verified_inputs=["raw files", "ordered domain indices", "per-graph structure/features"],
        reused_source_models=0, reused_score_models=0, reused_trajectories=0,
        policy="Fresh fits for every registered task under the same R4 runner"))


def expected_predictions(root, plan):
    for task in tasks(plan):
        yield root / "baselines" / (task.key + ".pt")
        for method in plan["methods"]:
            if method == "B0":
                continue
            path = job_dir(root, plan, task, method)
            yield path / "complete.json"
            for step in range(plan["model"]["adapt_rounds"] + 1):
                yield path / f"round{step:02d}.pt"


def evaluation_labels(root):
    obs = verify(root, "R4_OBS")
    required = {"protocol/r4-plan.json", "seals/R4_INPUT.json", "replay.json",
                "inputs/label_records.json", "inputs/domains.json", "inputs/registry.json"}
    if not required <= obs["files"].keys():
        raise PermissionError("Incomplete R4 observation metadata")
    plan = read_json(root / "protocol/r4-plan.json")
    summary = read_json(root / "replay.json")
    expected = plan["counts"]
    if any(summary[key] != expected[key] for key in ("sources", "scores", "trajectories", "attempts")):
        raise PermissionError("R4 replay counts differ from registration")
    if (summary.get("predictions_replayed") != expected["trajectories"] + expected["attempts"]
            or summary.get("score_endpoints") != expected["scores"] * len(plan["score"]["endpoints"])
            or sum(summary.get("side_statuses", {}).values()) != expected["trajectories"] * len(plan["side_attempts"])
            or summary.get("side_predictions_replayed", 0) !=
            3 * summary.get("side_statuses", {}).get("available", 0)):
        raise PermissionError("R4 replay omits required predictions or score endpoints")
    required.update(str(path.relative_to(root)) for path in expected_predictions(root, plan))
    for source in plan["domains"]:
        for seed in plan["seeds"]:
            prefix = source_dir(root, plan, source, seed)
            required.update(str((prefix / name).relative_to(root))
                            for name in ("source.pt", "frozen.pt", "complete.json"))
    for task in tasks(plan):
        prefix = score_dir(root, plan, task)
        required.update(str((prefix / name).relative_to(root))
                        for name in ("epoch0800.pt", "diagnostics.pt", "complete.json"))
    if not required <= obs["files"].keys():
        raise PermissionError("R4 observations omit registered models or prediction records")
    meta = read_json(root / "inputs/label_records.json")
    values = {}
    for number in plan["domains"]:
        raw = read_values(meta["path"], meta["offsets"], domain(root, number), meta["sha256"])
        values[number] = torch.tensor([meta["mapping"][str(y)] for y in raw], dtype=torch.long)
    return obs["digest"], values
