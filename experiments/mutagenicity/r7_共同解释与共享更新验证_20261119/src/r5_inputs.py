"""Authenticated R5 inputs and a separate, seal-gated target label reader.

``load_inputs`` never opens the raw graph-label file.  Its JSON ``manifest``
must be published unchanged inside R7_INPUT.  Only ``load_labels`` opens that
file, after R7_OBS (adaptation labels) or R7_ORACLE (holdout labels) verifies.
Holdout arrays returned here are for the coordinator to retain until fitting
finishes; they must not be passed to an adaptation worker.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


_FORMAT = "r7-r5-inputs-v1"
_SPLIT = dict(hash="sha256", key_template="r7-v1|{tu_dataset}|{zero_based_tu_graph_id_decimal}",
              byte_order="big", holdout_fraction=[3, 10], rule="10*h < 3*2**256",
              retain_domain_order=True, holdout_inputs_for_adaptation=False, stratify_on_labels=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _identity(value, *, compact=False):
    options = dict(sort_keys=True, allow_nan=False)
    if compact:
        options["separators"] = (",", ":")
    return _sha(json.dumps(value, **options).encode())


def _inside(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise PermissionError("Expected a relative evidence path without traversal")
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PermissionError("Evidence path is absent or escapes its registered root")
    return path


def _repository(repository):
    return Path(repository).resolve() if repository is not None else Path(__file__).resolve().parents[4]


def _ids(values, total, name):
    if (not isinstance(values, list) or not values
            or any(type(i) is not int or not 0 <= i < total for i in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Invalid or duplicate {name} graph IDs")
    return values


def split_target_ids(plan, dataset, ids):
    """Apply the registered label-blind hash rule, retaining domain order."""
    if plan.get("target_split") != _SPLIT:
        raise ValueError("Unrecognized R7 target split; register a new reader for a new rule")
    adaptation, holdout = [], []
    for graph_id in ids:
        if type(graph_id) is not int or graph_id < 0:
            raise ValueError("Target IDs must be nonnegative Python integers")
        value = int.from_bytes(hashlib.sha256(f"r7-v1|{dataset}|{graph_id}".encode()).digest(), "big")
        (holdout if 10 * value < 3 * 2 ** 256 else adaptation).append(graph_id)
    return adaptation, holdout


def _context(plan, setting, source, target, seed, repository):
    """Authenticate consumed files, not unrelated historical score/label outputs."""
    setting_id = setting["id"] if isinstance(setting, dict) else setting
    matches = [s for s in plan["real_settings"] if s["id"] == setting_id]
    if len(matches) != 1 or (isinstance(setting, dict) and setting != matches[0]):
        raise ValueError("Setting is not the registered R7 setting")
    spec = matches[0]
    if (type(source) is not int or type(target) is not int or source == target
            or not 0 <= source < len(spec["domain_counts"])
            or not 0 <= target < len(spec["domain_counts"])
            or type(seed) is not int or seed not in plan["real_model_seeds"]):
        raise ValueError("Unregistered source, target or model seed")
    index_path = _inside(repository, plan["provenance"]["r5_archive_index"])
    actual_path = _inside(repository, plan["provenance"]["r5_actual_config"])
    index_bytes, actual_bytes = index_path.read_bytes(), actual_path.read_bytes()
    entries = [r for r in json.loads(index_bytes) if r["setting"] == setting_id]
    if len(entries) != 1:
        raise ValueError("R5 archive registry must identify one setting")
    entry = entries[0]
    root = Path(entry["archive_root"]).resolve()
    consumed, seals = {}, {}
    for stage in ("INPUT", "OBS", "FINAL"):
        relative = f"seals/R5_{stage}.json"
        blob = _inside(root, relative).read_bytes()
        if _sha(blob) != entry["seal_file_sha256"][stage]:
            raise PermissionError("R5 seal bytes differ from the registered archive")
        seal = json.loads(blob)
        if (seal.get("stage") != f"R5_{stage}"
                or seal.get("digest") != entry["seal_digests"][stage]
                or _identity({k: seal[k] for k in ("stage", "files")}, compact=True) != seal["digest"]):
            raise PermissionError("R5 seal identity is invalid")
        seals[stage] = seal
        consumed[relative] = _sha(blob)

    def read(relative, stage="OBS"):
        blob = _inside(root, relative).read_bytes()
        actual = _sha(blob)
        if seals[stage]["files"].get(relative) != actual:
            raise PermissionError(f"Unsealed or changed R5 dependency: {relative}")
        # An input present in both seals must retain its original identity.
        if relative in seals["INPUT"]["files"] and seals["INPUT"]["files"][relative] != actual:
            raise PermissionError(f"R5 input identity changed: {relative}")
        consumed[relative] = actual
        return blob

    old_plan_bytes = read("protocol/plan.json", "INPUT")
    old_plan = json.loads(old_plan_bytes)
    actual = json.loads(actual_bytes)[setting_id]
    if actual["sha256"] != _sha(old_plan_bytes) or actual["config"] != old_plan:
        raise PermissionError("Archived R5 plan differs from recorded actual configuration")
    execution = json.loads(read("protocol/provenance.json", "INPUT"))
    if execution["git_commit"] != entry["execution_commit"]:
        raise PermissionError("R5 execution commit differs from archive registry")
    if (old_plan["dataset"] != spec["tu_dataset"] or old_plan["setting"]["id"] != setting_id
            or source not in old_plan["domains"] or target not in old_plan["domains"]
            or seed not in old_plan["seeds"]):
        raise ValueError("R5 and R7 task identities differ")
    total = sum(spec["domain_counts"])
    domains = json.loads(read("inputs/domains.json", "INPUT"))
    if set(domains) != {str(i) for i in range(len(spec["domain_counts"]))}:
        raise ValueError("Unexpected archived domain keys")
    indices = {}
    for number, count in enumerate(spec["domain_counts"]):
        ids = _ids(domains[str(number)], total, "domain")
        registered = [r for r in old_plan["setting"]["domain_indices"] if r["domain"] == number]
        if len(registered) != 1:
            raise ValueError("Missing or duplicate R5 domain registration")
        registered = registered[0]
        relative = spec["domain_index_pattern"].format(domain=number)
        blob = _inside(repository, relative).read_bytes()
        array = np.load(io.BytesIO(blob), allow_pickle=False)
        if (registered["path"] != relative or registered["sha256"] != _sha(blob)
                or registered["count"] != count or len(ids) != count
                or array.ndim != 1 or not np.issubdtype(array.dtype, np.integer)
                or array.tolist() != ids):
            raise PermissionError("Domain index order or identity differs from R5")
        indices[relative] = _sha(blob)
        a, h = split_target_ids(plan, spec["tu_dataset"], ids)
        if [len(a), len(h)] != spec["planned_adaptation_holdout_counts"][number]:
            raise ValueError("Hash fold counts differ from the registered R7 plan")
    if sorted(i for ids in domains.values() for i in ids) != list(range(total)):
        raise ValueError("Archived domains are not a disjoint exhaustive partition")
    roles = json.loads(read(f"roles/M{source}/seed{seed}.json", "INPUT"))
    fit = _ids(roles["fit"], total, "source-fit")
    validation = _ids(roles["validation"], total, "source-validation")
    if sorted(fit + validation) != sorted(domains[str(source)]):
        raise ValueError("Source roles overlap or do not partition the source domain")
    adaptation, holdout = split_target_ids(plan, spec["tu_dataset"], domains[str(target)])
    label_relative = "inputs/label_records.json"
    label_meta = json.loads(read(label_relative, "INPUT"))
    expected_label_path = Path(old_plan["input"]["raw_directory"]) / f"{spec['tu_dataset']}_graph_labels.txt"
    if Path(label_meta["path"]).resolve() != expected_label_path.resolve():
        raise PermissionError("Raw label path differs from the registered R5 input")
    label_spec = old_plan["input"]["files"][expected_label_path.name]
    expected_hash = label_spec["sha256"] if isinstance(label_spec, dict) else label_spec
    if label_meta["sha256"] != expected_hash:
        raise PermissionError("Raw label identity differs from the registered R5 input")
    mapping = label_meta["mapping"]
    if sorted(mapping.values()) != [0, 1] or any(not isinstance(k, str) for k in mapping):
        raise ValueError("Unexpected R5 binary class mapping")
    offsets = label_meta["offsets"]
    previous = 0
    if len(offsets) != total:
        raise ValueError("Label offset count differs from graph count")
    for offset in offsets:
        if (not isinstance(offset, list) or len(offset) != 2 or any(type(v) is not int for v in offset)
                or offset[0] < previous or offset[1] <= 0):
            raise ValueError("Invalid ordered label record offsets")
        previous = sum(offset)
    task_ids = dict(source=fit, validation=validation, adapt=adaptation, holdout=holdout)
    provenance = dict(archive_root=str(root), execution_commit=entry["execution_commit"],
                      r5_seal_digests=deepcopy(entry["seal_digests"]), r5_files=consumed,
                      repository_files={str(index_path.relative_to(repository)): _sha(index_bytes),
                                        str(actual_path.relative_to(repository)): _sha(actual_bytes), **indices})
    return dict(spec=spec, old_plan=old_plan, root=root, read=read, roles=roles, ids=task_ids,
                domains=domains, label_meta=label_meta, label_relative=label_relative,
                total=total, provenance=provenance)


def _tensor(value, shape, name, *, integer=False):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"Invalid {name} tensor shape")
    if integer and value.dtype != torch.int64:
        raise ValueError(f"{name} must be int64")
    if not integer and not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    value = value.detach().cpu().to(torch.int64 if integer else torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")
    return value


def _numpy(value):
    result = value.detach().cpu().numpy().copy()
    result.setflags(write=False)
    return result


def load_inputs(plan, setting, source, target, seed, *, repository=None):
    """Return frozen NumPy arrays, ordered IDs, provenance and a sealable manifest.

    ``source_y`` is the only label array loaded or returned.  Source head
    coefficients are promoted from their archived dtype to float64 before
    reconstructing z0, as registered in R7.  The input manifest contains no
    target label values and can be serialized with ordinary JSON.
    """
    repository = _repository(repository)
    context = _context(plan, setting, source, target, seed, repository)
    read, old_plan, ids = context["read"], context["old_plan"], context["ids"]
    config_hash = _identity({k: v for k, v in old_plan.items() if k != "workers"})
    directory = f"source/{old_plan['dataset']}/M{source}/seed{seed}/{config_hash}"
    checkpoint = torch.load(io.BytesIO(read(directory + "/source.pt")), map_location="cpu", weights_only=True)
    frozen = torch.load(io.BytesIO(read(directory + "/frozen.pt")), map_location="cpu", weights_only=True)
    history = checkpoint["history"]
    if (len(history) != old_plan["model"]["source_epochs"]
            or [r["epoch"] for r in history] != list(range(len(history)))
            or any(not np.isfinite(r["source_validation"]["ce"]) for r in history)
            or checkpoint["selected_epoch"] != min(history, key=lambda r: r["source_validation"]["ce"])["epoch"]):
        raise ValueError("Source checkpoint selection differs from registered source-validation rule")
    dimension = plan["features"]["output_dimension_real"]
    for key, old_key in (("conv", "encoder"), ("hidden", "hidden"), ("num_layers", "num_layers"),
                         ("out_dim", "emb_dim"), ("dropout", "dropout")):
        if checkpoint["encoder_spec"][key] != old_plan["model"][old_key]:
            raise ValueError("Source encoder architecture differs from the archived plan")
    if (checkpoint["encoder_spec"]["out_dim"] != dimension
            or plan["features"]["raw_cache_key"] != "h"
            or plan["features"]["source_logits_dtype"] != "float64"):
        raise ValueError("Unsupported R7 frozen representation contract")
    h = _tensor(frozen["h"], (context["total"], dimension), "frozen embeddings")
    head_state = checkpoint[plan["features"]["source_model_state_key"]]
    weight_key, bias_key = plan["features"]["source_head_state_keys"]
    weight = _tensor(head_state[weight_key], (2, dimension), "source weight")
    bias = _tensor(head_state[bias_key], (2,), "source bias")
    labels = torch.load(io.BytesIO(read(f"source_labels/M{source}/seed{seed}_fit.pt", "INPUT")),
                        map_location="cpu", weights_only=True)
    labels = _tensor(labels, (len(ids["source"]),), "source labels", integer=True)
    if labels.unique().tolist() != [0, 1]:
        raise ValueError("Both source-fit classes must be represented")
    if torch.bincount(labels, minlength=2).tolist() != context["roles"]["class_counts"]["fit"]:
        raise ValueError("Source-fit labels disagree with sealed role counts")
    baseline = torch.load(io.BytesIO(read(f"baselines/M{source}_to_M{target}/seed{seed}.pt")),
                          map_location="cpu", weights_only=True)
    z0 = h @ weight.T + bias
    parity = {}
    for pool, selected in (("source_fit", ids["source"]), ("source_validation", ids["validation"]),
                            ("target", context["domains"][str(target)])):
        archived_logp = _tensor(baseline[pool], (len(selected), 2), f"B0 {pool}")
        probability = z0[selected].softmax(-1)
        archived_probability = archived_logp.exp()
        torch.testing.assert_close(probability, archived_probability, rtol=1e-10, atol=1e-12)
        if not torch.equal(probability.argmax(1), archived_logp.argmax(1)):
            raise ValueError(f"Reconstructed B0 classes differ in {pool}")
        parity[pool] = dict(rows=len(selected), probability_max_absolute_error=float(
            (probability - archived_probability).abs().max()))
    provenance = context["provenance"]
    provenance.update(source_config_hash=config_hash, selected_source_epoch=checkpoint["selected_epoch"],
                      b0_parity=parity, b0_rtol=1e-10, b0_atol=1e-12,
                      id_hashes={name: _identity(value, compact=True) for name, value in ids.items()},
                      target_labels_loaded=False)
    manifest = dict(format=_FORMAT, r7_plan=deepcopy(plan), setting=context["spec"]["id"],
                    source=source, target=target, seed=seed, ids=deepcopy(ids),
                    provenance=deepcopy(provenance),
                    label_records=dict(relative_path=context["label_relative"],
                                       sha256=provenance["r5_files"][context["label_relative"]]))
    result = dict(source_y=_numpy(labels), source_head_weight=_numpy(weight), source_head_bias=_numpy(bias),
                  provenance=deepcopy(provenance), manifest=manifest)
    for name, selected in ids.items():
        result[name + "_ids"] = list(selected)
        result[name + "_h"] = _numpy(h[selected])
        result[name + "_z0"] = _numpy(z0[selected])
    return result


def load_labels(root, manifest_relative_path, pool, *, repository=None, authenticated_snapshot=False):
    """Read only the sealed task's A or H labels after the required R7 gates.

    No arbitrary IDs or raw path are accepted.  The independent R7 verifier
    validates stage completeness and chaining. Before R7_ORACLE, reads seek
    only to the authenticated adaptation offsets. After R7_ORACLE, whole-file
    authentication is available to verify the previously observed A values.
    """
    from .seals import verify_seal

    if pool not in ("adapt", "holdout", "validation"):
        raise ValueError("Target label pool must be 'adapt' or 'holdout'")
    root = Path(root).resolve()
    input_seal = verify_seal(root, "R7_INPUT")
    verify_seal(root, "R7_OBS")
    if pool in ("holdout", "validation") or authenticated_snapshot:
        verify_seal(root, "R7_ORACLE")
    path = _inside(root, manifest_relative_path)
    relative = str(path.relative_to(root))
    blob = path.read_bytes()
    if input_seal["files"].get(relative) != _sha(blob):
        raise PermissionError("Label request manifest was not sealed in R7_INPUT")
    manifest = json.loads(blob)
    if manifest.get("format") != _FORMAT:
        raise ValueError("Unknown R7 input manifest format")
    context = _context(manifest["r7_plan"], manifest["setting"], manifest["source"],
                       manifest["target"], manifest["seed"], _repository(repository))
    if manifest["ids"] != context["ids"]:
        raise PermissionError("Sealed label request IDs differ from the registered task and hash folds")
    provenance = manifest["provenance"]
    for key in ("archive_root", "execution_commit", "r5_seal_digests", "repository_files"):
        if provenance[key] != context["provenance"][key]:
            raise PermissionError("Label request provenance differs from authenticated R5 dependencies")
    expected = dict(relative_path=context["label_relative"],
                    sha256=context["provenance"]["r5_files"][context["label_relative"]])
    if manifest["label_records"] != expected:
        raise PermissionError("Label metadata path or identity differs from sealed R7 inputs")
    if pool == "validation":
        ids = context["ids"][pool]
        arch = f"source_labels/M{manifest['source']}/seed{manifest['seed']}_validation.pt"
        data = torch.load(io.BytesIO(context['read'](arch, 'INPUT')), map_location='cpu', weights_only=True)
        labels = _tensor(data, (len(ids),), "source validation labels", integer=True)
        if torch.bincount(labels, minlength=2).tolist() != context["roles"]["class_counts"][pool]:
            raise ValueError("Source validation labels disagree with sealed role counts")
        return _numpy(labels)
    metadata = context["label_meta"]
    records = [metadata["offsets"][graph_id] for graph_id in context["ids"][pool]]
    with Path(metadata["path"]).open("rb", buffering=0) as stream:
        if pool == "holdout" or authenticated_snapshot:
            raw = stream.read()
            if _sha(raw) != metadata["sha256"]:
                raise PermissionError("Raw label bytes differ from the authenticated R5 snapshot")
            selected = [raw[start:start + length] for start, length in records]
        else:
            selected = []
            for start, length in records:
                stream.seek(start)
                selected.append(stream.read(length))
    result = []
    for payload, (_, expected_length) in zip(selected, records):
        if len(payload) != expected_length:
            raise ValueError("Raw label record is truncated")
        value = str(int(payload))
        if value not in metadata["mapping"]:
            raise ValueError("Raw target label is outside the fixed source class mapping")
        result.append(metadata["mapping"][value])
    labels = np.asarray(result, dtype=np.int64)
    labels.setflags(write=False)
    return labels
