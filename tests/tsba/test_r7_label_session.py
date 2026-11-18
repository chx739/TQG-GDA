"""Generated R5/R7 evidence tests for scoped, cryptographically checked labels."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals


class ToyEvidence:
    """A complete tiny authenticated archive; no real graph or label files."""

    def __init__(self, directory, *, final_stage="R7_ORACLE"):
        base = Path(directory)
        self.repo, self.archive, self.root = (base / name for name in ("repo", "r5", "C"))
        self.repo.mkdir()
        self.archive.mkdir()
        self.root.mkdir()
        rawdir = base / "raw"
        rawdir.mkdir()
        self.raw = rawdir / "Generated_graph_labels.txt"
        self.raw.write_bytes(b"".join(f"{i % 2}\n".encode() for i in range(36)))
        self.domains = {str(d): list(range(d * 12, (d + 1) * 12)) for d in range(3)}
        indices = []
        for d in range(3):
            path = self.repo / f"domain{d}.npy"
            np.save(path, np.asarray(self.domains[str(d)], dtype=np.int64))
            indices.append(dict(domain=d, path=path.name, sha256=r5_inputs._sha(path.read_bytes()), count=12))
        old_plan = dict(dataset="Generated", setting=dict(id="toy", domain_indices=indices),
                        domains=[0, 1, 2], seeds=[7],
                        input=dict(raw_directory=str(rawdir), files={self.raw.name: r5_inputs._sha(self.raw.read_bytes())}))
        roles = dict(fit=list(range(8)), validation=list(range(8, 12)),
                     class_counts=dict(fit=[4, 4], validation=[2, 2]))
        payloads = {
            "protocol/plan.json": old_plan,
            "protocol/provenance.json": dict(git_commit="generated-source"),
            "inputs/domains.json": self.domains,
            "roles/M0/seed7.json": roles,
            "inputs/label_records.json": dict(path=str(self.raw), sha256=r5_inputs._sha(self.raw.read_bytes()),
                                              offsets=[[2 * i, 2] for i in range(36)], mapping={"0": 0, "1": 1}),
        }
        files = {}
        for name, value in payloads.items():
            path = self.archive / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            files[name] = r5_inputs._sha(path.read_bytes())
        self.val_path = self.archive / "source_labels/M0/seed7_validation.pt"
        self.val_path.parent.mkdir(parents=True)
        torch.save(torch.tensor([0, 1, 0, 1]), self.val_path)
        files[str(self.val_path.relative_to(self.archive))] = r5_inputs._sha(self.val_path.read_bytes())
        (self.archive / "seals").mkdir()
        seal_hashes, seal_digests = {}, {}
        for stage in ("INPUT", "OBS", "FINAL"):
            record = dict(stage=f"R5_{stage}", files=files if stage == "INPUT" else
                          {"protocol/plan.json": files["protocol/plan.json"]})
            record["digest"] = r5_inputs._identity(record, compact=True)
            path = self.archive / f"seals/R5_{stage}.json"
            path.write_text(json.dumps(record))
            seal_hashes[stage] = r5_inputs._sha(path.read_bytes())
            seal_digests[stage] = record["digest"]
        (self.repo / "registry.json").write_text(json.dumps([dict(
            setting="toy", archive_root=str(self.archive), execution_commit="generated-source",
            seal_file_sha256=seal_hashes, seal_digests=seal_digests)]))
        (self.repo / "actual.json").write_text(json.dumps(dict(toy=dict(
            sha256=files["protocol/plan.json"], config=old_plan))))
        plan = dict(target_split=deepcopy(r5_inputs._SPLIT), real_model_seeds=[7],
                    provenance=dict(r5_archive_index="registry.json", r5_actual_config="actual.json"))
        spec = dict(id="toy", tu_dataset="Generated", domain_counts=[12, 12, 12],
                    domain_index_pattern="domain{domain}.npy")
        spec["planned_adaptation_holdout_counts"] = [
            [len(x) for x in r5_inputs.split_target_ids(plan, "Generated", self.domains[str(d)])] for d in range(3)]
        plan["real_settings"] = [spec]
        self.manifests = []
        for target in (1, 2):
            context = r5_inputs._context(plan, "toy", 0, target, 7, self.repo)
            manifest = dict(format=r5_inputs._FORMAT, r7_plan=plan, setting="toy", source=0, target=target, seed=7,
                            ids=context["ids"], provenance=context["provenance"],
                            label_records=dict(relative_path=context["label_relative"],
                                               sha256=context["provenance"]["r5_files"][context["label_relative"]]))
            name = f"M0_to_M{target}/manifest.json"
            path = self.root / name
            path.parent.mkdir()
            path.write_text(json.dumps(manifest))
            self.manifests.append(name)
        (self.root / "input.bin").write_bytes(b"generated input")
        (self.root / "observed.bin").write_bytes(b"frozen generated predictions")
        (self.root / "oracle.bin").write_bytes(b"frozen generated oracle")
        if final_stage is not None:
            self.seal(final_stage)

    def seal(self, stage):
        members = dict(R7_INPUT=[*self.manifests, "input.bin"], R7_OBS=["observed.bin"], R7_ORACLE=["oracle.bin"])
        for name in seals.STAGES[:seals.STAGES.index(stage) + 1]:
            if not (self.root / "seals" / f"{name}.json").exists():
                seals.seal(self.root, name, members[name], expected=members[name])


class LabelSessionTests(unittest.TestCase):
    def test_identical_labels_with_two_full_chain_passes_for_the_whole_batch(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory)
            expected = {}
            for manifest in toy.manifests:
                for pool in ("adapt", "holdout", "validation"):
                    expected[manifest, pool] = r5_inputs.load_labels(
                        toy.root, manifest, pool, repository=toy.repo, authenticated_snapshot=True)
            with patch.object(seals, "digest", wraps=seals.digest) as hashes:
                with r5_inputs.evaluation_labels(toy.root, repository=toy.repo) as reader:
                    entry_hashes = hashes.call_count
                    self.assertGreater(entry_hashes, 0)
                    for repeat in range(3):
                        for manifest in toy.manifests:
                            for pool in ("adapt", "holdout", "validation"):
                                actual = reader.load(manifest, pool, authenticated_snapshot=True)
                                np.testing.assert_array_equal(actual, expected[manifest, pool])
                                self.assertFalse(actual.flags.writeable)
                    self.assertEqual(hashes.call_count, entry_hashes)
                self.assertEqual(hashes.call_count, 2 * entry_hashes)
            with self.assertRaises(PermissionError):
                reader.load(toy.manifests[0], "holdout")
            # Existing standalone authorization remains independent/full.
            with patch.object(seals, "verify_seal", wraps=seals.verify_seal) as verify:
                r5_inputs.load_labels(toy.root, toy.manifests[0], "holdout", repository=toy.repo)
                self.assertEqual([call.args[1] for call in verify.call_args_list],
                                 ["R7_INPUT", "R7_OBS", "R7_ORACLE"])

    def test_obs_permissions_do_not_upgrade_when_oracle_appears(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory, final_stage="R7_OBS")
            with self.assertRaises(FileNotFoundError):
                with r5_inputs.evaluation_labels(toy.root, repository=toy.repo):
                    self.fail("Missing ORACLE must fail before entering")
            with r5_inputs.evaluation_labels(toy.root, repository=toy.repo, stage="R7_OBS") as reader:
                expected = r5_inputs.load_labels(toy.root, toy.manifests[0], "adapt", repository=toy.repo)
                ids = json.loads((toy.root / toy.manifests[0]).read_text())["ids"]["adapt"]
                reads = []
                original_open = Path.open

                class SelectedRecordsOnly:
                    def __init__(self, stream):
                        self.stream, self.offset = stream, None

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.stream.close()

                    def seek(self, offset):
                        if offset not in {2 * i for i in ids}:
                            raise AssertionError("OBS attempted a record outside A")
                        self.offset = offset
                        return self.stream.seek(offset)

                    def read(self, size=-1):
                        if size != 2 or self.offset is None:
                            raise AssertionError("OBS attempted a whole-file/unscoped label read")
                        reads.append(self.offset // 2)
                        return self.stream.read(size)

                def guarded_open(path, *args, **kwargs):
                    stream = original_open(path, *args, **kwargs)
                    return SelectedRecordsOnly(stream) if path == toy.raw else stream

                with patch.object(Path, "open", guarded_open):
                    np.testing.assert_array_equal(reader.load(toy.manifests[0], "adapt"), expected)
                self.assertEqual(reads, ids)
                toy.seal("R7_ORACLE")
                for pool, snapshot in (("holdout", False), ("validation", False), ("adapt", True)):
                    with self.assertRaises(PermissionError):
                        reader.load(toy.manifests[0], pool, authenticated_snapshot=snapshot)
            with r5_inputs.evaluation_labels(toy.root, repository=toy.repo) as reader:
                self.assertGreater(len(reader.load(toy.manifests[0], "holdout")), 0)

    def test_manifest_and_seal_metadata_tampering_rejected_before_role_reads(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory)
            with r5_inputs.evaluation_labels(toy.root, repository=toy.repo) as reader:
                with patch.object(r5_inputs, "_context", wraps=r5_inputs._context) as context:
                    for path in (toy.root / toy.manifests[0], toy.root / "seals/R7_OBS.json"):
                        original = path.read_bytes()
                        path.write_bytes(original + b" ")
                        try:
                            with self.assertRaises(PermissionError):
                                reader.load(toy.manifests[0], "holdout")
                        finally:
                            path.write_bytes(original)
                        context.assert_not_called()
                    (toy.root / "unsealed.json").write_bytes((toy.root / toy.manifests[0]).read_bytes())
                    with self.assertRaises(PermissionError):
                        reader.load("unsealed.json", "holdout")
                    with self.assertRaises((ValueError, PermissionError)):
                        reader.load("../raw/Generated_graph_labels.txt", "holdout")
                    context.assert_not_called()

    def test_unrelated_sealed_artifact_mutation_aborts_scope_before_publication(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory)
            published = False
            with self.assertRaisesRegex(ValueError, "Sealed file has changed"):
                with r5_inputs.evaluation_labels(toy.root, repository=toy.repo) as reader:
                    reader.load(toy.manifests[0], "holdout")
                    (toy.root / "observed.bin").write_bytes(b"changed frozen predictions")
                    # Relevant bytes still authenticate, but the full exit scan
                    # must reject the batch before its results can be published.
                    reader.load(toy.manifests[1], "holdout")
                published = True
            self.assertFalse(published)
            with self.assertRaises(PermissionError):
                reader.load(toy.manifests[0], "holdout")

    def test_consumed_raw_and_source_label_bytes_are_not_cached(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory)
            with r5_inputs.evaluation_labels(toy.root, repository=toy.repo) as reader:
                reader.load(toy.manifests[0], "holdout")
                reader.load(toy.manifests[0], "validation")
                original = toy.raw.read_bytes()
                toy.raw.write_bytes(b"1\n" + original[2:])
                with self.assertRaisesRegex(PermissionError, "Raw label bytes differ"):
                    reader.load(toy.manifests[0], "holdout")
                with self.assertRaisesRegex(PermissionError, "Raw label bytes differ"):
                    reader.load(toy.manifests[0], "adapt", authenticated_snapshot=True)
                toy.raw.write_bytes(original)
                torch.save(torch.tensor([1, 0, 1, 0]), toy.val_path)
                with self.assertRaisesRegex(PermissionError, "changed R5 dependency"):
                    reader.load(toy.manifests[0], "validation")

    def test_cryptographically_sealed_but_wrong_pool_ids_still_rejected(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory, final_stage=None)
            path = toy.root / toy.manifests[0]
            manifest = json.loads(path.read_text())
            manifest["ids"]["holdout"] = list(manifest["ids"]["adapt"])
            path.write_text(json.dumps(manifest))
            toy.seal("R7_ORACLE")
            with r5_inputs.evaluation_labels(toy.root, repository=toy.repo) as reader:
                with self.assertRaisesRegex(PermissionError, "IDs differ"):
                    reader.load(toy.manifests[0], "holdout")

    def test_capability_cannot_be_serialized_or_used_in_another_process(self):
        with TemporaryDirectory() as directory:
            toy = ToyEvidence(directory)
            with seals.verified_seal_chain(toy.root, "R7_ORACLE") as capability:
                with self.assertRaises(TypeError):
                    pickle.dumps(capability)
                with patch.object(seals.os, "getpid", return_value=-1):
                    with self.assertRaises(PermissionError):
                        capability.read_bytes(toy.manifests[0], "R7_INPUT")
                self.assertTrue(capability.read_bytes(toy.manifests[0], "R7_INPUT"))


if __name__ == "__main__":
    unittest.main()
