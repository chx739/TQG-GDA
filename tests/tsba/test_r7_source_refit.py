"""Generated-graph engineering checks; no archived or raw research data access."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from scripts.run_task import synthetic_task
from tsba.config import Config
from tsba.encoder import GraphEncoder
from tsba.pretrain import pretrain_encoder
from tsba.utils import set_seed
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import source_refit_inputs as refit
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import matched_tsba
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.seals import seal


REPOSITORY = Path(__file__).resolve().parents[2]


def _tensor_bytes(value):
    handle = io.BytesIO()
    torch.save(value, handle)
    return handle.getvalue()


class ToyArchive:
    """A generated R5-shaped data context with read tracing, never a real archive."""

    def __init__(self, root):
        self.root = root / "toy-r5"
        (self.root / "seals").mkdir(parents=True)
        data = synthetic_task(41)
        graphs = data.source_train.graphs + data.source_validation.graphs + data.target.graphs
        records = [dict(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in graphs]
        self.ids = dict(source=list(range(12)), validation=list(range(12, 16)))
        self.domains = {"0": list(range(16)), "1": list(range(16, 21)), "2": list(range(21, 26))}
        self.roles = dict(fit=self.ids["source"], validation=self.ids["validation"],
                          class_counts=dict(fit=[6, 6], validation=[2, 2]))
        cfg = Config(seed=0, adapter_width=16)
        self.old = dict(dataset="Generated", setting=dict(id="generated"), domains=[0, 1, 2], seeds=[103],
                        model=cfg.to_dict(), adam=dict(betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False,
                        maximize=False, foreach=None, capturable=False, differentiable=False, fused=None),
                        weight=1., solver={}, methods=dict(B1=dict(group="M00", density="zero"),
                                                          A0=dict(group="M00", density="learned")))
        self.spec = dict(id="generated", tu_dataset="Generated", domain_counts=[16, 5, 5])
        self.plan = dict(real_settings=[self.spec], real_model_seeds=[103],
                         source_refit=dict(recipe=refit._RECIPE, pooling="sum"),
                         execution=dict(threads_per_worker=1),
                         features=dict(output_dimension_real=16, raw_cache_key="h", source_logits_dtype="float64",
                                       source_model_state_key="state", source_head_state_keys=["head.weight", "head.bias"]))
        self.blobs = {
            "inputs/graphs.pt": _tensor_bytes(records),
            "roles/M0/seed103.json": json.dumps(self.roles).encode(),
            "source_labels/M0/seed103_fit.pt": _tensor_bytes(data.source_train.labels),
            "source_labels/M0/seed103_validation.pt": _tensor_bytes(data.source_validation.labels),
        }
        # The only raw labels here are generated engineering fixtures. They are
        # not read during preparation, source training or load_inputs.
        raw = b"".join(f"{i % 2}\n".encode() for i in range(26))
        self.label_file = root / "generated-labels.txt"
        self.label_file.write_bytes(raw)
        self.label_meta = dict(path=str(self.label_file), sha256=r5_inputs._sha(raw),
                               offsets=[[2 * i, 2] for i in range(26)], mapping={"0": 0, "1": 1})
        self.blobs["inputs/label_records.json"] = json.dumps(self.label_meta).encode()
        files = {name: r5_inputs._sha(blob) for name, blob in self.blobs.items()}
        seal_blob = json.dumps(dict(stage="R5_INPUT", files=files)).encode()
        (self.root / "seals/R5_INPUT.json").write_bytes(seal_blob)
        self.provenance = dict(archive_root=str(self.root), execution_commit="generated-only",
                               r5_seal_digests=dict(INPUT="toy", OBS="toy", FINAL="toy"), repository_files={},
                               r5_files={"seals/R5_INPUT.json": r5_inputs._sha(seal_blob),
                                         "inputs/label_records.json": files["inputs/label_records.json"]})
        self.reads = []

    def context(self, plan, setting, source, target, seed, repository):
        assert source == 0 and seed == 103 and target in (1, 2)
        ids = {**deepcopy(self.ids), "adapt": self.domains[str(target)][:3],
               "holdout": self.domains[str(target)][3:]}
        provenance = deepcopy(self.provenance)

        def read(path, stage="OBS"):
            self.reads.append((path, stage))
            # A source worker must never ask for a different source/target's labels.
            if path not in self.blobs:
                raise PermissionError(f"Forbidden fixture read: {path}")
            blob = self.blobs[path]
            provenance["r5_files"][path] = r5_inputs._sha(blob)
            return blob

        return dict(spec=deepcopy(self.spec), old_plan=deepcopy(self.old), root=self.root,
                    roles=deepcopy(self.roles), ids=ids, domains=deepcopy(self.domains), total=26,
                    read=read, provenance=provenance, label_relative="inputs/label_records.json",
                    label_meta=deepcopy(self.label_meta))


class SourceRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sum_is_exact_stock_and_mean_changes_only_pooling(self):
        kwargs = dict(in_dim=3, edge_dim=1, hidden=4, num_layers=2, out_dim=4, dropout=.2)
        set_seed(103)
        stock = GraphEncoder(**kwargs)
        stock_rng = torch.random.get_rng_state().clone()
        for pooling in ("sum", "mean"):
            set_seed(103)
            model = refit.PoolingGraphEncoder(**kwargs, pooling=pooling)
            self.assertTrue(refit._same(stock.state_dict(), model.state_dict()))
            self.assertTrue(torch.equal(stock_rng, torch.random.get_rng_state()))
        batch = Batch.from_data_list(synthetic_task().source_train.graphs[:3])
        set_seed(103)
        actual = refit.PoolingGraphEncoder(**kwargs, pooling="sum")
        stock.freeze()
        actual.freeze()
        torch.testing.assert_close(stock(batch), actual(batch), rtol=0, atol=0)
        graph = synthetic_task().source_train.graphs[0]
        twin = Data(x=torch.cat([graph.x, graph.x]),
                    edge_index=torch.cat([graph.edge_index, graph.edge_index + graph.num_nodes], 1),
                    edge_attr=torch.cat([graph.edge_attr, graph.edge_attr]))
        mean = refit.PoolingGraphEncoder(**kwargs, pooling="mean")
        mean.load_state_dict(stock.state_dict())
        mean.freeze()
        output = mean(Batch.from_data_list([graph, twin]))
        torch.testing.assert_close(output[0], output[1], rtol=1e-6, atol=1e-7)
        mean.train()
        self.assertFalse(mean.training)
        self.assertFalse(any(p.requires_grad for p in mean.parameters()))
        with self.assertRaises(ValueError):
            refit.PoolingGraphEncoder(**kwargs, pooling="median")

    def test_sum_kernel_matches_stock_and_resumption_preserves_rng(self):
        cfg = Config(seed=103, hidden=4, num_layers=2, emb_dim=4, dropout=.25,
                     source_epochs=4, source_batch=5, adapter_width=4)
        data = synthetic_task()
        set_seed(cfg.seed)
        expected, expected_info = pretrain_encoder(cfg, data)
        # R5 spells out exactly these Adam defaults; check that explicit path.
        adam = dict(betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False,
                    maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual, info = refit._fit_source(cfg, data.source_train, data.source_validation,
                                             3, 1, "sum", adam, root / "sum", "toy")
            self.assertTrue(refit._same(expected.state_dict(), actual.state_dict()))
            self.assertEqual(expected_info["selected_epoch"], info["selected_epoch"])
            self.assertEqual(expected_info["history"], info["history"])
            refit._fit_source(replace(cfg, source_epochs=2), data.source_train, data.source_validation,
                              3, 1, "sum", adam, root / "resumed", "toy")
            resumed, replay = refit._fit_source(cfg, data.source_train, data.source_validation,
                                               3, 1, "sum", adam, root / "resumed", "toy")
            self.assertTrue(refit._same(actual.state_dict(), resumed.state_dict()))
            self.assertEqual(info, replay)
            refit._fit_source(cfg, data.source_train, data.source_validation,
                              3, 1, "mean", adam, root / "mean", "toy")
            for epoch in range(cfg.source_epochs):
                sum_epoch = refit._load(root / "sum" / "epochs" / f"epoch{epoch:03d}.pt")
                mean_epoch = refit._load(root / "mean" / "epochs" / f"epoch{epoch:03d}.pt")
                self.assertEqual(sum_epoch["order"], mean_epoch["order"])
                self.assertTrue(torch.equal(sum_epoch["permutation_rng"], mean_epoch["permutation_rng"]))
                self.assertTrue(torch.equal(sum_epoch["torch_rng"], mean_epoch["torch_rng"]))
                self.assertTrue(sum_epoch["optimizer"]["state"])
            initial_sum = refit._load(root / "sum/initialization.pt")
            initial_mean = refit._load(root / "mean/initialization.pt")
            self.assertTrue(refit._same(initial_sum, initial_mean))

    def test_staged_source_labels_retarget_tsba_and_broker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ToyArchive(root)
            plan = fixture.plan
            out = root / "phase/source-refit"
            with patch.object(r5_inputs, "_context", side_effect=fixture.context):
                preparation = refit.prepare_source(plan, "generated", 0, 103, output_root=out)
                self.assertEqual(fixture.reads, [])
                observed_target_forward = []
                original_encode = refit.encode

                def frozen_encode(encoder, graphs, batch_size, device):
                    observed_target_forward.append(True)
                    self.assertTrue(encoder.frozen)
                    self.assertFalse(encoder.training)
                    self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
                    self.assertTrue((Path(preparation["artifact_root"]) / "source.pt").is_file())
                    return original_encode(encoder, graphs, batch_size, device)

                with patch.object(refit, "encode", side_effect=frozen_encode):
                    complete = refit.train_source(plan, "generated", 0, 103, output_root=out)
                self.assertEqual(observed_target_forward, [True])
                self.assertEqual(fixture.reads, [("inputs/graphs.pt", "INPUT"),
                    ("source_labels/M0/seed103_fit.pt", "INPUT"),
                    ("source_labels/M0/seed103_validation.pt", "INPUT")])
                self.assertEqual(complete["epochs"], 100)
                fixture.reads.clear()
                self.assertEqual(complete, refit.train_source(plan, "generated", 0, 103, output_root=out))
                self.assertEqual(fixture.reads, [])
                first = refit.load_inputs(plan, "generated", 0, 1, 103, output_root=out, include_tsba=True)
                changed_phase = deepcopy(plan)
                changed_phase["phase"] = "C"
                second = refit.load_inputs(changed_phase, "generated", 0, 2, 103, output_root=out, include_tsba=True)
                self.assertEqual(fixture.reads, [("source_labels/M0/seed103_fit.pt", "INPUT")] * 2)
                np.testing.assert_array_equal(first["source_h"], second["source_h"])
                self.assertEqual(first["provenance"]["source_refit"]["artifact_digest"], complete["digest"])
                self.assertEqual(second["provenance"]["source_refit"]["artifact_digest"], complete["digest"])
                self.assertNotEqual(first["adapt_ids"], second["adapt_ids"])
                self.assertEqual(second["manifest"]["target"], 2)
                self.assertEqual(first["manifest"]["format"], r5_inputs._FORMAT)
                self.assertFalse(first["source_h"].flags.writeable)
                self.assertFalse(first["source_y"].flags.writeable)
                self.assertEqual(set(matched_tsba.blind_inputs(first)), matched_tsba._BLIND_KEYS)
                self.assertFalse(any("holdout" in key or "validation" in key for key in matched_tsba.blind_inputs(first)))
                for name in ("source", "validation", "adapt", "holdout"):
                    np.testing.assert_allclose(first[name + "_x"],
                                               (first[name + "_h"] - first["tsba_mean"]) / first["tsba_scale"], rtol=0, atol=0)
                    np.testing.assert_allclose(first[name + "_z0"], first[name + "_h"] @ first["source_head_weight"].T
                                               + first["source_head_bias"], rtol=1e-15, atol=1e-15)
                # A toy seal exercises the unmodified broker's acceptance of
                # the extended provenance and its existing ORACLE restriction.
                phase = root / "phase"
                (phase / "manifest.json").write_text(json.dumps(first["manifest"]))
                seal(phase, "R7_INPUT", ["manifest.json"], expected=["manifest.json"])
                (phase / "obs.json").write_text("{}")
                seal(phase, "R7_OBS", ["obs.json"], expected=["obs.json"])
                labels = r5_inputs.load_labels(phase, "manifest.json", "adapt")
                np.testing.assert_array_equal(labels, np.asarray(first["adapt_ids"]) % 2)
                with self.assertRaises(FileNotFoundError):
                    r5_inputs.load_labels(phase, "manifest.json", "holdout")
                # Changed cache bytes are rejected before being deserialized.
                with (Path(complete["artifact_root"]) / "frozen.pt").open("ab") as handle:
                    handle.write(b"tampered")
                with self.assertRaises(PermissionError):
                    refit.load_inputs(plan, "generated", 0, 1, 103, output_root=out)

    def test_recipe_and_graph_label_rejection(self):
        plan = dict(source_refit=dict(recipe=refit._RECIPE, pooling="mean"))
        with self.assertRaises(ValueError):
            refit._pooling(plan, "sum")
        plan["source_refit"]["hidden"] = 4
        with self.assertRaises(ValueError):
            refit._pooling(plan, None)
        graph = synthetic_task().source_train.graphs[0]
        with self.assertRaises(ValueError):
            refit._graphs([dict(x=graph.x, edge_index=graph.edge_index, y=torch.tensor(1))], 1)


if __name__ == "__main__":
    unittest.main()
