import dataclasses
import tempfile
import unittest
from pathlib import Path
import torch
from torch_geometric.data import Batch
from tqg_gda.tsba.config import Config
from tqg_gda.tsba.data import from_payload, load_task
from tqg_gda.tsba.encoder import GraphEncoder
from tqg_gda.tsba.pipeline import run_task, source_identity
from tqg_gda.tsba.predict import load_predictor
from tqg_gda.tsba.pretrain import graph_batches, pretrain_encoder
from tqg_gda.tsba.score import train_score
from workflows.tsba.run_task import synthetic_task

torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def small_config(self, **kw):
        values = dict(hidden=4, num_layers=2, emb_dim=2, source_epochs=2, source_batch=16,
                      score_hidden=4, score_epochs=2, adapt_rounds=2, target_batch=4)
        values.update(kw)
        return Config(**values)

    def test_target_labels_and_role_overlap_are_rejected(self):
        data = synthetic_task()
        data.target.labels = torch.zeros(len(data.target.graphs), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "target labels"):
            data.validate()
        data = synthetic_task()
        data.target.graphs[0].y = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, "label-free"):
            data.validate()
        data = synthetic_task()
        data.target.ids[0] = data.source_train.ids[0]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            data.validate()

    def test_input_archive_does_not_accept_target_label_field(self):
        record = dict(id="x", x=torch.ones((2, 1)), edge_index=torch.empty((2, 0), dtype=torch.long), label=0)
        payload = dict(schema_version=1, protocol="transductive", source_train=[], source_validation=[], target=[record])
        with self.assertRaisesRegex(ValueError, "target labels"):
            from_payload(payload)

    def test_gine_double_and_frozen_running_statistics(self):
        data = synthetic_task()
        encoder = GraphEncoder(3, 1, hidden=4, num_layers=2, out_dim=2, dropout=.5, conv="gine").double().freeze()
        state = {k: v.clone() for k, v in encoder.state_dict().items()}
        encoder.train()
        batch = Batch.from_data_list(data.target.graphs[:2])
        first, second = encoder(batch), encoder(batch)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for k, v in encoder.state_dict().items():
            torch.testing.assert_close(v, state[k], atol=0, rtol=0)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))

    def test_single_node_graph_batches_preserve_all_graphs_and_train(self):
        data = synthetic_task()
        for pool in (data.source_train, data.source_validation, data.target):
            for graph in pool.graphs:
                graph.x = graph.x[:1].clone()
                graph.edge_index = torch.empty((2, 0), dtype=torch.long)
                graph.edge_attr = torch.empty((0, 1))
        chunks = graph_batches(data.source_train.graphs, 1)
        self.assertEqual([i for chunk in chunks for i in chunk], list(range(len(data.source_train.graphs))))
        self.assertTrue(all(len(chunk) >= 2 for chunk in chunks))
        _, info = pretrain_encoder(self.small_config(source_epochs=1, source_batch=1), data)
        self.assertEqual(info["selected_epoch"], 0)

    def test_score_training_frozen_and_repeatable(self):
        cfg = self.small_config()
        x = torch.tensor([[-1., .2], [.5, 1.], [.1, -.4]], dtype=torch.float64)
        first, info = train_score(x, .4, cfg)
        second, other = train_score(x, .4, cfg)
        self.assertEqual(info, other)
        torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)
        self.assertTrue(all(not p.requires_grad for p in first.parameters()))

    def test_end_to_end_checkpoint_and_standalone_prediction(self):
        cfg, data = self.small_config(), synthetic_task()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "first"
            model, result = run_task(cfg, data, out)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["target_labels_used"])
            self.assertIsNone(result["target_accuracy"])
            predictions = model.predict(data.target.graphs)
            ck = torch.load(out / "model.pt", weights_only=True)
            self.assertFalse(any(k in ck for k in ("score", "source_labels", "source_graphs", "target")))
            loaded = load_predictor(out / "model.pt")
            torch.testing.assert_close(predictions, loaded.predict(data.target.graphs), atol=0, rtol=0)
            reused, again = run_task(cfg, data, Path(tmp) / "reuse", out / "source.pt")
            torch.testing.assert_close(predictions, reused.predict(data.target.graphs), atol=0, rtol=0)
            self.assertEqual(result["adaptation"], again["adaptation"])
            with self.assertRaises(FileExistsError):
                run_task(cfg, data, out)
            data.source_train.labels = 1 - data.source_train.labels
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                run_task(cfg, data, source_checkpoint=out / "source.pt")

    def test_identity_covers_roles_target_and_configuration(self):
        data, cfg = synthetic_task(), self.small_config()
        old = data.fingerprints()
        source_key = source_identity(cfg, old)
        data.target.graphs[0].x[0, 0] += .1
        changed = data.fingerprints()
        self.assertNotEqual(old["target"], changed["target"])
        self.assertEqual(source_key, source_identity(cfg, changed))
        self.assertNotEqual(source_key, source_identity(dataclasses.replace(cfg, source_epochs=3), changed))

    def test_missing_source_validation_preserves_label_budget(self):
        data = synthetic_task()
        data.source_validation.graphs, data.source_validation.ids = [], []
        data.source_validation.labels = torch.empty(0, dtype=torch.long)
        model, result = run_task(self.small_config(adapt_rounds=0), data)
        self.assertEqual(result["source_pretraining"]["selection"], "last_predeclared_epoch")
        self.assertIsNone(result["source_validation_after"])


if __name__ == "__main__":
    unittest.main()
