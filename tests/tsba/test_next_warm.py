"""Synthetic engineering checks for the isolated next-round warm-start arm."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from tsba.classifier import ResidualClassifier
from workflows.tsba.validation.archive import digest, save_tensor, write_json
from workflows.tsba.validation.real_data.probes import Probe, model_directory
from workflows.tsba.validation.real_data.next_round.warm import (
    fit_warm, rescore_random, run_warm, source_initialized_probe,
)


class WarmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.features = torch.randn((24, 3), dtype=torch.double,
                                    generator=torch.Generator().manual_seed(113))
        self.split = dict(budget_ids=list(range(12)), train_ids=list(range(8)),
                          validation_ids=list(range(8, 12)), audit_ids=[])
        self.labels = {i: i % 2 for i in range(12)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(27)
            self.source = ResidualClassifier(3, 2, 4, dtype=torch.double)
        self.initial = self._old_initial(self.features)
        self.plan = dict(seeds=[7], domains=[1], warm=dict(
            learning_rates=[.001, .0001], weight_decay=.001, updates=40,
            validation_cadence=20, outer_folds=1, supplementary_endpoints=[20, 40]),
            resources=dict(max_probe_seconds=120, max_rss_bytes=8 * 1024**3))
        self.info = dict(seed=7, domain=1, fold=0, lr=.001)
        self.evaluation = dict(B1=[12, 13], source_fit=[14, 15],
                               source_validation=[16, 17], source_diagnostic=[18, 19])

    def _old_initial(self, features):
        x = features[self.split["train_ids"]]
        model = Probe(x.mean(0), x.std(0, unbiased=False).clamp_min(1e-8))
        return dict(step=0, mean=model.mean.clone(), scale=model.scale.clone(),
                    model_state=copy.deepcopy(model.state_dict()), optimizer_state={})

    def _fit(self, directory, **kwargs):
        args = dict(directory=directory, features=self.features, labels=self.labels,
                    split=self.split, source_state=self.source.state_dict(),
                    old_initial=self.initial, model_info=self.info, plan=self.plan,
                    evaluation_ids=self.evaluation)
        args.update(kwargs)
        return fit_warm(**args)

    def _write_old(self, root, initial=None, selected=None):
        root.mkdir(parents=True)
        (root / "checkpoints").mkdir()
        initial = copy.deepcopy(self.initial if initial is None else initial)
        selected = copy.deepcopy(initial if selected is None else selected)
        selected["step"] = 20
        model = Probe(initial["mean"], initial["scale"])
        model.load_state_dict(selected["model_state"])
        ids = self.split["validation_ids"]
        value = float(F.cross_entropy(model(self.features[ids]),
                                      torch.tensor([self.labels[i] for i in ids])).detach())
        metadata = dict(self.split, status="success", selected_step=20,
                        selected_validation_ce=value, head="linear",
                        checkpoints={"selected": "checkpoints/step0020.pt"})
        save_tensor(root / "initial.pt", initial)
        save_tensor(root / "checkpoints/step0020.pt", selected)
        write_json(root / "metadata.json", metadata)
        return metadata

    def test_affine_mapping_reuses_old_scaler_and_preserves_float64_probabilities(self):
        rng_before = torch.random.get_rng_state().clone()
        model, error = source_initialized_probe(self.features, self.source.state_dict(), self.initial)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertLess(error, 1e-12)
        torch.testing.assert_close(model(self.features).softmax(-1),
                                   self.source(self.features).softmax(-1), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(model.mean, self.initial["mean"], rtol=0, atol=0)
        torch.testing.assert_close(model.scale, self.initial["scale"], rtol=0, atol=0)
        weight, bias = model.affine()
        torch.testing.assert_close(weight, self.source.linear.weight, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bias, self.source.linear.bias, rtol=1e-12, atol=1e-12)

    def test_fresh_adam_shared_initialization_checkpoints_and_endpoint_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._fit(root / "lr1")
            second = self._fit(root / "lr2", model_info=dict(self.info, lr=.0001))
            for directory, meta in ((root / "lr1", first), (root / "lr2", second)):
                self.assertEqual(meta["status"], "success", meta.get("failure"))
                history = json.loads((directory / "history.json").read_text())
                self.assertEqual([row["step"] for row in history["training"]], list(range(41)))
                self.assertEqual([row["step"] for row in history["validation"]], [0, 20, 40])
                expected = min(history["validation"], key=lambda row: row["validation_ce"])
                self.assertEqual(meta["selected_step"], expected["step"])
                zero = torch.load(directory / "checkpoints/step0000.pt", weights_only=False)
                final = torch.load(directory / "checkpoints/step0040.pt", weights_only=False)
                self.assertEqual(zero["optimizer_state"]["state"], {})
                self.assertTrue(all(int(v["step"]) == 40 for v in final["optimizer_state"]["state"].values()))
                self.assertEqual(final["validation_history"], history["validation"])
                predictions = torch.load(directory / "probabilities.pt", weights_only=False)
                self.assertEqual(set(predictions), {"0", "selected", "20", "40"})
                model = Probe(self.initial["mean"], self.initial["scale"])
                for endpoint, pools in predictions.items():
                    state = torch.load(directory / meta["checkpoints"][endpoint], weights_only=False)
                    model.load_state_dict(state["model_state"])
                    for name, pool in pools.items():
                        self.assertEqual(pool["ids"], self.evaluation[name])
                        torch.testing.assert_close(pool["probabilities"],
                                                   model(self.features[pool["ids"]]).softmax(-1))
            state1 = torch.load(root / "lr1/checkpoints/step0000.pt", weights_only=False)
            state2 = torch.load(root / "lr2/checkpoints/step0000.pt", weights_only=False)
            for key in state1["model_state"]:
                torch.testing.assert_close(state1["model_state"][key], state2["model_state"][key], rtol=0, atol=0)

    def test_step_zero_wins_exact_tie(self):
        features = torch.zeros_like(self.features)
        state = copy.deepcopy(self.source.state_dict())
        state["linear.weight"].zero_()
        state["linear.bias"].zero_()
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._fit(Path(tmp) / "tie", features=features, source_state=state,
                             old_initial=self._old_initial(features))
            self.assertEqual(meta["status"], "success", meta.get("failure"))
            self.assertEqual(meta["selected_step"], 0)

    def test_failures_are_explicit_and_partial_state_matches_completed_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # One preflight, then steps 0..20; fail before optimizer update 21.
            with patch("workflows.tsba.validation.real_data.next_round.warm._rss",
                       side_effect=[1] * 22 + [9 * 1024**3, 9 * 1024**3]):
                meta = self._fit(root / "limited")
            self.assertEqual(meta["status"], "failed")
            self.assertIn("warm_rss_limit", meta["failure"])
            self.assertEqual(meta["updates_completed"], 20)
            partial = torch.load(root / "limited/partial.pt", weights_only=False)
            self.assertEqual(partial["step"], 20)
            self.assertEqual(partial["training_history"][-1]["step"], 20)
            self.assertFalse((root / "limited/probabilities.pt").exists())
            non_affine = copy.deepcopy(self.source.state_dict())
            non_affine["U2.weight"].fill_(.1)
            failed = self._fit(root / "nonaffine", source_state=non_affine)
            self.assertIn("not_affine", failed["failure"])
            bad = copy.deepcopy(self.initial)
            bad["scale"] *= 2
            failed = self._fit(root / "wrongscale", old_initial=bad)
            self.assertEqual(failed["status"], "failed")

    def test_old_random_correction_selects_zero_without_training_or_old_file_writes(self):
        self.features[:, 0] = torch.tensor([2 * (i % 2) - 1 for i in range(24)])
        initial = self._old_initial(self.features)
        initial["model_state"]["network.weight"].zero_()
        initial["model_state"]["network.weight"][:, 0] = torch.tensor([-2., 2.])
        initial["model_state"]["network.bias"].zero_()
        selected = copy.deepcopy(initial)
        selected["model_state"]["network.weight"] *= -1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            original = self._write_old(old, initial, selected)
            before = {str(p.relative_to(old)): digest(p) for p in old.rglob("*") if p.is_file()}
            with patch("torch.optim.Adam", side_effect=AssertionError("No optimizer allowed")):
                result = rescore_random(root / "corrected", old, self.features, self.labels,
                                        self.split, self.info, self.evaluation)
            self.assertEqual(result["status"], "success", result.get("failure"))
            self.assertEqual(result["selected_step"], 0)
            self.assertEqual(result["original_selected_step"], 20)
            self.assertEqual(result["original_selected_validation_ce"], original["selected_validation_ce"])
            predictions = torch.load(root / "corrected/probabilities.pt", weights_only=False)
            torch.testing.assert_close(predictions["selected"]["B1"]["probabilities"],
                                       predictions["0"]["B1"]["probabilities"], rtol=0, atol=0)
            self.assertFalse(torch.equal(predictions["selected"]["B1"]["probabilities"],
                                         predictions["original_selected"]["B1"]["probabilities"]))
            self.assertEqual(before, {str(p.relative_to(old)): digest(p) for p in old.rglob("*") if p.is_file()})

    def test_run_accepts_A_only_and_archives_all_source_input_pools_before_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = root / "deps"
            (deps / "roles").mkdir(parents=True)
            (deps / "source/seed7").mkdir(parents=True)
            roles = dict(source=dict(domain_ids=list(range(14, 20))),
                         targets={"1": dict(A=list(range(12)), B=[12, 13], final=[20, 21])})
            write_json(deps / "roles.json", roles)
            write_json(deps / "roles/seed7.json", dict(fit_ids=[14, 15], validation_ids=[16, 17],
                       diagnostic_ids=[18, 19], folds=dict(targets={"1": {"0": self.split}})))
            save_tensor(deps / "source/seed7/frozen.pt", dict(x=self.features,
                        initial_head=self.source.state_dict()))
            save_tensor(deps / "source/seed7/predictions.pt", dict(ids=torch.arange(24),
                        probabilities=self.source(self.features).softmax(-1).detach()))
            self._write_old(model_directory(deps, "target", 7, 1, 0, "H", "linear", "true"))
            with self.assertRaisesRegex(PermissionError, "only_declared_target_A"):
                run_warm(root, self.plan, {1: dict(self.labels, **{}) | {20: 0}})
            self.assertFalse((root / "warm").exists())
            events = []
            results = run_warm(root, self.plan, {1: self.labels}, events.append)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(events), 2)
            self.assertTrue(all(row["status"] == "success" for row in results), results)
            probabilities = torch.load(root / results[0]["path"] / "probabilities.pt", weights_only=False)
            self.assertEqual(probabilities["selected"]["source_all"]["ids"], list(range(14, 20)))
            self.assertEqual(probabilities["selected"]["B1"]["ids"], [12, 13])
            (root / "seals").mkdir()
            write_json(root / "seals/OBS.json", {})
            with self.assertRaisesRegex(PermissionError, "closed_after_OBS"):
                run_warm(root, self.plan, {1: self.labels})


if __name__ == "__main__":
    unittest.main()
