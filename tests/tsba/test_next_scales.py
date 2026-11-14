"""Synthetic frozen-scale checks; these tests never load research artifacts."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.common.synthetic.archive import manifest, write_json
from experiments.common.real_data.data import canonical_sha
from experiments.common.real_data.density import (
    GaussianKDE, bootstrap_counts, finite_support, valley_observation,
)
import experiments.mutagenicity.r1b_20261111.scales as scales


class FrozenScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.plan = dict(seeds=[3405], domains=[1], scales=dict(bootstrap_repeats=200,
            settings=[dict(support_multiplier=r, bandwidth_multiplier=b) for r, b in scales.SETTINGS]),
            resources=dict(max_scale_cell_seconds=60, max_rss_bytes=8 * 1024**3))
        self.settings = scales._settings(self.plan)
        self.x = torch.tensor([[1., 0.], [-1., 0.], [1., 5.], [-1., 5.],
                               [1., 7.], [-1., 7.], [1.6, 0.], [-1.6, 0.],
                               [.6, 2.], [-.6, 2.], [0., 0.], [0., 1.]], dtype=torch.float64)
        self.ids = list(range(len(self.x)))
        self.point = torch.zeros(2, dtype=torch.float64)
        self.normal = torch.tensor([1., 0.], dtype=torch.float64)
        self.counts = bootstrap_counts(len(self.x), 200, 234)
        self.anchor = dict(origin_id=0, success=True, failure=None, cluster_id=0,
                           point=self.point, normal=self.normal)

    def test_radius_only_changes_support_and_never_queries_or_density(self):
        actual = scales.observe_anchor(self.x, self.ids, self.ids, self.anchor,
                                       1., self.counts, self.settings)
        baseline, expanded, widest = actual["settings"][:3]
        self.assertEqual(baseline["support"]["positive_ids"], [8])
        self.assertEqual(expanded["support"]["positive_ids"], [2, 8])
        self.assertEqual(widest["support"]["positive_ids"], [2, 4, 8])
        for record in (baseline, expanded, widest):
            self.assertNotIn(0, record["support"]["positive_ids"])
            self.assertNotIn(6, record["support"]["positive_ids"])
            self.assertNotIn(7, record["support"]["negative_ids"])
            for key in ("query_points", "logp", "ess", "V_bootstrap", "g_bootstrap"):
                torch.testing.assert_close(record["main"][key], baseline["main"][key], rtol=0, atol=0)
        self.assertEqual(widest["support"]["positive_B_n"], 3)
        self.assertEqual(widest["support"]["coverage_fraction_target_pool"], 7 / 12)

    def test_bandwidth_only_changes_kde_with_fixed_support_and_queries(self):
        actual = scales.observe_anchor(self.x, self.ids, self.ids, self.anchor,
                                       1., self.counts, self.settings)["settings"]
        baseline, expanded = actual[0], actual[3]
        self.assertEqual(baseline["support"]["positive_ids"], expanded["support"]["positive_ids"])
        torch.testing.assert_close(baseline["main"]["query_points"], expanded["main"]["query_points"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(baseline["main"]["logp"], expanded["main"]["logp"]))
        torch.testing.assert_close(expanded["main"]["query_points"],
                                   torch.stack((self.point, self.normal, -self.normal)), rtol=0, atol=0)
        self.assertTrue(torch.all(expanded["loo"]["origin_kernel_weight"] == 0))
        self.assertTrue(torch.all(expanded["main"]["origin_kernel_weight"] > 0))

    def test_baseline_parity_for_support_main_and_loo_with_old_draws(self):
        for pool, drop in (("main", None), ("loo", 0)):
            old = valley_observation(GaussianKDE(self.x, 1.), self.point, self.normal,
                                     1., 1., self.counts, drop)
            old.update(bandwidth_multiplier=1., shoulder_sigma=1.)
            if pool == "main":
                self.anchor["panels"] = [old]
            else:
                self.anchor["loo"] = old
        observation = scales.observe_anchor(self.x, self.ids, self.ids, self.anchor,
                                            1., self.counts, self.settings)["settings"][0]
        old_support = finite_support(self.x, self.ids, 0, self.point, self.normal, 1., 1., self.ids)
        for key, value in old_support.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(observation["support"][key], value, rtol=0, atol=0)
            else:
                self.assertEqual(observation["support"][key], value)
        for pool in ("main", "loo"):
            self.assertEqual(observation[pool]["old_baseline_parity"]["status"], "matches")
        direct = GaussianKDE(self.x[1:], 1.).evaluate(observation["loo"]["query_points"])
        torch.testing.assert_close(observation["loo"]["logp"], direct["logp"][0], rtol=1e-12, atol=1e-12)
        weight = torch.softmax(GaussianKDE(self.x, 1.).log_kernel(observation["main"]["query_points"]), -1)
        torch.testing.assert_close(observation["main"]["max_kernel_weight"], weight.max(-1).values)

    def _fixture(self, root):
        deps = root / "deps"
        (deps / "source/seed3405").mkdir(parents=True)
        (deps / "geometry/seed3405_domain1/banks").mkdir(parents=True)
        x = torch.randn((96, 2), generator=torch.Generator().manual_seed(28), dtype=torch.float64)
        ids, query_ids = list(range(96)), list(range(64))
        torch.save(dict(x=x), deps / "source/seed3405/frozen.pt")
        write_json(deps / "roles.json", dict(targets={"1": dict(ids=ids, A=query_ids, B=list(range(64, 96)), query_ids=query_ids)}))
        torch.save(dict(counts=bootstrap_counts(96, 200, 99), sigma=1., target_ids=ids),
                   deps / "geometry/seed3405_domain1/bootstrap.pt")
        anchors = [dict(origin_id=i, success=False, failure="original_projection_failure", cluster_id=None)
                   for i in query_ids]
        anchors[0].update(success=True, failure=None, cluster_id=0, point=self.point, normal=self.normal)
        for bank in scales.BANKS:
            torch.save(dict(sigma=1., target_ids=ids, origin_ids=query_ids, anchors=anchors, clusters={}),
                       deps / "geometry/seed3405_domain1/banks" / f"{bank}.pt")
        return deps

    def test_runner_has_no_class_input_or_label_reads_and_preserves_failures(self):
        self.assertNotIn("labels", inspect.signature(scales.run_scales).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = self._fixture(root)
            original_load = scales._load
            reads = []

            def guarded(path):
                path = Path(path)
                reads.append(str(path.relative_to(deps)))
                self.assertTrue(path.is_relative_to(deps))
                self.assertTrue(path.name in ("frozen.pt", "bootstrap.pt") or path.parent.name == "banks")
                return original_load(path)

            with patch.object(scales, "_load", side_effect=guarded):
                rows = scales.run_scales(root, self.plan)
            self.assertEqual(len(rows), 9 * 6)
            self.assertEqual(len(reads), 11)
            for row in rows:
                self.assertEqual(row["denominator"], 64)
                self.assertEqual(row["projected"], 1)
                self.assertEqual(row["failure_counts"], {"original_projection_failure": 63})
            bank = original_load(root / "scales/seed3405_domain1/source.pt")
            self.assertEqual(bank["origin_ids"], list(range(64)))
            self.assertEqual(len(bank["anchors"]), 64)
            self.assertFalse(bank["labels_used"])
            with self.assertRaises(PermissionError):
                scales.evaluate_scales(root, self.plan, {1: {i: i % 2 for i in range(64, 96)}})
            (root / "seals").mkdir()
            seal = dict(stage="OBS", files={k: v["sha256"] for k, v in manifest(root).items()})
            write_json(root / "seals/OBS.json", dict(seal, digest=canonical_sha(seal)))
            with self.assertRaises(PermissionError):
                scales.run_scales(root, self.plan)
            with self.assertRaises(PermissionError):
                scales.evaluate_scales(root, self.plan, {1: {i: i % 2 for i in range(96)}})
            evaluated = scales.evaluate_scales(root, self.plan, {1: {i: i % 2 for i in range(64, 96)}})
            self.assertEqual(len(evaluated["banks"]), 9 * 6 * 2)
            for row in evaluated["banks"]:
                self.assertEqual(row["denominator"], 64)
                self.assertEqual(row["eligible"], 0)
                self.assertIsNone(row["joint_positive_fraction_eligible"])
            self.assertFalse(evaluated["summary"]["scale_selection_performed"])
            self.assertEqual({g["group"] for g in evaluated["groups"]}, {"source", "reference_all", "linear", "mlp"})
            self.assertEqual(len(evaluated["seed_aggregates"]), 4 * 6 * 2)
            with (root / "scales/index.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaises(PermissionError):
                scales.evaluate_scales(root, self.plan, {1: {i: i % 2 for i in range(64, 96)}})

    def test_density_failure_keeps_independent_support_visible_in_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            with patch.object(scales, "density_at_frozen_queries", side_effect=ValueError("synthetic_density_failure")):
                rows = scales.run_scales(root, self.plan)
            self.assertTrue(all(r["main_observed"] == 0 for r in rows))
            (root / "seals").mkdir()
            seal = dict(stage="OBS", files={k: v["sha256"] for k, v in manifest(root).items()})
            write_json(root / "seals/OBS.json", dict(seal, digest=canonical_sha(seal)))
            evaluated = scales.evaluate_scales(root, self.plan, {1: {i: i % 2 for i in range(64, 96)}})
            query = evaluated["queries"][0]
            self.assertIn("positive_unlabeled_n", query)
            self.assertIn("positive_B_n", query)
            self.assertIsNone(query["V"])
            self.assertFalse(query["eligible"])
            self.assertIn("synthetic_density_failure", query["failure"])

    def test_missing_dependency_preserves_all_fixed_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = self._fixture(root)
            (deps / "geometry/seed3405_domain1/bootstrap.pt").unlink()
            rows = scales.run_scales(root, self.plan)
            self.assertEqual(len(rows), 9 * 6)
            self.assertTrue(all(r["denominator"] == 64 and r["projected"] == 1 for r in rows))
            self.assertTrue(all(sum(r["failure_counts"].values()) == 64 for r in rows))
            bank = scales._load(root / "scales/seed3405_domain1/source.pt")
            torch.testing.assert_close(bank["anchors"][0]["point"], self.point, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
