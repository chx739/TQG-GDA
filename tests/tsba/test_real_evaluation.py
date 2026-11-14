"""Synthetic-only checks of independent R1 evidence arithmetic and denominators."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.common.statistics import classification_metrics, wilson, validated_nonnegative_errorbars; from experiments.mutagenicity.representation_boundary_support_20261111.src.evaluate import direction_from_ci, evaluate_panel, reference_consensus, nll_screen, strict_fold_summary, summarize_bank, summarize_direction, verify_observations_sealed


class RealEvaluationTests(unittest.TestCase):
    def test_graph_metrics_class_imbalance_and_missing_class(self):
        probabilities = torch.tensor([[.8, .2], [.8, .2], [.4, .6], [.5, .5]], dtype=torch.float64)
        labels = torch.tensor([0, 1, 1, 1])
        row = classification_metrics(probabilities, labels)
        self.assertEqual(row["confusion"], [[1, 0], [2, 1]])
        self.assertAlmostEqual(row["balanced_accuracy"], (1 + 1/3)/2)
        self.assertAlmostEqual(row["accuracy"], .5)
        self.assertEqual(row["ba_one_graph_resolution_class0_pp"], 50)
        self.assertAlmostEqual(row["nll"], -(math.log(.8)+math.log(.2)+math.log(.6)+math.log(.5))/4)
        missing = classification_metrics(probabilities[:1], labels[:1])
        self.assertIsNone(missing["balanced_accuracy"])
        self.assertAlmostEqual(missing["nll"], -math.log(.8))

    def test_stable_log_probability_not_probability_clipping(self):
        logits = torch.tensor([[1000., 0.]], dtype=torch.float64)
        row = classification_metrics(logits.softmax(1), torch.tensor([1]), logits.log_softmax(1))
        self.assertEqual(row["nll"], 1000.)
        with self.assertRaises(ValueError):
            classification_metrics([[.3, .3]], [0])

    def test_missing_fold_cannot_be_averaged_away(self):
        result = strict_fold_summary([.9, .8, .7, None])
        self.assertIsNone(result["mean"])
        self.assertAlmostEqual(result["available_mean"], .8)
        self.assertEqual(result["available_count"], 3)
        with self.assertRaises(ValueError):
            strict_fold_summary([.8]*3)

    def test_missing_seed_cannot_waive_nll_safety_gate(self):
        result = nll_screen([.02, .02, .02, .02, None])
        self.assertFalse(result["complete"])
        self.assertFalse(result["passes"])
        self.assertTrue(nll_screen([-.02]*5)["passes"])
        self.assertTrue(nll_screen([.02]*5)["systematic_worsening"])

    def test_wilson_does_not_invent_opposite_majorities(self):
        self.assertGreater(wilson(10, 10)[0], .5)
        self.assertLess(wilson(0, 10)[1], .5)
        self.assertLess(wilson(7, 10)[0], .5)
        self.assertEqual(wilson(0, 0), [None, None])

    def test_wilson_errorbar_endpoints_clip_roundoff_only_without_changing_statistics(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            for n in (1, 10, 31, 271):
                for successes in (0, n, n//2):
                    interval = wilson(successes, n)
                    saved = interval.copy()
                    value = successes/n
                    errors = validated_nonnegative_errorbars(value, interval)
                    self.assertTrue(all(length[0] >= 0 for length in errors))
                    self.assertEqual(interval, saved)
                    ax.errorbar([successes], [value], yerr=errors)
            fig.canvas.draw()
        finally:
            plt.close(fig)
        self.assertEqual(validated_nonnegative_errorbars(0., [1e-17, .5])[0], [0.])
        self.assertEqual(validated_nonnegative_errorbars(1., [.5, 1.-1e-16])[1], [0.])
        with self.assertRaises(ValueError):
            validated_nonnegative_errorbars(0., [.01, .5])
        with self.assertRaises(ValueError):
            validated_nonnegative_errorbars(.5, [.7, .3])
        with self.assertRaises(ValueError):
            validated_nonnegative_errorbars(.5, [float("nan"), .7])

    def panel(self):
        positive, negative = list(range(40)), list(range(40, 80))
        b = {i: 0 if i < 40 else 1 for i in range(20)}
        b.update({i: 1 for i in range(40, 60)})
        return dict(support=dict(positive_ids=positive, negative_ids=negative,
                    positive_B_ids=list(range(20)), negative_B_ids=list(range(40, 60))),
                    ess=[20., 10., 10.], V=.3, V_ci=[.2, .4]), b

    def test_density_joint_support_ess_and_semantics_are_separate(self):
        panel, b = self.panel()
        row = evaluate_panel(panel, b)
        self.assertTrue(row["joint_positive"])
        self.assertTrue(row["semantic_joint_positive"])
        self.assertEqual(row["label_state"], "correct")
        reversed_row = evaluate_panel(panel, {i: 1-y for i, y in b.items()})
        self.assertTrue(reversed_row["joint_positive"])
        self.assertFalse(reversed_row["semantic_joint_positive"])
        self.assertEqual(reversed_row["label_state"], "reversed")
        panel["ess"][1] = 4.99
        concentrated = evaluate_panel(panel, b)
        self.assertFalse(concentrated["eligible"])
        self.assertEqual(concentrated["valley_state"], "kernel_concentrated")

    def test_fixed_total_and_distinct_graph_cluster_support(self):
        panel, b = self.panel()
        row = evaluate_panel(panel, b)
        rows = [dict(row, success=True, origin_id=i, cluster_id=i%8) for i in range(16)]
        rows += [dict(success=False, origin_id=i, cluster_id=None) for i in range(16, 64)]
        summary = summarize_bank(rows)
        self.assertTrue(summary["passes"])
        self.assertEqual(summary["eligible"], 16)
        self.assertEqual(summary["joint_positive_fraction_total"], .25)
        self.assertEqual(summary["joint_positive_fraction_eligible"], 1.)
        self.assertEqual(summary["distinct_B_covered"], 40)
        self.assertEqual(summary["max_B_reuse"], 16)
        for r in rows[:16]:
            r["cluster_id"] = 0
        self.assertFalse(summarize_bank(rows)["passes"])

    def roots(self, count=8, sign=-1):
        return [dict(status="detected_unique" if i < count else "missing_reference_head",
                     t=sign*.5 if i < count else None, architecture="linear" if i < 4 else "mlp") for i in range(8)]

    def test_reference_missing_heads_and_near_coincidence(self):
        self.assertEqual(reference_consensus(self.roots(6), 1.)["status"], "consensus")
        self.assertEqual(reference_consensus(self.roots(5), 1.)["status"], "reference_uncertain")
        near = self.roots()
        for row in near:
            row["t"] = .04
        self.assertEqual(reference_consensus(near, 1.)["usable_count"], 0)
        with self.assertRaises(ValueError):
            reference_consensus(self.roots()[:7], 1.)

    def test_loo_losing_direction_uses_original_identifiable_denominator(self):
        panel, b = self.panel()
        row = evaluate_panel(panel, b)
        anchors = [dict(row, success=True, origin_id=i, cluster_id=i%8) for i in range(16)]
        anchors += [dict(success=False, origin_id=i, cluster_id=None) for i in range(16, 64)]
        scans = [dict(origin_id=i, success=True, sigma_g_ci=[.1, .2], roots=self.roots(),
                      loo=dict(sigma_g_ci=[.1, .2], ess_center=10.)) for i in range(16)]
        scans += [dict(origin_id=i, success=False, roots=[]) for i in range(16, 64)]
        summary, _ = summarize_direction(anchors, scans, 1.)
        self.assertEqual(summary["state"], "support")
        for scan in scans[:5]:
            scan["loo"]["sigma_g_ci"] = [-.0001, .0001]
        summary, _ = summarize_direction(anchors, scans, 1.)
        self.assertEqual(summary["loo_original_identifiable_denominator"], 16)
        self.assertEqual(summary["loo_opposite_fraction"], 0.)
        self.assertEqual(summary["loo_unidentifiable_fraction"], 5/16)
        self.assertEqual(summary["state"], "inconclusive")

    def test_broker_preflight_rejects_unsealed_observation_before_label_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "probes").mkdir()
            (root / "probes/new.pt").write_bytes(b"late observation")
            seal = dict(stage="S2", files={"roles.json": "x", "seals/S1.json": "y"})
            with self.assertRaises(PermissionError):
                verify_observations_sealed(root, {"seeds": [], "tasks": []}, seal)
            (root / "probes/new.pt").unlink()
            with self.assertRaises(PermissionError):
                verify_observations_sealed(root, {"seeds": [3405], "tasks": [[0,1]]}, seal)


if __name__ == "__main__":
    unittest.main()
