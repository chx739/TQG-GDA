"""R5 input, observation and same-state regression checks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import test_r4
from experiments.mutagenicity.r5_原始方法与请求价值_20261114.src import protocol, trajectory, diagnostics, evaluate, run
from experiments.mutagenicity.r5_原始方法与请求价值_20261114.src.shared import io, p4, training, assert_nested_equal

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/mutagenicity/r5_原始方法与请求价值_20261114"


class R5Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.request = json.loads((ROOT / "configs/proteins-plan.json").read_text())

    def test_proteins_exact_update_budget_and_identity(self):
        plan = protocol.resolve(self.request, "proteins_edge")
        self.assertEqual(plan["model"]["score_epochs"], 2400)
        self.assertEqual(len(p4.tasks(plan)), 60)
        self.assertEqual(plan["methods"]["A0"]["group"], "M00")
        wrong = copy.deepcopy(self.request)
        wrong["settings"][0]["score_epochs"] = 800
        with self.assertRaises(ValueError):
            protocol.resolve(wrong, "proteins_edge")

    def test_proteins_reconstruction_does_not_read_class_labels(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            for name, text in {
                "A": "1,2\n2,1\n2,1\n2,2\n3,4\n4,3\n",
                "graph_indicator": "1\n1\n2\n2\n",
                "node_labels": "0\n1\n2\n0\n",
                "graph_labels": "do not parse",
                "node_attributes": "do not parse",
            }.items():
                (raw / f"PROTEINS_{name}.txt").write_text(text)
            plan = protocol.resolve(self.request, "proteins_edge")
            plan["input"]["raw_directory"] = tmp
            plan["setting"]["total_graphs"] = 2
            graphs, hashes = protocol.graphs(plan)
            self.assertEqual(len(graphs), 2)
            self.assertIsNone(graphs[0]["edge_attr"])
            self.assertEqual(graphs[0]["edge_index"].shape, (2, 2))
            self.assertEqual(graphs[1]["x"].shape, (2, 3))
            self.assertTrue(all("y" not in graph for graph in graphs))

    def test_unsealed_evaluation_never_reads_labels(self):
        with TemporaryDirectory() as tmp, patch.object(evaluate.p3, "read_values") as reader:
            with self.assertRaises(FileNotFoundError):
                evaluate.target_labels(Path(tmp), protocol.resolve(self.request, "proteins_edge"))
            reader.assert_not_called()

    def test_partial_seal_never_reads_labels(self):
        plan = protocol.resolve(self.request, "proteins_edge")
        with TemporaryDirectory() as tmp, patch.object(evaluate.p3, "read_values") as reader:
            root = Path(tmp)
            io.write_json(root / "protocol/plan.json", plan)
            io.write_json(root / "inputs/domains.json", {str(i):list(range(278)) for i in range(4)})
            io.write_json(root / "replay.json", plan["counts"])
            io.seal(root, "R5_INPUT", ["inputs", "protocol"])
            io.seal(root, "R5_OBS", ["inputs", "protocol", "replay.json", "seals/R5_INPUT.json"])
            with self.assertRaises(PermissionError):
                evaluate.target_labels(root, plan)
            reader.assert_not_called()

    def test_numerical_failure_is_terminal_but_interruption_can_resume(self):
        def numerical(*args):
            raise FloatingPointError("nonfinite fixture")
        def interrupted(*args):
            raise InterruptedError("engineering interruption")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(InterruptedError):
                run.guarded(root, {}, None, interrupted)
            run.check_failure(root)
            with self.assertRaises(FloatingPointError):
                run.guarded(root, {}, None, numerical)
            with self.assertRaises(RuntimeError):
                run.check_failure(root)

    def test_single_class_and_empty_subsets(self):
        pred = torch.tensor([[0., -1.], [0., -2.]], dtype=torch.float64).log_softmax(1)
        row = evaluate.subset_metrics(pred, torch.zeros(2, dtype=torch.long))
        self.assertEqual(row["macro_f1"], .5)
        self.assertEqual(row["correct"], 2)
        self.assertEqual(evaluate.subset_metrics(pred[:0], torch.zeros(0, dtype=torch.long)), {"n":0})

    def test_pair_means_preserve_missing_seeds_and_exact_ties(self):
        plan = protocol.resolve(self.request, "proteins_edge")
        rows = [dict(source=t.source, target=t.target, seed=t.seed, contrast="A0−B1", n=279,
                     correct_difference=(1 if (t.source,t.target)==(0,1) else -1 if (t.source,t.target)==(0,2) else 0),
                     balanced_accuracy=0., macro_f1=0., nll=0.) for t in p4.tasks(plan)]
        seeds, summary = evaluate.aggregate_pairs(plan, rows)
        avg = next(r for r in summary if r["scope"]=="Avg" and r["contrast"]=="A0−B1" and r["metric"]=="accuracy")
        self.assertEqual((avg["mean"], avg["ties"]), (0.,5))
        seeds, summary = evaluate.aggregate_pairs(plan, rows[1:])
        avg = next(r for r in summary if r["scope"]=="Avg" and r["contrast"]=="A0−B1" and r["metric"]=="accuracy")
        self.assertIsNone(avg["mean"])
        self.assertEqual((avg["available_pairs"],avg["missing_seeds"]), (59,1))

    def test_three_way_side_pairs_include_both_controls(self):
        rows = [dict(dataset="fixture", split="fixture", source=0,target=1,seed=3405,attempt=1,
                     pool="target",origin="fresh",status="available", direction=d,n=4,full_target_n=4,
                     correct=2+d,accuracy=(2+d)/4,balanced_accuracy=(2+d)/4,macro_f1=.5,nll=1.-d*.1,
                     vs_0_wrong_to_right=1,vs_0_right_to_wrong=0,
                     **{"vs_-1_wrong_to_right":2,"vs_-1_right_to_wrong":0}) for d in (-1,0,1)]
        pairs = evaluate.side_pairs(rows)
        self.assertEqual([r["contrast"] for r in pairs],["normal_minus_0","normal_minus_-1"])
        self.assertEqual([r["accuracy_pp"] for r in pairs],[25.,50.])
        self.assertEqual([r["net_correct"] for r in pairs],[1,2])

    def test_proteins_actual_score_endpoint_filenames(self):
        plan = protocol.resolve(self.request, "proteins_edge")
        fixture = test_r4.R4Test()
        fixture.setUp()
        fixture.plan = plan
        task = p4.Task(2,3,3405)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture.prepare_adapt(root, task)
            (root / "inputs/domains.json").unlink()
            io.write_json(root / "inputs/domains.json", {"3":list(range(16,295))})
            with patch.object(training, "epoch_update", return_value=.1) as updater, \
                 patch.object(training, "validation", return_value=.1):
                training.train_score(root, plan, task)
            self.assertEqual(updater.call_count, 2400)
            for updates, epoch in ((900,300),(1800,600),(3600,1200),(7200,2400)):
                self.assertEqual(io.load(p4.score_dir(root, plan, task)/f"epoch{epoch:04d}.pt")["updates"], updates)
            score = training.learned_score(root, plan, task)
            self.assertEqual(score(torch.zeros(1,16,dtype=torch.float64)).shape, (1,16))

    def test_passive_side_and_resume_match_original_m00(self):
        fixture = test_r4.R4Test()
        fixture.setUp()
        plan = fixture.plan
        plan["methods"]["A0"] = dict(group="M00", density="learned")
        plan["model"].update(score_epochs=2, adapt_rounds=2)
        plan["score"].update(endpoints=[9, 18], recovery_epochs=1)
        plan["side_attempts"] = [1, 2]
        task = p4.Task(2, 3, 3405)
        fixture.plan = plan
        with TemporaryDirectory() as tmp:
            root, original = Path(tmp)/"new", Path(tmp)/"original"
            for path in (root, original):
                torch.manual_seed(82)
                fixture.prepare_adapt(path, task)
                training.train_score(path, plan, task)
            from experiments.mutagenicity.r4_全方向迁移对照_20261112.src.trajectory import adapt
            adapt(original, plan, (task, "A0"))
            trajectory.adapt(root, plan, (task, "A0"), stop_after=1)
            trajectory.adapt(root, plan, (task, "A0"))
            for step in (1, 2):
                a = io.load(p4.job_dir(root, plan, task, "A0")/f"round{step:02d}.pt")
                b = io.load(p4.job_dir(original, plan, task, "A0")/f"round{step:02d}.pt")
                assert_nested_equal(a["after"], b["after"])
                assert_nested_equal(a["predictions"], b["predictions"])
                self.assertEqual(a["side"]["status"], "available")
                self.assertLess(a["r5_side"]["candidates"][1]["increment_residual"], 1e-7)
                torch.testing.assert_close(a["r5_side"]["candidates"][1]["increment"],
                                           -a["r5_side"]["candidates"][-1]["increment"], atol=1e-10, rtol=1e-7)
                io.atomic_tensor(root / "diagnostics" / task.key / f"side{step:02d}.pt", a["r5_side"])
            rows = evaluate.side_rows(root, plan, task, torch.arange(1085)%2,
                    dict(dataset="fixture",split="fixture",origin="fresh",source=2,target=3,seed=3405))
            pairs = evaluate.side_pairs(rows)
            self.assertEqual(len(pairs), 20)
            for attempt in (1,2):
                chosen = {r["pool"]:r for r in pairs if r["attempt"]==attempt and r["contrast"]=="normal_minus_0"}
                self.assertAlmostEqual(chosen["target"]["nll"], sum(chosen[p]["nll_contribution"]
                    for p in ("initially_correct", "initially_wrong")), places=12)
