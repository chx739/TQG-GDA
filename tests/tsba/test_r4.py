"""R4 regression checks for non-M0 tasks, checkpoints and exact accuracy pairs."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tsba.classifier import ResidualClassifier
from experiments.mutagenicity.r2_20261111.src.common import atomic_tensor, batches, load, seal, write_json
from experiments.mutagenicity.r2_20261111.src.score import assert_nested_equal
from experiments.mutagenicity.r3_20261112.src.source_roles import split_source_roles
from experiments.mutagenicity.r4_20261112.src.evaluate import METRICS, correction, summarize
from experiments.mutagenicity.r4_20261112.src.protocol import (
    Task, batch_path, evaluation_labels, job_dir, labels_path, score_dir, source_dir, split_domain, tasks,
)
from experiments.mutagenicity.r4_20261112.src.training import train_score, train_source
from experiments.mutagenicity.r4_20261112.src.trajectory import adapt

REPO = Path(__file__).resolve().parents[2]
PLAN = json.loads((REPO / "experiments/mutagenicity/r4_20261112/configs/r4-plan.json").read_text())


class R4Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(82)
        self.plan = deepcopy(PLAN)

    def test_registry_and_non_m0_sega_rounding(self):
        cases = tasks(self.plan)
        paths = {job_dir(Path("/tmp"), self.plan, task, m) for task in cases for m in ("B1","B2","A1","A2","A3")}
        self.assertEqual((len(cases), len(paths)), (60, 300))
        self.assertEqual(len({score_dir(Path("/tmp"), self.plan, task) for task in cases}), 60)
        self.assertEqual(len({source_dir(Path("/tmp"), self.plan, task.source, task.seed) for task in cases}), 20)
        ids = np.arange(10, 1095)
        row, fit, val = split_domain(ids, [0]*670+[1]*415, {"0":0,"1":1},
                                     split_source_roles, 3405, 1095, (.4,.4))
        self.assertEqual((len(fit),len(val)), (868,217))
        self.assertEqual(sorted(row["fit"]+row["validation"]), ids.tolist())

    def test_incomplete_observations_cannot_open_evaluation_labels(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            write_json(root/"protocol/r4-plan.json", self.plan)
            write_json(root/"replay.json", self.plan["counts"])
            write_json(root/"inputs/label_records.json", {"path":"must-not-open"})
            write_json(root/"inputs/domains.json", {})
            write_json(root/"inputs/registry.json", [])
            seal(root,"R4_INPUT",["inputs"])
            seal(root,"R4_OBS",["protocol","replay.json","inputs","seals/R4_INPUT.json"])
            with self.assertRaises(PermissionError):
                evaluation_labels(root)

    def test_exact_accuracy_pair_and_direction_weight(self):
        rows = []
        for task in tasks(self.plan):
            n = self.plan["domain_counts"][task.target]
            for method in self.plan["methods"]:
                gain = 1 if method == "A2" and task.source == 0 and task.target == 3 else 0
                correct = n//2 + gain
                values = {key:.5 for key in METRICS}
                values["accuracy"] = correct/n
                values["nll"] -= .001 * gain
                rows.append(dict(source=task.source,target=task.target,seed=task.seed,method=method,
                                 attempt=30,pool="target",correct=correct,n=n,**values))
        pairs, summary = summarize(self.plan, rows)
        avg = next(r for r in summary if r["method_or_contrast"]=="A2−B0" and r["scope"]=="Avg")
        self.assertEqual(avg["accuracy"]["mean"], 1/(1085*12))
        self.assertEqual(avg["accuracy"]["wins"], 5)
        self.assertEqual(avg["nll"]["wins"], 5)
        self.assertEqual(sum(r["correct_difference"]==0 for r in pairs if r["contrast"]=="A2−B0"),55)

    def test_corrections_conserve_count_and_class(self):
        y = torch.tensor([0,0,1,1])
        before = torch.tensor([[0.,1.],[1.,0.],[1.,0.],[0.,1.]])
        after = torch.tensor([[1.,0.],[0.,1.],[0.,1.],[0.,1.]])
        r = correction(after,before,y)
        self.assertEqual((r["wrong_to_right"],r["right_to_wrong"],r["net_correct"]),(2,1,1))
        self.assertEqual((r["net_correct_class0"],r["net_correct_class1"]),(0,1))

    def prepare_adapt(self, root, task):
        x = torch.randn(1101,16,dtype=torch.float64)*.1
        model = ResidualClassifier(16,2,16,dtype=torch.float64)
        target = list(range(16,1101))
        out = source_dir(root,self.plan,task.source,task.seed)
        atomic_tensor(out/"frozen.pt",dict(x=x,initial_head=model.state_dict()))
        write_json(root/"inputs/domains.json", {str(task.target):target})
        fit = list(range(12))
        write_json(root/"roles/M2/seed3405.json",dict(fit=fit,validation=list(range(12,16))))
        atomic_tensor(labels_path(root,2,3405,"fit"), torch.arange(12)%2)
        atomic_tensor(batch_path(root,task),batches(fit,target,3405,rounds=2,size=8))
        from tsba.boundary import build_boundaries
        from experiments.mutagenicity.r2_20261111.src.adapt import ZeroScore
        from experiments.mutagenicity.r4_20261112.src.protocol import config
        from dataclasses import asdict
        b = build_boundaries(model,x[target[:8]],ZeroScore(),1.,config(self.plan,3405))
        atomic_tensor(out/"geometry_M3.pt",dict(sigma=1.,boundaries=asdict(b)))

    def test_m3_score_resume_preserves_optimizer_and_rng(self):
        self.plan["model"]["score_epochs"] = 2
        self.plan["score"].update(endpoints=[9,18],recovery_epochs=1)
        task = Task(2,3,3405)
        with TemporaryDirectory() as d:
            one,two = Path(d)/"one",Path(d)/"two"
            torch.manual_seed(82)
            self.prepare_adapt(one,task)
            torch.manual_seed(82)
            self.prepare_adapt(two,task)
            train_score(one,self.plan,task)
            def interrupted(path, payload):
                atomic_tensor(path,payload)
                if path.name=="epoch0001.pt":
                    raise InterruptedError("after publication")
            with patch("experiments.mutagenicity.r4_20261112.src.training.atomic_tensor",side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    train_score(two,self.plan,task)
            train_score(two,self.plan,task)
            for name in ("epoch0002.pt","diagnostics.pt"):
                a,b=[load(score_dir(root,self.plan,task)/name) for root in (one,two)]
                assert_nested_equal(a,b)
            self.assertEqual(load(score_dir(two,self.plan,task)/"epoch0002.pt")["updates"],18)

    def test_non_m0_trajectory_reads_only_its_source_and_resumes(self):
        self.plan["model"]["adapt_rounds"]=2
        self.plan["side_attempts"]=[1,2]
        task=Task(2,3,3405)
        with TemporaryDirectory() as d:
            one,two=Path(d)/"one",Path(d)/"two"
            torch.manual_seed(82)
            self.prepare_adapt(one,task)
            torch.manual_seed(82)
            self.prepare_adapt(two,task)
            with patch("experiments.mutagenicity.r4_20261112.src.trajectory.load", wraps=load) as reader:
                adapt(one,self.plan,(task,"A3"))
                paths=[p.args[0] for p in reader.call_args_list if "source_labels" in p.args[0].parts]
                self.assertEqual(paths,[labels_path(one,2,3405,"fit")])
            adapt(two,self.plan,(task,"A3"),stop_after=1)
            path=job_dir(two,self.plan,task,"A3")/"round01.pt"
            inode=path.stat().st_ino
            adapt(two,self.plan,(task,"A3"))
            self.assertEqual(path.stat().st_ino,inode)
            for step in range(3):
                a,b=[load(job_dir(root,self.plan,task,"A3")/f"round{step:02d}.pt") for root in (one,two)]
                assert_nested_equal(a,b)
                self.assertEqual(a["predictions"]["target"].shape,(1085,2))

    def test_source_epoch_resume_preserves_optimizer_and_selection(self):
        self.plan["model"].update(source_epochs=2,source_batch=4)
        self.plan["model"]["adapt_rounds"]=2
        s,seed=2,3405
        with TemporaryDirectory() as d:
            one,two=Path(d)/"one",Path(d)/"two"
            data=[dict(x=torch.randn(3,2),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                       edge_attr=torch.zeros(4,1)) for _ in range(22)]
            fit,val=list(range(8)),list(range(8,12))
            domains={"0":[12,13,14],"1":[15,16,17],"2":list(range(12)),"3":[18,19,20,21]}
            for root in (one,two):
                atomic_tensor(root/"inputs/graphs.pt",data)
                write_json(root/"inputs/domains.json",domains)
                write_json(root/"roles/M2/seed3405.json",dict(fit=fit,validation=val))
                for name,ids in (("fit",fit),("validation",val)):
                    atomic_tensor(labels_path(root,s,seed,name),torch.arange(len(ids))%2)
                for target in (0,1,3):
                    atomic_tensor(batch_path(root,Task(s,target,seed)),batches(fit,domains[str(target)],seed,rounds=2,size=3))
            train_source(one,self.plan,(s,seed))
            def interrupted(path,payload):
                atomic_tensor(path,payload)
                if path.name=="epoch000.pt":
                    raise InterruptedError("after source epoch")
            with patch("experiments.mutagenicity.r4_20261112.src.training.atomic_tensor",side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    train_source(two,self.plan,(s,seed))
            train_source(two,self.plan,(s,seed))
            for name in ("epochs/epoch001.pt","source.pt","frozen.pt"):
                a,b=[load(source_dir(root,self.plan,s,seed)/name) for root in (one,two)]
                assert_nested_equal(a,b)
