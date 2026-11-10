"""R0 synthetic fixtures only: no Mutagenicity labels or actual training runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from workflows.tsba.validation.archive import digest, write_json
from workflows.tsba.validation.real_data.data import (
    LabelBroker, canonical_sha, label_offsets, probe_folds, reconstruct_graphs,
    rng_seed, source_split,
)
from workflows.tsba.validation.real_data.probes import Probe, fit_probe, load_probe


class InputsTests(unittest.TestCase):
    def test_raw_conversion_matches_pyg_on_synthetic_fixture(self):
        from torch_geometric.io import read_tu_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "A": "1,2\n2,1\n1,2\n2,1\n1,1\n3,4\n4,3\n",
                "graph_indicator": "1\n1\n2\n2\n",
                "node_labels": "2\n3\n2\n4\n",
                "edge_labels": "4\n4\n5\n5\n4\n6\n6\n",
                # Synthetic only: permitted read_tu_data comparison.
                "graph_labels": "-1\n1\n",
            }
            for suffix, value in files.items():
                (root / f"Tiny_{suffix}.txt").write_text(value)
            graphs, size, record = reconstruct_graphs(root, "Tiny", 2, (3,3))
            expected, slices, _ = read_tu_data(str(root), "Tiny")
            for i, actual in enumerate(graphs):
                for key in ("x", "edge_index", "edge_attr"):
                    lo, hi = slices[key][i:i+2]
                    value = expected[key][:,lo:hi] if key == "edge_index" else expected[key][lo:hi]
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            self.assertEqual([r["undirected_edges"] for r in record], [1,1])
            torch.testing.assert_close(size[:,1], torch.full((2,), torch.log(torch.tensor(2.)).item(), dtype=torch.double))
            self.assertNotIn("y", graphs[0])
            # The input rebuild still succeeds if graph labels are unparseable.
            (root / "Tiny_graph_labels.txt").write_text("NEVER_PARSE\nSENTINEL\n")
            reconstruct_graphs(root, "Tiny", 2, (3,3))
            (root / "Tiny_A.txt").write_text("1,3\n3,1\n1,3\n3,1\n1,1\n3,4\n4,3\n")
            with self.assertRaisesRegex(ValueError, "Cross-graph"):
                reconstruct_graphs(root, "Tiny", 2, (3,3))

    def test_source_roles_exact_counts_and_rng(self):
        ids = list(range(1084))
        labels = [0]*453+[1]*631
        split = source_split(ids, labels, 3405)
        self.assertEqual(split, source_split(ids, labels, 3405))
        # Correct counts depend on class sizes. Protocol proportions use exact
        # Python round and floor rather than forcing a desired total.
        self.assertEqual({k:len(split[k]) for k in ("fit_ids","validation_ids","diagnostic_ids")},
                         {"fit_ids":780,"validation_ids":108,"diagnostic_ids":196})
        groups = [set(split[k]) for k in ("fit_ids","validation_ids","diagnostic_ids")]
        self.assertEqual(set.union(*groups), set(ids))
        self.assertEqual(sum(map(len,groups)),len(ids))
        second = source_split(ids, labels, 3406)
        self.assertNotEqual(split["fit_ids"],second["fit_ids"])
        self.assertEqual([len(split[k]) for k in ("fit_ids","validation_ids","diagnostic_ids")],
                         [len(second[k]) for k in ("fit_ids","validation_ids","diagnostic_ids")])
        targets = {"1":{"A":list(range(2000,2271))}}
        folds = probe_folds(split["diagnostic_ids"], targets, "test-protocol")
        for fold in folds["source"].values():
            self.assertEqual((len(fold["budget_ids"]),len(fold["train_ids"]),len(fold["validation_ids"]),len(fold["audit_ids"])), (147,118,29,49))
            self.assertFalse(set(fold["budget_ids"]) & set(fold["audit_ids"]))
        with self.assertRaisesRegex(ValueError,"three"):
            source_split([0,1,2,3], [0,0,1,1], 1)

    def test_broker_denies_unsealed_final_tampering_and_role_crossover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("inputs","roles","seals","logs"):
                (root/name).mkdir()
            raw = root / "raw_labels"
            # Final value is deliberately not a valid integer.
            raw.write_bytes(b"-1\n1\n-1\n1\n-1\n1\nNEVER_PARSE\n")
            roles = dict(source=dict(domain_ids=[0,1,2,3]),targets={"1":dict(ids=[4,5,6],A=[4],B=[5],final=[6])})
            write_json(root/"roles.json",roles)
            write_json(root/"roles/seed7.json",dict(fit_ids=[0,1],validation_ids=[2],diagnostic_ids=[3]))
            write_json(root/"inputs/broker.json",dict(path=str(raw),offsets=label_offsets(raw),sha256=digest(raw)))
            broker = LabelBroker(root)
            self.assertEqual(broker.read([0,1,2,3],"source_allocator","test").tolist(),[-1,1,-1,1])
            write_json(root/"inputs/class_mapping.json",dict(raw_to_class={"-1":0,"1":1}))
            self.assertEqual(broker.read([0,1],"source_fit","test",seed=7).tolist(),[0,1])
            for role, ids, extra in (("source_allocator",[0],{}),("source_fit",[3],dict(seed=7)),
                                      ("source_diagnostic",[3],dict(seed=7)),("target_A",[4],dict(domain=1)),
                                      ("target_B",[5],dict(domain=1)),("target_B",[6],dict(domain=1))):
                with self.assertRaises(PermissionError):
                    broker.read(ids,role,"test",**extra)
            frozen = root/"frozen"
            frozen.write_text("immutable-model")
            s1 = dict(stage="S1",files={"frozen":digest(frozen)})
            s1["digest"] = canonical_sha(s1)
            write_json(root/"seals/S1.json",s1)
            self.assertEqual(broker.read([4],"target_A","test",domain=1,seal=s1["digest"]).tolist(),[0])
            with self.assertRaises(PermissionError):
                broker.read([5],"target_B","test",domain=1,seal=s1["digest"])
            s2 = dict(stage="S2",files={"seals/S1.json":digest(root/"seals/S1.json"),"frozen":digest(frozen)})
            s2["digest"] = canonical_sha(s2)
            write_json(root/"seals/S2.json",s2)
            self.assertEqual(broker.read([5],"target_B","test",domain=1,seal=s2["digest"]).tolist(),[1])
            with self.assertRaises(PermissionError):
                broker.read([6],"target_B","test",domain=1,seal=s2["digest"])
            frozen.write_text("model-was-modified")
            with self.assertRaisesRegex(PermissionError,"modified"):
                broker.read([4],"target_A","test",domain=1,seal=s1["digest"])
            events = [json.loads(s) for s in (root/"logs/label-access.jsonl").read_text().splitlines()]
            self.assertFalse(events[-1]["permitted"])
            self.assertEqual(events[-1]["ids"],[4])


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_affine_and_scaler_chain_rule(self):
        model = Probe(torch.tensor([2.,-3.]),torch.tensor([.5,4.]),"linear")
        x = torch.tensor([1.3,2.1],dtype=torch.double,requires_grad=True)
        w,b = model.affine()
        torch.testing.assert_close(model(x),w@x+b)
        grad = torch.autograd.grad((model(x)[0]-model(x)[1]),x)[0]
        torch.testing.assert_close(grad,w[0]-w[1])

    def test_selection_checkpoint_predictions_and_shared_initialization(self):
        plan = dict(protocol_id="synthetic",probes=dict(lr=.01,weight_decay=.001,updates=40,
                    validation_cadence=20,supplementary_endpoints=[20,40]),
                    resources=dict(max_probe_seconds=120,max_rss_bytes=8*1024**3))
        generator = torch.Generator().manual_seed(4)
        features = torch.randn((20,3),generator=generator,dtype=torch.double)
        labels = {i:i%2 for i in range(16)}
        split = dict(budget_ids=list(range(16)),train_ids=list(range(12)),validation_ids=list(range(12,16)),audit_ids=[16,17])
        info = dict(seed=7,domain=0,origin="source",fold=0,feature="H",head="linear",condition="true")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"source"
            meta = fit_probe(directory,features,labels,split,info,plan,{"B1":[18,19],"audit":[16,17]})
            self.assertEqual(meta["status"],"success",meta.get("failure"))
            model = load_probe(directory)
            state = torch.load(directory/"initial.pt",weights_only=False)
            torch.testing.assert_close(state["mean"],features[:12].mean(0))
            torch.testing.assert_close(state["scale"],features[:12].std(0,unbiased=False))
            predictions = torch.load(directory/"probabilities.pt",weights_only=False)
            torch.testing.assert_close(predictions["selected"]["B1"]["probabilities"],model(features[[18,19]]).softmax(-1))
            history = json.loads((directory/"history.json").read_text())
            chosen = min(history["validation"],key=lambda r:r["validation_ce"])
            self.assertEqual(meta["selected_step"],chosen["step"])
            final = torch.load(directory/"checkpoints/step0040.pt",weights_only=False)
            self.assertEqual(final["step"],40)
            self.assertIn("state",final["optimizer_state"])
            # Source and target initialization keys deliberately share domain0.
            target_info = dict(info,domain=1,origin="target")
            target = Path(tmp)/"target"
            target_meta = fit_probe(target,features,labels,split,target_info,plan,{"B1":[18,19]})
            self.assertEqual(target_meta["status"],"success")
            other = torch.load(target/"initial.pt",weights_only=False)
            for key in state["model_state"]:
                torch.testing.assert_close(state["model_state"][key],other["model_state"][key])
            self.assertEqual(meta["initialization_seed"],target_meta["initialization_seed"])
            bad = dict(labels)
            for i in split["validation_ids"]:
                bad[i]=0
            failed = fit_probe(Path(tmp)/"failed",features,bad,split,info,plan,{})
            self.assertEqual(failed["status"],"failed")
            self.assertIn("missing_class",failed["failure"])
            self.assertFalse((Path(tmp)/"failed/probabilities.pt").exists())

    def test_rng_canonical_spec(self):
        import hashlib
        key = json.dumps(["p","purpose",1,2,3,"H16","linear",-1],ensure_ascii=True,separators=(",",":"))
        expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big") % (2**63-1)
        self.assertEqual(rng_seed("p","purpose",1,2,3,"H16","linear"),expected)


if __name__ == "__main__":
    unittest.main()
