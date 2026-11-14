"""S5 access boundary and exact initial-asset/real-head interfaces, synthetic IDs."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from test_s1 import make_fixture
from sega_gda.data import SourceGraphs,sha256
from sega_gda.real_data import RealTaskGraphs,InputProtocolRequired
from sega_gda.pilot import ExactInitialAssetDistance,verify_geometric_handoff
from sega_gda.models import SourceClassifier
from sega_gda.initialization import InitializedClassifier
from sega_gda.metric import LocalMetricField
from sega_gda.geometry_head import RiemannianClassifier


class S5Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_pending_protocol_stops_before_any_target_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pending.json';p.write_text(json.dumps(dict(status='PENDING_INPUT_PROTOCOL')))
            with self.assertRaises(InputProtocolRequired):RealTaskGraphs(p,None)

    def test_target_label_stripped_before_separate_and_validation_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,split=make_fixture(root)
            target=root/'target.npy';np.save(target,np.array([12]))
            source=SourceGraphs(cache,split,True);pp=source.fit_preprocessor()
            p=root/'task.json';p.write_text(json.dumps(dict(status='LOCKED_DEVELOPMENT',protocol_authority='synthetic_unit_test',
                target_policy='transductive_unlabeled',data_cache=str(cache),source_split_manifest=str(split),
                source_split_sha256=sha256(split),target_index_file=str(target),target_index_sha256=sha256(target),
                dataset='fixture',use_node_attributes=True)))
            task=RealTaskGraphs(p,pp)
            g=task.graph('fixture:00000012')
            self.assertIsNone(g.y);self.assertNotIn('y',task.inputs)
            task.source.data.y[12]=-999  # Deliberately meaningless target label.
            again=task.graph('fixture:00000012')
            torch.testing.assert_close(g.x,again.x,atol=0,rtol=0)
            self.assertIsNone(again.y)
            with self.assertRaises(PermissionError):task.graph('fixture:00000008')
            self.assertEqual(len(task.validation_graphs()),4)

    def test_exact_asset_specialization_and_actual_head_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache,split=make_fixture(Path(tmp));data=SourceGraphs(cache,split,False)
            model=SourceClassifier(dict(input_dim=2,hidden_dim=2,num_layers=1,edge_dim=0,readout='mean'),2).double()
            pairs=torch.tensor([[[-1.,0.],[1.,0.]]],dtype=torch.float64)
            scale=(pairs[:,0]-pairs[:,1]).square().sum(-1).mean()
            initial=InitializedClassifier(model,scale,1.)
            field=LocalMetricField(torch.zeros(1,2,dtype=torch.float64),torch.ones(1,dtype=torch.float64))
            geometric=RiemannianClassifier.from_s1(initial,pairs,field,chunk_size=8)
            loader=DataLoader(data.materialize(data.train_ids,data.fit_preprocessor()),batch_size=4)
            proof=verify_geometric_handoff(model,geometric,loader,'cpu')
            self.assertEqual(proof['prediction_mismatches'],0)
            asset=ExactInitialAssetDistance(geometric.distance)
            x,y=torch.randn(4,2,dtype=torch.float64),torch.randn(4,2,dtype=torch.float64)
            with torch.no_grad():
                torch.testing.assert_close(asset(x,y),geometric.distance(x,y),atol=1e-10,rtol=1e-8)
                field.B.copy_(torch.tensor([[[.2,.1],[.1,-.2]]],dtype=torch.float64))
            with self.assertRaises(ValueError):asset(x,y)


if __name__=='__main__':unittest.main()
