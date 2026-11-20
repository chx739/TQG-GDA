from fractions import Fraction
from hashlib import sha256
from inspect import getmodule
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
from scipy import sparse
import torch

from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import (
    source_attribute_screen as run, source_attributes as a)
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts import source_attribute_screen as cli


class SourceAttributeTests(TestCase):
    def test_code_inventory_covers_implementations_used_for_reading_and_writing(self):
        for function in (a.sha, run.inputs.npz_member, run.write_table):
            with self.subTest(function=function.__name__):
                self.assertIn(getmodule(function), cli.CODE_MODULES)

    def test_raw_hash_and_type_alignment_keep_negative_and_even_median(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            for suffix, payload in (
                ('graph_indicator',b'1\n2\n1\n1\n2\n1\n'),
                ('node_labels',b'0\n1\n0\n1\n2\n0\n'),
                ('node_attributes',b'-9\n3\n-7\n10\n4\n-2\n'),
            ):
                name = f'PROTEINS_{suffix}.txt'
                (root/name).write_bytes(payload)
                files[name] = dict(sha256=sha256(payload).hexdigest(),bytes=len(payload))
            plan = dict(dataset='PROTEINS',input=dict(raw_directory=tmp,node_label_values=[0,1,2],files=files))
            owners, types, values, _ = a.raw_columns(plan,2)
            graph = dict(x=torch.tensor(np.eye(3)[[0,0,1,0]]))
            median, n = a.graph_attributes(0,graph,owners,types,values)
            np.testing.assert_array_equal(median,[-7.,10.,np.nan])
            self.assertEqual(n,4)
            np.testing.assert_array_equal(a.graph_medians(types[owners==1],values[owners==1]),[np.nan,3.,4.])
            np.testing.assert_array_equal(a.graph_medians(np.array([0,0]),np.array([-9.,-2.])),[-5.5,np.nan,np.nan])
            graph['x'] = graph['x'][[0,2,1,3]]
            with self.assertRaises(PermissionError):
                a.graph_attributes(0,graph,owners,types,values)
            (root/'PROTEINS_node_attributes.txt').write_bytes(b'invalid')
            with self.assertRaises(PermissionError):
                a.raw_columns(plan,2)

    def test_missing_and_constant_D_coordinates_are_exactly_zero(self):
        d = np.array([[-3.,np.nan,.1],[-1.,np.nan,.1],[np.nan,np.nan,.1]])
        scaler = a.scale_attributes(d)
        self.assertEqual(scaler,dict(mean=[-2.,0.,float(np.mean(d[:,2]))],scale=[1.,0.,0.],present=[2,0,3]))
        new = np.array([[np.nan,100.,5.],[-4.,1.,9.],[1.,2.,3.]])
        np.testing.assert_array_equal(a.transform_attributes(new,scaler),[[0,0,0],[-2,0,0],[3,0,0]])

    def test_augmented_design_only_appends_three_D_standardized_features(self):
        f = {(0,b'known'):1.}
        novel = {(0,b'unseen'):1.}
        features = {0:(f,np.array([0.,1.]),np.array([-1.,2.,np.nan])),
                    1:(f,np.array([2.,1.]),np.array([1.,2.,np.nan])),
                    2:(novel,np.array([3.,5.]),np.array([3.,4.,1.]))}
        model = a.source_design(features,dict(D=[0,1],C=[2],E=[2]))
        x = a.design_for(features,[2],model)
        np.testing.assert_array_equal(x['Base'].toarray(),[[0,2,4]])
        np.testing.assert_array_equal(x['Attribute'].toarray(),[[0,2,4,3,0,0]])

    def test_single_class_has_same_R9_smoothed_constant(self):
        new = a.fit_new({}, {}, np.zeros(2,dtype=int), np.array([1,0]), {})
        np.testing.assert_allclose(a.predict_new(new,sparse.csr_matrix((3,10))),np.full(3,np.log(.2)))

    def test_pair_net_is_counted_by_graph(self):
        x = {'e_ids':np.arange(3, dtype=np.int64)}
        for name in run.METHODS:
            z = [-1.,2.,3.] if name=='Attribute' else [1.,-2.,3.]
            x[name] = (np.array(z)>0).astype(np.int64)
            x[f'margin_{name}'] = np.array(z,dtype=float)
        result,delta = run.evaluate(x,np.array([0,1,0]))
        self.assertEqual(result['Attribute'][0],Fraction(2,3))
        self.assertEqual((delta['Base']['corrected'],delta['Base']['harmed'],delta['Base']['changed']),(2,0,2))

    def test_gate_uses_exact_ties_and_each_criterion(self):
        rows = [dict(source=None,right=base,accuracy_exact='1/200') for base in run.BASES]
        rows += [dict(source=i,right='Base',accuracy_exact='1/200' if i<3 else '0') for i in range(4)]
        self.assertTrue(run.gate(rows)['advance_source_baseline'])
        for row in rows[:6]:
            prev = row['accuracy_exact']
            row['accuracy_exact'] = '0'
            self.assertFalse(run.gate(rows)['advance_source_baseline'])
            row['accuracy_exact'] = prev
        rows[2]['accuracy_exact'] = str(Fraction(1,200)-Fraction(1,10**30))
        self.assertFalse(run.gate(rows)['advance_source_baseline'])

    def test_failed_OBS_does_not_open_E_or_assign_score_receipt(self):
        with TemporaryDirectory() as td, mock.patch.object(run.data,'ScopedLabels') as labels:
            out = Path(td)
            p = run.Inputs({}, {}, {}, {}, {}, None, {})
            with self.assertRaises(FileNotFoundError):
                run.score(out,dict(geometry_obs_sha256='x',roles_obs_sha256='y'),p,'broken',execution={})
            labels.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
