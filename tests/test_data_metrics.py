import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cela.data import Graph, build_tu_index, load_task
from cela.metrics import equal_positive_corrections, metrics


class DataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # Five TU graphs, one isolated node in train and one in val. The val/A/H
        # distributions differ visibly from train to catch accidentally pooled fits.
        self.put('graph_indicator', '1\n1\n2\n3\n4\n4\n5\n')
        self.put('node_labels', '7\n3\n7\n9\n9\n8\n9\n')
        self.put('node_attributes', '200,0\n200,1\n200,2\n200,-100\n200,100\n200,0\n200,0\n')
        self.put('A', '2,1\n1,2\n5,6\n6,5\n')
        self.put('edge_labels', '6\n6\n6\n6\n')
        self.put('graph_labels', 'FORBIDDEN\n')
        self.manifest = dict(dataset='T', roles={'train':[0,1], 'val':[2], 'A':[3], 'H':[4]},
                             typed_edges=True, node_type_provenance='Fixture categorical attributes',
                             source_labels={'train':'train.json','val':'val.json'})
        self.config = {'dataset':'T', 'raw_dir':str(self.root), 'manifest':str(self.root/'task.json')}
        (self.root/'train.json').write_text('{"0":4,"1":-2}')
        (self.root/'val.json').write_text('{"2":4}')
        self.index_version = 0
        self.prepare()

    def prepare(self):
        self.index_version += 1
        name = f'index{self.index_version}.json'
        self.manifest['tu_index'] = name
        (self.root/'task.json').write_text(json.dumps(self.manifest))
        build_tu_index(self.config, self.root/name)

    def put(self, field, content):
        (self.root/f'T_{field}.txt').write_text(content)

    def get(self, roles, pre=None):
        (self.root/'task.json').write_text(json.dumps(self.manifest))
        return load_task(self.config, roles=roles, preprocessing=pre)

    def test_source_only_fit_unknown_types_and_separate_permissions(self):
        self.manifest['continuous_node_attributes'] = dict(columns=[1,0], exogenous=True,
                                                           provenance='External assay in columns 1 and 0')
        self.prepare()
        t = self.get(['train','val'])
        pre = t.preprocessing
        self.assertEqual(pre['node_values'], [3,7])
        self.assertEqual(pre['common_types'], [7,3])
        self.assertEqual(pre['edge_values'], [6])
        self.assertEqual(pre['class_values'], [-2,4])
        self.assertEqual(t.labels['train'].tolist(), [1,0])
        self.assertEqual(pre['continuous_mean'], [1.,200.])
        np.testing.assert_allclose(pre['continuous_scale'], [np.sqrt(2/3),1.])
        self.assertEqual(tuple(t.graphs['train'][1].edges.shape), (0,2))
        self.assertEqual(t.graphs['train'][0].edges.tolist(), [[0,1]])
        self.assertEqual(t.graphs['train'][0].edge_type.tolist(), [0])
        self.assertEqual(t.graphs['val'][0].x[0,:3].tolist(), [0,0,1])
        self.assertFalse(hasattr(t.graphs['val'][0], 'y'))
        for fname in ('train.json','val.json','T_graph_labels.txt'):
            (self.root/fname).unlink()
        a = self.get(['A'], pre)
        self.assertEqual(a.labels, {})
        self.assertEqual(a.graphs['A'][0].node_type.tolist(), [9,8])
        self.assertAlmostEqual(a.graphs['A'][0].x[0,-2].item(), 99/np.sqrt(2/3), places=4)
        self.assertEqual(self.get(['H'], pre).graphs['H'][0].id, '4')
        with self.assertRaisesRegex(ValueError, 'frozen'):
            self.get(['A'])
        self.manifest['roles']['train'] = [1,0]
        with self.assertRaisesRegex(ValueError, 'frozen'):
            self.get(['H'],pre)

    def test_capabilities_and_structure_checks(self):
        self.put('node_attributes','UNREADABLE\n')  # default features are categorical only
        t = self.get(['train'])
        self.manifest['continuous_node_attributes'] = dict(columns=[0], exogenous=False, provenance='degree')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.get(['train'])
        self.manifest.pop('continuous_node_attributes')
        self.put('edge_attributes','1\n')
        with self.assertRaisesRegex(ValueError, 'continuous edge'):
            self.get(['train'])
        (self.root/'T_edge_attributes.txt').unlink()
        self.put('edge_labels','6\n6\n6\n8\n')
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.get(['A'],t.preprocessing)
        self.put('edge_labels','6\n6\n8\n8\n')
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            self.get(['A'],t.preprocessing)
        self.put('A','1,2\n1,2\n5,6\n6,5\n')
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.get(['train'])
        self.put('A','1,2\n2,3\n5,6\n6,5\n')
        with self.assertRaisesRegex(ValueError, 'cross'):
            self.prepare()
        self.manifest['roles']['H'] = [3]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.get(['train'])

    def test_no_categorical_or_edge_labels(self):
        (self.root/'T_node_labels.txt').unlink()
        self.manifest['node_type_provenance'] = None
        (self.root/'T_edge_labels.txt').unlink()
        self.manifest['typed_edges'] = False
        self.put('A', '')
        self.prepare()
        t = self.get(['train'])
        g = t.graphs['train'][0]
        self.assertEqual(g.x.tolist(), [[1.], [1.]])
        self.assertEqual(g.node_type.tolist(), [0,0])
        self.assertEqual(t.preprocessing['num_edge_types'], 1)
        self.assertEqual(tuple(g.edges.shape), (0,2))
        (self.root/'train.json').write_text('{"0":-2,"0":4,"1":-2}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.get(['train'])

    def test_categorical_provenance_required_or_explicitly_omitted(self):
        del self.manifest['node_type_provenance']
        with self.assertRaisesRegex(ValueError, 'node_type_provenance'):
            self.prepare()
        self.manifest['node_type_provenance'] = None
        self.prepare()
        task = self.get(['train'])
        self.assertEqual(task.graphs['train'][0].x.tolist(), [[1.], [1.]])

    def test_index_confines_raw_reads_and_detects_moved_rows(self):
        # Add one H edge and index a valid import. Changing only those H edge
        # bytes to an invalid endpoint must leave train and A entirely usable.
        self.put('graph_indicator','1\n1\n2\n3\n4\n4\n5\n5\n')
        self.put('node_labels','7\n3\n7\n9\n9\n8\n9\n9\n')
        self.put('A','2,1\n1,2\n5,6\n6,5\n7,8\n8,7\n')
        self.put('edge_labels','6\n6\n6\n6\n6\n6\n')
        self.prepare()
        before = self.get(['train','val'])
        adapt_before = self.get(['A'], before.preprocessing)
        self.put('A','2,1\n1,2\n5,6\n6,5\n7,0\n0,7\n')
        real_open = Path.open
        def forbid_global_y(path,*args,**kwargs):
            if path.name.endswith('_graph_labels.txt'):
                raise AssertionError('accessed TU graph labels')
            return real_open(path,*args,**kwargs)
        with patch.object(Path,'open',forbid_global_y):
            pre = self.get(['train','val']).preprocessing
            self.assertEqual(before.preprocessing, pre)
            adapt_after = self.get(['A'],pre)
            torch.testing.assert_close(adapt_before.graphs['A'][0].edges, adapt_after.graphs['A'][0].edges)
            with self.assertRaisesRegex(ValueError, 'differ from manifest'):
                self.get(['H'],pre)

    def test_explicit_index_and_layout_requirements(self):
        self.manifest.pop('tu_index')
        with self.assertRaisesRegex(ValueError, 'requires tu_index'):
            self.get(['train'])
        self.put('A','2,1\n5,6\n1,2\n6,5\n')
        with self.assertRaisesRegex(ValueError, 'grouped by nondecreasing'):
            self.prepare()

    def test_graph_contract(self):
        args = (torch.ones(3,1), torch.ones(3, dtype=torch.long))
        for edges in ([[0,0]], [[1,0]], [[0,3]], [[0,1],[0,1]]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                Graph('g',*args,torch.tensor(edges),torch.zeros(len(edges), dtype=torch.long))


class MetricTests(unittest.TestCase):
    def test_multiclass_missing_class_and_alternative_actions(self):
        p = [[.8,.1,.1],[.2,.6,.2],[.1,.3,.6],[.1,.7,.2]]
        t = metrics(p,[0,1,2,0]); self.assertAlmostEqual(t['balanced_accuracy'], 5/6)
        self.assertAlmostEqual(t['auroc'], (5/8 + 2/3 + 1)/3)
        self.assertAlmostEqual(t['nll'], -np.log([.8,.6,.6,.1]).mean())
        missing = metrics(p[:2],[0,1],decisions=[1,1]); self.assertIsNone(missing['balanced_accuracy'])
        self.assertIsNone(missing['auroc']); self.assertEqual(missing['per_class_auroc'],[1.,1.,None])
        self.assertEqual(missing['per_class_recall'],[0.,1.,None]); self.assertAlmostEqual(missing['macro_f1'], 2/9)
        self.assertEqual(metrics([[.5,.5],[.5,.5]],[0,1])['auroc'], .5)
        for malformed in ([[.2,.2]], [[-.1,1.1]], [[1.,float('nan')]]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                metrics(malformed,[1])
        with self.assertRaises(ValueError):
            metrics([[.8,.2]],[1.0])

    def test_saturated_log_probabilities_preserve_tail_ranks_and_nll(self):
        # Every positive probability rounds to exactly 1 or 0. The complement
        # distinguishes positive-class ranks even when log p(class) itself is 0.
        log_p = np.array([[0., -1200.], [0., -900.], [-900., 0.], [-1200., 0.]])
        p = np.exp(log_p)
        y = [0, 1, 0, 1]
        self.assertEqual(metrics(p, y)['auroc'], .5)
        precise = metrics(p, y, log_probabilities=log_p)
        self.assertEqual(precise['auroc'], .75)
        self.assertEqual(precise['nll'], 450.)
        # In the three-class case, summing BOTH other columns decides which
        # of the saturated winners had stronger odds, even when their runners-up differ.
        log_three = np.array([[0.,-1200.,-1300.], [0.,-1400.,-900.], [-1000.,0.,-1200.]])
        multi = metrics(np.exp(log_three), [0,1,2], log_probabilities=log_three)
        self.assertEqual(multi['per_class_auroc'][0], 1.)
        ref = log_p[[0,1,3,2]]
        row = equal_positive_corrections(np.exp(ref), p, y, positive_count=1,
                                         reference_log_probabilities=ref, candidate_log_probabilities=log_p)
        self.assertEqual(row['net_corrections'], 2)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            metrics(p, y, log_probabilities=log_p[::-1])

    def test_equal_positive_rank_corrections_and_reordering_ties(self):
        # Candidate q=2; threshold changes alone cannot score a matched-count gain.
        p=np.array([[.4,.6],[.6,.4],[.6,.4],[.8,.2]])
        q=np.array([[.9,.1],[.1,.9],[.2,.8],[.9,.1]])
        ids=['10','20','2','31']; y=[0,1,1,0]
        row=equal_positive_corrections(p,q,y,ids=ids)
        self.assertEqual((row['positive_count'],row['changed'],row['net_corrections']), (2,2,2))
        order=[2,3,1,0]; self.assertEqual(row, equal_positive_corrections(p[order],q[order],np.array(y)[order],ids=np.array(ids)[order]))
        self.assertEqual(equal_positive_corrections(p,p,y,ids=ids,positive_count=0)['changed'],0)
        self.assertEqual(equal_positive_corrections(p,q,y,ids=ids,positive_count=4)['changed'],0)
        with self.assertRaisesRegex(ValueError, 'unique'):
            equal_positive_corrections(p,q,y,ids=['a']*4)


if __name__ == '__main__':
    unittest.main()
