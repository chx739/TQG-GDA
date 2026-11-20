from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from scipy import sparse

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    geometry_models as geom, reference_geometry as run, wl)


SOLVER = dict(max_iterations=2000, gradient_tolerance=1e-10, function_tolerance=1e-14,
              stationarity_tolerance=1e-7)


def records(x):
    return {i:{(0,bytes([j])):v for j,v in enumerate(row) if v}
            for i,row in enumerate(x)}


def source_fixture(x):
    sizes = np.column_stack((np.linspace(-1,1,len(x)), np.ones(len(x))))
    return {i:(r,sizes[i]) for i,r in records(x).items()}


class ReferenceGeometryTests(TestCase):
    def test_full_and_diagonal_kernel_use_population_law(self):
        x = np.array([[1.,0,2],[0,2.,0],[1,1.,0],[0,1.,3],[2,0.,1]])
        rows = records(x)
        queries, refs = [0,1], [2,3,4]
        p = geom.reference_features(rows,queries,refs)
        cov = np.cov(x[refs],rowvar=False,bias=True)
        np.testing.assert_allclose(p@p.T,x[queries]@cov@x[queries].T,atol=1e-14)
        vocab = wl.vocabulary(rows,queries)
        f = wl.sparse_rows(rows,queries,vocab)
        extra = geom.reference_blocks(rows,queries,vocab,f,refs,refs)['TargetDiag']
        np.testing.assert_allclose((extra@extra.T).toarray(),x[queries]@np.diag(cov.diagonal())@x[queries].T)
        np.testing.assert_allclose(geom.column_variance(sparse.csr_matrix(x)),x.var(0))

    def test_full_similarity_transmits_words_absent_from_D(self):
        x = np.array([[1.,0,0],[0,1,0],[1,0,1],[0,1,0],[0,0,1],[0,0,0]])
        rows = records(x)
        d,u,query = [0,1],[2,3],[4,5]
        d_only = {i:{w:v for w,v in row.items() if w in wl.vocabulary(rows,d)} for i,row in rows.items()}
        src = geom.reference_features(rows,d,u)
        target = geom.reference_features(rows,query,u)
        np.testing.assert_allclose(src@target.T, [[.25,0],[-.25,0]],atol=1e-14)
        np.testing.assert_array_equal(src@geom.reference_features(d_only,query,u).T,np.zeros((2,2)))
        action = (src[0]@target.T>0).astype(int)
        np.testing.assert_array_equal(action,[1,0])
        self.assertEqual(np.mean(action==np.array([1,0])),1)
        self.assertEqual(np.mean(action==np.array([0,1])),0)

    def test_reference_permutation_and_duplicate_F_preserve_logistic_margin(self):
        rng = np.random.default_rng(911)
        f = sparse.csr_matrix(rng.normal(size=(16,4)))
        extra = rng.normal(size=(16,3))
        z = rng.normal(size=(16,2))
        y = np.arange(16)%2
        plan = dict(C=[1.],solver=SOLVER)
        def margins(x):
            candidates = structural.fit_candidates(x[:12],x[12:],y[:12],y[12:],plan)
            return scores(x,structural.select_model(candidates))
        aug = geom.matrix(f,z,extra,0.75)
        np.testing.assert_allclose(margins(aug),margins(geom.matrix(f,z,extra[:,[2,0,1]],0.75)),atol=1e-7)
        np.testing.assert_allclose(margins(geom.matrix(f,z)),margins(geom.matrix(f,z,f,1.)),atol=1e-7)

    def test_actual_fit_matches_D_variance_shares_base_C_and_falls_back(self):
        rnd = np.random.default_rng(41)
        x = rnd.random((14,3))
        x[8:11]=[1,0,0]  # Constant U, both target arms must fall back.
        features = source_fixture(x)
        row = dict(setting='fixture',source=2,roles=dict(D=list(range(5)),C=[5,6,7],U=[8,9,10],E=[11,12,13]))
        d,c = np.array([0,0,1,0,1]),np.array([0,1,1])
        model = geom.fit(features,row,d,c,SOLVER)
        out = geom.decisions(model,features,row['roles']['E'])
        self.assertEqual(model['m'],3)
        self.assertEqual(model['candidates']['SourceFull']['selected_C'],model['selected_base_C'])
        self.assertAlmostEqual(model['geometry']['SourceFull']['D_matched_variance'],
                               model['D_wl_variance'],places=13)
        for name in ('TargetFull','TargetDiag'):
            self.assertTrue(model['geometry'][name]['fallback'])
            np.testing.assert_array_equal(out[f'margin_{name}'],out['margin_Base'])
        self.assertEqual(out['EqualPositive'].sum(),out['TargetFull'].sum())

    def test_hash_subselection_constant_D_and_tie_order(self):
        ids = [3,19,5,22,71,8]
        chosen = geom.select_references(ids,3,'fixture',2)
        self.assertEqual(chosen,geom.select_references(list(reversed(ids)),3,'fixture',2))
        self.assertEqual(len(set(chosen)),3)
        x = np.ones((10,2))
        role = dict(setting='fixture',source=2,roles=dict(D=[0,1],C=[2,3,4],U=[5,6],E=[7,8,9]))
        model = geom.fit(source_fixture(x),role,np.zeros(2,dtype=int),np.array([0,1,0]),SOLVER)
        result = geom.decisions(model,source_fixture(x),role['roles']['E'])
        self.assertEqual(model['status'],'single_class_D')
        np.testing.assert_allclose(result['margin_Base'],np.full(3,np.log(1/5)))
        np.testing.assert_array_equal(result['TargetFull'],[0,0,0])
        np.testing.assert_array_equal(geom.equal_positive(np.array([9,4,2]),np.ones(3),2),[0,1,1])

    def test_no_E_reader_before_successful_obs_replay(self):
        with TemporaryDirectory() as tmp, mock.patch.object(run.data,'ScopedLabels') as labels:
            ref = run.Reference({}, {}, {}, {})
            out = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                run.score(out,dict(source_obs_sha256='old'),ref,None,{},'missing',
                          execution={'obs_sha256':'missing'})
            labels.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
            with mock.patch.object(run,'observed',return_value=({},{})):
                labels.side_effect = RuntimeError('fake E port reached')
                with self.assertRaisesRegex(RuntimeError,'fake E port reached'):
                    run.score(out,{},ref,SimpleNamespace(by_setting={}),{},'verified',
                              execution={'obs_sha256':'verified'})
                self.assertIn('verified',(out/'score-execution.json').read_text())
                with self.assertRaises(FileExistsError):
                    run.score(out,{},ref,SimpleNamespace(by_setting={}),{},'verified',
                              execution={'obs_sha256':'verified'})
                labels.assert_called_once()

    def test_exact_advance_uses_all_controls_and_source_group_signs(self):
        ref = run.Reference(dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge']),{},{},{})
        rows=[dict(setting=s,source=None,left='TargetFull',right=r,accuracy_exact='1/200')
              for s in ref.plan['settings'] for _,r in run.PAIRS]
        rows += [dict(setting=s,source=i,left='TargetFull',right='Base',accuracy_exact='1/200')
                 for i in range(3) for s in ref.plan['settings']]
        self.assertTrue(run.gate(ref,rows)['continue_real_four_domain_pilot'])
        proteins = next(r for r in rows if r['setting']=='proteins_edge' and r['right']=='TargetDiag')
        proteins['accuracy_exact']='0'
        self.assertFalse(run.gate(ref,rows)['continue_real_four_domain_pilot'])
        proteins['accuracy_exact']='1/200'
        base = next(r for r in rows if r['setting']=='proteins_edge' and r['source'] is None and r['right']=='Base')
        base['accuracy_exact']=str(Fraction(1,200)-Fraction(1,10**30))
        self.assertFalse(run.gate(ref,rows)['continue_real_four_domain_pilot'])
