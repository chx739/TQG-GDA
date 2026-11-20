from fractions import Fraction
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
import torch
from scipy.optimize import minimize

from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    residual_inputs as data, residual_models as m, weighted_residuals as w)
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import roc_rate
from experiments.common.io import write_json


CFG = dict(max_relative_weight=4.,regularization_per_calibration_graph=.01,neighbor_count=8,
           weight_solver=dict(max_iterations=10000,max_first_order_gap=1e-10))


class ResidualTests(TestCase):
    def test_cut_minimizes_count_gap_and_preserves_equal_sizes(self):
        s,v = [9,1,7,5],[8,2,6,4]
        sizes = dict.fromkeys(s+v,2) | {gid:1 for gid in [9,8,2]} | {gid:3 for gid in [7,6,4]}
        cut, directions = data.split(s,v,sizes)
        self.assertEqual(cut,1)
        small,large = (directions[name] for name in data.DIRECTIONS)
        self.assertEqual(small,dict(D=[9],C=[8,2],U=[1,7,5],E=[6,4]))
        self.assertEqual(large,dict(D=small['U'],C=small['E'],U=small['D'],E=small['C']))
        self.assertIsNone(data.split(s,v,dict.fromkeys(s+v,1))[0])

    def test_phase_and_direction_grants_follow_actual_byte_offsets(self):
        small = dict(status='ok',setting='test',roles=dict(D=[0],C=[1],U=[3],E=[2]))
        large = dict(status='ok',setting='test',roles=dict(D=[3],C=[2],U=[0],E=[1]))
        registry = dict(small=small,large=large)
        raw = b'1\n-1\n1\n-1\n'
        grant = dict(offsets=[(0,1),(2,2),(5,1),(7,2)],mapping={'1':0,'-1':1})
        fit = data.ScopedLabels({},registry,'fit')
        with mock.patch.object(fit,'label_file',return_value=(io.BytesIO(raw),grant)) as opener:
            with fit.direction('small'):
                with self.assertRaises(PermissionError):
                    fit.load('small','U',[3])
                with self.assertRaises(PermissionError):
                    fit.load('large','D',[3])
                opener.assert_not_called()
                np.testing.assert_array_equal(fit.load('small','D',[0]),[0])
                np.testing.assert_array_equal(fit.load('small','C',[1]),[1])
            with fit.direction('large'):
                np.testing.assert_array_equal(fit.load('large','D',[3]),[1])
                np.testing.assert_array_equal(fit.load('large','C',[2]),[0])
                with self.assertRaises(PermissionError):
                    fit.load('large','C',[2])
        data.validate_receipt(registry,fit.actual,'fit')
        score = data.ScopedLabels({},registry,'score')
        with mock.patch.object(score,'label_file') as opener, score.direction('small'):
            with self.assertRaises(PermissionError):
                score.load('small','D',[0])
            opener.assert_not_called()
        with self.assertRaises(PermissionError):
            data.validate_receipt(registry,fit.actual,'score')

    def test_prepare_never_opens_source_members_with_class_counts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = range(1,5)
            graphs = []
            for n in sizes:
                arcs = [(i,i+1) for i in range(n-1)]
                edges = torch.tensor(arcs+[(b,a) for a,b in arcs],dtype=torch.int64).reshape(-1,2).T
                graphs.append(dict(x=torch.eye(3)[[0]*n],edge_attr=None,edge_index=edges))
            hashes = [dict(id=i,input_sha256={k:None if v is None else hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                            for k,v in g.items()}) for i,g in enumerate(graphs)]
            np.save(root/'split_0.npy',np.array([0,1,2,3]))
            sha = hashlib.sha256((root/'split_0.npy').read_bytes()).hexdigest()
            source = mock.Mock(count=4,graphs=graphs,graph_hashes=hashes,receipt={'inputs/graphs.pt':'graphs'},
                              plan=dict(setting=dict(domain_indices=[dict(domain=0,path='split_0.npy',count=4,sha256=sha)])))
            source.spec = dict(domain_index_pattern='split_{domain}.npy',domain_counts=[4])
            for attr in ('_read','role','source_graphs'):
                setattr(source,attr,mock.Mock(side_effect=AssertionError('Whole-source class counts must remain unopened')))
            archive = object.__new__(data.Archive)
            archive.repo = root
            archive.by_setting = {'proteins_edge':source}
            archive.scope = dict(settings=['proteins_edge'],domains=[0],seed=3405)
            archive.fit_access = {'proteins_edge-M0-seed-3405':dict(source_ids=[0,2],validation_ids=[1,3],
                                                                   graph_container_sha256='graphs')}
            roles,_,topology = archive.prepare()
            self.assertTrue(all(row['status']=='ok' for row in roles.values()))
            self.assertEqual(next(iter(topology.values()))['cut'],1)
            source._read.assert_not_called()
            source.role.assert_not_called()
            source.source_graphs.assert_not_called()

    def test_independent_quadratic_program_and_active_limit(self):
        rng = np.random.default_rng(91)
        f = rng.normal(size=(9,5))
        mu = f[0].copy()
        n = len(f)
        uniform = np.full(n,1/n)
        gram = f@f.T
        cross = f@mu
        weights, diag = m.match(gram,cross,mu@mu,CFG)
        def objective(x):
            return np.square(x@f-mu).sum()+.01*n*np.square(x-uniform).sum()
        def jac(x):
            return 2*((x@f-mu)@f.T+.01*n*(x-uniform))
        ref = minimize(objective,uniform,jac=jac,method='SLSQP',bounds=[(0,4/n)]*n,
                       constraints=[dict(type='eq',fun=lambda x:x.sum()-1.,jac=lambda x:np.ones(n))],
                       options=dict(ftol=1e-13,maxiter=1000))
        self.assertTrue(ref.success)
        self.assertAlmostEqual(objective(weights),ref.fun,places=10)
        self.assertAlmostEqual(weights.sum(),1.)
        self.assertGreater(diag['capped_count'],0)
        self.assertLessEqual(diag['first_order_gap'],1e-10)
        self.assertAlmostEqual(diag['distance_matched'],np.linalg.norm(weights@f-mu))
        self.assertAlmostEqual(diag['effective_n'],1/np.square(weights).sum())

    def test_target_only_coordinates_count_in_distance(self):
        features = {1:({(0,b'C'):1.},None),2:({(0,b'U'):1.},None)}
        gram,cross,um = m.wl_geometry(features,[1],[2])
        mu = np.asarray(um.mean(axis=0)).ravel()
        weights, checks = m.match(gram,cross[0],mu@mu,CFG)
        np.testing.assert_array_equal(weights,[1.])
        self.assertAlmostEqual(checks['distance_uniform'],np.sqrt(2))
        self.assertAlmostEqual(checks['distance_matched'],np.sqrt(2))

    def test_control_variate_difference_and_clipping_are_separate(self):
        r,gap,residual = m.estimates(np.array([.8,.2]),np.array([0,1]),np.array([.9]),np.array([.7]),
                                    np.array([0.,1.]),np.array([[0.,1.]]))
        self.assertAlmostEqual(r['WeightedResidual'],1.7)
        self.assertAlmostEqual(r['WeightedLabels'],1.)
        self.assertAlmostEqual(gap,.7)
        self.assertAlmostEqual(residual,0.)
        self.assertAlmostEqual(r['Neighbors8'],1.)

    def test_single_class_D_fallback_and_empty_U_stratum(self):
        single = m.fit({},[5,6],np.ones(2,dtype=int),None)
        self.assertEqual(single['prior_exact'],'5/6')
        out = m.margins(single,{},[2,7])
        np.testing.assert_array_equal(out[0],out[1])
        np.testing.assert_array_equal(out[0]>0.,[True,True])
        roles = dict(D=[1],C=[2,3],U=[4,5],E=[6,7])
        features = {i:({(0,b'x'):1.},None) for i in range(1,8)}
        outputs = [(np.array([-.2,.3]),np.zeros(2)),(np.array([-.5,.6]),np.ones(2)),
                   (np.array([.9,-.9]),np.array([-1.,1.]))]
        with mock.patch.object(m,'fit',return_value=single),mock.patch.object(m,'margins',side_effect=outputs):
            _,est,act = m.fitted_predictions(features,roles,np.array([1]),np.array([0,1]),CFG,None)
        row = est['groups'][0]
        self.assertEqual((row['u_n'],row['e_n'],row['status']),(0,1,'fallback_WL'))
        self.assertEqual(len(set(row['clipped'].values())),1)
        for name in m.ESTIMATORS:
            self.assertEqual(act[name][0],1)
            self.assertEqual(row['thresholds'][name],dict(cut=0.,u_predicted_positive=0))

    def test_exact_c_prior_keeps_lower_rate_in_a_mathematical_tie(self):
        raw = dict.fromkeys(m.ESTIMATORS,.1)
        rates = m.clipped_rates(raw,source_prior=Fraction(1,10),constant=None,has_u=True)
        self.assertEqual(rates['CPrior'],Fraction(1,10))
        self.assertEqual(roc_rate.quantile(np.arange(5,dtype=np.float64),rates['CPrior']),(float('inf'),0))

    def test_constant_d_corrections_share_exact_policy_with_direct_labels(self):
        d,c,u,e=range(2),range(2,8),range(8,15),range(15,17)
        roles=dict(zip(data.ROLES,map(list,(d,c,u,e)),strict=True))
        features={i:({(0,b'x'):1.},None) for i in range(17)}
        _,estimate,actions=m.fitted_predictions(features,roles,np.zeros(2,dtype=int),np.array([0,1]*3),CFG,None)
        row=estimate['groups'][0]
        self.assertEqual(row['exact_rate']['UniformResidual'],'1/2')
        self.assertEqual(row['exact_rate']['CPrior'],'1/2')
        self.assertEqual(row['exact_rate']['WLMean'],'1/6')
        self.assertEqual(row['exact_rate']['WeightedResidual'],row['exact_rate']['WeightedLabels'])
        self.assertEqual(row['weighted_model_gap'],0.)
        for name in ('UniformResidual','CPrior'):
            np.testing.assert_array_equal(actions[name],[0,0])

    def test_uniform_matching_and_neighbor_fallback_share_c_prior_at_exact_ties(self):
        d,c,u,e=range(2),range(2,42),range(42,49),range(49,51)
        roles=dict(zip(data.ROLES,map(list,(d,c,u,e)),strict=True))
        features={i:({(0,b'c' if i in c else b'u'):1.},None) for i in range(51)}
        _,estimate,actions=m.fitted_predictions(features,roles,np.zeros(2,dtype=int),np.array([0,1]*20),CFG,None)
        row=estimate['groups'][0]
        self.assertTrue(row['matching_uniform'])
        self.assertTrue(row['neighbors_uniform'])
        for name in ('WeightedResidual','WeightedLabels','UniformResidual','CPrior','Neighbors8'):
            self.assertEqual(row['exact_rate'][name],'1/2')
            np.testing.assert_array_equal(actions[name],[0,0])
        # Tied positive support is uniform only when the top-k rule keeps
        # every C graph. A row with no positive similarities falls back.
        edges=np.array([[.3,.3,.3],[.1,.1,.3],[0.,0.,0.]])
        np.testing.assert_array_equal(m.uniform_neighbors(edges,3),[True,False,True])
        np.testing.assert_array_equal(m.uniform_neighbors(edges,2),[False,False,True])

    def test_shared_wl_probabilities_reduce_residual_even_with_two_d_classes(self):
        d,c,u,e=range(3),range(3,5),range(5,12),range(12,14)
        roles=dict(zip(data.ROLES,map(list,(d,c,u,e)),strict=True))
        features={i:({(0,b'd' if i in d else b'b' if i==4 else b'a'):1.},np.zeros(2)) for i in range(14)}
        optimizer=dict(max_iterations=4000,gradient_tolerance=1e-10,
                       function_tolerance=1e-15,stationarity_tolerance=1e-7)
        fit,estimate,actions=m.fitted_predictions(features,roles,np.array([1,1,0]),np.array([0,1]),CFG,optimizer)
        self.assertEqual(fit['status'],'ok')
        row=estimate['groups'][1]
        self.assertFalse(row['matching_uniform'])
        self.assertFalse(row['neighbors_uniform'])
        self.assertAlmostEqual(row['shared_wl_probability'],2/3,places=7)
        self.assertEqual(row['exact_rate']['UniformResidual'],'1/2')
        self.assertEqual(row['exact_rate']['WeightedResidual'],row['exact_rate']['WeightedLabels'])
        np.testing.assert_array_equal(actions['UniformResidual'],[0,0])
        self.assertIsNone(m.common_wl_probability(np.array([.5]),np.array([np.nextafter(.5,1.)])))
        raw=dict.fromkeys(m.ESTIMATORS,.6) | {'SizeMean':.9}
        rates=m.clipped_rates(raw,source_prior=Fraction(1,2),constant=None,has_u=True,shared_wl_probability=.6)
        self.assertEqual(rates['SizeMean'],Fraction.from_float(.9))

    def test_stratified_and_pool_error_are_distinct(self):
        records = [
            dict(exact_rate={'arm':'0'},raw={'arm':0.},status='fit'),
            dict(exact_rate={'arm':'1'},raw={'arm':1.},status='fit')]
        scored,_ = w.proportion_errors(records,np.array([0,0,1,1]),np.array([1,1,0,0]),'arm')
        self.assertEqual(scored,dict(strat_mae=1,pool_bias=0,pool_abs_error=0))

    def test_unsealed_directions_never_open_score_grants(self):
        cfg=dict(settings=['proteins_edge'],domains=[0],seed=3405,r10_fit_sha256='test')
        with TemporaryDirectory() as directory,mock.patch.object(w.data,'ScopedLabels') as open_labels:
            with self.assertRaises(FileNotFoundError):
                w.score(Path(directory),cfg,None,{}, {},'not-sealed')
            open_labels.assert_not_called()

    def test_full_grid_fit_seal_then_score_label_reads(self):
        cfg = CFG | dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],domains=list(range(4)),
                         seed=3405,r10_fit_sha256='fixture',minimum_u_mae_gain_fraction=[1,100],
                         minimum_e_accuracy_gain_fraction=[1,200])
        tasks = data.grid(cfg)
        roles,features,topology = {},{},{}
        for setting in cfg['settings']:
            for source in cfg['domains']:
                start=4*source
                sizes={start:1,start+1:2,start+2:1,start+3:2}
                cut,assignments=data.split([start,start+1],[start+2,start+3],sizes)
                group=f'{setting}-M{source}-seed-3405'
                topology[group]=dict(cut=cut,source_ids=[start,start+1],validation_ids=[start+2,start+3],sizes=[1,2,1,2])
                features[setting,source]={i:({(0,b'x'):1.},None) for i in sizes}
                for direction,fields in assignments.items():
                    key=data.key(setting,source,direction,3405)
                    roles[key]=dict(**tasks[key],cut=cut,roles=fields,status='ok',reason=None)
        raw=b'1\n'*16
        meta=dict(offsets=[(2*i,1) for i in range(16)],mapping={'1':0},sha256=hashlib.sha256(raw).hexdigest())
        archive=mock.Mock(by_setting={s:mock.Mock(receipt={}) for s in cfg['settings']},plan=dict(solver={}))
        orig=data.ScopedLabels.load
        events=[]
        with TemporaryDirectory() as directory:
            out=Path(directory)
            write_json(out/'executed-config.json',dict(fixture=True))
            def reads(session,key,role,ids):
                sealed=(out/'OBS.json').is_file()
                self.assertEqual(sealed,session.phase=='score')
                events.append((session.phase,key,role,tuple(ids)))
                return orig(session,key,role,ids)
            with mock.patch.object(data.ScopedLabels,'load',new=reads), mock.patch.object(
                    data.ScopedLabels,'label_file',side_effect=lambda _: (io.BytesIO(raw),meta)):
                with redirect_stdout(io.StringIO()):
                    observation=w.freeze(out,cfg,archive,roles,features,topology)
                self.assertEqual(len(events),48)
                self.assertTrue(all(event[0]=='fit' for event in events))
                w.score(out,cfg,archive,roles,topology,observation)
            self.assertTrue((out/'FINAL.json').is_file())
            self.assertEqual(len(events),96)
            for phase in ('fit','score'):
                self.assertEqual([event for event in events if event[0]==phase],
                                 [(phase,key,r,tuple(entry['roles'][r])) for key,entry in roles.items()
                                  for r in data.PHASE_ROLES[phase]])
                actual=json.loads((out/f'{phase}-label-access.json').read_bytes())
                data.validate_receipt(roles,actual,phase)

    def test_unavailable_directions_stay_in_denominator(self):
        cfg=dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],domains=[0,1,2,3],seed=3405)
        rows={k:dict(**v,status='unavailable') for k,v in data.grid(cfg).items()}
        summary, _, _, gate = w.summarize(cfg,rows,{},{},{})
        self.assertEqual((summary[0]['available'],summary[0]['expected']), (0,8))
        self.assertIsNone(summary[0]['accuracy'])
        self.assertFalse(gate['register_real_domain_validation'])
