"""S4 state transitions and dual acceptance on the actual S2/S3 finite engine."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from s4_fixture import make_fixture
from sega_gda.evolution import EvolutionTrainer,acceptance_conditions,_unique_pairs,trust_change

RESULTS={}


def snapshot(engine):
    return {k:v.detach().clone() for k,v in engine.model.state_dict().items()}


class S4Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_content_only_updates_content_with_auxiliary_gradients(self):
        engine=make_fixture()
        engine.metric.project_(torch.tensor([[[.3,.1],[.1,-.3]],[[-.15,.2],[.2,.15]]],dtype=torch.float64))
        before=snapshot(engine)
        result=engine.content_step()
        after=snapshot(engine)
        changed=[k for k in before if not torch.equal(before[k],after[k])]
        self.assertTrue(any(k.startswith('encoder.') for k in changed))
        self.assertIn('q',changed);self.assertIn('beta',changed)
        self.assertTrue(all(k.startswith('encoder.') or k in ('q','beta') for k in changed))
        self.assertIsNone(engine.metric.B.grad)
        for key in ('alignment','relation'):
            self.assertGreater(result['encoder_component_gradient_norms'][key],1e-8)
        self.assertTrue(result['pi_detached'])
        self.assertIsNone(engine.pi_state)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError): engine.save_checkpoint(Path(tmp)/'mid.pt')
        RESULTS['content']=dict(changed_keys=changed,encoder_component_gradient_norms=result['encoder_component_gradient_norms'],B_bitwise_unchanged=True)

    def test_space_freezes_content_and_uses_full_geometric_feedback(self):
        engine=make_fixture()
        engine.content_step()
        feedback=engine.freeze_feedback()
        self.assertTrue(all(not u.requires_grad for u in feedback.coordinates.values()))
        self.assertFalse(feedback.q.requires_grad);self.assertFalse(feedback.beta.requires_grad)
        before=snapshot(engine);solves=engine.ot_solve_count
        with patch.object(engine,'_solve',side_effect=AssertionError('Candidate re-solved OT')):
            log=engine.space_update(feedback)
        self.assertEqual(solves,engine.ot_solve_count)
        after=snapshot(engine)
        changed=[k for k in before if not torch.equal(before[k],after[k])]
        self.assertEqual(changed,['distance.solver.metric.B'])
        self.assertTrue(log['accepted'])
        self.assertGreater(log['classification_logit_max_change'],1e-8)
        for name in ('classification','alignment','relation'):
            self.assertGreater(log['component_B_gradient_norms'][name],1e-9)
        engine.metric.validate()
        self.assertTrue(all(p.grad is None for p in engine.content_parameters))
        for entry in log['attempts']:
            if entry['reason']=='ACCEPTED':
                self.assertLessEqual(entry['delta'],engine.config['delta_space']**2)
                self.assertLessEqual(entry['phi'],entry['armijo_rhs'])
        RESULTS['space']=log

    def test_trust_rejection_all_fail_keeps_legal_content(self):
        engine=make_fixture()
        engine.config.update(delta_space=1e-12,backtrack_attempts=2)
        engine.content_step()
        before=snapshot(engine)
        feedback=engine.freeze_feedback()
        old=engine.metric.B.detach().clone()
        terms=engine.space_terms(feedback)
        components=[torch.autograd.grad(terms[name],engine.metric.B,retain_graph=True)[0]
                    for name in ('classification','alignment','relation')]
        gradient=components[0]+engine.config['alignment_weight']*components[1]+engine.config['relation_weight']*components[2]
        full,=torch.autograd.grad(engine._total(terms),engine.metric.B)
        torch.testing.assert_close(gradient,full,atol=1e-14,rtol=1e-12)
        original=engine.metric.project_
        candidates=[]
        def project(candidate):
            candidates.append(candidate.detach().clone())
            return original(candidate)
        with patch.object(engine.metric,'project_',side_effect=project):
            log=engine.space_update(feedback)
        self.assertFalse(log['accepted'])
        self.assertEqual(len(log['attempts']),2)
        self.assertTrue(all(not entry['trust_ok'] for entry in log['attempts']))
        self.assertTrue(all(torch.equal(v,engine.model.state_dict()[k]) for k,v in before.items()))
        self.assertEqual(engine.content_version,1)
        for candidate,entry in zip(candidates,log['attempts']):
            torch.testing.assert_close(candidate,old-entry['alpha']*gradient,atol=0,rtol=0)
        RESULTS['trust_rejection']=log

    def test_acceptance_equations_group_sampling_and_no_tolerance(self):
        # Independent scalar checks guard squared-mean vs mean-square and boundary slack.
        self.assertEqual(acceptance_conditions(2.,1.8,.01,.1,.1,.5,1.)[:2],(True,True))
        self.assertFalse(acceptance_conditions(2.,1.8000000000001,.01,.1,.1,.5,1.)[1])
        self.assertFalse(acceptance_conditions(2.,1.7,.01000000000001,.1,.1,.5,1.)[0])
        previous={k:torch.zeros(n,dtype=torch.float64) for k,n in [('ss',2),('tt',1),('st',3)]}
        current={'ss':torch.tensor([-1.,1.]),'tt':torch.tensor([2.]),'st':torch.tensor([3.,3.,3.])}
        delta,groups=trust_change(previous,current,dict(ss=.2,tt=.3,st=.5))
        self.assertAlmostEqual(delta,5.9)
        self.assertEqual(groups,dict(ss=1.,tt=4.,st=9.))
        import random,itertools
        ids=['s:0','s:1','s:2','s:3']
        self.assertEqual(set(_unique_pairs(ids,100,random.Random(3))),set(itertools.combinations(ids,2)))
        self.assertEqual(len(set(_unique_pairs(ids,4,random.Random(3)))),4)
        self.assertEqual(len(set(_unique_pairs(ids,100,random.Random(3),['t:0','t:1']))),8)

    def test_sufficient_decrease_rejection_on_feasible_actual_metric(self):
        # Separate legitimate numerical fixture. A decrease alone is insufficient.
        config=json.loads((ROOT/'configs/s4_evolution_debug.json').read_text())
        config.update(spectral_bound=4.,space_alpha=1000.,c_A=.49,delta_space=1.,backtrack_attempts=1)
        engine=make_fixture(config)
        before=snapshot(engine)
        log=engine.space_update(engine.freeze_feedback())
        entry=log['attempts'][0]
        self.assertTrue(entry['trust_ok'])
        self.assertFalse(entry['armijo_ok'])
        self.assertEqual(entry['reason'],'ARMIJO')
        self.assertLess(entry['phi'],log['phi_before'])
        self.assertGreater(entry['phi'],entry['armijo_rhs'])
        self.assertFalse(log['accepted'])
        self.assertTrue(all(torch.equal(v,engine.model.state_dict()[k]) for k,v in before.items()))
        RESULTS['armijo_rejection']=log

    def test_two_round_closed_loop_and_current_correspondence(self):
        engine=make_fixture()
        first=engine.run_round();second=engine.run_round()
        self.assertEqual(engine.round,2)
        self.assertEqual(engine.content_step_count,4)
        self.assertEqual(engine.ot_solve_count,1+2*(2+2))
        for row in (first,second):
            self.assertTrue(row['space']['accepted'])
            self.assertGreater(row['next_pi_max_change'],1e-9)
            self.assertGreater(row['space']['classification_logit_max_change'],1e-9)
            self.assertGreater(row['space']['feedback_correspondence_id'],row['content'][-1]['correspondence_id'])
            self.assertGreater(row['next_correspondence_id'],row['space']['feedback_correspondence_id'])
        self.assertEqual(engine.pi_state['content_version'],engine.content_version)
        self.assertEqual(engine.pi_state['geometry_version'],engine.geometry_version)
        self.assertEqual(engine.pi_state['event'],'next')
        RESULTS['two_rounds']=[first,second]

    def test_checkpoint_round_boundary_exact_recovery_and_rules(self):
        engine=make_fixture();engine.run_round()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'round1.pt';engine.save_checkpoint(path)
            expected=engine.run_round();expected_state=snapshot(engine)
            restored=EvolutionTrainer.load_checkpoint(path,engine.lookup,engine.data_identity,expected_config=engine.config)
            actual=restored.run_round()
            self.assertEqual(expected,actual)
            for k,v in expected_state.items(): self.assertTrue(torch.equal(v,restored.model.state_dict()[k]),k)
            self.assertEqual(engine.optimizer.state_dict()['param_groups'],restored.optimizer.state_dict()['param_groups'])
            for a,b in zip(engine.optimizer.state_dict()['state'].values(),restored.optimizer.state_dict()['state'].values()):
                for k in a: self.assertTrue(torch.equal(a[k],b[k]))
            self.assertTrue(torch.equal(engine.batch_rng.get_state(),restored.batch_rng.get_state()))
            self.assertEqual(engine.probe_rng.getstate(),restored.probe_rng.getstate())
            with self.assertRaises(ValueError):
                EvolutionTrainer.load_checkpoint(path,engine.lookup,dict(kind='changed'))
            bad=copy.deepcopy(engine.config);bad['c_A']=.2
            with self.assertRaises(ValueError):
                EvolutionTrainer.load_checkpoint(path,engine.lookup,engine.data_identity,expected_config=bad)
        RESULTS['checkpoint']=dict(exact_round_logs=True,model_bitwise_equal=True,optimizer_bitwise_equal=True,sampler_states_equal=True)

    def test_empty_probe_group_and_stale_feedback_are_explicit(self):
        engine=make_fixture()
        batch=engine.make_batch()
        feedback=engine.freeze_feedback(batch,dict(ss=(),tt=(('t:0','t:1'),),st=(('s:0','t:0'),)))
        before=engine.metric.B.detach().clone()
        log=engine.space_update(feedback)
        self.assertEqual(log['reason'],'INSUFFICIENT_PROBE_PAIRS')
        self.assertTrue(torch.equal(before,engine.metric.B))
        engine.content_step()
        with self.assertRaises(RuntimeError): engine.space_update(feedback)


if __name__=='__main__': unittest.main()
