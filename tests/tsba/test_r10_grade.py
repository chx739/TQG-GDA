"""Step equivalence to unmodified upstream graph-mode GRADE on generated graphs."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import training


ROUND = Path(__file__).resolve().parents[2]/'experiments/mutagenicity/r10_原生图适配基线与增量_20261121'
PLAN = json.loads((ROUND/'configs/r10-grade.json').read_text())
UPSTREAM = ROUND/'src/upstream'


def load_wrapper():
    def compile_source(name, bindings):
        source = UPSTREAM/name
        module = ast.parse(source.read_bytes(), filename=str(source))
        # Supply import dependencies that are outside the JS graph experiment.
        # Execute the entire upstream class and its unmodified method bodies.
        module.body = [node for node in module.body
                       if not isinstance(node, ast.ImportFrom) or not node.level]
        exec(compile(module,str(source),'exec'), bindings)
        return bindings

    base = compile_source('base.py', {'logger': lambda **_: None})['BaseGDA']
    def unused(*args, **kwargs):
        raise AssertionError('The JS graph path called an unrelated upstream branch')

    wrappers = compile_source('grade.py', dict(BaseGDA=base, MMD=unused, eval_macro_f1=unused,
        logger=lambda **_:None, eval_micro_f1=lambda y,p:float((y==p).float().mean()),
        GRADEBase=training.GRADEBase, GradReverse=training.GradReverse))
    return wrappers['GRADE']


def model_kwargs(configuration):
    return dict(in_dim=3,hid_dim=configuration['hidden'],num_classes=2,mode='graph',
        num_layers=configuration['layers'],dropout=configuration['dropout'],disc='JS',
        weight=configuration['domain_weight'],weight_decay=configuration['weight_decay'],
        lr=configuration['learning_rate'],epoch=configuration['epochs'],device='cpu',batch_size=0)


def graph_bank():
    return {n:dict(x=torch.eye(3)*float(n-8),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]))
            for n in (10,11,20,21,22)}


class GradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_js_graph_forward_all_gradients_and_adam_step(self):
        cfg = PLAN['model']
        bank = graph_bank()
        source = Batch.from_data_list(training.dataset(bank,[10,11],torch.tensor([0,1])))
        target = Batch.from_data_list(training.dataset(bank,[20,21,22]))
        self.assertEqual((len(source),len(target)),(2,3))
        self.assertNotIn('y',target)
        training.seed_process(3405)
        ours = training.new_model(cfg,3,2,'cpu')
        wrapped = load_wrapper()(**model_kwargs(cfg))
        wrapped.grade = deepcopy(ours)
        alpha = training.alpha_at(71,200)
        torch.manual_seed(910)
        old_loss,old_s,old_a = wrapped.forward_model(source,target,alpha)
        loss_source = F.nll_loss(F.log_softmax(old_s,dim=1),source.y)
        domain = (old_loss-loss_source)/cfg['domain_weight']
        torch.manual_seed(910)
        result = training.js_forward(ours,source,target,alpha,cfg['domain_weight'])
        torch.testing.assert_close(result['source_logits'],old_s,atol=0,rtol=0)
        torch.testing.assert_close(result['target_logits'],old_a,atol=0,rtol=0)
        torch.testing.assert_close(result['loss'],old_loss,atol=0,rtol=0)
        torch.testing.assert_close(result['source_ce'],loss_source,atol=0,rtol=0)
        torch.testing.assert_close(result['domain_ce'],domain,atol=1e-5,rtol=0)
        opt = training.optimizer_for(ours,cfg)
        ref = torch.optim.Adam(wrapped.grade.parameters(),lr=.001,weight_decay=.001)
        old_loss.backward()
        result['loss'].backward()
        for (name,value),(ref_name,other) in zip(ours.named_parameters(),wrapped.grade.named_parameters()):
            self.assertEqual(name,ref_name)
            torch.testing.assert_close(value.grad,other.grad,atol=0,rtol=0)
        opt.step()
        ref.step()
        self.assertEqual(training.state_hash(ours),training.state_hash(wrapped.grade))

    def test_shuffled_full_batch_fit_one_step_matches_original_loop(self):
        plan = deepcopy(PLAN)
        plan['model']['epochs'] = 1
        cfg = plan['model']
        bank = graph_bank()
        s,a = [10,11],[20,21,22]
        source = training.dataset(bank,s,torch.tensor([0,1]))
        adapt = training.dataset(bank,a)
        training.seed_process(3405)
        author = load_wrapper()(**model_kwargs(cfg))
        author.fit(source,adapt)
        ours,initial,trace = training.fit_model(source,adapt,s,a,plan,3405,3,2,lambda _:None)
        self.assertEqual(training.state_hash(ours),training.state_hash(author.grade))
        self.assertEqual(trace[0]['updates'],1)
        source_only, init_s, track_s = training.fit_model(source,None,s,[],plan,3405,3,2,lambda _:None)
        self.assertEqual(initial,init_s)
        state = training.state_hash(source_only)
        for h in ([22,20],[21]):
            frozen = training.infer(source_only,bank,h,plan)
            self.assertEqual(frozen['ids'].tolist(),h)
            self.assertEqual(training.state_hash(source_only),state)
        self.assertIsNone(track_s[0]['domain_ce'])

    def test_structure_changes_adaptation_gradient_and_a_labels_are_rejected(self):
        cfg = PLAN['model']
        bank = graph_bank()
        source = Batch.from_data_list(training.dataset(bank,[10,11],torch.tensor([0,1])))
        target = Batch.from_data_list(training.dataset(bank,[20,21]))
        modified = deepcopy(bank)
        modified[20]['edge_index'] = torch.tensor([[0,2],[2,0]])
        alt = Batch.from_data_list(training.dataset(modified,[20,21]))
        training.seed_process(173)
        model = training.new_model(cfg,3,2,'cpu')
        def gradient(a):
            model.zero_grad()
            torch.manual_seed(1701)
            training.js_forward(model,source,a,.7,.005)['loss'].backward()
            return model.convs[0].lin.weight.grad.clone()
        self.assertGreater(torch.max(torch.abs(gradient(target)-gradient(alt))).item(),0.)
        alt.y = torch.tensor([0,1])
        with self.assertRaises(PermissionError):
            training.js_forward(model,source,alt,.7,.005)

    def test_missing_observation_cannot_open_holdout_label_scope(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import scoring

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(scoring.r5_inputs, 'evaluation_labels') as labels:
                with self.assertRaises(FileNotFoundError):
                    scoring.score(root, root, {'tasks':{}}, PLAN, root)
                labels.assert_not_called()
            with self.assertRaises(FileNotFoundError):
                scoring.validation.completion_record(root, 'fake', 3405, None, PLAN)

    def test_completion_rejects_wrong_job_identity_and_empty_manifest(self):
        from tempfile import TemporaryDirectory
        from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import validation

        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root/'runs/grade-M1-3405'
            model.mkdir(parents=True)
            path=model/'DONE.json'
            for value in (dict(job=model.name,seed=3406,target=3,attempt='attempt-001',files={}),
                          dict(job=model.name,seed=3405,target=1,attempt='attempt-001',files={})):
                path.write_text(json.dumps(value))
                with self.assertRaises(PermissionError):
                    validation.completion_record(root,model.name,3405,1,PLAN)

    def test_worker_launch_identity_mismatch_fails_before_h_loading(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import validation
        from experiments.common.io import digest

        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'INPUT.json').write_text('{}')
            name='source-only-3405'
            attempt=root/'runs'/name/'attempt-001'
            attempt.mkdir(parents=True)
            (attempt.parent/'DONE.json').write_text('{}')
            process=dict(job=name,pid=19,ppid=4,seed=3405,target=None,attempt='attempt-001',
                         input_sha256=digest(root/'INPUT.json'),command=['python','unexpected-wrapper'])
            (attempt/'summary.json').write_text(json.dumps(process))
            (attempt/'process.json').write_text(json.dumps(process))
            witness=({(name,19,4):{'command':process['command']}}, {(name,19,4):{'returncode':0}}, [])
            with patch.object(validation.data,'read_inputs',return_value=({}, {'plan':PLAN})), \
                 patch.object(validation,'validate_code'), patch.object(validation,'workers',return_value=witness), \
                 patch.object(validation,'completion_record',return_value=(attempt,{'attempt':'attempt-001'},[1,3])), \
                 patch.object(validation.torch,'load') as loads:
                with self.assertRaisesRegex(PermissionError,'worker command'):
                    validation.validate_jobs(root,root,seeds=[3405],allow_missing=True)
                loads.assert_not_called()
