"""Masking, composition subtraction, exact H tangent, and registered gate regressions."""
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from itertools import product, permutations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import normalization
from experiments.mutagenicity.r14_节点类型与分类更新_20261121.src import activations as a, experiment, recovery, scoring
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel


def example(conv, sizes):
    torch.manual_seed(725)
    encoder = GraphEncoder(3, 1, hidden=6, num_layers=3, out_dim=4, dropout=.25, conv=conv)
    model = SourceModel(encoder, 2)
    model.requires_grad_(False).eval()
    model.encoder.freeze()
    graphs = []
    for delta, n in enumerate(sizes):
        edges = torch.tensor([(i,j) for i in range(n) for j in (i-1,i+1) if 0 <= j < n],dtype=torch.int64).reshape(-1,2).T
        x = torch.nn.functional.one_hot((torch.arange(n)+delta)%3,3).float()
        graphs.append(Data(x=x,edge_index=edges,edge_attr=torch.rand(edges.shape[1],1)))
    return model,graphs


def masked_fixture():
    return recovery.Masked(ids=np.array([5,7,9]), owner=np.array([0,0,0,1,2,2]),
        nodes=np.array([4,5,2,1,3,7]), types=np.array([0,1,0,7,1,1]),
        normalized=np.array([[1.,2.],[2.,2.],[3.,-1.],[-2.,1.],[1.,-.5],[-3.,.3]]),
        bn_output=np.array([[0.,3.],[2.,2.],[5.,0.],[-2.,1.],[1.,-.5],[-3.,.3]]))


class R14Tests(unittest.TestCase):
    def test_masking_clears_one_row_and_keeps_the_graph(self):
        _, graph = example('gine',[6])
        original = graph[0]
        masked = recovery.masked_graph(original,3)
        np.testing.assert_array_equal(masked.x[3],np.zeros(3))
        np.testing.assert_array_equal(masked.x[[0,1,2,4,5]],original.x[[0,1,2,4,5]])
        np.testing.assert_array_equal(masked.edge_index,original.edge_index)
        np.testing.assert_array_equal(masked.edge_attr,original.edge_attr)
        self.assertEqual(float(original.x[3].sum()),1.)
        expected=sorted(range(6),key=lambda n: sha256(f'r14-node-v1|TEST|99|{n}'.encode()).hexdigest())[:4]
        self.assertEqual(recovery.positions('TEST',99,6,4),expected)

    def test_masked_pool_has_same_total_weight_per_graph_and_ordered_ids(self):
        model, graphs = example('gin',[1,2,3,5])
        masked = recovery.masked_activations(model,graphs,[3,4,5,6],'TEST',4,3)
        np.testing.assert_array_equal(masked.sizes,[1,2,3,4])
        np.testing.assert_allclose(np.bincount(masked.owner,weights=masked.weights),1.)
        for i,g in enumerate(graphs):
            np.testing.assert_array_equal(g.x[masked.nodes[masked.owner==i]].argmax(1),masked.types[masked.owner==i])

    def test_gradient_matches_autograd_and_enumerated_label_permutations(self):
        masked = masked_fixture()
        head = dict(seen=[0,1],weights=[[1.,-.2],[-.3,.7]],intercept=[.2,0.])
        grad, metrics = recovery.gradients(masked,head)
        true,hist,unknown,pure = recovery.targets(masked,head['seen'])
        self.assertEqual(metrics['graphs_two_true_types'],1)
        self.assertEqual(metrics['unknown_masks'],1)
        np.testing.assert_array_equal(grad['Binding'][pure],0.)
        np.testing.assert_array_equal(true[unknown],[[.5,.5]])
        self.assertAlmostEqual(float(metrics['unknown_graph_weight']),1/3)
        v,w,b = (torch.tensor(x,dtype=torch.float64) for x in (masked.normalized,head['weights'],head['intercept']))

        def autograd(target):
            theta = torch.zeros(4,dtype=torch.float64,requires_grad=True)
            z = torch.relu(torch.tensor(masked.bn_output)+v*theta[:2]+theta[2:])@w.T+b
            loss = -(torch.tensor(target)*torch.log_softmax(z,1)).sum(1)
            return torch.autograd.grad((loss*torch.tensor(masked.weights)).sum()/len(masked.ids),theta)[0].numpy()

        np.testing.assert_allclose(grad['True'].mean(0),autograd(true),atol=1e-14)
        np.testing.assert_allclose(grad['Hist'].mean(0),autograd(hist),atol=1e-14)
        permuted = []
        for order in permutations(range(3)):
            candidate=true.copy()
            candidate[:3]=true[list(order)]
            permuted.append(autograd(candidate))
        np.testing.assert_allclose(grad['Hist'].mean(0),np.mean(permuted,axis=0),atol=1e-14)

    def test_exact_zero_both_relu_stages_right_tangent(self):
        model=SourceModel(GraphEncoder(1,1,hidden=1,num_layers=1,out_dim=1),2).double().eval()
        model.encoder.freeze()
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor([[0.],[1.]],dtype=torch.float64))
            model.head.bias.zero_()
            last=model.encoder.norms[0]
            last.running_mean.zero_()
            last.running_var.fill_(1.)
            last.weight.fill_(1.)
            last.bias.zero_()
            for layer in (model.encoder.readout[0],model.encoder.readout[2]):
                layer.weight.fill_(1.)
                layer.bias.zero_()
        cached=a.HeldBatch(model,torch.zeros((2,1),dtype=torch.float64),np.zeros((2,1)),
             torch.empty((1,0),dtype=torch.float64),np.zeros((1,1)),torch.zeros(2,dtype=torch.long),1,np.zeros((1,2)))
        for sign, expected in [(1.,2.),(-1.,0.)]:
            direction=np.array([0.,sign])
            v,fd,step=cached.local(direction,1e-5)
            np.testing.assert_allclose(v,[expected])
            np.testing.assert_allclose(fd,v)
            np.testing.assert_allclose(a.margin(cached.finite(a.theta(model)+direction*step))/step,v)

    def test_cached_finite_gin_and_gine_match_full_updated_model(self):
        for conv in ('gin','gine'):
            model,graphs=example(conv,[2,4,7])
            learned,stats,changed=normalization.recalibrate(model,graphs,'node',2)
            fixed=dict(base=dict(mean_and_variance=stats,changed_buffers=changed),base_theta=a.theta(learned).tolist())
            restored=experiment.restore_base(model,fixed)
            h=a.held_batches(restored,graphs,2)
            direction=np.linspace(-1,2,12)
            direction=direction/np.linalg.norm(direction)
            theta=a.theta(learned)
            step=.05*np.linalg.norm(theta)
            z,v,err,_=experiment.predict_limited(h,theta,direction,step,1e-5,dict(atol=2e-10,rtol=2e-10))
            full=deepcopy(restored).eval()
            with torch.inference_mode():
                g,b=np.split(theta+step*direction,2)
                full.encoder.norms[-1].weight.copy_(torch.tensor(g))
                full.encoder.norms[-1].bias.copy_(torch.tensor(b))
            np.testing.assert_array_equal(z,normalization.logits(full,graphs,2))
            self.assertLess(err,2e-10)
            actual={k for k,t in full.state_dict().items() if not torch.equal(t,restored.state_dict()[k])}
            self.assertEqual(actual,{'encoder.norms.2.weight','encoder.norms.2.bias'})
            self.assertEqual(len(v),3)

    def test_head_fit_matches_manual_weighted_fit_objective(self):
        m=masked_fixture()
        config=dict(ftol=1e-15,gtol=1e-10,max_iterations=6000,stationarity_tolerance=2e-7)
        f=recovery.fit_type_head(m,config)
        self.assertEqual(f['seen'],[0,1,7])
        z=torch.tensor(recovery.node_scores(m,f))
        targets=torch.tensor(np.searchsorted(f['seen'],m.types))
        objective=(m.weights @ torch.nn.functional.cross_entropy(z,targets,reduction='none').numpy()
                   +.5*np.square(f['weights']).sum())/len(m.ids)
        self.assertAlmostEqual(objective,f['mean_penalized_loss'],places=12)

    def test_single_source_and_single_pool_type_force_zero(self):
        m=masked_fixture()
        single=recovery.Masked(m.ids,m.owner,m.nodes,np.zeros(len(m.types),int),m.normalized,m.bn_output)
        f=recovery.fit_type_head(single,{})
        g,diag=recovery.gradients(m,f)
        for v in recovery.directions(g,diag,len(f['seen']),1e-12).values():
            self.assertEqual(v['fallback'],'single_source_type')
            np.testing.assert_array_equal(v['unit'],0.)
        f=dict(seen=[0,1],weights=[[1.,-.2],[-.3,.7]],intercept=[.2,0.])
        g,diag=recovery.gradients(single,f)
        self.assertTrue(all(v['fallback']=='single_pool_type' for v in recovery.directions(g,diag,2,1e-12).values()))

    def test_unsealed_scoring_cannot_open_h_port(self):
        plan=dict(settings=['x'],domains=[0,1])
        with TemporaryDirectory() as tmp, patch.object(scoring.r11,'GraphLabelSession') as port:
            with self.assertRaises(FileNotFoundError):
                scoring.score(Path(tmp),Path(tmp),plan,{},[3405],'fit','obs',pilot=True)
            port.assert_not_called()

    def test_gate_uses_exact_fraction_and_requires_proteins(self):
        settings=['mutagenicity_edge','mutagenicity_node','proteins_edge']
        plan=dict(settings=settings,domains=[0,1,2,3],phase1_seed=3405)
        cfg=dict(minimum_binding_gain_fraction=[1,200],minimum_positive_directions=7)
        keys=[(s,a,b,3405,'Binding',arm) for s,a,b,arm in product(
            settings,range(4),range(4),scoring.GATE_ARMS) if a!=b]
        values={k:Fraction(1,200) for k in keys}
        self.assertTrue(scoring.gate(plan,cfg,values)['extend'])
        for k in keys:
            if k[0]=='proteins_edge' and k[-1]=='Hist':
                values[k]=Fraction((-1)**sum(k[1:3]),307) if k[1]<k[2] else Fraction(-(-1)**sum(k[1:3]),307)
        self.assertFalse(scoring.gate(plan,cfg,values)['extend'])
