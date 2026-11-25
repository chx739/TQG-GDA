import itertools
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from cela.model import Model, Trace, edge_log_probs, order_logits, source_loss
from cela.structure import NUM_TOKENS, block_bounds, prefix_layout


def graph(x, edges, marks, types=None):
    n = len(x)
    return SimpleNamespace(
        id="example", x=x, node_type=torch.arange(n) % 3 if types is None else types,
        edges=torch.as_tensor(edges, dtype=torch.long).reshape(-1, 2),
        edge_type=torch.as_tensor(marks, dtype=torch.long),
    )


class ModelTests(unittest.TestCase):
    def test_declared_single_mark_uses_typed_prefix_messages(self):
        typed = Model(1, 2, 1, [], hidden=8, typed_edges=True)
        untyped = Model(1, 2, 1, [], hidden=8, typed_edges=False)
        self.assertEqual(len(typed.encoder.edge_embeddings), 2)
        self.assertEqual(len(untyped.encoder.edge_embeddings), 0)

    def setUp(self):
        torch.manual_seed(33)
        self.model = Model(3, 3, 2, (0, 1), hidden=8, blocks=3).double()
        self.g = graph(torch.randn(6, 3, dtype=torch.double),
                       [(0, 3), (1, 3), (3, 5), (1, 4), (0, 2), (0, 4), (2, 3)], [0, 1, 0, 1, 0, 0, 1])
        self.order = torch.tensor([4, 6, 5, 0, 1, 2, 3])
        self.eta = torch.linspace(-1., 2., NUM_TOKENS, dtype=torch.double)
        self.components = torch.arange(NUM_TOKENS) % 2

    def test_blocks_and_prefix_roles(self):
        self.assertEqual(list(block_bounds(9, 4)), [(0, 3), (3, 6), (6, 8), (8, 9)])
        self.assertEqual(list(block_bounds(2, 4)), [(0, 1), (1, 2)])
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        layout = prefix_layout(5, edges, np.array([0, 0, 1]), np.array([2, 3, 4]),
                               np.zeros(5, dtype=int), 1)
        np.testing.assert_equal(layout.group_key[layout.group], [1, 2, 0])
        np.testing.assert_equal(layout.token, [5, 4, 2])

    def test_typed_order_cache_counts(self):
        for par in self.model.parameters():
            torch.nn.init.zeros_(par)
        cache = self.model.trace(self.g, self.order)
        n, m = len(self.g.x), len(self.g.edges)
        available = 2 * (math.comb(n, 2) - torch.arange(m, dtype=torch.double))
        torch.testing.assert_close(cache.log_w.exp().sum((1, 2)), available)
        direct = -available.log()[:, None].expand(m, self.model.num_classes)
        torch.testing.assert_close(edge_log_probs(cache), direct)

    def test_reverse_cache_values_and_training_gradients(self):
        cache = self.model.trace(self.g, self.order)
        ref = self.model.trace(self.g, self.order, reference=True)
        torch.testing.assert_close(cache.log_w, ref.log_w)
        self.assertTrue(cache.log_w.isneginf().any())
        for name in ("a", "true_k", "true_j", "true_u", "prior_logits"):
            torch.testing.assert_close(getattr(cache, name), getattr(ref, name))
        params = tuple(self.model.parameters())
        fast = torch.autograd.grad(source_loss(cache, 2, 9), params)
        slow = torch.autograd.grad(source_loss(ref, 2, 9), params)
        for left, right in zip(fast, slow):
            self.assertTrue(left.isfinite().all())
            torch.testing.assert_close(left, right)
        for layer in (self.model.shared[0], self.model.shared[2]):
            self.assertGreater(fast[next(i for i, p in enumerate(params) if p is layer.weight)].norm(), 0)
        grad_w = fast[next(i for i, p in enumerate(params) if p is self.model.w)]
        self.assertGreater(grad_w.norm(), 0)
        self.assertAlmostEqual(float(grad_w.sum()), 0., places=12)

    def test_corresponding_relabel_is_invariant_and_suffix_cannot_change_prefix(self):
        mapping = torch.tensor([1, 4, 5, 0, 3, 2])
        new_edges = mapping[self.g.edges].sort(dim=1).values
        inv = mapping.argsort()
        relabeled = graph(self.g.x[inv], new_edges, self.g.edge_type, self.g.node_type[inv])
        original = self.model.trace(self.g, self.order)
        rel = self.model.trace(relabeled, self.order)
        torch.testing.assert_close(original.log_w, rel.log_w)
        torch.testing.assert_close(order_logits(original, self.eta, self.components),
                                   order_logits(rel, self.eta, self.components))
        altered = graph(self.g.x, self.g.edges[self.order].clone(), self.g.edge_type[self.order].clone())
        altered.edges[-2:] = torch.tensor([[0, 5], [2, 5]])
        altered.edge_type[-2:] = 1 - altered.edge_type[-2:]
        changed = self.model.trace(altered, torch.arange(len(altered.edges)))
        torch.testing.assert_close(original.prior_logits, changed.prior_logits)
        for prob in (lambda t: edge_log_probs(t), lambda t: edge_log_probs(t, self.eta, self.components)):
            torch.testing.assert_close(prob(original)[:-2], prob(changed)[:-2])

    @torch.no_grad()
    def test_all_typed_sequences_sum_to_one_per_class(self):
        x, pairs = self.g.x[:4], list(itertools.combinations(range(4), 2))
        source = torch.zeros(3, dtype=torch.double)
        adapted = torch.zeros_like(source)
        for sequence in itertools.permutations(pairs, 2):
            for marks in itertools.product((0, 1), repeat=2):
                tr = self.model.trace(graph(x, sequence, marks), [0, 1])
                source += edge_log_probs(tr).sum(0).exp()
                adapted += edge_log_probs(tr, self.eta, self.components).sum(0).exp()
        torch.testing.assert_close(source, torch.ones_like(source))
        torch.testing.assert_close(adapted, torch.ones_like(adapted))

    def test_empty_graph_and_invalid_orders(self):
        for n in (0, 3):
            empty = graph(self.g.x[:n], [], [])
            tr = self.model.trace(empty, [])
            self.assertEqual(edge_log_probs(tr).shape, (0, 3))
            torch.testing.assert_close(order_logits(tr, self.eta, self.components), tr.prior_logits)
        with self.assertRaisesRegex(ValueError, "permutation"):
            self.model.trace(self.g, [0] * len(self.g.edges))
        duplicate = graph(self.g.x, [(0, 1), (0, 1)], [0, 1])
        with self.assertRaisesRegex(ValueError, "repeated"):
            self.model.trace(duplicate, [0, 1])

    def test_float32_source_double_zero_tilt_recovers_exactly(self):
        tr = self.model.float().trace(self.g, self.order)
        self.assertEqual(tr.log_w.dtype, torch.float32)
        eta = torch.zeros(NUM_TOKENS, dtype=torch.float64, requires_grad=True)
        source = edge_log_probs(tr)
        for components in (None, self.components):
            log_p = edge_log_probs(tr, eta, components)
            self.assertEqual(log_p.dtype, torch.float64)
            self.assertTrue(torch.equal(log_p, source))
            gradient, = torch.autograd.grad(log_p.sum(), eta)
            self.assertTrue(gradient.isfinite().all())

    def test_cross_component_strata_and_log_odds(self):
        # One abstract candidate in every (k,j) cell, including unseen comparisons.
        W = torch.tensor([[1., 3., 2., 4.], [3., 1., 4., 1.]], dtype=torch.double)
        a = torch.tensor([[0., 0.4, -0.1], [-0.2, 0., 0.5]], dtype=torch.double)
        src = W[:, :, None] * a.exp()[:, None, :]
        src /= src.sum((0, 1))
        eta = self.eta[:4]
        comp = torch.tensor([0, 0, 1, 1])
        tilt = src * eta.exp()[None, :, None]
        normalized = torch.zeros_like(src)
        for i in range(2):
            j = comp == i
            normalized[:, j] = tilt[:, j] * (src[:, j].sum((0, 1)) / tilt[:, j].sum((0, 1)))
        for k, j in itertools.product(range(2), range(4)):
            tr = Trace(torch.zeros(3, dtype=torch.double), W.log()[None], a[None],
                       torch.tensor([k]), torch.tensor([j]), W[k, j].log()[None])
            adapted = edge_log_probs(tr, eta, comp)[0]
            torch.testing.assert_close(adapted.exp(), normalized[k, j])
            torch.testing.assert_close(edge_log_probs(tr, eta + torch.tensor([5, 5, -2, -2]), comp)[0], adapted)
            torch.testing.assert_close(edge_log_probs(tr, torch.zeros_like(eta), comp), edge_log_probs(tr))
            src_l = src[:, comp == comp[j]].sum((0, 1))
            tilt_l = tilt[:, comp == comp[j]].sum((0, 1))
            corr = src_l.log() - tilt_l.log()
            observed = adapted - edge_log_probs(tr)[0]
            torch.testing.assert_close(observed[0] - observed, corr[0] - corr)
            # Within a selected (group,component), token choice is independent of class/prior.
            in_comp = comp == comp[j]
            conditional = normalized[k, j] / normalized[k, in_comp].sum(0)
            expected = (W[k, j] * eta[j].exp()) / (W[k, in_comp] * eta[in_comp].exp()).sum()
            torch.testing.assert_close(conditional, expected.expand_as(conditional))

    def test_section_313_ranking_reversal(self):
        log3 = math.log(3)
        a = torch.tensor([[[0., log3], [log3, 0.]]], dtype=torch.double)
        eta = torch.tensor([-log3 / 2, log3 / 2], dtype=torch.double)
        comp = torch.zeros(2, dtype=torch.long)
        for mass, expected in (([[.05, .45], [.9, .1]], [21 / 26, 25 / 34]),
                               ([[.9, .1], [.1, .9]], [.75, 9 / 11])):
            w = torch.tensor(mass, dtype=torch.double).log()[None]
            tr = Trace(torch.zeros(2, dtype=torch.double), w, a, torch.tensor([0]), torch.tensor([0]),
                       w[0, 0, :1])
            for posterior, p in zip((order_logits(tr), order_logits(tr, eta, comp)), expected):
                self.assertAlmostEqual(float(posterior.softmax(0)[1]), p, places=12)


if __name__ == "__main__":
    unittest.main()
