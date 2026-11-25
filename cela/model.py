"""Exact ordered edge-choice model, with prefix-local neural scores and reverse masses."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .structure import NUM_TOKENS, block_bounds, pair_ids, prefix_layout


@dataclass
class Trace:
    prior_logits: Tensor       # [C], conditioned on exogenous attributes and budget
    log_w: Tensor              # [m,K,J], absent cells have mass zero (log = -inf)
    a: Tensor                  # [m,K,C], group indices are local to each block
    true_k: Tensor
    true_j: Tensor
    true_u: Tensor


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def _pool(x):
    if len(x) == 0:
        return x.new_zeros(x.shape[1] * 2)
    return torch.cat((x.mean(0), x.amax(0)))


class PrefixGIN(nn.Module):
    """Two prefix GIN layers with learned eps and GINE messages for typed graphs."""

    def __init__(self, input_dim, hidden, num_edge_types, typed_edges):
        super().__init__()
        self.input = nn.Linear(input_dim + 1, hidden)
        self.layers = nn.ModuleList([_mlp(hidden, hidden, hidden) for _ in range(2)])
        self.eps = nn.Parameter(torch.zeros(2))
        self.edge_embeddings = nn.ModuleList(
            nn.Embedding(num_edge_types, hidden) for _ in range(2) if typed_edges
        )

    def forward(self, x, degree, edges, edge_type):
        h = self.input(torch.cat((x, degree.log1p().unsqueeze(1)), 1)).relu()
        src, dst = edges.T
        row, col = torch.cat((src, dst)), torch.cat((dst, src))
        for i, layer in enumerate(self.layers):
            messages = h[row]
            if len(self.edge_embeddings):
                emb = self.edge_embeddings[i](edge_type)
                messages = (messages + torch.cat((emb, emb))).relu()
            neighbors = torch.zeros_like(h).index_add_(0, col, messages)
            h = layer((1 + self.eps[i]) * h + neighbors).relu()
        return h


def _cell_lse(u, cells, size):
    """Segment logsumexp with -inf on empty segments and finite backward there."""
    offsets = u.new_full((size,), -torch.inf)
    offsets.scatter_reduce_(0, cells, u.detach(), reduce="amax", include_self=True)
    sums = u.new_zeros(size).scatter_add_(0, cells, (u - offsets[cells]).exp())
    return offsets + sums.clamp_min(1).log()


def _mass_history(scores, cells, removed, tail, num_groups, reference):
    """Restore all types of each deleted pair by adding positive masses in log space."""
    cell_count = num_groups * NUM_TOKENS
    if reference:
        # Independent slow forward deletion for the numerical reference tests.
        live = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
        snapshots = []
        for row in removed.unbind():
            indices = live.nonzero().flatten()
            snapshots.append(_cell_lse(scores[indices].flatten(), cells[indices].flatten(), cell_count))
            live[row] = False
        return torch.stack(snapshots).view(-1, num_groups, NUM_TOKENS)

    mass = _cell_lse(scores[tail].flatten(), cells[tail].flatten(), cell_count)
    snapshots = []
    # Within a node pair each mark has a different stratum, hence distinct indices.
    all_keys = cells[removed].flip((0,))
    all_scores = scores[removed].flip((0,))
    for keys, values in zip(all_keys.unbind(), all_scores.unbind()):
        mass = mass.scatter(0, keys, torch.logaddexp(mass[keys], values))
        snapshots.append(mass)
    return torch.stack(snapshots[::-1]).view(-1, num_groups, NUM_TOKENS)


class Model(nn.Module):
    def __init__(self, input_dim, num_classes, num_edge_types, common_types,
                 hidden=64, blocks=4, *, typed_edges=None):
        super().__init__()
        self.hidden = int(hidden)
        self.num_classes = int(num_classes)
        self.num_edge_types = int(num_edge_types)
        self.blocks = int(blocks)
        self.common_types = tuple(int(t) for t in common_types)
        if len(set(self.common_types)) != len(self.common_types) or len(self.common_types) > 4:
            raise ValueError("common_types must be the at most four distinct source-fitted types")
        if min(self.num_classes, self.num_edge_types, self.blocks, self.hidden) < 1:
            raise ValueError("class count, edge type count, block count and hidden size must be positive")

        edge_dim = min(8, hidden)
        num_coarse = 1 + len(self.common_types)
        if typed_edges is None:
            typed_edges = num_edge_types > 1
        self.encoder = PrefixGIN(input_dim, hidden, num_edge_types, typed_edges)
        self.type_emb = nn.Embedding(num_coarse * (num_coarse + 1) // 2, edge_dim)
        self.path_emb = nn.Embedding(3, edge_dim)
        self.mark_emb = nn.Embedding(num_edge_types, edge_dim)
        # f consists of pair features, graph context, edge mark embedding and token.
        self.feature_dim = hidden * 5 + 4 + edge_dim + NUM_TOKENS
        self.shared = _mlp(self.feature_dim, hidden, 1)
        self.w = nn.Parameter(torch.zeros(NUM_TOKENS))
        self.group_head = _mlp(2 * self.feature_dim + 2 * hidden + 4 + 1 + edge_dim * 3,
                               hidden, num_classes)
        self.prior_node = _mlp(input_dim, hidden, hidden)
        self.prior_head = _mlp(hidden * 2 + 3, hidden, num_classes)

    def _x_budget(self, graph):
        x = graph.x.to(device=self.w.device, dtype=self.w.dtype)
        n, m = len(x), len(graph.edges)
        budget = x.new_tensor([math.log1p(n), math.log1p(m), m / max(1, n * (n - 1) // 2)])
        return x, budget

    def prior_logits(self, graph):
        x, budget = self._x_budget(graph)
        return self.prior_head(torch.cat((_pool(self.prior_node(x)), budget)))

    def _scores(self, layout, h, context):
        """Apply the MLP to the exact feature vector without tiling pair features per mark."""
        def ids(a):
            return torch.as_tensor(a, dtype=torch.long, device=h.device)

        hu, hv = h[ids(layout.u)], h[ids(layout.v)]
        pair = torch.cat((hu + hv, (hu - hv).abs(), hu * hv), 1)
        token, group = ids(layout.token), ids(layout.group)
        # Linear([pair,context,mark,onehot(token)]) is a sum of four projections.
        widths = (pair.shape[1], len(context), self.mark_emb.embedding_dim, NUM_TOKENS)
        wp, wc, wm, wj = self.shared[0].weight.split(widths, dim=1)
        base = F.linear(pair, wp, self.shared[0].bias) + F.linear(context, wc) + wj.T[token]
        mark = self.mark_emb.weight
        activations = (base[:, None, :] + F.linear(mark, wm)).relu()
        scores = self.shared[2](activations).squeeze(-1) + (self.w - self.w.mean())[token, None]

        # Pair groups have identical pair features and token statistics for each mark.
        ng, nt = len(layout.group_key), self.num_edge_types
        count = pair.new_tensor(layout.group_count)
        count_j = pair.new_tensor(layout.token_count)
        mean = pair.new_zeros(ng, pair.shape[1]).index_add_(0, group, pair) / count[:, None]
        maxval = pair.new_full(mean.shape, -torch.inf).scatter_reduce_(
            0, group[:, None].expand_as(pair), pair, reduce="amax", include_self=True
        )
        context_k = context.expand(ng, nt, -1)
        mark_k = mark.expand(ng, nt, -1)

        def expand_groups(value):
            return value[:, None, :].expand(-1, nt, -1)

        def summarize(statistic, frequencies):
            return torch.cat((expand_groups(statistic), context_k, mark_k,
                              expand_groups(frequencies)), dim=2)

        keys = ids(layout.group_key)
        z = torch.cat((
            summarize(mean, count_j / count[:, None]),
            summarize(maxval, (count_j > 0).to(pair.dtype)),
            context_k, expand_groups(count.log1p()[:, None]),
            expand_groups(self.type_emb(keys // 3)), expand_groups(self.path_emb(keys % 3)), mark_k,
        ), dim=2)
        a = self.group_head(z.flatten(0, 1))
        a = a - a.mean(1, keepdim=True)
        k = group[:, None] * nt + torch.arange(nt, device=h.device)
        cells = k * NUM_TOKENS + token[:, None]
        return scores, a, cells

    def trace(self, graph, order, *, reference=False):
        """Score one order (integer positions into graph.edges), retaining autograd."""
        x, budget = self._x_budget(graph)
        n, m = len(x), len(graph.edges)
        prior = self.prior_head(torch.cat((_pool(self.prior_node(x)), budget)))

        def array(t):
            return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

        def ids(a):
            return torch.as_tensor(a, dtype=torch.long, device=x.device)

        order = array(order)
        if (order.shape != (m,) or (m and order.dtype.kind not in "iu")
                or not np.array_equal(np.sort(order), np.arange(m))):
            raise ValueError("order must be a permutation of all edge indices")
        if m == 0:
            empty_ids = ids([])
            return Trace(prior, x.new_empty(0, 0, NUM_TOKENS), x.new_empty(0, 0, self.num_classes),
                         empty_ids, empty_ids, x.new_empty(0))

        edges = array(graph.edges)[order]
        marks = array(graph.edge_type)[order]
        if (edges.shape != (m, 2) or edges.dtype.kind not in "iu" or edges.min() < 0
                or edges.max() >= n or np.any(edges[:, 0] >= edges[:, 1])):
            raise ValueError("edges must contain integer pairs 0 <= u < v < n")
        real_ids = pair_ids(edges[:, 0], edges[:, 1], n)
        if np.unique(real_ids).size != m:
            raise ValueError("repeated unordered node pair")
        if (marks.shape != (m,) or marks.dtype.kind not in "iu"
                or marks.min() < 0 or marks.max() >= self.num_edge_types):
            raise ValueError("edge types must be integer indices in the source mark vocabulary")

        raw_types = array(graph.node_type)
        coarse = np.full(n, len(self.common_types), dtype=np.int64)
        for i, node_type in enumerate(self.common_types):
            coarse[raw_types == node_type] = i

        all_u, all_v = np.triu_indices(n, k=1)
        live = np.ones(len(all_u), dtype=bool)
        edge_index, edge_type = ids(edges), ids(marks)
        blocks = []
        for start, stop in block_bounds(m, self.blocks):
            rows = np.flatnonzero(live)
            layout = prefix_layout(n, edges[:start], all_u[rows], all_v[rows], coarse,
                                   len(self.common_types) + 1)
            degree = x.new_tensor(layout.degree)
            h = self.encoder(x, degree, edge_index[:start], edge_type[:start])
            context = torch.cat((_pool(h), budget, x.new_tensor([start / m])))
            scores, a, cells = self._scores(layout, h, context)

            # Suffix choices only index outputs and implement the reverse mass cache.
            # Neither the encoder nor the stratum scores see a remaining true edge.
            removed = np.searchsorted(rows, real_ids[start:stop])
            keep = np.ones(len(rows), dtype=bool)
            keep[removed] = False
            removed = ids(removed)
            log_w = _mass_history(scores, cells, removed, ids(np.flatnonzero(keep)), len(a), reference)
            observed_type = edge_type[start:stop]
            selected = cells[removed, observed_type]
            blocks.append((log_w, a, selected // NUM_TOKENS, selected % NUM_TOKENS,
                           scores[removed, observed_type]))
            live[real_ids[start:stop]] = False

        max_k = max(len(a) for _, a, *_ in blocks)
        log_w = torch.cat([F.pad(w, (0, 0, 0, max_k - len(a)), value=-torch.inf)
                           for w, a, *_ in blocks])
        a = torch.cat([F.pad(a, (0, 0, 0, max_k - len(a))).expand(len(w), -1, -1)
                       for w, a, *_ in blocks])
        k, j, u = [torch.cat(items) for items in zip(*(entry[2:] for entry in blocks))]
        return Trace(prior, log_w, a, k, j, u)

    def forward(self, graph, order):
        return order_logits(self.trace(graph, order))


def edge_log_probs(trace, eta=None, components=None):
    """Class-conditional log probabilities of the actual typed edge at each step."""
    m, _, c = trace.a.shape
    if not m:
        return trace.a.new_empty(0, c, dtype=torch.float64)
    step = torch.arange(m, device=trace.a.device)
    a = trace.a.double()
    logits = a[:, :, None, :] + trace.log_w.double()[:, :, :, None]
    log_z = torch.logsumexp(logits, dim=(1, 2))
    observed = trace.true_u.double()[:, None] + a[step, trace.true_k]
    source = observed - log_z
    if eta is None:
        return source

    eta = torch.as_tensor(eta, dtype=torch.float64, device=trace.a.device)
    # Use the source normalization precision for both component log-sums.
    same_component = logits
    if components is not None:
        component = torch.as_tensor(components, dtype=torch.long, device=trace.a.device)
        chosen = component[None, :] == component[trace.true_j, None]
        same_component = same_component.masked_fill(~chosen[:, None, :, None], -torch.inf)
    # Without component labels this is the global, fully connected case.
    # The selected component always contains at least the actual edge.
    shift = torch.logsumexp(same_component, dim=(1, 2)) - torch.logsumexp(
        same_component + eta[None, None, :, None], dim=(1, 2)
    )
    return source + (eta[trace.true_j, None] + shift)


def order_logits(trace, eta=None, components=None, nu=None):
    logits = trace.prior_logits.double() + edge_log_probs(trace, eta, components).sum(0, dtype=torch.float64)
    return logits if nu is None else logits + torch.as_tensor(nu, device=logits.device)


def source_loss(trace, label, mean_source_edges, lambda_g=1.0):
    """One graph/order's CE and scaled joint NLL using the global source edge mean."""
    p_log = edge_log_probs(trace).sum(0, dtype=torch.float64)
    prior = trace.prior_logits.double()
    joint = prior.log_softmax(0)[label] + p_log[label]
    graph_ce = -(prior + p_log).log_softmax(0)[label]
    return graph_ce - lambda_g / (1 + mean_source_edges) * joint
