"""GIN/GINE backbone adapted from the user-supplied SEGA-GDA package.

TQG keeps a linear head; no metric field, geometric classifier or OT layer.
"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool


def center_logits(logits):
    return logits - logits.mean(dim=-1, keepdim=True)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, edge_dim=0, readout='mean'):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) < 1 or edge_dim < 0:
            raise ValueError('Invalid encoder dimensions')
        if readout not in ('mean', 'sum'):
            raise ValueError('readout must be mean or sum')
        self.spec = dict(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         edge_dim=edge_dim, readout=readout)
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            dim = input_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=edge_dim) if edge_dim
                               else GINConv(mlp, train_eps=True))

    def forward(self, batch):
        h = batch.x
        for conv in self.layers:
            h = conv(h, batch.edge_index, batch.edge_attr) if self.spec['edge_dim'] else conv(h, batch.edge_index)
            h = torch.relu(h)
        pool = global_mean_pool if self.spec['readout'] == 'mean' else global_add_pool
        return pool(h, batch.batch, size=batch.num_graphs)


class GraphClassifier(nn.Module):
    def __init__(self, encoder_spec, num_classes):
        super().__init__()
        if num_classes < 2:
            raise ValueError('Use at least two logits, including binary classification')
        self.encoder = GraphEncoder(**encoder_spec)
        self.head = nn.Linear(encoder_spec['hidden_dim'], num_classes)
        self.num_classes = num_classes

    def raw_logits(self, batch):
        return self.head(self.encoder(batch))

    def forward(self, batch):
        return center_logits(self.raw_logits(batch))
