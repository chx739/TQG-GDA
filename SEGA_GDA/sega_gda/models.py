"""Shared GIN/GINE graph encoder and the S1 linear classification head."""
import torch
from torch import nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, edge_dim=0, readout='mean'):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) < 1 or edge_dim < 0:
            raise ValueError('Invalid encoder dimensions')
        if readout not in ('mean', 'sum'):
            raise ValueError('Readout must be fixed to mean or sum before warmup')
        self.spec = dict(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         edge_dim=edge_dim, readout=readout)
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            dim = input_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            conv = GINEConv(mlp, train_eps=True, edge_dim=edge_dim) if edge_dim else GINConv(mlp, train_eps=True)
            self.layers.append(conv)

    def forward(self, batch):
        h = batch.x
        for conv in self.layers:
            if self.spec['edge_dim']:
                if batch.edge_attr is None:
                    raise ValueError('GINE requires the existing edge attributes')
                h = conv(h, batch.edge_index, batch.edge_attr)
            else:
                h = conv(h, batch.edge_index)
            h = torch.relu(h)
        pool = global_mean_pool if self.spec['readout'] == 'mean' else global_add_pool
        # These ARE the source coordinates. No later normalization/projection.
        return pool(h, batch.batch, size=batch.num_graphs)


class SourceClassifier(nn.Module):
    def __init__(self, encoder_spec, num_classes):
        super().__init__()
        self.encoder = GraphEncoder(**encoder_spec)
        self.head = nn.Linear(encoder_spec['hidden_dim'], num_classes)

    def forward(self, batch):
        return self.head(self.encoder(batch))
