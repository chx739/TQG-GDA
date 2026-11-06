"""SEGA-GDA(1) encoder: per-layer sum pooling, concatenation and projection.

Architecture matches references/sega-gda/sega/encoder.py. Input casts follow
parameter dtype to permit FP64 TQG probes; FP32 computation is otherwise identical.
"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, GINEConv, global_add_pool

from .config import EncoderConfig


def _mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, num_layers=3,
                 out_dim=16, dropout=0., conv='gin'):
        super().__init__()
        EncoderConfig(conv, hidden, num_layers, out_dim, dropout)
        if in_dim < 1 or edge_dim < 0 or (conv == 'gine' and edge_dim < 1):
            raise ValueError('GIN needs node features; GINE additionally needs edge features')
        self.spec = dict(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden, num_layers=num_layers,
                         out_dim=out_dim, dropout=dropout, conv=conv)
        self.conv_type = conv
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for layer in range(num_layers):
            i = in_dim if layer == 0 else hidden
            self.convs.append(GINEConv(_mlp(i, hidden), edge_dim=edge_dim, train_eps=True)
                              if conv == 'gine' else GINConv(_mlp(i, hidden), train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.readout = nn.Sequential(nn.Linear(hidden * num_layers, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))

    def forward(self, data):
        dtype = next(self.parameters()).dtype
        x, edge_index, batch = data.x.to(dtype), data.edge_index, data.batch
        ea = data.edge_attr.to(dtype) if getattr(data, 'edge_attr', None) is not None else None
        if self.conv_type == 'gine' and ea is None:
            raise ValueError('GINE requires edge attributes')
        pooled = []
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, ea) if self.conv_type == 'gine' else conv(x, edge_index)
            x = torch.relu(bn(x))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            pooled.append(global_add_pool(x, batch, size=data.num_graphs))
        return self.readout(torch.cat(pooled, dim=-1))
