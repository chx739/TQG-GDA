"""SEGA's GIN/GINE + per-layer sum pooling + projection architecture.

Adapted from references/sega-gda/sega/encoder.py. Input dtype follows the
parameters, allowing float64 GINE as well as float32 source pretraining.
"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, GINEConv, global_add_pool


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, num_layers=3, out_dim=16,
                 dropout=0.0, conv="gin"):
        super().__init__()
        if conv not in ("gin", "gine") or (conv == "gine" and edge_dim < 1):
            raise ValueError("GIN/GINE requires compatible input features")
        self.spec = dict(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden,
                         num_layers=num_layers, out_dim=out_dim, dropout=dropout, conv=conv)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for layer in range(num_layers):
            width = in_dim if layer == 0 else hidden
            mlp = nn.Sequential(nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True) if conv == "gine"
                              else GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.readout = nn.Sequential(nn.Linear(hidden * num_layers, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))
        self.frozen = False

    def freeze(self):
        self.frozen = True
        self.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode=True):
        return super().train(False if self.frozen else mode)

    def forward(self, data):
        dtype = next(self.parameters()).dtype
        x = data.x.to(dtype)
        ea = data.edge_attr.to(dtype) if data.edge_attr is not None else None
        pooled = []
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, ea) if self.spec["conv"] == "gine" else conv(x, data.edge_index)
            x = torch.relu(bn(x))
            x = nn.functional.dropout(x, self.spec["dropout"], self.training)
            pooled.append(global_add_pool(x, data.batch))
        return self.readout(torch.cat(pooled, dim=-1))
