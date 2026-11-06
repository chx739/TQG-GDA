"""Shared graph encoder f_phi: G -> R^d (Sec. 3.1.1). GIN / GINE with graph-level readout."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, global_add_pool


def _mlp(i: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 64, num_layers: int = 3,
                 out_dim: int = 16, dropout: float = 0.0, conv: str = "gin"):
        super().__init__()
        self.conv_type = conv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            i = in_dim if layer == 0 else hidden
            if conv == "gine":
                self.convs.append(GINEConv(_mlp(i, hidden), edge_dim=edge_dim, train_eps=True))
            elif conv == "gin":
                self.convs.append(GINConv(_mlp(i, hidden), train_eps=True))
            else:
                raise ValueError(conv)
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        # jumping-knowledge readout: concat of per-layer sum pooling, then a projection to R^d
        self.readout = nn.Sequential(nn.Linear(hidden * num_layers, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        ea = data.edge_attr.float() if getattr(data, "edge_attr", None) is not None else None
        pooled = []
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, ea) if self.conv_type == "gine" else conv(x, edge_index)
            x = torch.relu(bn(x))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            pooled.append(global_add_pool(x, batch))
        return self.readout(torch.cat(pooled, dim=-1))
