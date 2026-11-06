"""TQG §§3.2: SEGA encoder followed by one trainable linear head."""
import torch
from torch import nn

from .encoder import GraphEncoder
from .models import center_logits


class GraphClassifier(nn.Module):
    def __init__(self, encoder_spec, num_classes, *, dtype=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError('TQG requires at least two logits')
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError('TQG construction requires FP32 or FP64')
        # PyG GINE also constructs its own edge projection. Set the dtype
        # before ALL random initialization, not just after creating the model.
        # The task runner is sequential (its RNG configuration is process-wide).
        previous = torch.get_default_dtype()
        try:
            if dtype is not None:
                torch.set_default_dtype(dtype)
            self.encoder = GraphEncoder(**encoder_spec)
            self.head = nn.Linear(self.encoder.spec['out_dim'], num_classes)
        finally:
            torch.set_default_dtype(previous)
        self.num_classes = num_classes

    def raw_logits(self, batch):
        return self.head(self.encoder(batch))

    def forward(self, batch):
        return center_logits(self.raw_logits(batch))
