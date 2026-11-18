"""Predict with a frozen source model plus a shared affine logit correction."""

import torch
from torch import nn
from torch_geometric.data import Batch

from ..encoder import GraphEncoder
from ..pretrain import SourceModel, graph_batches
from ..utils import finite


def update_features(embeddings, mean, scale):
    x = (embeddings.to(mean) - mean) / scale
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)


class SharedUpdatePredictor(nn.Module):
    def __init__(self, source, mean, scale, correction):
        super().__init__()
        dimension = source.encoder.spec["out_dim"]
        if mean.shape != (dimension,) or scale.shape != mean.shape or bool((scale <= 0).any()):
            raise ValueError("invalid shared-update standardization shape or scale")
        if correction.shape != (source.head.out_features, dimension + 1):
            raise ValueError("invalid shared-update correction shape")
        finite(mean, scale, correction)
        self.source = source.requires_grad_(False)
        self.source.encoder.freeze()
        device = source.head.weight.device
        for name, tensor in (("mean", mean), ("scale", scale), ("correction", correction)):
            self.register_buffer(name, tensor.detach().to(device=device, dtype=torch.float64).clone())
        self.eval()

    def logits_from_embeddings(self, h):
        z0 = self.source.head(h).to(self.correction)
        u = update_features(h, self.mean, self.scale)
        return z0 + u @ self.correction.T

    def forward(self, batch):
        return self.logits_from_embeddings(self.source.encoder(batch))

    @torch.no_grad()
    def predict(self, graphs, batch_size=128):
        self.eval()
        if not graphs:
            return self.correction.new_empty((0, self.source.head.out_features))
        device = self.source.head.weight.device
        batches = (Batch.from_data_list([graphs[i] for i in ids]).to(device)
                   for ids in graph_batches(graphs, batch_size))
        probabilities = torch.cat([self(batch).softmax(-1) for batch in batches])
        finite(probabilities)
        return probabilities


def save_shared_predictor(model, path, metadata):
    torch.save(dict(format="shared-update-predictor-v1", encoder_spec=model.source.encoder.spec,
                    classes=model.source.head.out_features,
                    state={name: value.cpu() for name, value in model.state_dict().items()},
                    metadata=metadata), path)


def load_shared_predictor(path, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != "shared-update-predictor-v1":
        raise ValueError("unsupported shared-update predictor format")
    state = artifact["state"]
    source = SourceModel(GraphEncoder(**artifact["encoder_spec"]), artifact["classes"])
    source.to(dtype=state["source.head.weight"].dtype)
    model = SharedUpdatePredictor(source, state["mean"], state["scale"], state["correction"])
    model.load_state_dict(state)
    return model.to(device)
