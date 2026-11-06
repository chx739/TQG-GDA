"""Self-contained inference: no source library, score model, or target pool (§3.17)."""
import torch
from torch import nn
from torch_geometric.data import Batch
from .classifier import Coordinates, ResidualClassifier
from .encoder import GraphEncoder
from .pretrain import graph_batches
from .utils import finite


class Predictor(nn.Module):
    def __init__(self, encoder, coordinates, classifier):
        super().__init__()
        self.encoder = encoder.freeze()
        self.coordinates, self.classifier = coordinates, classifier
        self.eval()

    def forward(self, batch):
        return self.classifier(self.coordinates(self.encoder(batch)))

    @torch.no_grad()
    def predict(self, graphs, batch_size=128):
        self.eval()
        device = next(self.encoder.parameters()).device
        if not graphs:
            return self.coordinates.mean.new_empty((0, self.classifier.linear.out_features))
        out = [self(Batch.from_data_list([graphs[i] for i in ids]).to(device)).softmax(-1)
               for ids in graph_batches(graphs, batch_size)]
        probs = torch.cat(out)
        finite(probs)
        return probs


def save_predictor(model, path, metadata):
    torch.save(dict(format="tsba-predictor-v1", encoder_spec=model.encoder.spec,
                    encoder_state={k: v.cpu() for k, v in model.encoder.state_dict().items()},
                    coordinates={k: v.cpu() for k, v in model.coordinates.state_dict().items()},
                    classifier_state={k: v.cpu() for k, v in model.classifier.state_dict().items()},
                    classes=model.classifier.linear.out_features, width=model.classifier.first.out_features,
                    metadata=metadata), path)


def load_predictor(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["format"] != "tsba-predictor-v1":
        raise ValueError("unsupported predictor format")
    encoder = GraphEncoder(**ck["encoder_spec"])
    encoder.load_state_dict(ck["encoder_state"])
    coordinates = Coordinates(ck["coordinates"]["mean"], ck["coordinates"]["scale"])
    classifier = ResidualClassifier(ck["encoder_spec"]["out_dim"], ck["classes"], ck["width"],
                                    dtype=coordinates.mean.dtype)
    classifier.load_state_dict(ck["classifier_state"])
    return Predictor(encoder, coordinates, classifier).to(device).requires_grad_(False)
