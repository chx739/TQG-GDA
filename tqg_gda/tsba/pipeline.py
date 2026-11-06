"""Four-stage TSBA pipeline following SEGA's modular orchestration paradigm."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .adapt import adapt_classifier
from .classifier import Coordinates, ResidualClassifier
from .encoder import GraphEncoder
from .predict import Predictor, save_predictor
from .pretrain import SourceModel, pretrain_encoder, encode
from .score import bandwidth, train_score
from .utils import identity, set_seed, finite, NumericalError, file_hash


SOURCE_KEYS = ("seed", "encoder", "hidden", "num_layers", "emb_dim", "dropout", "source_epochs",
               "source_lr", "source_batch")


def source_identity(cfg, fingerprints):
    return identity(dict(config={k: getattr(cfg, k) for k in SOURCE_KEYS},
                         train=fingerprints["source_train"], validation=fingerprints["source_validation"]))


@torch.no_grad()
def source_readout_metrics(model, x, labels):
    if not len(x):
        return None
    out = model(x)
    ce = F.cross_entropy(out, labels, reduction="none")
    finite(out, ce)
    return dict(ce=float(ce.mean()), accuracy=float((out.argmax(1) == labels).float().mean()),
                class_ce={str(c): float(ce[labels == c].mean()) for c in labels.unique().tolist()})


def run_task(cfg, data, output_dir=None, source_checkpoint=None, logger=None):
    dimensions = data.validate()
    set_seed(cfg.seed)
    fingerprints = data.fingerprints()
    source_key = source_identity(cfg, fingerprints)
    metadata = dict(config=cfg.to_dict(), data=fingerprints, protocol=data.protocol,
                    source_identity=source_key, source_checkpoint_sha256=None if source_checkpoint is None
                    else file_hash(source_checkpoint))
    metadata["run_identity"] = identity(metadata)
    out = None if output_dir is None else Path(output_dir)
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)  # never silently reuse unrelated results
    try:
        # A: source semantics, frozen complete-graph coordinates.
        if source_checkpoint is None:
            source, source_info = pretrain_encoder(cfg, data, logger)
        else:
            ck = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
            if ck.get("format") != "tsba-source-v1" or ck.get("source_identity") != source_key:
                raise ValueError("source checkpoint architecture/data/roles/config identity mismatch")
            source = SourceModel(GraphEncoder(**ck["encoder_spec"]), dimensions[2]).to(cfg.device)
            source.load_state_dict(ck["state"])
            source.encoder.freeze()
            source.eval()
            source_info = ck["info"]
        if out is not None:
            torch.save(dict(format="tsba-source-v1", source_identity=source_key, encoder_spec=source.encoder.spec,
                            state={k: v.cpu() for k, v in source.state_dict().items()}, info=source_info), out / "source.pt")
        dtype = getattr(torch, cfg.adaptation_dtype)
        h_source = encode(source.encoder, data.source_train.graphs, cfg.source_batch, cfg.device).to(dtype)
        coordinates = Coordinates.fit(h_source, cfg.coordinate_eps)
        x_source = coordinates(h_source)
        x_target = coordinates(encode(source.encoder, data.target.graphs, cfg.source_batch, cfg.device))
        x_val = coordinates(encode(source.encoder, data.source_validation.graphs, cfg.source_batch, cfg.device))
        y_source = data.source_train.labels.to(cfg.device)
        y_val = data.source_validation.labels.to(cfg.device)
        classifier = ResidualClassifier.from_source(source.head, coordinates,
                                                    cfg.adapter_width or min(16, cfg.emb_dim), cfg.seed)
        before = source_readout_metrics(classifier, x_source, y_source)
        with torch.no_grad():
            initial_probs = classifier(x_target).softmax(-1)
        # B: independent target-only score training, one fixed bandwidth.
        sigma = bandwidth(x_target, cfg.kappa_sigma)
        score_info = None
        if sigma is None:
            adaptation = dict(accepted=0, stop_reason="fewer_than_two_distinct_target_points", history=[])
        else:
            score, score_info = train_score(x_target, sigma, cfg)
            # C: refresh snapshot -> local proposal -> true-forward acceptance -> commit.
            adaptation = adapt_classifier(classifier, x_source, y_source, x_target, score, sigma, cfg, logger)
        # D: retain only encoder, fixed transform and final accepted classifier.
        predictor = Predictor(source.encoder, coordinates, classifier)
        with torch.no_grad():
            final_probs = classifier(x_target).softmax(-1)
        finite(final_probs)
        result = dict(**metadata, status="numerical_error" if adaptation["stop_reason"] == "numerical_error"
                      else "completed", source_pretraining=source_info, sigma=sigma, score=score_info,
                      adaptation=adaptation, source_before=before,
                      source_after=source_readout_metrics(classifier, x_source, y_source),
                      source_validation_after=source_readout_metrics(classifier, x_val, y_val),
                      target_prediction_changes=int((initial_probs.argmax(1) != final_probs.argmax(1)).sum()),
                      target_labels_used=False, target_accuracy=None)
        if out is not None:
            save_predictor(predictor, out / "model.pt", metadata)
            torch.save(dict(ids=data.target.ids, initial_probabilities=initial_probs.cpu(),
                            probabilities=final_probs.cpu()), out / "predictions.pt")
            (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return predictor, result
    except (NumericalError, ValueError, RuntimeError) as exc:
        if out is not None:
            (out / "failure.json").write_text(json.dumps(dict(**metadata, status="failed", error=str(exc)), indent=2) + "\n")
        raise
