"""Graph task entry for adaptation through shared explanations."""

import json
from pathlib import Path

import torch

from ..pretrain import SOURCE_KEYS, encode, prepare_source, source_identity
from ..utils import file_hash, finite, identity, set_seed
from .explanations import ExplanationSet
from .predict import SharedUpdatePredictor, corrected_logits, save_shared_predictor, update_features
from .solver import solve_shared_update


def _write_result(directory, filename, result):
    (directory / filename).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def _frozen_data(source_cfg, update_cfg, source, data):
    h_source = encode(source.encoder, data.source_train.graphs, source_cfg.source_batch, source_cfg.device)
    h_target = encode(source.encoder, data.target.graphs, source_cfg.source_batch, source_cfg.device)
    train = h_source.to(dtype=torch.float64)
    mean = train.mean(0)
    scale = (train - mean).square().mean(0).sqrt().clamp_min(update_cfg.coordinate_floor)
    u_source = update_features(h_source, mean, scale)
    u_target = update_features(h_target, mean, scale)
    with torch.no_grad():
        z0 = source.head(h_target).to(dtype=torch.float64)
    finite(mean, scale, u_source, u_target, z0)
    return mean, scale, u_source, u_target, z0


def run_shared_task(source_cfg, update_cfg, data, output_dir=None, source_checkpoint=None, logger=None):
    _, _, classes = data.validate()
    set_seed(source_cfg.seed)
    fingerprints = data.fingerprints()
    source_key = source_identity(source_cfg, fingerprints)
    metadata = dict(method="shared-update", source_identity=source_key, data=fingerprints,
                    protocol=data.protocol,
                    source_config={name: getattr(source_cfg, name) for name in (*SOURCE_KEYS, "device")},
                    update_config=update_cfg.to_dict(),
                    source_checkpoint_sha256=file_hash(source_checkpoint) if source_checkpoint is not None else None)
    metadata["run_identity"] = identity(metadata)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    try:
        source, source_info = prepare_source(source_cfg, data, classes, source_key, source_checkpoint, out, logger)
        mean, scale, u_source, u_target, z0 = _frozen_data(source_cfg, update_cfg, source, data)
        explanations = ExplanationSet(u_source[:, :-1].cpu().numpy(), data.source_train.labels.cpu().numpy(),
                                      u_target[:, :-1].cpu().numpy(), update_cfg)
        correction, adaptation = solve_shared_update(z0.cpu().numpy(), u_target.cpu().numpy(), explanations, update_cfg)
        model = SharedUpdatePredictor(source, mean, scale, torch.as_tensor(correction))
        initial = z0.softmax(-1)
        probabilities = corrected_logits(z0, u_target, model.correction).softmax(-1)
        finite(initial, probabilities)
        result = dict(**metadata, status="completed", source_pretraining=source_info, adaptation=adaptation,
                      target_labels_used=False, target_accuracy=None,
                      target_prediction_changes=int((initial.argmax(1) != probabilities.argmax(1)).sum()))
        if out is not None:
            save_shared_predictor(model, out / "model.pt", metadata)
            torch.save(dict(ids=data.target.ids, initial_probabilities=initial.cpu(),
                            probabilities=probabilities.cpu()), out / "predictions.pt")
            _write_result(out, "result.json", result)
        return model, result
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        if out is not None:
            _write_result(out, "failure.json", dict(**metadata, status="failed", error=str(exc)))
        raise
