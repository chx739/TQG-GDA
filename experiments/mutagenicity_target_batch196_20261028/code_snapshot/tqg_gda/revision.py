"""Stopped response dictionaries and the predeclared revision method family."""
from dataclasses import dataclass, field

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .feedback import ProbeConfig, Snapshot, build_feedback
from .diagnostics import heldout_response_features
from .matching import MatchingConfig, matching_weights, normalize_embedding, permute_weights
from .residual_solver import uniform_ridge, weighted_ridge, select_ridge_cv, project_residual


@dataclass
class ResponseDictionary:
    snapshot: Snapshot
    probe: ProbeConfig
    feedback: object

    @classmethod
    def build(cls, model, source, target, generator, *, K=4, response='odd', direction='target'):
        probe = ProbeConfig(count=K, responses='linear' if response == 'linear_JVP' else response,
                            direction=direction)
        fb = build_feedback(model, source, source.y, target, probe, generator)
        return cls(Snapshot(model), probe, fb)

    def features(self, batch):
        return heldout_response_features(self.snapshot, batch, self.feedback, self.probe)

    @torch.no_grad()
    def embedding(self, batch):
        params = {n[len('encoder.'):]: p for n, p in self.snapshot.parameters.items()
                  if n.startswith('encoder.')}
        buffers = {n[len('encoder.'):]: p.clone() for n, p in self.snapshot.buffers.items()
                   if n.startswith('encoder.')}
        return functional_call(self.snapshot.model.encoder, (params, buffers), (batch,), strict=True).detach()


@dataclass
class Prediction:
    correction: torch.Tensor
    raw_correction: torch.Tensor
    reference: torch.Tensor
    gate: torch.Tensor
    weights: torch.Tensor
    coefficients: torch.Tensor | None = None
    soft_target: torch.Tensor | None = None
    source_ce_weights: torch.Tensor | None = None
    statistics: dict = field(default_factory=dict)


class ResidualPredictor:
    def __init__(self, dictionary, source_batch, config, cv_permutation, frozen_model=None,
                 source_labels=None):
        self.dictionary, self.config = dictionary, config
        self.source = source_batch
        self.phi = dictionary.feedback.source_features
        self.source_logits = dictionary.snapshot.logits(source_batch).detach()
        labels = source_batch.y if source_labels is None else source_labels
        self.labels = labels.detach()
        self.residual = (F.one_hot(labels, self.source_logits.shape[1]).to(self.source_logits)
                         - self.source_logits.softmax(-1)).detach()
        self.frozen_model = frozen_model
        self.cv_scores = None
        needs_ridge = (config.get('predictor', 'weighted_ridge') == 'weighted_ridge'
                       and config.get('method', 'tqg') != 'weighted_source_query_CE')
        self.ridge = None
        if needs_ridge:
            if config.get('ridge') == 'source_CV':
                self.ridge, self.cv_scores = select_ridge_cv(self.phi, self.residual, cv_permutation)
            else:
                self.ridge = float(config.get('ridge', .01))
        self.source_embedding = None
        matching = config.get('matching', 'uniform')
        if matching == 'current_embedding':
            self.source_embedding = normalize_embedding(dictionary.embedding(source_batch))
        elif matching == 'frozen_embedding':
            if frozen_model is None:
                raise ValueError('Frozen embedding control requires the warmup model')
            with torch.no_grad():
                self.source_embedding = normalize_embedding(frozen_model.encoder(source_batch))

    @torch.no_grad()
    def predict(self, batch, *, phi=None, permutation_generator=None):
        phi = self.dictionary.features(batch) if phi is None else phi
        logits = self.dictionary.snapshot.logits(batch).detach()
        n, nt = len(self.residual), len(logits)
        uniform = logits.new_full((n, nt), 1/n)
        if self.phi.shape[-1]:
            response_match = matching_weights(self.phi, phi, MatchingConfig())
        else:
            response_match = dict(weights=uniform, gate=torch.zeros(nt, dtype=torch.bool),
                                  statistics={'uninformative_signature': True})
        matching = self.config.get('matching', 'uniform')
        weights = response_match['weights']
        match_stats = response_match['statistics']
        if matching == 'uniform':
            weights = uniform
        elif matching == 'permuted_response_weights':
            if permutation_generator is None:
                raise ValueError('Permutation control needs its isolated generator')
            weights = permute_weights(weights, permutation_generator)
        elif matching in ('frozen_embedding', 'current_embedding'):
            embedding = (self.dictionary.embedding(batch) if matching == 'current_embedding'
                         else self.frozen_model.encoder(batch).detach())
            match = matching_weights(self.source_embedding, normalize_embedding(embedding), MatchingConfig())
            weights, match_stats = match['weights'], match['statistics']
        elif matching != 'response_local':
            raise ValueError('Unknown matching mode ' + matching)
        gate = (response_match['gate'] if self.config.get('gate') else
                torch.ones(nt, dtype=torch.bool, device=logits.device))
        predictor = self.config.get('predictor', 'weighted_ridge')
        coefficients, soft_target, source_weights = None, None, None
        if self.config.get('method') == 'weighted_source_query_CE':
            source_weights = (weights * gate.to(weights)).mean(1).detach()
            raw = torch.zeros_like(logits)
        elif predictor == 'weighted_mean_residual':
            raw = torch.einsum('ij,ic->jc', weights, self.residual)
        elif predictor == 'weighted_source_labels':
            soft_target = torch.einsum('ij,ic->jc', weights,
                                      F.one_hot(self.labels, logits.shape[1]).to(logits))
            raw = soft_target - logits.softmax(-1)
        elif predictor == 'global_mean':
            raw = self.residual.mean(0).expand_as(logits)
        elif predictor == 'zero':
            raw = torch.zeros_like(logits)
        elif predictor == 'weighted_ridge':
            coefficients = (uniform_ridge(self.phi, self.residual, self.ridge).expand(nt,-1)
                            if matching == 'uniform' else
                            weighted_ridge(self.phi, self.residual, weights, self.ridge))
            raw = torch.einsum('jcm,jm->jc', phi, coefficients)
        else:
            raise ValueError('Unknown residual predictor ' + predictor)
        correction = raw
        if self.config.get('projection') and soft_target is None and source_weights is None:
            correction = project_residual(logits.softmax(-1), raw)
        if any(not torch.isfinite(t).all() for t in (weights, raw, correction)):
            raise FloatingPointError('Nonfinite revision prediction')
        stats = dict(ridge=self.ridge, cv_scores=self.cv_scores, matching=matching,
                     predictor=predictor, gate_rate=float(gate.double().mean()),
                     ESS_mean=float((1/weights.square().sum(0)).mean()),
                     max_weight=float(weights.max()),
                     weight_entropy=float(-(weights*weights.clamp_min(1e-300).log()).sum(0).mean()),
                     L1_from_uniform=float((weights-uniform).abs().sum(0).mean()),
                     raw_correction_rms=float(raw.square().sum(-1).mean().sqrt()),
                     correction_rms=float(correction.square().sum(-1).mean().sqrt()),
                     matching_statistics=match_stats,
                     response_columns=self.phi.shape[-1], valid_directions=len(self.dictionary.feedback.directions))
        return Prediction(correction.detach(), raw.detach(), (logits+correction).detach(),
                          gate.detach(), weights.detach(), coefficients,
                          soft_target.detach() if soft_target is not None else None,
                          source_weights, stats)


def auxiliary_loss(model, source, target, prediction):
    if prediction.source_ce_weights is not None:
        return (F.cross_entropy(model(source), source.y, reduction='none')
                * prediction.source_ce_weights.detach()).sum()
    logits = model(target)
    if prediction.soft_target is not None:
        per_graph = -(prediction.soft_target.detach() * logits.log_softmax(-1)).sum(-1)
    else:
        per_graph = .5 * (logits-prediction.reference.detach()).square().sum(-1)
    return (per_graph * prediction.gate.detach().to(per_graph)).mean()
