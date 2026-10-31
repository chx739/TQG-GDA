"""Metrics with explicit undefined values; no NaN in JSON experiment records."""
import numpy as np
import torch
from torch.nn import functional as F


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(logits, labels):
    logits, labels = logits.detach().cpu(), labels.detach().cpu().long()
    if not len(labels) or not torch.isfinite(logits).all():
        raise ValueError('Need nonempty labels and finite logits')
    classes = logits.shape[1]
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Evaluation label outside source class vocabulary')
    predictions = logits.argmax(-1)
    cm = torch.bincount(labels * classes + predictions, minlength=classes ** 2).reshape(classes, classes)
    counts = cm.sum(1)
    recall = [float(cm[c, c] / counts[c]) if counts[c] else None for c in range(classes)]
    observed = [v for v in recall if v is not None]
    return dict(ce=float(F.cross_entropy(logits, labels)), accuracy=float((predictions == labels).double().mean()),
                balanced_accuracy=sum(observed) / len(observed), confusion=cm.tolist(),
                class_counts=counts.tolist(), class_recall=recall, graphs=len(labels),
                auroc=binary_auc(labels.numpy(), logits.softmax(-1)[:, 1].numpy()) if classes == 2 else None)


@torch.no_grad()
def evaluate_source(model, task, batch_size=32):
    model.eval()
    parameter = next(model.parameters())
    logits, labels = [], []
    ids = task.validation_ids
    for start in range(0, len(ids), batch_size):
        batch = task.batch(ids[start:start + batch_size], 'validation', parameter.device, parameter.dtype)
        logits.append(model(batch)); labels.append(batch.y)
    return classification_metrics(torch.cat(logits), torch.cat(labels))
