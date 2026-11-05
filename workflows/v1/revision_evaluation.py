"""OFFLINE development-label evaluator. Never imported by the training method."""
from pathlib import Path
import json

import torch
from torch_geometric.data.separate import separate

from tqg_gda.metrics import classification_metrics


class DevelopmentLabels:
    def __init__(self, context, domain, task, audit_log):
        self.allowed = set(context.development_ids(domain))
        self.task, self.audit_log = task, Path(audit_log)
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, ids):
        if not ids or not set(ids) <= self.allowed:
            raise PermissionError('Only previously frozen development IDs may expose target labels')
        values = []
        for i in ids:
            graph = separate(self.task._raw.__class__, self.task._raw, i,
                             self.task.slices, decrement=False)
            label = int(graph.y.item())
            if label not in self.task.class_map:
                raise ValueError('Unknown target class')
            values.append(self.task.class_map[label])
        with self.audit_log.open('a') as out:
            out.write(json.dumps(dict(ids=list(ids), role='target_development_only'))+'\n')
        return torch.tensor(values, dtype=torch.long)


@torch.no_grad()
def full_metrics(logits, labels):
    row = classification_metrics(logits, labels)
    p = logits.softmax(-1)
    row['brier'] = float((p-torch.nn.functional.one_hot(labels, logits.shape[1])).square().sum(-1).mean())
    confidence, predicted = p.max(-1)
    ece = 0.
    for i in range(15):
        mask = (confidence > i/15) & (confidence <= (i+1)/15)
        if mask.any():
            ece += float(mask.double().mean() *
                         (confidence[mask].mean()-(predicted[mask] == labels[mask]).double().mean()).abs())
    row['calibration_ece'] = ece
    row['predicted_class_proportions'] = [float((predicted == c).double().mean()) for c in range(logits.shape[1])]
    return row
