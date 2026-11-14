"""Protocol-independent serialization and binary classification statistics."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch

def finite(value):
    return value is not None and math.isfinite(float(value))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def difference(a, b):
    return float(a - b) if finite(a) and finite(b) else None


def clean(value):
    """JSON-safe values; missing observations remain null rather than zero."""
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    rows = [clean(row) for row in rows]
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"))
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def classification_metrics(probabilities, labels, log_probabilities=None):
    """Binary graph metrics. A missing class invalidates BA, not NLL."""
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    if p.shape != (len(y), 2) or y.ndim != 1 or not len(y):
        raise ValueError("Expected nonempty N x 2 probabilities and N labels")
    if not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("Binary labels required")
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or bool((p > 1).any()):
        raise ValueError("Invalid probabilities")
    if not torch.allclose(p.sum(1), torch.ones(len(y), dtype=p.dtype), atol=1e-6, rtol=1e-6):
        raise ValueError("Probabilities must sum to one")
    predicted = p.argmax(1)  # Ties intentionally select class 0.
    confusion = torch.bincount(2 * y + predicted, minlength=4).reshape(2, 2)
    counts = confusion.sum(1)
    recall = [ratio(int(confusion[c, c]), int(counts[c])) for c in range(2)]
    lp = p.clamp_min(1e-300).log() if log_probabilities is None else torch.as_tensor(log_probabilities, dtype=torch.float64)
    if lp.shape != p.shape or not bool(torch.isfinite(lp).all()) or not torch.allclose(lp.exp(), p, atol=1e-10, rtol=1e-8):
        raise ValueError("Log probabilities do not match frozen probabilities")
    nll = -lp[torch.arange(len(y)), y]
    return dict(n=len(y), n_class0=int(counts[0]), n_class1=int(counts[1]),
                accuracy=float((predicted == y).double().mean()),
                balanced_accuracy=sum(recall) / 2 if all(finite(x) for x in recall) else None,
                nll=float(nll.mean()), recall0=recall[0], recall1=recall[1],
                confusion=confusion.tolist(), predicted_class0=float((predicted == 0).double().mean()),
                predicted_class1=float((predicted == 1).double().mean()),
                ba_one_graph_resolution_class0_pp=ratio(50, int(counts[0])),
                ba_one_graph_resolution_class1_pp=ratio(50, int(counts[1])))


def wilson(successes, total, z=1.95996398454):
    if total == 0:
        return [None, None]
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0., center - width), min(1., center + width)]


def validated_nonnegative_errorbars(value, interval):
    """Convert an interval for plotting, clipping only roundoff-sized errors.

    Wilson endpoints can differ from an observed proportion of zero or one by
    a few floating-point ulps. Matplotlib rejects even these tiny negative
    lengths. Keep the saved statistical interval intact and correct only its
    display lengths; a materially inconsistent interval still raises.
    """
    if len(interval) != 2 or not all(finite(v) for v in (value, *interval)):
        raise ValueError("Finite value and interval required for error bars")
    value, lower, upper = map(float, (value, *interval))
    if lower > upper:
        raise ValueError("Error-bar interval is reversed")
    tolerance = 16 * math.ulp(max(1., abs(value), abs(lower), abs(upper)))
    lengths = (value-lower, upper-value)
    if any(length < -tolerance for length in lengths):
        raise ValueError("Value is materially outside its error-bar interval")
    return [[max(0., length)] for length in lengths]
