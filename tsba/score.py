"""One fixed-noise denoising score model, trained only on target inputs (§3.4–3.5)."""
import math
import torch
from torch import nn
from .utils import finite, NumericalError


def bandwidth(x, kappa, chunk=256):
    """Positive nearest-neighbor median; exact duplicates never act as neighbors."""
    x = x.detach()
    finite(x)
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("bandwidth requires nonempty [samples, dimensions] coordinates")
    if bool((x == x[:1]).all()):
        return None
    nearest = []
    for block in x.split(chunk):
        distance = torch.cdist(block, x, compute_mode="donot_use_mm_for_euclid_dist")
        distance = distance.masked_fill(distance == 0, float("inf"))
        nearest.append(distance.min(dim=1).values)
    distances = torch.cat(nearest)
    # Distinct points were checked above: inf here is a numerical error, not
    # the intended duplicate-point degeneracy (including cdist overflow).
    finite(distances)
    sigma = kappa * torch.quantile(distances, 0.5) / math.sqrt(x.shape[1])
    finite(sigma)
    if float(sigma) <= 0:
        raise NumericalError("positive-neighbor bandwidth underflow")
    return float(sigma)


class ScoreNetwork(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


def denoising_loss(model, x, sigma, noise):
    return (sigma * model(x + sigma * noise) + noise).square().mean()


def train_score(x, sigma, cfg):
    x = x.detach()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + 101)
        model = ScoreNetwork(x.shape[1], cfg.score_hidden).to(dtype=x.dtype)
    model = model.to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.score_lr)
    rng = torch.Generator(device=x.device).manual_seed(cfg.seed + 102)
    val_rng = torch.Generator(device=x.device).manual_seed(cfg.seed + 103)
    # A separate fixed noise draw on the unlabeled pool validates denoising only.
    val_noise = torch.randn(x.shape, generator=val_rng, device=x.device, dtype=x.dtype)
    best, best_state, best_epoch = math.inf, None, None
    history = []
    for epoch in range(cfg.score_epochs):
        order = torch.randperm(len(x), generator=rng, device=x.device)
        total = 0.0
        for ids in order.split(cfg.score_batch):
            noise = torch.randn(x[ids].shape, generator=rng, device=x.device, dtype=x.dtype)
            loss = denoising_loss(model, x[ids], sigma, noise)
            finite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters()))
            opt.step()
            finite(*(p for p in model.parameters()))
            total += float(loss.detach()) * len(ids)
        with torch.no_grad():
            validation = sum(float(denoising_loss(model, xb, sigma, eb)) * len(xb)
                             for xb, eb in zip(x.split(cfg.score_batch), val_noise.split(cfg.score_batch))) / len(x)
        finite(validation)
        history.append(dict(epoch=epoch, train_loss=total / len(x), noise_validation_loss=validation))
        if not cfg.score_validation or validation < best:
            best, best_epoch = validation, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.requires_grad_(False).eval()
    return model, dict(selected_epoch=best_epoch, noise_validation_loss=best, history=history,
                       validation_scope="independent noise on the same unlabeled target pool")
