"""One fixed-noise denoising score model, trained only on target inputs (§3.4–3.5)."""
import math
import torch
from torch import nn
from .utils import finite, NumericalError


# §3.4：ellT=median_j min_{k:||xj−xk||>0}||xj−xk||，
# sigma=kappa_sigma·ellT/sqrt(d)。仅使用目标输入，整个适配阶段固定 sigma。
def bandwidth(x, kappa, chunk=256):
    """Positive nearest-neighbor median; exact duplicates never act as neighbors."""
    x = x.detach()
    finite(x)
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("bandwidth requires nonempty [samples, dimensions] coordinates")
    if bool((x == x[:1]).all()):
        return None
    # 按实际坐标判定重复，不能把距离计算下溢得到的零误认为相同表示。
    # 分组索引只需 [N]；避免在每个距离块展开 [chunk,N,d] 的坐标比较。
    _, groups = torch.unique(x, dim=0, return_inverse=True)
    nearest = []
    for block, block_groups in zip(x.split(chunk), groups.split(chunk)):
        distance = torch.cdist(block, x, compute_mode="donot_use_mm_for_euclid_dist")
        duplicates = block_groups[:, None] == groups[None, :]
        if bool(((distance == 0) & ~duplicates).any()):
            raise NumericalError("positive-neighbor distance underflow")
        # 自身和完全重复表示不充当正距离邻居；重复样本仍保留在训练池中。
        distance = distance.masked_fill(duplicates, float("inf"))
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


# §3.5：s_ω:R^d→R^d，拟合 ∇_x log p_T,sigma(x)。
# 两层平滑 SiLU MLP；不与编码器或分类器共享参数，不输出类别概率。
class ScoreNetwork(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


# §3.5：L_score=E[(1/d)||sigma·s_ω(x+sigma·ε)+ε||²]，ε~N(0,I)。
# square().mean() 同时平均样本和坐标维度；这里加 ε，符号不可反转。
def denoising_loss(model, x, sigma, noise):
    return (sigma * model(x + sigma * noise) + noise).square().mean()


# §3.5、§3.13：输入坐标 detach，只训练 ω。Adam 仅用于此独立阶段。
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
    # 独立噪声用于无标签去噪验证；固定噪声便于比较快照。
    # 同一目标输入池上的验证不等于独立样本泛化，更不能据此选择目标分类最优模型。
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
    # §3.6：选定 ω* 后冻结，分类适配期间不再重训。
    model.requires_grad_(False).eval()
    return model, dict(selected_epoch=best_epoch, noise_validation_loss=best, history=history,
                       validation_scope="independent noise on the same unlabeled target pool")
