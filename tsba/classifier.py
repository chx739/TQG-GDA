"""Fixed coordinates and probability-preserving residual classifier (§3.2–3.3)."""
import torch
from torch import nn
from torch.func import functional_call
from .utils import finite


# §3.2 固定整图坐标：mean=μH∈R^d，scale=sH 是全局标量。
# 统计量作为 buffer 保存，不作为可训练参数。
class Coordinates(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())

    @classmethod
    def fit(cls, source_h, eps):
        finite(source_h)
        # μH=(1/nS)Σ_i h_i；sH=sqrt((1/(nS·d))Σ_i||h_i−μH||²+εH²)。
        # 下面第二次 mean 同时平均样本与维度，不是逐维标准差。
        mean = source_h.mean(0)
        scale = ((source_h - mean).square().mean() + eps ** 2).sqrt()
        finite(scale)
        return cls(mean, scale)

    def forward(self, h):
        # §3.2：x=(h−μH)/sH；to 同时对齐统计量的设备和精度。
        return (h.to(self.mean) - self.mean) / self.scale


# §3.3：f_ψ(x)=P_C[Wx+b+U2·SiLU(U1x+a)]。
# linear 对应 W,b；U1.weight 对应矩阵 U1，U1.bias 对应 a；U2 无偏置。
class ResidualClassifier(nn.Module):
    def __init__(self, dim, classes, width, *, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(dim, classes, device=device, dtype=dtype)
        self.U1 = nn.Linear(dim, width, device=device, dtype=dtype)
        self.U2 = nn.Linear(width, classes, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.U1.bias)
        # U2=0 使残差初始为零；U1 保留随机初始化，后续属于可训练 ψ。
        nn.init.zeros_(self.U2.weight)

    @classmethod
    def from_source(cls, linear, coordinates, width, seed):
        # CPU initialization isolates random adapter features from prior training's RNG usage.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(linear.in_features, linear.out_features, width, dtype=coordinates.mean.dtype)
        model = model.to(coordinates.mean.device)
        with torch.no_grad():
            w = linear.weight.to(coordinates.mean)
            # 由 h=sH·x+μH 得 W0h+b0=(sH·W0)x+(W0·μH+b0)。
            # 因此以下复制在数学上保持源模型的初始概率。
            model.linear.weight.copy_(coordinates.scale * w)
            model.linear.bias.copy_(w @ coordinates.mean + linear.bias.to(w))
        return model

    def forward(self, x):
        logits = self.linear(x) + self.U2(nn.functional.silu(self.U1(x)))
        # P_C=I−11ᵀ/C：减去类别平均 logit，不改变 softmax 或两类 margin。
        return logits - logits.mean(-1, keepdim=True)


# §3.9、§3.12–3.13：将 ψ 展平为 [P]，候选用独立向量求值。
# 只有 commit 修改真实模型；不反传穿过求解过程做元学习。
class ParameterMap:
    """Functional candidate evaluation: rejected steps never mutate the live model."""
    def __init__(self, model):
        self.model = model
        self.entries = [(name, p.shape, p.numel()) for name, p in model.named_parameters()]

    def vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()]).clone()

    def mapping(self, vector):
        result, offset = {}, 0
        for name, shape, size in self.entries:
            result[name] = vector[offset:offset + size].view(shape)
            offset += size
        return result

    def logits(self, vector, x):
        # 以候选 ψ 临时替换参数进行前向；保留对 vector 的响应，不覆盖当前参数。
        return functional_call(self.model, self.mapping(vector), (x,))

    # §3.12：仅由验收通过的分支调用，提交 ψ_next=ψ_bar+α·δ。
    def commit(self, vector):
        finite(vector)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(self.mapping(vector)[name])
