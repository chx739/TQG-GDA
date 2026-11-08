"""Source CE initialization and frozen graph encoding (§3.2).

SEGA's source-stage/eval-checkpoint organization is retained. TSBA uses a
linear class head; SEGA reference-classifier checkpoints are not interchangeable.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from .encoder import GraphEncoder
from .utils import finite


# §3.2：z0(G)=W0·h_theta0(G)+b0；encoder 对应 h，head 对应 W0,b0。
class SourceModel(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.spec["out_dim"], classes)

    def forward(self, graph_batch):
        return self.head(self.encoder(graph_batch))


# 工程批处理：按既定顺序使用每张图一次；合并不足两个节点的批次以满足 BN。
def graph_batches(graphs, size, order=None):
    order = list(range(len(graphs))) if order is None else order
    chunks = [list(order[i:i + size]) for i in range(0, len(order), size)]
    # BatchNorm needs >=2 nodes. Merge every undersized batch, preserving the
    # given order and using each graph exactly once, including size=1 inputs.
    merged, pending = [], []
    for chunk in chunks:
        pending.extend(chunk)
        if sum(graphs[i].num_nodes for i in pending) >= 2:
            merged.append(pending)
            pending = []
    if pending:
        if merged:
            merged[-1].extend(pending)
        else:
            merged.append(pending)
    return merged


@torch.no_grad()
def source_metrics(model, pool, batch_size, device):
    if not pool.graphs:
        return None
    model.eval()
    losses, correct = [], 0
    for ids in graph_batches(pool.graphs, batch_size):
        batch = Batch.from_data_list([pool.graphs[i] for i in ids]).to(device)
        labels = pool.labels[ids].to(device)
        logits = model(batch)
        finite(logits)
        losses.append(F.cross_entropy(logits, labels, reduction="none"))
        correct += int((logits.argmax(1) == labels).sum())
    return dict(ce=float(torch.cat(losses).mean()), accuracy=correct / len(pool.graphs))


# §3.2：只用合法源训练标签，通过普通 CE 建立类别语义和表示。
# 此阶段更新编码器与线性头；适配阶段才冻结编码器。
def pretrain_encoder(cfg, data, logger=None):
    dim, edge_dim, classes = data.validate()
    encoder = GraphEncoder(dim, edge_dim, cfg.hidden, cfg.num_layers, cfg.emb_dim, cfg.dropout, cfg.encoder)
    model = SourceModel(encoder, classes).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.source_lr)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    best, state, selected = float("inf"), None, None
    history = []
    for epoch in range(cfg.source_epochs):
        model.train()
        order = torch.randperm(len(data.source_train.graphs), generator=rng).tolist()
        total = 0.0
        for ids in graph_batches(data.source_train.graphs, cfg.source_batch, order):
            batch = Batch.from_data_list([data.source_train.graphs[i] for i in ids]).to(cfg.device)
            if batch.num_nodes < 2:
                raise ValueError("SEGA BatchNorm architecture requires >=2 source nodes per training batch")
            labels = data.source_train.labels[ids].to(cfg.device)
            # L_src=−(1/BS)Σ_i log p_theta,W,b(y_i^S|G_i^S)，PyTorch 默认取批次均值。
            loss = F.cross_entropy(model(batch), labels)
            finite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters() if p.grad is not None))
            opt.step()
            finite(*(p for p in model.parameters()))
            total += float(loss.detach()) * len(ids)
        # 源验证只用于源初始化选模，不加入梯度训练；无验证池时保留最后预定 epoch。
        validation = source_metrics(model, data.source_validation, cfg.source_batch, cfg.device)
        history.append(dict(epoch=epoch, train_ce=total / len(order), source_validation=validation))
        if validation is None or validation["ce"] < best:
            best = validation["ce"] if validation is not None else total / len(order)
            selected = epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if logger:
            logger(f"source epoch {epoch}: train CE {total / len(order):.6g}")
    model.load_state_dict(state)
    model.encoder.freeze()
    model.eval().zero_grad(set_to_none=True)
    return model, dict(selected_epoch=selected, selection="source_validation_ce" if data.source_validation.graphs
                      else "last_predeclared_epoch", history=history)


@torch.no_grad()
# §3.2：冻结 theta0 并缓存整图 h；no_grad/detach 切断后续适配到 GNN 的梯度。
def encode(encoder, graphs, batch_size, device):
    encoder.freeze()
    outputs = [encoder(Batch.from_data_list([graphs[i] for i in ids]).to(device))
               for ids in graph_batches(graphs, batch_size)]
    if not outputs:
        return torch.empty((0, encoder.spec["out_dim"]), device=device, dtype=next(encoder.parameters()).dtype)
    result = torch.cat(outputs).detach()
    finite(result)
    return result
