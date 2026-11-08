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


# §3.16 总流程：源模型 → 固定坐标 → 目标 score → 分类适配 → 推断模型。
# 输入 data 含源训练/源验证/无标签目标池；cfg 是预先固定配置。
# 返回 predictor（可独立推断）和 result（过程记录）；适配函数原位更新 classifier。
def run_task(cfg, data, output_dir=None, source_checkpoint=None, logger=None):
    # §3.1：先检查数据角色与特征接口；下面的身份哈希用于复现，不是训练目标。
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
        # 阶段一 / §3.2：训练或复用 z0(G)=W0 h_theta0(G)+b0。
        # 源模型是编码器加线性头，不是 SEGA 的参照库分类器。
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
        # §3.2：encode 内部冻结编码器（含 BN 统计并关闭 dropout）。
        # h_source:[nS,d] 是整图表示；后续适配不更新 theta0。
        h_source = encode(source.encoder, data.source_train.graphs, cfg.source_batch, cfg.device).to(dtype)
        # μH、sH 仅由源训练表示估计；x(G)=(h_theta0(G)−μH)/sH。
        # 目标与源验证使用同一个变换，不单独标准化、不参与拟合统计量。
        coordinates = Coordinates.fit(h_source, cfg.coordinate_eps)
        x_source = coordinates(h_source)
        x_target = coordinates(encode(source.encoder, data.target.graphs, cfg.source_batch, cfg.device))
        x_val = coordinates(encode(source.encoder, data.source_validation.graphs, cfg.source_batch, cfg.device))
        y_source = data.source_train.labels.to(cfg.device)
        y_val = data.source_validation.labels.to(cfg.device)
        # §3.3：W=sH·W0，b=W0·μH+b0，U2=0；初始概率与源模型一致。
        # 适配变量仅为分类头 ψ=(W,b,U1,a,U2)。
        classifier = ResidualClassifier.from_source(source.head, coordinates,
                                                    cfg.adapter_width or min(16, cfg.emb_dim), cfg.seed)
        before = source_readout_metrics(classifier, x_source, y_source)
        with torch.no_grad():
            initial_probs = classifier(x_target).softmax(-1)
        # B: independent target-only score training, one fixed bandwidth.
        # 阶段二 / §3.4–3.5：仅凭目标输入确定 sigma，再拟合平滑密度 score。
        # score 是输入坐标的对数密度梯度，不是目标分类损失对参数的梯度。
        sigma = bandwidth(x_target, cfg.kappa_sigma)
        score_info = None
        if sigma is None:
            adaptation = dict(accepted=0, stop_reason="fewer_than_two_distinct_target_points", history=[])
        else:
            score, score_info = train_score(x_target, sigma, cfg)
            # C: refresh snapshot -> local proposal -> true-forward acceptance -> commit.
            # 阶段三 / §3.6–3.14：内部反复构造边界请求、求解并验收。
            # 传入源标签 y_source；没有目标标签。返回日志，classifier 留下最后接受的参数。
            adaptation = adapt_classifier(classifier, x_source, y_source, x_target, score, sigma, cfg, logger)
        # D: retain only encoder, fixed transform and final accepted classifier.
        # 阶段四 / §3.17：G → h_theta0(G) → x(G) → f_ψ*(x) → softmax。
        # 最终 predictor 不需要 score 网络、源参照库或目标池。
        predictor = Predictor(source.encoder, coordinates, classifier)
        with torch.no_grad():
            final_probs = classifier(x_target).softmax(-1)
        finite(final_probs)
        # 预测改变数量仅说明决策变化，不代表准确率提高；本入口不计算目标准确率。
        result = dict(**metadata, status="numerical_error" if adaptation["stop_reason"] == "numerical_error"
                      else "completed", source_pretraining=source_info, sigma=sigma, score=score_info,
                      adaptation=adaptation, source_before=before,
                      source_after=source_readout_metrics(classifier, x_source, y_source),
                      source_validation_after=source_readout_metrics(classifier, x_val, y_val),
                      target_prediction_changes=int((initial_probs.argmax(1) != final_probs.argmax(1)).sum()),
                      target_labels_used=False, target_accuracy=None)
        # 可选落盘：model.pt 是推断模型，predictions.pt 是概率，result.json 是记录。
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
