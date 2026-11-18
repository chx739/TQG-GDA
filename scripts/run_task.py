"""Run a predeclared graph adaptation task or a generated-graph smoke check."""
import argparse
import json
from pathlib import Path
import torch
from tsba.config import Config
from tsba.data import from_payload, load_task
from tsba.pipeline import run_task


def synthetic_task(seed=0):
    rng = torch.Generator().manual_seed(seed)
    payload = dict(schema_version=1, protocol="transductive", source_train=[], source_validation=[], target=[])
    for role, count in (("source_train", 12), ("source_validation", 4), ("target", 10)):
        for i in range(count):
            center = (-1 if i % 2 == 0 else 1) + (0.3 if role == "target" else 0)
            record = dict(id=f"synthetic/{role}/{i}", x=torch.randn((4, 3), generator=rng) * .2 + center,
                          edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
                          edge_attr=torch.ones((6, 1)))
            if role != "target":
                record["label"] = i % 2
            payload[role].append(record)
    return from_payload(payload)


def _run_shared(args, overrides, parser):
    from tsba.pretrain import SOURCE_KEYS
    from tsba.shared_update import SharedUpdateConfig

    try:
        from tsba.shared_update.pipeline import run_shared_task
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        parser.error("shared-update requires SciPy; install tsba-gda[shared-update]")
    if not isinstance(overrides, dict) or overrides.keys() - {"source", "adaptation"}:
        parser.error("shared-update config must contain only source and adaptation objects")
    source, adaptation = overrides.get("source", {}), overrides.get("adaptation", {})
    if not isinstance(source, dict) or not isinstance(adaptation, dict):
        parser.error("source and adaptation must be JSON objects")
    if source.keys() - {*SOURCE_KEYS, "device"}:
        parser.error("source contains an option outside source training; see the shared-update guide")
    defaults = dict(hidden=8, num_layers=2, emb_dim=4, source_epochs=3, source_batch=16) if args.synthetic else {}
    source_cfg = Config(**(defaults | source))
    update_cfg = SharedUpdateConfig(**({"seed": source_cfg.seed} | adaptation))
    data = synthetic_task(source_cfg.seed) if args.synthetic else load_task(args.input)
    model, result = run_shared_task(source_cfg, update_cfg, data, args.output, args.source_checkpoint, logger=print)
    print(json.dumps(dict(method="shared-update", status=result["status"], run_identity=result["run_identity"],
                          adaptation_status=result["adaptation"]["status"],
                          applied_update=bool(model.correction.count_nonzero()),
                          stop_reason=result["adaptation"]["stop_reason"])))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tsba", "shared-update"), default="tsba")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="explicit source/validation/unlabeled target tensor archive")
    inputs.add_argument("--synthetic", action="store_true", help="generated-graph engineering check only")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--source-checkpoint", type=Path)
    args = parser.parse_args(argv)
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if args.method == "shared-update":
        return _run_shared(args, overrides, parser)
    if args.synthetic:
        cfg_dict = dict(hidden=8, num_layers=2, emb_dim=4, source_epochs=3, source_batch=16,
                        score_hidden=8, score_epochs=5, adapt_rounds=3, target_batch=10)
        cfg_dict.update(overrides)
        cfg = Config(**cfg_dict)
        data = synthetic_task(cfg.seed)
    else:
        cfg = Config(**overrides)
        data = load_task(args.input)
    _, result = run_task(cfg, data, args.output, args.source_checkpoint, logger=print)
    print(json.dumps(dict(status=result["status"], run_identity=result["run_identity"],
                          accepted=result["adaptation"]["accepted"],
                          stop_reason=result["adaptation"]["stop_reason"])))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
