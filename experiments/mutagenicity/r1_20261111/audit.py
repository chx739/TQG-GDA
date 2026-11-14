"""Pre-S2 checkpoint reload audit; no labels or new model fitting."""
import json
from pathlib import Path
import torch
from tsba.classifier import ResidualClassifier
from experiments.common.synthetic.archive import write_json
from experiments.common.real_data.probes import load_probe


def replay_predictions(root, plan):
    root = Path(root)
    if (root / 'seals/S2.json').exists():
        raise PermissionError('Perform model replay before opening B evaluation')
    size = torch.load(root / 'inputs/size.pt', weights_only=True)
    frozen = {}
    source_checks, probe_checks, maximum_error = 0, 0, 0.
    with torch.no_grad():
        for seed in plan['seeds']:
            path = root / 'source' / f'seed{seed}'
            if not (path / 'frozen.pt').exists():
                continue
            frozen[seed] = torch.load(path / 'frozen.pt', weights_only=True)
            data = frozen[seed]
            model = ResidualClassifier(data['x'].shape[1], 2, 16, dtype=torch.float64)
            model.load_state_dict(data['initial_head'])
            expected = torch.load(path / 'predictions.pt', weights_only=True)
            actual = model(data['x']).softmax(-1)
            torch.testing.assert_close(actual, expected['probabilities'], rtol=0, atol=0)
            source_checks += 1
        for meta_path in sorted((root / 'probes').rglob('metadata.json')):
            meta = json.loads(meta_path.read_text())
            if meta['status'] != 'success':
                continue
            path = meta_path.parent
            features = frozen[meta['seed']]['x'] if meta['feature'] == 'H' else size
            saved = torch.load(path / 'probabilities.pt', weights_only=True)
            for endpoint, pools in saved.items():
                model = load_probe(path, endpoint)
                for pool in pools.values():
                    logits = model(features[pool['ids']])
                    actual = logits.softmax(-1)
                    error = float((actual - pool['probabilities']).abs().max())
                    maximum_error = max(maximum_error, error)
                    torch.testing.assert_close(actual, pool['probabilities'], rtol=0, atol=0)
                    torch.testing.assert_close(logits.log_softmax(-1), pool['log_probabilities'], rtol=0, atol=0)
                    probe_checks += 1
    record = dict(source_checkpoint_predictions_checked=source_checks,
                  probe_endpoint_pool_predictions_checked=probe_checks,
                  maximum_probability_error=maximum_error, exact_match=True,
                  stage='S1_before_S2', labels_used=False)
    write_json(root / 'prediction-replay.json', record)
    return record
