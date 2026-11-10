"""Export aggregate support/ESS diagnostics from existing frozen bank records.

No new neighbors, density queries or labels are read. Origin kernel-weight
summaries are descriptive follow-up checks, not predeclared screening rules.
"""
import argparse
import json
from pathlib import Path
import torch
from ..archive import write_json
from .evaluate import write_csv


def summaries(root):
    rows = []
    root = Path(root)
    for path in sorted((root / 'geometry').glob('*/banks/*.pt')):
        cell = path.parent.parent.name
        seed, domain = int(cell.split('_')[0][4:]), int(cell.split('_')[1][6:])
        bank = torch.load(path, map_location='cpu', weights_only=True)
        index = {int(i): p for p, i in enumerate(bank['target_ids'])}
        for which in [0, 1, 2, 3, 4, 'loo']:
            observations = []
            for anchor in bank['anchors']:
                panels = anchor.get('panels', [])
                panel = anchor.get('loo') if which == 'loo' else panels[which] if len(panels) > which else None
                if panel is not None:
                    observations.append((anchor, panel))
            result = dict(seed=seed, domain=domain, bank=path.stem, panel=str(which),
                          denominator=len(bank['anchors']), observed=len(observations))
            if observations:
                ess = torch.stack([p['ess'] for _, p in observations])
                positive = [len(p['support']['positive_ids']) for _, p in observations]
                negative = [len(p['support']['negative_ids']) for _, p in observations]
                B_positive = [len(p['support']['positive_B_ids']) for _, p in observations]
                B_negative = [len(p['support']['negative_B_ids']) for _, p in observations]
                result.update(center_ess_median=float(ess[:, 0].quantile(.5)), center_ess_max=float(ess[:, 0].max()),
                    minimum_three_point_ess_max=float(ess.min(1).values.max()),
                    three_point_ess_passes=int((ess >= 5).all(1).sum()),
                    center_ess_passes=int((ess[:, 0] >= 5).sum()),
                    unlabeled_supported=sum(p['support']['unlabeled_supported'] for _, p in observations),
                    B_supported=sum(p['support']['B_supported'] for _, p in observations),
                    both_sides_nonempty=sum(a > 0 and b > 0 for a, b in zip(positive, negative)),
                    max_one_side_unlabeled=max(positive + negative),
                    max_smaller_side_unlabeled=max(min(a, b) for a, b in zip(positive, negative)),
                    max_one_side_B=max(B_positive + B_negative),
                    slope_identifiable=sum(p['direction_identifiable'] for _, p in observations))
                if which == 0:
                    weights = torch.stack([p['log_kernel'][0].softmax(0)[index[a['origin_id']]] for a, p in observations])
                    result.update(origin_center_kernel_weight_median=float(weights.quantile(.5)),
                                  origin_center_kernel_weight_gt90pct=int((weights > .9).sum()))
            rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = summaries(args.root)
    write_json(args.output / 'support-ess.json', rows)
    write_csv(args.output / 'support-ess.csv', rows)


if __name__ == '__main__':
    main()
