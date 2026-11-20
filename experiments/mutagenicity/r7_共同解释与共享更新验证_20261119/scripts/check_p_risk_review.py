"""Reproduce the post-review candidate and shared-draw interval checks."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from experiments.common.io import digest
from ..src import fit
from ..src.p_risk_replay import read_npz
from ..src.paired import _cohort
from ..src.r5_inputs import evaluation_labels
from ..src.seals import verified_seal_chain


def weighted_auc(margins, y, weights):
    """Each tied positive/negative pair earns one half, also after resampling."""
    order = np.argsort(margins, kind='stable')
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(margins[order]))]
    pos = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1-y[order]), starts, axis=1)
    wins = (pos * (np.cumsum(neg, axis=1) - 0.5*neg)).sum(axis=1)
    return wins / (pos.sum(axis=1)*neg.sum(axis=1))


def run(config_path):
    check = json.loads(config_path.read_bytes())
    root = Path(check['input'])
    repo = Path(__file__).resolve().parents[4]
    nll, auc = {}, {}
    worst, count, cohort = 0., 0, None
    with verified_seal_chain(root, 'R7_FINAL') as scope, evaluation_labels(root, repository=repo) as labels:
        if json.loads((root/'seals/R7_FINAL.json').read_bytes())['digest'] != check['source_final_digest']:
            raise ValueError('Unexpected source FINAL seal')
        plan = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        e = plan['evaluation']
        for seed in check['seeds']:
            d = f"blocks/{check['setting']}-M{check['source']}_to_M{check['target']}-seed-{seed}"
            width = fit.width_key(check['tau'])
            std = read_npz(scope, f'{d}/standardization.npz', 'R7_OBS')
            inputs = read_npz(scope, f'{d}/fit_input.npz', 'R7_INPUT')
            obs = read_npz(scope, f'{d}/{width}-observed.npz', 'R7_OBS')
            family = fit.explanation_family(plan, std['source_u'][:, :-1], inputs['source_y'],
                std['adapt_u'][:, :-1], fit.config(plan, seed, check['tau']), std['v0'],
                source_mean=std['mean'], source_scale=std['scale'])
            change = log_softmax(obs['adapt_logits_J'], axis=1) - log_softmax(obs['adapt_logits_B0'], axis=1)
            update = obs['w_J'][1] - obs['w_J'][0]
            for prefix in ('', 'audit_'):
                for theta, moment in zip(obs[prefix+'theta'], obs[prefix+'stats'], strict=True):
                    q = family._target_terms(theta)[-1]
                    direct = -(q*change).sum(axis=1).mean()
                    indirect = -change[:, 0].mean() - moment[1] @ update
                    worst = max(worst, abs(direct-indirect), abs(q.T @ std['adapt_u']/len(q)-moment).max())
                    count += 1
            held = read_npz(scope, f'{d}/{width}-heldout.npz', 'R7_OBS')
            ids = np.asarray(json.loads(scope.read_bytes(f'{d}/manifest.json','R7_INPUT'))['ids']['holdout'])
            order = np.argsort(ids)
            y = labels.load(f'{d}/manifest.json','holdout',authenticated_snapshot=True)[order]
            if cohort is None:
                cohort = _cohort(check['setting'],check['target'],ids[order],y,e['cohort_bootstrap_replicates'],
                    e['cohort_bootstrap_seed'],e['paired_interval']['minimum_graphs_per_class_for_interval'])
            np.testing.assert_array_equal(ids[order],cohort['ids'])
            np.testing.assert_array_equal(y,cohort['y'])
            w = cohort['weights']
            b, j = (held['holdout_logits_'+key][order] for key in ('B0','J'))
            delta = (log_softmax(b, axis=1)-log_softmax(j, axis=1))[np.arange(len(y)),y]
            nll[seed] = w @ delta
            auc[seed] = weighted_auc(j[:,1]-j[:,0], y,w)-weighted_auc(b[:,1]-b[:,0], y,w)
    if worst > check['atol']:
        raise ValueError('Candidate moments or risks differ from their per-graph reconstruction')
    def ci(vals):
        return np.quantile(vals,[.025,.975]).tolist()
    return dict(source_final=check['source_final_digest'], config_sha256=digest(config_path),
                script_sha256=digest(Path(__file__)), candidates=count, max_error=worst,
                draws=cohort['audit']['draw_counts_sha256'], nll_ci=ci(np.mean(list(nll.values()),axis=0)),
                auc_delta_ci=ci(np.mean(list(auc.values()),axis=0)), nll_ci_by_seed={s:ci(x) for s,x in nll.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), indent=2))
