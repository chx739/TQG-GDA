"""Post-run descriptive audit of saved score/source contributions; no fitting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.density_and_single_step_diagnostics_20261111.src.common import verify_seal


def norm(value):
    return float(value.norm())


def ratio(a, b):
    return a/b if b else None


def rms(value):
    return float(value.square().mean().sqrt()) if value.numel() else None


def summarize(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise PermissionError('Post-run analysis must not modify the experiment')
    obs, final = verify_seal(root,'OBS'), verify_seal(root,'FINAL')
    index = json.loads((root/'steps/index.json').read_text())
    by_key = {(r['seed'],r['domain'],r['estimator'],r['bandwidth_multiplier'],r['arm']):r for r in index}
    inputs, rows, static_rows = {}, [], []
    def load(rel):
        expected = obs['files'].get(rel)
        actual = digest(root/rel)
        if actual != expected:
            raise PermissionError('Unsealed input: '+rel)
        inputs[rel] = actual
        return torch.load(root/rel,weights_only=False)
    for record in index:
        if record['arm'] != 'normal' or record['status'] != 'success':
            continue
        key = tuple(record[k] for k in ('seed','domain','estimator','bandwidth_multiplier'))
        normal, matched, reverse = [load(by_key[key+(a,)]['path']) for a in ('normal','matched_source','reverse')]
        sigma = record['sigma0']
        zeta = normal['boundaries'].zeta
        item = dict(zip(('seed','domain','estimator','bandwidth_multiplier'),key),
            parameter_score_source_norm_ratio=ratio(norm(normal['score_proposal_delta']),norm(normal['source_proposal_delta'])),
            boundary_score_source_norm_ratio=ratio(norm(normal['proposal_score_movement']),norm(normal['proposal_source_movement'])),
            score_source_rhs_norm_ratio=ratio(norm(normal['score_rhs']),norm(normal['source_rhs'])),
            requested_rms_sigma=rms(zeta/sigma),views={})
        for view in ('small_step','production_backtrack'):
            a,b,c = [p['views'][view] for p in (normal,matched,reverse)]
            increment = a['actual_margin_movement']-b['actual_margin_movement']
            probabilities = [p['probabilities'][f"B{record['domain']}"]['probabilities'].argmax(1) for p in (a,b,c)]
            item['views'][view] = dict(normal_alpha=a['alpha'],matched_alpha=b['alpha'],reverse_alpha=c['alpha'],
                normal_vs_matched_prediction_disagreements=int((probabilities[0]!=probabilities[1]).sum()),
                normal_vs_reverse_prediction_disagreements=int((probabilities[0]!=probabilities[2]).sum()),
                normal_source_CE_change=a['after']['source_ce']-a['before']['source_ce'],
                normal_actual_rms_sigma=rms(a['actual_margin_movement']/sigma),
                matched_actual_rms_sigma=rms(b['actual_margin_movement']/sigma),
                normal_minus_matched_margin_rms_sigma=rms(increment/sigma),
                incremental_margin_request_norm_ratio=ratio(norm(increment),norm(zeta)),
                incremental_margin_request_cosine=ratio(float(increment@zeta),norm(increment)*norm(zeta)),
                normal_request_cosine=ratio(float(a['actual_margin_movement']@zeta),norm(a['actual_margin_movement'])*norm(zeta)),
                normal_request_rmse_ratio=ratio(norm(a['actual_margin_movement']-zeta),norm(zeta)),
                zero_step_request_rmse_ratio=1.)
        rows.append(item)
    for record in json.loads((root/'scores/index.json').read_text()):
        if record['status'] != 'success':
            continue
        value = load(record['path']+'/static.pt')
        empirical, learned = value['empirical_gamma'],value['learned_gamma']
        static_rows.append({k:record[k] for k in ('seed','domain','bandwidth_multiplier')} | dict(
            empirical_normal_rms_sigma=rms(record['sigma0']*empirical),
            learned_normal_rms_sigma=rms(record['sigma0']*learned),
            normal_relative_L2_error=ratio(norm(learned-empirical),norm(empirical)),
            normalization='relative to saved finite-pool empirical normal score, not population truth'))
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['domain'],row['estimator'],row['bandwidth_multiplier']].append(row)
    summaries=[]
    for key,items in sorted(grouped.items()):
        item=dict(zip(('domain','estimator','bandwidth_multiplier'),key),seed_count=len(items),views={})
        for metric in ('parameter_score_source_norm_ratio','boundary_score_source_norm_ratio','score_source_rhs_norm_ratio','requested_rms_sigma'):
            values=[r[metric] for r in items]
            item[metric]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),min=min(values),max=max(values))
        for view in ('small_step','production_backtrack'):
            item['views'][view]={}
            for metric in items[0]['views'][view]:
                values=[r['views'][view][metric] for r in items]
                valid=[v for v in values if v is not None]
                item['views'][view][metric]=dict(mean=statistics.mean(valid) if len(valid)==5 else None,
                    available_count=len(valid),expected_count=5,min=min(valid) if valid else None,max=max(valid) if valid else None)
        summaries.append(item)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(scope='Post-run arithmetic on presealed candidates/probabilities; no new model, labels, queries or density estimate',
        execution_OBS=obs['digest'],execution_FINAL=final['digest'],source_code_sha256=digest(Path(__file__)),inputs=inputs,
        rows=rows,summary=summaries,static_relative_errors=static_rows)
    write_json(output,result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.output)
    print(json.dumps(dict(combinations=len(result['rows']), static_models=len(result['static_relative_errors']))))
