"""Descriptive P0 secondary metrics; never modifies the frozen decision rules."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from scipy.stats import spearmanr


def run(stage):
    stage=Path(stage).resolve()
    group=defaultdict(list)
    for seed in range(3405,3410):
        folder=stage/'logs'/str(seed)
        gradients=[json.loads(x) for x in (folder/'gradient_blocks.jsonl').read_text().splitlines()]
        true={(r['task'],r['snapshot_step'],r['block_id']):r for r in gradients if r['arm']=='TRUE'}
        for line in (folder/'geometry_stats.jsonl').read_text().splitlines():
            r=json.loads(line);t=true[(r['task'],r['snapshot_step'],r['block_id'])]
            group[(r['task'],seed)].append((r['statistics']['calibration']['mean_error'],t['e_A']))
    correlations=[]
    for (task,seed),values in group.items():
        coefficient=float(spearmanr([v[0] for v in values],[v[1] for v in values]).statistic)
        correlations.append(dict(task=task,seed=seed,blocks=len(values),proxy_vs_gradient_error_spearman=coefficient))
    groups=defaultdict(dict)
    for row in csv.DictReader((stage/'one_step_blocks.csv').open()):
        if row['arm'] in ('S','S+Q') and float(row['eta'])==.0003:
            key=(row['task'],row['seed'],row['snapshot_step'],row['block_id'])
            groups[key][row['arm']]=row
    approximation=defaultdict(list)
    for (task,seed,step,block),arms in groups.items():
        actual=float(arms['S+Q']['CE_B'])-float(arms['S']['CE_B'])
        predicted=float(arms['S+Q']['first_order_CE_B'])-float(arms['S']['first_order_CE_B'])
        approximation[task].append((actual,predicted))
    first_order=[]
    for task,values in approximation.items():
        selected=[(a,p) for a,p in values if abs(a)>1e-8]
        first_order.append(dict(task=task,all_blocks=len(values),selected_blocks=len(selected),
            median_absolute_remainder=statistics.median(abs(a-p) for a,p in values),
            median_relative_remainder=statistics.median(abs(a-p)/abs(a) for a,p in selected) if selected else None,
            sign_agreement=sum((a>0)==(p>0) for a,p in selected)/len(selected) if selected else None))
    report=dict(scope='Descriptive secondary analysis of already saved P0 outputs; no new labels or updated decision thresholds',
        correlation_unit='Within task/model seed across 24 blocks; not an independent-sample p-value',
        correlations=correlations,first_order_incremental_feedback=first_order,
        first_order_selection_note='Post-hoc descriptive numerical floor |actual CE increment|>1e-8; not used for primary criteria')
    (stage/'secondary_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    run(parser.parse_args().stage)
