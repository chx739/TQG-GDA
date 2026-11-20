"""Aggregate pair-matched Other-Binding endpoints and setting-level graph counts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.common.io import digest
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run as storage
from ..src import scoring

IDENTITY = ('setting','source','target','seed')
METRICS = ('accuracy','balanced_accuracy','recall_0','recall_1','sampled_nll','auroc','predicted_class_1')
COUNTS = ('n','positive_count','corrected','harmed','net_corrections')


def number(field):
    return None if not field else float(field)


def aggregate(rows):
    blocks = {}
    for row in rows:
        key = tuple(row[k] for k in IDENTITY)
        group = blocks.setdefault(key, {})
        if row['method'] in group:
            raise ValueError('Duplicate sealed classification endpoint')
        group[row['method']] = row
    foreign = []
    for key, by_arm in blocks.items():
        own = [by_arm[f'Other_Binding_{i}'] for i in (1,2)]
        if own[0]['n'] != own[1]['n']:
            raise ValueError('Foreign arms require the same set of paired H graph rows')
        average = dict(zip(IDENTITY,key),method='Other_Binding_mean')
        for name in (*METRICS,*COUNTS):
            x,y = (number(r[name]) for r in own)
            average[name] = None if x is None or y is None else (x+y)/2
        foreign.extend((*own,average))
    averages = [row for row in foreign if row['method']=='Other_Binding_mean']
    summary = []
    groups = sorted({(r['setting'],r['method']) for r in rows+averages})
    for setting,method in groups:
        matches = [r for r in rows+averages if r['setting']==setting and r['method']==method]
        metrics = {key: [v if not isinstance(v,str) else number(v)
                         for v in (r[key] for r in matches)] for key in METRICS}
        summary.append(dict(setting=setting,method=method,blocks=len(matches),
            **{f'total_{k}':sum(float(r[k]) for r in matches) for k in COUNTS},
            **{k:None if None in values else float(np.mean(values)) for k,values in metrics.items()}))
    return foreign, summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    obs_sha = digest(args.run/'OBS.json')
    final_sha = digest(args.run/'FINAL.json')
    members = storage.verified_members(args.run,'FINAL',final_sha,scoring.FINAL_MEMBERS,{'obs_sha256':obs_sha})
    rows = list(csv.DictReader(members['endpoints.csv'].decode().splitlines()))
    alien, summary = aggregate(rows)
    args.output.mkdir(parents=True,exist_ok=False)
    write_table(args.output/'other-binding-endpoints.csv',alien)
    write_table(args.output/'setting-counts.csv',summary)
    sha = storage.seal(args.output,'SUMMARY',['other-binding-endpoints.csv','setting-counts.csv'],{'final_sha256':final_sha})
    print(json.dumps(dict(summary_sha256=sha,other_rows=len(alien),setting_rows=len(summary))))


if __name__=='__main__':
    main()
