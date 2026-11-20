"""Summarize registered direction-level rates with equal direction weights."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import sys

out = Path(sys.argv[1])
with (out / 'directions-by-radius.csv').open() as f:
    by_depth = list(csv.DictReader(f))
with (out / 'directions-union.csv').open() as f:
    combined = list(csv.DictReader(f))

def summarise(rows, layer):
    data = defaultdict(list)
    for row in rows:
        rad = row.get('radius', layer)
        for feature, value in row.items():
            if feature == 'eligible' or feature == 'oov' or feature.startswith(('raw_hit_', 'hit_', 'delta_', 'graph_')):
                data[row['setting'], rad, feature].append(float(value))
                data['all', rad, feature].append(float(value))
        if 'hit_local' in row:
            for group in ('all', row['setting']):
                data[group,rad,'hit_foreign_mean'].append((float(row['hit_foreign1']) + float(row['hit_foreign2'])) / 2)
                data[group,rad,'raw_minus_primary_local'].append(float(row['raw_hit_local']) - float(row['hit_local']))
    return [dict(setting=g, radius=r, metric=metric, directions=len(values),
                 mean=fmean(values), minimum=min(values), maximum=max(values),
                 positive_directions=sum(v>0 for v in values), negative_directions=sum(v<0 for v in values))
            for (g, r, metric), values in sorted(data.items())]

summary = summarise(by_depth, None) + summarise(combined, 'union')
with (out / 'summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=summary[0], lineterminator='\n')
    w.writeheader()
    w.writerows(summary)
receipts = json.loads((out / 'checks.json').read_bytes())['groups']
for group in ('all', 'mutagenicity_edge','mutagenicity_node','proteins_edge'):
    subset = [r for r in receipts if group=='all' or r['setting']==group]
    print(group, {k:sum(r[k] for r in subset) for k in ['n_s','n_v_read','n_q_read','n_r_read','raw_graphs_read']},
          'k', [r['k'] for r in subset])
for r in summary:
    selected = r['metric'] in ('oov','eligible','hit_local','hit_validation','hit_foreign_mean','delta_local_validation',
                               'delta_local_foreign_mean','raw_minus_primary_local','graph_any_local',
                               'graph_changed_local_validation','graph_changed_local_foreign1','graph_changed_local_foreign2')
    if (r['setting']=='all' or r['metric'] in ('hit_local','delta_local_foreign_mean')) and selected:
        print(r['setting'], r['radius'], r['metric'],
              *[f"{r[k]*100:.4f}" for k in ('mean','minimum','maximum')],r['positive_directions'],r['negative_directions'])
