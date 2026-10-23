"""Standalone, seed-level plots of the frozen pilot; no training or target label access."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-root',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args();root=args.execution_root
    output=root/'figures';output.mkdir(exist_ok=True)
    paths=[]
    results=root/'results.csv'
    if results.exists():
        with results.open() as stream:rows=list(csv.DictReader(stream))
        rows=[r for r in rows if r.get('checkpoint_role','fixed_final') in ('fixed_final','final','fixed_step_200')]
        tasks=sorted(set(r['task'] for r in rows))
        fig,axes=plt.subplots(1,len(tasks),figsize=(6*max(1,len(tasks)),4),squeeze=False)
        for axis,task in zip(axes[0],tasks):
            subset=[r for r in rows if r['task']==task]
            arms=sorted(set(r.get('arm') or r.get('config_name') for r in subset))
            for position,arm in enumerate(arms):
                current=[r for r in subset if (r.get('arm') or r.get('config_name'))==arm]
                values=[100*float(r.get('balanced_accuracy',r.get('primary_metric'))) for r in current]
                axis.errorbar(position,statistics.mean(values),yerr=statistics.stdev(values) if len(values)>1 else 0,
                    fmt='o',capsize=4,color='tab:blue')
                axis.scatter([position+.04*(i-(len(values)-1)/2) for i in range(len(values))],values,s=12,color='0.5')
            axis.set_xticks(range(len(arms)),[a.replace('_',' ') for a in arms],rotation=35,ha='right',fontsize=8)
            axis.set_title(task);axis.set_ylabel('Balanced accuracy (%)');axis.grid(axis='y',alpha=.2)
        fig.tight_layout()
        for suffix in ('png','pdf'):
            path=output/f'classification.{suffix}';fig.savefig(path,dpi=180);paths.append(str(path))
        plt.close(fig)
    target=root/'diagnostics/target_summary.json'
    if target.exists():
        groups=json.loads(target.read_text())['groups']
        groups=[g for g in groups if g['domain']=='target_heldout' and g['query_count']==16
                and g['label_condition']=='true_source_labels' and g['direction'] in ('target','random','source')]
        tasks=sorted(set(g['task'] for g in groups));states=sorted(set(g['state'] for g in groups))
        fig,axes=plt.subplots(len(states),len(tasks),figsize=(5*len(tasks),3.5*len(states)),squeeze=False)
        for row,state in enumerate(states):
            for col,task in enumerate(tasks):
                ax=axes[row,col]
                for x,direction in enumerate(('target','random','source')):
                    group=next(g for g in groups if g['state']==state and g['task']==task and g['direction']==direction)
                    metric=group['metrics']['gradient_cosine_full']
                    ax.errorbar(x,metric['mean'],yerr=metric['seed_std'],fmt='o',capsize=4)
                    values=[s['gradient_cosine_full'] for s in group['per_seed']]
                    ax.scatter([x+.04*(i-2) for i in range(len(values))],values,s=12,color='0.5')
                ax.axhline(0,color='0.5',linewidth=.8);ax.set_xticks(range(3),['Target probes','Random','Source probes'])
                ax.set_title(task+' / '+state.replace('_',' '),fontsize=9)
                ax.set_ylabel('Feedback / target CE gradient cosine');ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        for suffix in ('png','pdf'):
            path=output/f'mechanism.{suffix}';fig.savefig(path,dpi=180);paths.append(str(path))
        plt.close(fig)
    (output/'plot_manifest.json').write_text(json.dumps(dict(files=paths,uncertainty='sample SD across model seeds; blocks averaged first'),indent=2)+'\n')
    print(json.dumps(dict(files=paths)))


if __name__=='__main__':main()
