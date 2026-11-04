"""Standalone figures for the bounded optimization experiment."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_a4(stage):
    root=Path(stage)/'A4';out=root/'figures';out.mkdir(exist_ok=True)
    with (root/'summary.csv').open() as f:rows=list(csv.DictReader(f))
    for convention in ['FIXED','NORMALIZED']:
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
        for i,sig in enumerate(['H','HA']):
            for j,task in enumerate(['0_to_1','0_to_2','0_to_3']):
                ax=axes[i,j]
                for td,pool,label in [('S','S_audit','Source to source'),('S','B','Source to target'),('T','B','Target to target (diagnostic)')]:
                    rs=sorted([r for r in rows if (r['signature'],r['task'],r['convention'],r['train_domain'],r['pool'])==(sig,task,convention,td,pool)],key=lambda r:int(r['n']))
                    ax.errorbar([int(r['n']) for r in rs],[float(r['nll']) for r in rs],
                        yerr=[float(r['nll_sd']) for r in rs],marker='o',capsize=2,label=label)
                ax.set_title(f'{sig}: {task.replace("_to_"," -> ")}');ax.set_xticks([32,64,128,147]);ax.grid(alpha=.2)
                if j==0:ax.set_ylabel('NLL (lower is better)')
                if i==1:ax.set_xlabel('Reader training graphs')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
        fig.suptitle(f'A4: nested-CV selected kernels, {convention}; mean +/- model-seed SD')
        fig.tight_layout(rect=(0,.07,1,.95))
        for extension in ['png','pdf']:fig.savefig(out/f'learning_curves_{convention.lower()}.{extension}',dpi=160)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();plot_a4(a.stage)
