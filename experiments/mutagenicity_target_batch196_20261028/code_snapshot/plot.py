"""Standalone scientific plot for the frozen target-batch study."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arms=['B_frozen','B_fit','B16','B32','B64','B196']
    labels=['Frozen','Fit SGD','TQG 16','TQG 32','TQG 64','TQG 196']
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    for domain,ax in zip((1,2,3),axes):
        groups=[summary[f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else 'fixed_200']['balanced_accuracy'] for arm in arms]
        means=np.array([g['mean'] for g in groups])*100
        errors=np.array([g['sd'] for g in groups])*100
        ax.bar(np.arange(len(arms)),means,yerr=errors,capsize=3,color=['#a0a9b2','#8999aa','#6b9bc1','#448aab','#df8a42','#b95a59'])
        ax.set_xticks(np.arange(len(arms)),labels,rotation=35,ha='right')
        ax.set_ylim(max(0,min(means-errors)-2),min(100,max(means+errors)+2))
        ax.set_title(f'Mutagenicity 0 → {domain}');ax.set_ylabel('Balanced accuracy (%)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Target batch size · fixed 200 SGD steps · five seeds · development only')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(folder/f'target_batch_accuracy.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
