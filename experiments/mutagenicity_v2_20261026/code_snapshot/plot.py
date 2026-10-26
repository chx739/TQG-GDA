"""Standalone scientific figure, BA mean and sample SD over five seeds."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arms=['B_frozen','B_fit','B_full','B_pseudo','KRR','V2_repr','V2_permute','V2_joint']
    labels=['Frozen','Fit SGD','Full SGD','Pseudo','KRR','h only','Shuffled','V2 joint']
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for domain,ax in zip([1,2,3],axes):
        groups=[summary[f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else 'fixed_200']['balanced_accuracy'] for arm in arms]
        means=np.array([m['mean'] for m in groups])*100
        errors=np.array([m['sd'] for m in groups])*100
        ax.bar(np.arange(len(arms)),means,yerr=errors,capsize=3,
            color=['#87929e']*4+['#7a9bbd','#7a9bbd','#7a9bbd','#d47b40'])
        ax.set_xticks(np.arange(len(arms)),labels,rotation=48,ha='right')
        ax.set_ylim(max(0,min(means-errors)-3),min(100,max(means+errors)+3))
        ax.set_title(f'Mutagenicity 0 → {domain}')
        ax.set_ylabel('Balanced accuracy (%)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fixed 200 SGD steps · five seeds · development only')
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(folder/f'balanced_accuracy.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
