"""Static experiment plot; consumes sealed, already-computed metrics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary,destination):
    destination.mkdir(parents=True,exist_ok=True)
    arms=['B_frozen','B_fit','B_full','B_pseudo','Q_query_O','Q_query_OE','Q_all_O','Q_all_OE']
    labels=['Frozen','Fit CE','Full CE','Pseudo','Query odd','Query odd+even','All odd','All odd+even']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for domain,ax in zip([1,2,3],axes):
        values=[summary[f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else 'fixed_200']['balanced_accuracy'] for arm in arms]
        ax.bar(np.arange(8),[100*v['mean'] for v in values],yerr=[100*v['sd'] for v in values],capsize=3)
        ax.set_xticks(np.arange(8),labels,rotation=65,ha='right');ax.set_title(f'0 → {domain}')
        ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    axes[0].set_ylabel('Balanced accuracy (%)');fig.tight_layout()
    fig.savefig(destination/'balanced_accuracy.png',dpi=180);fig.savefig(destination/'balanced_accuracy.pdf');plt.close(fig)
