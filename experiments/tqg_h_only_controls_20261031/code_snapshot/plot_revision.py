"""Export reproducible diagnostic/classification figures from saved summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revision_common import STAGE, Budget, read


def main():
    budget=Budget('scientific_figures')
    status='failed'
    try:
        output=STAGE/'figures'
        output.mkdir(exist_ok=True)
        source=read(STAGE/'diagnostics/D1/source_summary.json')
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
        for axis,task in zip(axes,('0_to_2','0_to_3')):
            for columns in (2,4,8):
                means,errors=[],[]
                for n in (8,16,32):
                    group=next(g for g in source['groups'] if g['task']==task and
                               g['configuration']==f'fixed__target__n{n}__M{columns}__ridge0.01')
                    metric=group['metrics']['source_audit_residual_mse']
                    means.append(metric['mean']);errors.append(metric['seed_std'])
                axis.errorbar([8,16,32],means,yerr=errors,marker='o',label=f'{columns} odd columns')
            axis.set(title=task,xlabel='Source labels used to fit coefficients',ylabel='Source audit residual MSE')
            axis.set_xticks([8,16,32]);axis.legend(fontsize=8)
        fig.savefig(output/'D1_sample_size.png',dpi=180);fig.savefig(output/'D1_sample_size.pdf');plt.close(fig)
        summary=read(STAGE/'results_summary.json')
        arms=['B_frozen','B_fit','B_full','T0','T1','T2','T3','T4','T5']
        if summary['stage_B_executed']:
            arms+=['T6','T7','T9','T10','T11','T12','T13','T14','T15','T16']
        fig,axes=plt.subplots(2,1,figsize=(max(9,len(arms)*.6),7),constrained_layout=True)
        for axis,task in zip(axes,('0_to_2','0_to_3')):
            means,errors=[],[]
            for arm in arms:
                group=next(g for g in summary['groups'] if (g['task'],g['arm'],g['selection'])==(task,arm,'fixed_200'))
                metric=group['metrics']['balanced_accuracy']
                means.append(100*metric['mean']);errors.append(100*metric['seed_std'])
            axis.bar(arms,means,yerr=errors,capsize=2)
            axis.set(title=task,ylabel='Balanced accuracy (%)')
            axis.tick_params(axis='x',rotation=45)
        fig.savefig(output/'classification.png',dpi=180);fig.savefig(output/'classification.pdf');plt.close(fig)
        status='completed'
    finally:
        budget.finish(status)


if __name__=='__main__':
    main()
