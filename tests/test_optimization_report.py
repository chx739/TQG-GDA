import numpy as np
import pandas as pd

from workflows.v2.optimization.optimization_feedback_report import paired_blocks,seed_summary


def test_incomplete_null_does_not_silently_use_two_repetitions():
    rows=[]
    for arm in ['FULL','BASE','BASE_M']+[f'{p}_{i}' for p in ['CPL','CRL','CPL_M','CRL_M','FULL_PERM_M','FULL_RAND_M'] for i in [1,2,3]]:
        rows.append(dict(task='x',seed=1,snapshot_step=0,block_id=0,profile='p',arm=arm,c_A=.5,
                         control_valid=arm!='FULL_PERM_M_3',valid_comparison=True))
    table=paired_blocks(pd.DataFrame(rows),'c_A')
    assert np.isnan(table.iloc[0]['DELTA_FULL_PERM_M_AVG'])
    assert table.iloc[0]['DELTA_FULL_RAND_M_AVG']==0


def test_snapshot_weight_and_missing_coverage_are_preserved():
    rows=[dict(task='x',seed=1,profile='p',snapshot_step=s,block_id=b,FULL=value)
          for s,values in [(0,[1.,3.]),(50,[4.,np.nan]),(200,[6.,6.])] for b,value in enumerate(values)]
    _,seeds,coverage=seed_summary(pd.DataFrame(rows),['FULL'])
    assert seeds.iloc[0].FULL==4 # mean(2,4,6), not mean of five valid blocks
    assert coverage.iloc[0].FULL==5/6
    for row in rows:
        if row['snapshot_step']==50:row['FULL']=np.nan
    _,seeds,coverage=seed_summary(pd.DataFrame(rows),['FULL'])
    assert np.isnan(seeds.iloc[0].FULL)
    assert coverage.iloc[0].FULL==4/6
