"""Run only the new acceptance suites and real-data role/provenance checks."""
import json
import sys
import unittest
from pathlib import Path

import torch

from revision_common import Context, Budget, STAGE, DESIGN, read, atomic_json
from tqg_gda.utils import sha256


def main():
    ctx=Context()
    budget=Budget('D0_acceptance')
    status='failed'
    try:
        suite=unittest.TestSuite()
        for pattern in ('test_revision_math.py','test_d1_design.py','test_revision_integration.py','test_revision_training.py'):
            suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=pattern))
        result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise AssertionError('D0 acceptance failed')
        role_checks=[]
        for domain in (2,3):
            for all_source in (False,True):
                task=ctx.task(domain,all_source)
                for seed in ctx.plan['seeds']:
                    model,state,path=ctx.source(domain,seed,all_source)
                    role_checks.append(dict(domain=domain,seed=seed,all_source=all_source,
                                            checkpoint=str(path),sha256=sha256(path)))
                source_sets=[set(task.fit_ids),set(task.query_ids),set(task.validation_ids)]
                if any(a&b for i,a in enumerate(source_sets) for b in source_sets[i+1:]):
                    raise ValueError('Source data roles overlap')
                target=task.batch(task.target_ids[:2],'target')
                if target.y is not None:
                    raise PermissionError('Target labels exposed by training input view')
        preserved=read(DESIGN/'preserved_inputs.json')
        if any(sha256(path)!=digest for path,digest in preserved.items()):
            raise ValueError('Original plan/evidence changed')
        atomic_json(STAGE/'D0_acceptance.json',dict(passed=True,tests=result.testsRun,
            commit=ctx.commit,real_checkpoint_role_checks=role_checks,target_label_scores_computed=False,
            original_evidence_unchanged=True,real_training_jobs_started=0))
        status='completed'
    finally:
        budget.finish(status)


if __name__=='__main__':
    main()
