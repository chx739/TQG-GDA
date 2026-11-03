# Recorded commands; these output paths already exist. Reproduce only in a fresh, equivalent run layout.
# Initial code b6fc5d6: prepare; unittest test_assumption_diagnostics; a0; pilot.
# Formal code 7222b65: same unit checks; prepare; run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# All commands used /home/hxchen/anaconda3/envs/bfn4jet/bin/python.
# assumptions_experiment.py <prepare|a0|pilot|run> --repo /home/hxchen/TQG-GDA --stage ..
# assumptions_summarize.py --stage ..
# assumptions_interpretation.py --stage /home/hxchen/runs/2026-11-03-tqg-assumptions/stage3_execution
# scripts/archive_assumptions.py --stage <stage> --repo <repo> --dest <fresh archive>
