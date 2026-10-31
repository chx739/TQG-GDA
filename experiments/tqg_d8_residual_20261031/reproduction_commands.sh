#!/usr/bin/env bash
# Original execution cwd: /home/hxchen/runs/2026-10-31-tqg-d8-residual-mechanisms/stage3_execution/code
# Requires initialized new stage2_method, execution_protocol.md, local dataset and referenced geometry caches.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/hxchen/anaconda3/envs/bfn4jet/bin/python mechanism_experiment.py prepare --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-d8-residual-mechanisms/stage3_execution --config configs/mechanism_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python mechanism_experiment.py d0 --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-d8-residual-mechanisms/stage3_execution --config configs/mechanism_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python mechanism_experiment.py pilot --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-d8-residual-mechanisms/stage3_execution --config configs/mechanism_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python mechanism_experiment.py run --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-d8-residual-mechanisms/stage3_execution --config configs/mechanism_diagnostics.json
PYTHONPATH=. /home/hxchen/anaconda3/envs/bfn4jet/bin/python ../analysis_code/mechanism_summarize.py --stage /home/hxchen/runs/2026-10-31-tqg-d8-residual-mechanisms/stage3_execution
# Run analysis_code/mechanism_interpretation.py --stage ... once to add descriptive interpretation and coverage-labelled plots.
