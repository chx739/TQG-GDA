#!/usr/bin/env bash
# Requires existing archived checkpoints, Mutagenicity local cache and referenced P0/D34 geometry.
# Run from this experiment's frozen code directory, with a NEW empty stage initialized as described in candidate_protocol.md.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Exact original stage path retained for provenance; use a fresh destination for a reproduction.
# Python: /home/hxchen/anaconda3/envs/bfn4jet/bin/python
# Repo: /home/hxchen/TQG-GDA
# Stage: /home/hxchen/runs/2026-10-31-tqg-h-only-controls/stage3_execution
/home/hxchen/anaconda3/envs/bfn4jet/bin/python h_only_experiment.py prepare --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-h-only-controls/stage3_execution --config configs/h_only_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python h_only_experiment.py d0 --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-h-only-controls/stage3_execution --config configs/h_only_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python h_only_experiment.py pilot --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-h-only-controls/stage3_execution --config configs/h_only_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python h_only_experiment.py run --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-10-31-tqg-h-only-controls/stage3_execution --config configs/h_only_diagnostics.json
mkdir -p ../figures
/home/hxchen/anaconda3/envs/bfn4jet/bin/python h_only_report.py --stage /home/hxchen/runs/2026-10-31-tqg-h-only-controls/stage3_execution
# Then run analysis_code/h_only_interpretation.py --stage ... once; it adds descriptive stability only.
