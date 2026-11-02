#!/usr/bin/env bash
# Original cwd: /home/hxchen/runs/2026-11-02-tqg-label-query-validation/stage3_execution/code
# Requires a NEW initialized stage, source dataset, archived checkpoints and H geometry cache.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/hxchen/anaconda3/envs/bfn4jet/bin/python label_query_experiment.py prepare --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-11-02-tqg-label-query-validation/stage3_execution --config configs/label_query_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python label_query_experiment.py d0 --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-11-02-tqg-label-query-validation/stage3_execution --config configs/label_query_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python label_query_experiment.py pilot --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-11-02-tqg-label-query-validation/stage3_execution --config configs/label_query_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python label_query_experiment.py run --repo /home/hxchen/TQG-GDA --stage /home/hxchen/runs/2026-11-02-tqg-label-query-validation/stage3_execution --config configs/label_query_diagnostics.json
/home/hxchen/anaconda3/envs/bfn4jet/bin/python label_query_summarize.py --stage /home/hxchen/runs/2026-11-02-tqg-label-query-validation/stage3_execution
# Then analysis_code/label_query_interpretation.py --stage ... once; it adds descriptions and input-only spectra.
