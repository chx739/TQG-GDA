# Recorded execution; use a NEW run layout, matching cache SHA and data_roles manifest.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Frozen commit 3755f8b2af40bf65407485d3684e2bb149f13a05; original Python /home/hxchen/anaconda3/envs/bfn4jet/bin/python
# assumptions_experiment.py <prepare|a0|pilot|run> --repo <repo> --stage <stage> --config configs/assumptions_proteins.json
# assumptions_regression.py --repo <repo> --output <stage>/D0/mutagenicity_regression.json
# assumptions_summarize.py --stage <stage>
# assumptions_proteins_interpretation.py --stage <stage> --repo <repo>
# scripts/archive_assumptions.py --stage <stage> --repo <repo> --dest <fresh archive>
