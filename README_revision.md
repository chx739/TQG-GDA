# TQG revision-v1 execution

This directory is a local clone of the frozen original pilot commit. Original
code/evidence remain in the sibling `stage3_execution`; new outputs are under
`stage3_revision_v1`. The method contract is the immutable `stage2_revision_v1`.

Use `/home/hxchen/anaconda3/envs/bfn4jet/bin/python` with
`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1`. CPU, one thread, float64 are fixed.
The original runtime dependency directory supplies TensorBoard; no network
installation or external logging service is used.

After inspecting and committing the complete new code:

```bash
python setup_revision.py
python run_revision_pipeline.py
```

The pipeline runs D0, profiles and resumes the predetermined D1 blocks, constructs
all stopped D1/D2 predictions, and only then starts the separate development-label
evaluators. It trains all 80 A jobs, and all 120 B jobs only if the locked D2 gate
passes. The T5/T6 timing profiles are the first ten updates of their real jobs;
they resume with the exact saved Adam/model/random states.

Training handlers: `training_revision.py`, `experiment_revision.py`.
Source-only D1: `diagnostics_sample_size.py`.
D2 construction: `diagnostics_local.py --phase source`.
Offline evaluators: `evaluate_sample_size.py`, `diagnostics_local.py --phase target`,
`summarize_revision.py`. These evaluators never expose final labels.

The pipeline and individual stages refuse changed source checkpoints, changed
data roles, dirty code, unsealed predictions, and incompatible recovery artifacts.
Progress is visible in `logs/`, `live_process.json`, and `command_log.jsonl`.
Do not infer a live process from a JSON record alone; inspect its actual PID.

The ledger counts the summed measured elapsed time of serial compute stages,
including diagnostics/profiles/evaluation. Proposed scope was explicitly requested
by the user; the locked cap is four CPU elapsed hours, zero GPU hours. The
80-percent review threshold stops execution before launching further work.

This is an exploratory local package, not independent dataset confirmation.
Dataset-size causality, recent external baselines, GCN robustness, and paper-stage
results are explicitly contingent future work in the source design.
