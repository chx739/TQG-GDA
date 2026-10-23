# Approved PROTEINS TQG pilot

The pilot config is `configs/pilot/controller.json`; the reviewed scientific plan is
`../../stage2_method/experiment_plan.yaml`. All newly generated evidence lives one
level above this code repository; legacy code and artifacts are never modified.

Use the pinned Python environment and set `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1`. The controller appends `../runtime_deps` to its Python
search path after the original environment packages for isolated TensorBoard dependencies. Commit the code first. Every controller phase requires a clean Git tree.

```
python experiment.py prepare
python experiment.py train
# Run source-only diagnostics and create ../source_evidence.json before evaluation.
python experiment.py evaluate
```

`train` runs exactly 10 source warmups and 60 adaptations. The 5 fit-only warmups
and 5 full-source warmups are reused across both target domains. Source-only,
target/random/source-probe and confidence pseudo-label arms have paired model
seeds and isolated, identical source-fit sampling streams. Adaptations start with
fresh Adam state. Full-source CE uses all 250 source training labels. The primary
checkpoint is the fixed step200 model; source-validation selection includes step0
and is stored separately. No training function calls a target label evaluator.

Jobs have frozen input fingerprints, complete configs, data and code hashes,
Git commit, CPU hardware, stdout/stderr, live TensorBoard scalar streams, source
history, checkpoints and peak process memory. Failed jobs remain in place and
are never overwritten or automatically reused. Successful jobs are reused only
if all input identities and registered output hashes still match. A failed job
requires explicit inspection and a new attempt directory/commit rather than a
silent overwrite. No automatic change to batch size or method is permitted.

Approval must be present in `../approval.json` with `approved:true`,
`cpu_wall_hours:2`, `gpu_hours:0`. The cumulative wall deadline begins when the
first new job is launched and survives controller restarts. A timeout kills the
active subprocess; all existing outputs remain. Warnings occur at50%; at80% the
next operation requires explicit continuation approval. No GPU is used. Diagnostic
and evaluation controllers must share this compute ledger/deadline.

`training_manifest.json` describes 70 jobs and 60 diagnostic records. `seal`
rechecks every output before creating `training_seal.json`. Development target
labels can be scored only after all70 training jobs and source evidence are
sealed. `evaluate` emits130 rows:120 adaptation checkpoint rows plus10 frozen
source rows. Each task has139 development graphs. Final target labels are excluded
from training, selection and evaluation by protocol; the raw cache physically
contains those labels. ECE uses10 equal-width confidence bins; balanced accuracy
is the primary endpoint. These rows are exploratory pilot results, not a
substitute for independent datasets, recent-method reproduction or a paper-ready
novelty argument.
