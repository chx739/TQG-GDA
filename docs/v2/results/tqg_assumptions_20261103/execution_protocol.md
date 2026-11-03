# A0/A1/A2-a execution protocol — frozen before target-B evaluation

This package implements only the first batch in the 2026-11-03 assumption plan. A2-b and A3–A6 are not included. The original method.md and thresholds are preserved.

## Resolved implementation details

- CPU FP64, 5 model seeds, initial source-fit snapshot, historical target-A16 block0 V. No GNN training. Signatures for query196, A271 and B271 use that same fixed V.
- Source outer folds: SHA256("outer|graph_id"), ordered then strided into 4 equal folds. Target147 per fold: SHA256("target_train|domain|fold|graph_id"). Inner folds: SHA256("inner|fold|graph_id"), ordered then strided into 3 folds. Neither selection uses labels.
- H/HA RBF bandwidths use all196 source query inputs, fixed before folds and shared by SS/ST/TT. LR/MLP concatenate h and flattened A for HA, standardize using only the current inner training subset or outer147 for refit. No evaluation features determine those means or scales.
- Null0 is true labels; null1–3 permute the 147 training labels, preserving class proportion, paired across signature/probe families. Each null performs its own inner CV. No null changes the original backbone probabilities p.
- Binary residual KRR predicts contrast-coordinate (Y−p)Q, solves K+λI with no hidden division by n, then adds Q times predicted contrast residual to original p. Save raw probabilities, un-clipped residual MSE, clipping rate and clipped NLL/Brier.
- LR objective is mean binary CE + L2/2 times squared weights (intercept unpenalized), SciPy L-BFGS-B max500, ftol1e-12, gtol1e-8, maxls40. Nonconvergence remains flagged.
- MLP: full-batch Adam, 200 steps, lr0.001, hidden32/ReLU, float64, weight decay on all parameters including biases, no dropout. Equal initialization across penalties/label variants for a model/outer/inner fold. Batched models have independent losses/parameters/Adam moments; checked against individual fits. Exactly 4 selected refits per selection block; no unselected outer fits.
- A0 reconstructs the original target-V and source signatures; compares H/HA W, feedback and c_A with old results at atol/rtol1e-8. Readouts are sealed before replay diagnostic target-A labels. Historical target-B projection energy scalars are reused, not recomputed with new label access.
- Probe target-A labels are used only in diagnostic T→T training, never in GNN adaptation. Source/target evaluation predictions, support groups, all refitted models and inner CV logs are sealed before the B label request. B labels cannot affect optimization, selection, stopping or support thresholds. Final labels are rejected.
- Pilot: seed3405/domain1/outer folds0–2 =144 selection units (5%),1728 fits. No B labels or B scores in the pilot. Runtime extrapolation times1.3 must fit8 CPU hours. Formal workers are new, start with the same initialization, and exclude pilot results.
- Main NLL/BA/Brier are per-fold then per-model aggregates, never 4-fold pseudo-replication. At least4/5 seeds means positive paired difference, with >=0.02 required on the model mean. Same-domain learnability requires improvement over BOTH prior and 3-null mean. Transfer gap needs T→T learnability as specified in method.md.
- A2 support candidate needs every outer fold of every model to have >=30 B graphs in both near/far groups; otherwise unsupported due insufficient support strata, not failed transfer. Distances normalized by original bandwidths; ESS undefined on underflow is null.
- Local compatibility uses the common near subset (intersection across 4 folds AND 5 models) for each task/signature; needs >=30 graphs. Both S→T and T→T must beat prior/null on that subset. Average per-graph paired NLL difference across folds then seeds; graph-level paired bootstrap1000 draws, 90% percentile CI. If CI not wholly in[−0.02,0.02], do not claim compatibility. This is a descriptive diagnostic with shared graphs, not a formal independence/power guarantee.
- All model families/signatures/tasks reported, no best-on-B selection. Existing dev is historically observed, no claims of a new blind test.

## Budget and monitoring

Pilot and A0 are additional engineering costs, reported separately from formal fits. Formal CPU cap8 core-hours, 5 equal allocations (1.6 hours each), plus memory cap8GiB across workers (1.6GiB each). A0+pilot cap accounting is checked before formal launch; if formal allocation needs lowering it must be written before evaluation. Progress JSON and heartbeat every15seconds; all failures retained, max5 fixes for a root cause. No network training dependencies.

## Output and freeze

Keep inner_cv.jsonl, history.jsonl/TensorBoard, selected probe states, per-graph predictions, full input-ID roles, calibration replay, label access, code hashes and commits. Save reports and per-seed tables; differentiate engineering and scientific outcomes. Freeze code and this protocol before running prepare/A0/pilot/formal. Do not edit frozen code in place; a correction requires a new attempt/commit and disclosure.
