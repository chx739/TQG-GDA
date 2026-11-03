# A0/A1/A2-a run report

Status: completed. Reused 5 initial source-fit GNNs (15 task references); no GNN updates or final labels. All four outer folds, three inner folds, H/HA, KRR/LR/MLP, true labels and 3 nulls completed. CPU FP64. See 结果解读.md and results_summary.json for gates, not just best means.

## Provenance and artifacts

Execution commit: `7222b6589345af1d992feb230e1fc9dab1faed3d`. Method SHA: `ac3cf818b583844ad5e28cae98b2b8f4d6362a97c38a8e0f7280ca4c6683fdf9`. 2880 selected refits / 34560 total fits; 10080 fold metrics. Logical rows and nulls are not independent model seeds. Formal CPU seconds: 349.159.

## Deviations and implementation resolutions

No outcome-based change to task, seeds, families, grids or gates. Full-batch probe optimization, exact LR objective, null pairing, equal training sizes, support minimum counts and common-subset bootstrap were fixed in execution_protocol.md before B evaluation. Kernel bandwidth uses full source-query input geometry; no held-out labels. MLP batching validated against independent fitting. Historical B projection scalars reused to avoid redundant B gradient access.

## Reproduction

From frozen code/, with matching input/cache SHA and local references: assumptions_experiment.py prepare → a0 → pilot → run, all with --repo <source repository> --stage <stage directory>. Output directories must be fresh. Pin all library versions in experiment_manifest.json. Postprocess with assumptions_summarize.py --stage <stage>. Code and dataset caches have separate roles; dataset not published.

## Interpretation limits

Target-trained probes are diagnostic oracles, not deployable UDA methods. A measured gap does not uniquely establish conditional shift. No-detected-gap is not equivalence; insufficient support or power remains inconclusive. Existing development inputs/labels have been observed by earlier studies. A2-b/A3–A6 and external-dataset validation are not performed here.

## Post-result interpretation (no refits)

Same-signature LR/MLP learn within each domain, but every task has a source-to-target gap. H support strata locate larger far-region gaps on all tasks. This motivates separately registered local source-bank and kernel-complexity studies; it does not uniquely identify conditional shift, prove task sufficiency, or validate gradient feedback. HA+LR can improve predictive NLL even though the original HA residual KRR and prior gradient controls are weak. One H/LR task passes only the local descriptive equivalence margin. See 结果解读.md for exact qualifications.

One inner LR null fit has a finite ABNORMAL termination, retained. Original raw KRR probabilities and clipping rates are reported; high clipped NLL is not all assigned to transfer mismatch. All post-result correlation and interpretation code is archived separately from frozen execution code, with no model fitting or B-based selection.
