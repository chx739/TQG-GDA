# A0/A1/A2-a run report

Status: completed. Reused 5 initial source-fit GNNs (15 task references); no GNN updates or final labels. All four outer folds, three inner folds, H/HA, KRR/LR/MLP, true labels and 3 nulls completed. CPU FP64. See 结果解读.md and results_summary.json for gates, not just best means.

## Provenance and artifacts

Execution commit: `3755f8b2af40bf65407485d3684e2bb149f13a05`. Method SHA: `6f8e10a6bd0c26811a96cd5f69bd38c57a3a296940e85d1dbeb361fefc283149`. 2880 selected refits / 34560 total fits; 10080 fold metrics. Logical rows and nulls are not independent model seeds. Formal CPU seconds: 298.744.

## Deviations and implementation resolutions

No outcome-based change to task, seeds, families, grids or gates. Full-batch probe optimization, exact LR objective, null pairing, equal training sizes, support minimum counts and common-subset bootstrap were fixed in execution_protocol.md before B evaluation. Kernel bandwidth uses full source-query input geometry; no held-out labels. MLP batching validated against independent fitting. Historical B projection scalars reused to avoid redundant B gradient access.

## Reproduction

From frozen code/, with matching input/cache SHA and local references: assumptions_experiment.py prepare → a0 → pilot → run, all with --repo <source repository> --stage <stage directory>. Output directories must be fresh. Pin all library versions in experiment_manifest.json. Postprocess with assumptions_summarize.py --stage <stage>. Code and dataset caches have separate roles; dataset not published.

## Interpretation limits

Target-trained probes are diagnostic oracles, not deployable UDA methods. A measured gap does not uniquely establish conditional shift. No-detected-gap is not equivalence; insufficient support or power remains inconclusive. Existing development inputs/labels have been observed by earlier studies. A2-b/A3–A6 and external-dataset validation are not performed here.

## PROTEINS-specific protocol

Dataset is PROTEINS, not Mutagenicity. GIN node3/edge0, initial source-fit models selected within700 updates. Source250=fit199+query51, validation28. Four unequal input-ID outer folds train38/39 and audit13/12; target69A/70B, training38/39 exactly matched per fold. No graph-size natural domain claim. A0 creates NEW v2 geometry and independently checks math/production, not historical replay. B projection is computed only after sealed predictions using the already-permitted B labels. Source sample size/backbone/pretraining differ from Mutagenicity, so cross-dataset differences cannot be assigned to data size alone. Same 30-graph support minimum and original gates; inadequate strata are inconclusive. Binary BA is null for a single-class fold; NLL remains primary.

## Cross-dataset descriptive comparison

See cross_dataset_comparison.json. Preserve the preregistered within-domain learnability prerequisites and 30-per-stratum support cutoff. PROTEINS uses fewer labeled query examples and a different backbone/pretraining budget; no size-only causal claim. Correlations over15 shared-model task references are descriptive, not new evidence of conditional-gradient correctness. No candidate was refitted or selected using B results.
