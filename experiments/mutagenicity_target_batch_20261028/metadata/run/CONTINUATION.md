# COMPLETE — target-batch ablation

Training and evaluation finished. Do not restart training.

30 new batch32/64 adaptations,30 reused batch16/source-SGD records,5 shared fit warmups,135 evaluation rows. All three target domains and seeds3405–3409 complete. Main criterion: both batch32 and64 support only1/3tasks, below preset2/3. Larger batches lower feedback norm CV but do not yield a stable cross-task classification gain.

Execution commit587072a4371e16da22cf5b6e9fc4db8a02dd90ab; plan SHA96ff0ddd2ea3b4b8ea554fada43e4e72648e208ba9aab33985a56a5d96236e9f. No numeric-core changes, no failures/restarts, no query nonconvergence or numeric fallback. Independent audit passed, source/query sequences and target prefixes verified. Final target labels not read.

Run root helpers post_run_audit.py and analyze_results.py provide independent artifact verification and descriptive postprocessing; neither trains models nor reads graph labels. JSONL and TensorBoard remain in stage3_execution/jobs. Published artifact paths and synchronization status are recorded by the main TQG-GDA repository history.
