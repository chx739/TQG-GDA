# A4-a execution report

A4-a is complete; the bounded research goal remains active. A2-b/A6 are separate pending phases. No backbone was trained or updated.

Three Mutagenicity tasks, five source-fit model seeds, four outer folds, three inner folds, training sizes32/64/128/147, H/HA, source/target training, true plus three independently selected shuffled-label readers. Bandwidth multiplies s, not s². Both fixed and sample-normalized ridge penalties were retained.

Counts: 960 fitting units, 276480 inner fits, 92160 outer refits, 368640 total logical fits. Shared matrix factorizations reduce physical computation; these are not GNN fits. The formal execution commit is 588fa47640e2aeee5a05e9b45dcf533a267d8e1a, source commit42c7404. Protocol SHA51a5161eab4bfc95bcd3d6ba9c7b583268aa5dbf5938c1e8d6227f74e5b908a8.

Independent verification replays all CV selections, all stored outer predictions, normal equations and230400 curve rows. All15 prediction/model/CV-prefix seals precede B label evaluation. Final label reads0. Source/target inner labels are kept separate; target-trained probes are diagnostic only.

There is1 finite JOINT observer not reaching its preset solver tolerance out of7680; retained unchanged. This is an observer of selected configurations, not a silently substituted fitted prediction.

The first report writer encountered a NumPy integer JSON serialization error after the independent audit finished. Explicit integer conversion repaired report export only; no fits, predictions, hyperparameters or labels were rerun for recovery. The error log and recovery record are retained.

The results do not establish successful gradient feedback. Source readability can improve while cross-domain prediction remains weak. Next: execute the frozen local-bank matrix and the predefined LOCAL/KERNEL feedback gates, with source-only LR teachers and label-attribution nulls.

Raw reports, parameters and predictions are preserved per worker. The complete source feature inputs are referenced to the earlier tqg_assumptions_20261103 archive; raw graph data and existing GNN checkpoints are not duplicated.
