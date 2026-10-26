# TQG v2 Mutagenicity execution snapshot

From-scratch template: reuse the verified graph-classification framework because target Jacobian calibration and single-step parameter feedback require a custom loop. HF and image templates do not match PyG graph inputs or this update rule. The numerical core is copied bytewise from the working repository. No network access is needed.

Source pretraining is reused, never repeated. See ../experiment_manifest.json and ../../stage2_method/experiment_plan.yaml for data roles, checkpoints, seeds, budgets and frozen settings. All output goes outside code, under this run.
