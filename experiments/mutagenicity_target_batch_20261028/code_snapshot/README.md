# Frozen target-batch ablation executor

Template: from-scratch, reusing the verified PyG TQG v2 loop. HF/image trainers cannot express full-parameter target queries and this custom stopped parameter feedback. No numeric core changes.

New jobs: batch32/64 × domains1/2/3 × seeds3405–3409 = 30. Reference batch16 and source SGD are reused only after config/data/source-checkpoint/code hashes match. Five fit warmups are reused.

From this directory, in the pinned PyTorch/PyG environment:

```bash
python experiment.py prepare
python -u launch.py
```

This snapshot expects `../../stage2_method/experiment_plan.yaml` and data_roles, with the matching plan copy under configs. Data cache and archived references must exist at paths specified by the plan. Local TensorBoard fallback is documented in experiment.py; no training-time network access.

FP64, one CPU thread per seeded worker, no mixed precision. Source and target roles/RNG are isolated. Same target permutations with different prefix lengths make batch-size comparisons paired. Original commits are retained for reused jobs; evaluation_commit denotes this snapshot. Final labels are prohibited; development labels are read only after every job is sealed.

All outputs go to the parent stage directory. Every metric is written to JSONL and local TensorBoard. Checkpoints retain model, optimizer and sampling RNG. Loss nondecrease only warns because adaptation is stochastic and nonstationary. Numeric errors stop; up to five recovery attempts, without changing batch size or any scientific hyperparameter.
