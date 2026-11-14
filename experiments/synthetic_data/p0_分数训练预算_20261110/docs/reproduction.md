# P0 immutable evidence

Execution commit: 4a34240. Analysis commit: 7fcb41c. Source tar files are git archives of those commits.
Python: /home/hxchen/anaconda3/envs/bfn4jet/bin/python
Environment: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; CPU float64; one Torch thread; execution deterministic algorithms enabled.

Commands (run in a clean checkout of the respective commit; use new exclusive output paths):

```
python -m unittest discover -s tests/tsba -v
python -m experiments.synthetic_data.p0_分数训练预算_20261110.scripts.run_p0 --parent /home/hxchen/experiments/tsba/20261110-static-score/run-001 --original /home/hxchen/experiments/tsba/20261109-mechanisms/run-001 --output NEW_RUN
python -m experiments.synthetic_data.p0_分数训练预算_20261110.scripts.summarize_p0 --run NEW_RUN --output NEW_ANALYSIS
```

`tests.log` preserves the first engineering test attempt: GH256/512 curvature discrepancy exceeded 1e-9, and Torch softplus's default linear threshold caused a derivative-check discrepancy. Before freezing or running experiments, quadrature was increased to1024/cross-check512 and the independent autodiff check changed to exact logaddexp. `tests-fixed.log`: all59 tests pass. No experiment reruns or discarded training cells. Analysis completed once; its console log is empty on success; analysis-001/audit.json is the structured audit.

The prior static/original archives are dependencies, verified before and after execution using published manifests. Their data are not duplicated in full here. Fixed P0 inputs, 6400-step schedules, validation noise, all15 endpoint model/Adam checkpoints, secondary states, per-query predictions, scalar oracle inputs and full trajectories are included. Zero learned-score classifier updates. The scalar density-only trajectory is a known-distribution diagnostic, not a learned-score loop or a TSBA performance bound.
