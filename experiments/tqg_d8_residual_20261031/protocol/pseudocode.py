# TQG v2 diagnostic plan — NON-RUNNABLE PSEUDOCODE, NOT A TRAINING ENTRY POINT.
# See experiment_plan.yaml and method.md for exact roles and seeds.
#
# freeze(plan_sha, execution_commit, data_roles, checkpoint_hashes)
# run_D0_engineering_and_permission_checks(); halt_on_failure()
# for task in [0_to_1, 0_to_2, 0_to_3]:
#   A_pool, B_pool = fixed_input_id_hash_split(original_development_ids)
#   assert no_ids_overlap_final(A_pool, B_pool)
#   for model_seed in [3405, 3406, 3407, 3408, 3409]:
#     for t in [0, 50, 200]:
#       theta = source_fit_best_if_zero_else_paired_source_SGD_checkpoint(t)
#       assert source_role_and_checkpoint_provenance_match()
#       baseline_B_prediction = unlabeled_forward(theta, B_pool)
#       for block_id in range(8):
#         fit_ids, query_ids, A_ids = shared_predeclared_ids(task, seed, t, block_id)
#         # NO target labels may enter this branch.
#         V, responses, signatures, kernels, W = stopped_unlabeled_query(theta, inputs)
#         seal_unlabeled_query(theta_hash, IDs, W_hash, config)
#         r_source = stack_graph_major(Q_C.T @ (Y_i - p_i) for i in query_ids)
#         g_true = -V @ W @ r_source
#         g_perm = three_label_permutations_same_theta_V_and_W()
#         g_rand = three_norm_matched_random_vectors_in_same_V_space()
#         # Only now explicitly enter isolated DEVELOPMENT-ORACLE scope.
#         g_A = direct_autograd_CE(theta, read_allowed_dev_labels(A_ids))
#         g_B = weighted_full_pool_gradient(theta, read_allowed_dev_labels(B_pool))
#         save_D1_raw_metrics_and_zero_cases()
#         if block_id < 4:
#           g_source = source_fit_CE_gradient(theta, fit_ids)
#           for eta in [0.0001, 0.0003, 0.001]:
#             for arm in predeclared_D2_arms:
#               temporary_theta = clone_then_one_plain_SGD_step(theta, arm, eta)
#               evaluate_development_A_B_only(temporary_theta)
#               discard_temporary_theta(); assert_original_theta_unchanged()
# aggregate_nulls_then_blocks_then_snapshots_within_each_model_seed()
# summarize_mean_sample_SD_and_paired_differences_over_five_seeds()
# apply_frozen_criteria(); choose_permitted_next_branch(); keep_all_failures()
# D3-D8 are separate amendments/packages, not silently nested here.
# D9 normal adaptation receives no development gradients or Oracle checkpoints.
# No final label reader exists anywhere in this plan.
