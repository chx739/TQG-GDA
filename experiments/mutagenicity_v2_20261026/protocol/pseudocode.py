# For each seed/domain/arm: restore fixed source model; run 200 paired-schedule SGD steps.
# Each v2 step: full target Jacobian -> V -> complete source/target signatures -> kernel -> W -> d=W r_S -> CE+parameter feedback.
# Seal every trained checkpoint; only then evaluate frozen development IDs, aggregate and report.
