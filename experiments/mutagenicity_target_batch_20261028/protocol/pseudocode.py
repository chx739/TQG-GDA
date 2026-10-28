# Frozen numeric method: source fit CE plus stopped TQG v2 query loss.
# Reuse audited batch16/B_fit; for five seeds and three target domains:
#   from the same source checkpoint, run batch32 then batch64 for 200 SGD steps.
#   freeze all completions before evaluating fixed development labels.
