# Mutagenicity execution

Use mutagenicity_experiment.py prepare, then commit a clean tree, run acceptance, then worker --slot 0 and worker --slot 1 concurrently. Source epochs and model follow SEGA source_development.json; TQG adaptation is 200 steps. After all 105 adaptations and 10 warmups finish, run mutagenicity_evaluate.py. It seals training before importing the development-label reader at execution. Final labels are never read. See configs/mutagenicity/sega_full_query.json; all outputs are outside code in the parent stage3_execution.
