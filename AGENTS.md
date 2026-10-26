# Project working agreement

- After completing code changes or experiments, update the relevant documentation and results, run appropriate checks, then commit and push to this project's GitHub origin. The user has explicitly authorized this as the default workflow; do not ask for confirmation again for routine updates.
- Include experiment histories and bounded model archives with provenance, checksums, and evaluation-to-checkpoint links. Keep datasets, credentials, temporary files, and incomplete runs out of Git. Preserve existing experiment artifacts instead of rewriting them.
- Push the current intended branch without force. Verify the remote commit and report the push status and commit to the user. If synchronization fails, state what remains local.
- Distinguish engineering checks from research evidence. Preserve source/query/validation roles and target-label access boundaries; do not use final target labels for tuning or routine verification.
