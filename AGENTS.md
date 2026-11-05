# Project working agreement

- After completing code changes or experiments, update the relevant documentation and results, run appropriate checks, then commit and push to this project's GitHub origin. The user has explicitly authorized this as the default workflow; do not ask for confirmation again for routine updates.
- Include experiment histories and bounded model archives with provenance, checksums, and evaluation-to-checkpoint links. Keep datasets, credentials, temporary files, and incomplete runs out of Git. Preserve existing experiment artifacts instead of rewriting them.
- Push the current intended branch without force. Verify the remote commit and report the push status and commit to the user. If synchronization fails, state what remains local.
- Distinguish engineering checks from research evidence. Preserve source/query/validation roles and target-label access boundaries; do not use final target labels for tuning or routine verification.
- Prefer GitHub URLs for user-facing input files, published data and reports when they exist in this repository; use local paths only for unpublished artifacts. Resolve report symlinks to the actual file for GitHub links.
- Exception to automatic publication: `docs/v2/results/assumptions_brief_20261105.md` is a local-only meeting draft at the user's request. Keep it untracked and ignored; do not force-add or republish it without explicit user authorization.
- Put experiment orchestration and reporting in `workflows/v1/` or the relevant `workflows/v2/<phase>/` package; do not add experiment scripts to the repository root. Use package-qualified imports and `python -m` entry points. Keep reusable algorithms in `tqg_gda/` and cross-experiment CLI/archive utilities in `scripts/`.
- When moving live code, update its imports, repository-root resolution, subprocess entry points, code-screen/publication paths, tests and navigation. Leave historical experiment snapshots and frozen protocols byte-for-byte intact.
