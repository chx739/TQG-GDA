# Project working agreement

- The user explicitly authorized renaming `main` to `SEGA-ref`, creating `TSBA`, implementing the supplied TSBA design, and an independent review by a subagent with no inherited conversation. `SEGA-ref` preserves baseline `ab0859065ebcdd4cfd72dbce13ed2d995d8f4587`; implementation belongs on `TSBA`.
- Follow `docs/tsba/TSBA_Graph_Level_GDA_Design.md` and the numerical choices in `docs/tsba/implementation.md`. Retain SEGA's modular organization and proposal/nonlinear-check/accept-or-backtrack workflow. Do not silently add objectives, pseudo-label tasks, hard source-risk constraints, or encoder adaptation.
- Legacy TQG is archived on `TQG`. On 2026-11-14 the user authorized removing historical `experiments/**/*.npz`. Preserve remaining historical evidence; do not rewrite history without new explicit authorization.
- Keep the files under `references/sega-gda/` byte-for-byte identical to their source entries in `SEGA-GDA(1).zip`. Their provenance and checksums are in `references/sega-gda.provenance.json`. These are reference materials, not a newly implemented TQG method. Record limitations in project documentation rather than modifying this snapshot.
- Do not add raw datasets, credentials, temporary outputs, model runs or the source ZIP to Git. The attachment's raw dataset files are intentionally omitted from the reference import.
- Do not run the reference package's real-data pipeline as a routine check: it reads whole-target labels. Preserve the historical development/final boundaries; final target labels must not be used for tuning or routine verification. Engineering checks and research evidence must remain distinct.
- The meeting draft `docs/v2/results/assumptions_brief_20261105.md` remains local-only, untracked and ignored. Do not publish it without explicit user authorization.
- After authorized changes, update relevant documentation, run appropriate checks, commit and synchronize to the project's GitHub origin without force. Verify the remote commit and report any local-only work or publication block. Do not bypass approval or access-control rejections.
- Prefer GitHub URLs for published code, data and reports; use local paths for unpublished artifacts. Resolve report symlinks before publishing links.
- Reusable TSBA code belongs in `tsba/`, workflows in `workflows/tsba/`, and tests in `tests/tsba/`. The implementation request authorizes synthetic engineering checks; real-data research runs require a specified data protocol.

- On 2026-11-09 the user explicitly authorized updating and publishing the review navigation as `docs/tsba/code-review-guide.md`. Keep that published guide aligned with code and tests. The old `docs/tsba/review-guide.md` name remains ignored as a legacy local copy; do not republish under the old name.

- Keep method/usage documentation in `docs/`, and each TSBA experiment evidence bundle in `experiments/<dataset>/<round-date>/`. Preserve historical JSON/CSV/log bytes when reorganizing; record actual execution separately from planned configuration.
