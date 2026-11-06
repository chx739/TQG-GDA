# Project working agreement

- `main` is the new design-preparation line based on the user-supplied SEGA-GDA package. The user has requested no concrete algorithm changes and no TQG implementation at this stage. Wait for an explicit subsequent implementation request before adding either.
- The previous TQG implementation and all its experimental evidence are preserved on `history/tqg-v2-20261106` at `eb39fd7c5433420bd51f14e6330ba5067d164456`. Do not rewrite its commits or experiment artifacts.
- Keep the files under `references/sega-gda/` byte-for-byte identical to their source entries in `SEGA-GDA(1).zip`. Their provenance and checksums are in `references/sega-gda.provenance.json`. These are reference materials, not a newly implemented TQG method. Record limitations in project documentation rather than modifying this snapshot.
- Do not add raw datasets, credentials, temporary outputs, model runs or the source ZIP to Git. The attachment's raw dataset files are intentionally omitted from the reference import.
- Do not run the reference package's real-data pipeline as a routine check: it reads whole-target labels. Preserve the historical development/final boundaries; final target labels must not be used for tuning or routine verification. Engineering checks and research evidence must remain distinct.
- The meeting draft `docs/v2/results/assumptions_brief_20261105.md` remains local-only, untracked and ignored. Do not publish it without explicit user authorization.
- After authorized changes, update relevant documentation, run appropriate checks, commit and synchronize to the project's GitHub origin without force. Verify the remote commit and report any local-only work or publication block. Do not bypass approval or access-control rejections.
- Prefer GitHub URLs for published code, data and reports; use local paths for unpublished artifacts. Resolve report symlinks before publishing links.
- When future implementation is authorized, keep reusable algorithms in `tqg_gda/`, experiment workflows in phase packages under `workflows/`, and shared CLI/archive utilities in `scripts/`. This agreement does not authorize creating those implementations now.
