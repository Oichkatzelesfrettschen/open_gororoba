//! # Reproducible Original Source Manifest
//!
//! Purpose
//! - Every external source used by the workspace should be recoverable by a stable command.
//! - This document defines how we discover links, archive them, and reproduce each download.
//!
//! Primary Procedure
//! 1. Run the workspace source discovery script:
//! >  - `scripts/discover_original_sources.sh`
//! 2. Review the generated files in `logs/source_discovery_*`:
//! >  - `source_discovery_<TS>.md` (human-readable summary)
//! >  - `source_discovery_<TS>.tsv` (machine-friendly table)
//! >  - `source_discovery_<TS>.jsonl` (appendable event style)
//! 3. Validate that each required artifact has:
//! >  - repository URL
//! >  - commit or tag reference
//! >  - source retrieval command
//! >  - checksum command
//! 4. Before a major merge or release, re-run discovery and diff against prior manifest.
//!
//! Canonical repository origins for this workspace
//! - `lambda-research`: `https://github.com/Oichkatzelesfrettschen/lambda-research.git`
//! - `lambda-synthesis-experiments`: `https://github.com/Oichkatzelesfrettschen/lambda-synthesis-experiments.git`
//! - `LambdaLearner`: `https://github.com/Oichkatzelesfrettschen/LambdaLearner.git`
//!
//! Deterministic checkout pattern for repository inputs
//! - `git clone <origin_url> /tmp/<repo_name>`
//! - `git -C /tmp/<repo_name> fetch --all --prune`
//! - `git -C /tmp/<repo_name> checkout <commit_or_branch>`
//! - `mv /tmp/<repo_name> archive/intake_lane_retirement/<TS>/merge_in/<repo_name>`
//!
//! Canonical URL Rules
//! - Prefer official project repositories and release pages over mirrors.
//! - Prefer fixed references over mutable tags:
//! - fixed commits for reproducibility
//! - release tags only when the tag hash is also captured
//! - If a direct artifact URL is mutable (for example `latest`), replace with a specific version URL.
//!
//! Reproducible Download Commands
//! - Git repositories:
//! - `git clone <origin> <local_dir>`
//! - `git -C <local_dir> checkout <commit>`
//! - Python package metadata (requirements files):
//! - `python3 -m pip download -r <requirements-file> --dest downloads/`
//! - Node package metadata:
//! - `npm install`
//! - Optional per-package tarball:
//! >   - `npm view <package>@<version> dist.tarball`
//! - Large archives:
//! - `curl -L -o <file> <artifact_url>`
//! - `sha256sum <file>`
//!
//! Retention and Audit
//! - Keep at least 4 recent source manifests.
//! - Keep every manifest tied to a pull, backup, or release event.
//! - Link manifest paths in task logs and roadmap notes.
//!
//! Quality expectation
//! - Missing or unresolved source URLs are tracked as blockers.
//! - Checksum command should be explicit before approving any artifact claim.
//!
