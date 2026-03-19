//! # Python Multithreading Policy
//!
//! ## Rule
//! All Python scripts in this workspace must be multithreaded or parallelized to avoid avoidable single-thread bottlenecks.
//!
//! Accepted compliance signals:
//! - `threading`
//! - `concurrent.futures` with `ThreadPoolExecutor` or `ProcessPoolExecutor`
//! - `multiprocessing`
//!
//! ## Enforcement
//! - Verification script: `scripts/verify_python_multithreading_policy.sh`
//! - Exemptions registry: `docs/PYTHON_THREADING_EXEMPTIONS.tsv`
//! - Latest verification table: `docs/python_multithreading_policy_latest.tsv`
//! - Verification artifacts are written to `logs/python_multithreading_policy_<TS>.{md,tsv}`.
//!
//! ## Exemption Rules
//! A file may be exempt only when it has an explicit row in `docs/PYTHON_THREADING_EXEMPTIONS.tsv` with:
//! - `path` (workspace-relative)
//! - `reason`
//! - `status` = `active`
//! - `owner`
//! - `review_date`
//!
//! Typical temporary exemptions:
//! - package marker files (`__init__.py`)
//! - test-only scaffolding
//! - legacy IO-bound scripts pending threaded refactor
//!
//! ## Change Management
//! - New Python files should be compliant by default.
//! - If a new exemption is required, add it in `docs/PYTHON_THREADING_EXEMPTIONS.tsv` with a clear reason and review date.
//! - Exemptions should be reduced over time as scripts are upgraded.
//!
//! ## Current Baseline
//! - Latest policy report: `logs/python_multithreading_policy_20260213_163120.md`
//! - Current counts: `11` compliant, `40` exempt, `0` non-compliant.
//!
