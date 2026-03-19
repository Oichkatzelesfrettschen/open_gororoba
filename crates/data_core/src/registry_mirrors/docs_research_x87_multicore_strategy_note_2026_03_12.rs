//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/research_narratives.toml -->
//!
//! # x87 Multicore Strategy Note (2026-03-12)
//!
//! This note summarizes the refreshed repo-local worker sweep of the pinned-core
//! `x87-strategy-bench` lane.
//!
//! Artifacts:
//!
//! - `reports/benchmarks/x87_strategy_multicore_65536.csv`
//! - `reports/benchmarks/x87_strategy_multicore_65536.md`
//!
//! Command:
//!
//! ```texttext
//! make x87-strategy-bench \
//!   LEN=65536 \
//!   REPEATS=5 \
//!   WORKER_COUNTS=1,2,4,6 \
//!   OUT=reports/benchmarks/x87_strategy_multicore_65536.csv \
//!   SUMMARY=reports/benchmarks/x87_strategy_multicore_65536.md
//! ```texttext
//!
//! The generated CSV and Markdown artifacts now embed the local run context
//! (host, CPU model, repeat count, RNG seed, detected physical workers, worker
//! sweep, and stability heuristic) so the benchmark evidence is self-describing.
//!
//! Observed worker sweep:
//!
//! - Detected one heavy worker per physical core: 6 workers total.
//! - Swept worker counts: 1, 2, 4, 6.
//!
//! What the refreshed worker sweep says:
//!
//! 1. Serial AVX2 is still the raw-throughput winner for the easy positive-sum
//!    case at this small size.
//! 2. Chunking a cancellation-heavy sum across cores changes the answer even when
//!    each chunk uses x87. This confirms that "multicore x87" is not the same
//!    thing as preserving one global FP80 reduction order.
//! 3. The hybrid lane (`avx2_per_chunk_x87_final`) is a useful system-level
//!    option, but it is still a different numerical contract from full-stream x87.
//! 4. For the random-dot workload, the fastest exact parallel lane stayed
//!    `x87_per_chunk` at 2 workers.
//! 5. For the synthetic ill-conditioned dot used here, the fastest exact parallel
//!    lane stayed `avx2_per_chunk` at 2 workers. Treat that as a
//!    workload-specific observation, not a universal guarantee.
//! 6. For the easy positive-sum workload, the fastest exact parallel lane was
//!    `avx2_per_chunk_x87_final` at 2 workers.
//! 7. The repo-local worker sweep still showed scheduling noise at higher
//!    worker counts on this machine state. In this run, `x87_per_chunk` at 6
//!    workers was unstable on `sum_positive` and `dot_random`, so the summary
//!    marks those rows explicitly instead of treating them as decision-grade
//!    winners.
//!
//! Current engineering guidance:
//!
//! - If the requirement is true FP80 semantics for one reduction stream, keep that
//!   stream x87 and do not split it across cores unless order changes are allowed.
//! - If the requirement is "better than plain f64 at high throughput", use the
//!   AVX2 or hybrid lanes and benchmark on the real workload.
//! - Use multicore x87 primarily for independent FP80 tasks, not for trying to
//!   make one shared reduction stream both parallel and semantically identical.
//!
