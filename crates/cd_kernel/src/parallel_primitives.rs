//! Pinned multicore reduction helpers for x87 and AVX2 strategies.
//!
//! The key policy is "one heavy numeric worker per physical core". These
//! helpers deliberately avoid SMT oversubscription and keep the semantic split
//! explicit:
//!
//! - `X87PerChunk`: strict "chunk-local x87 oracle" work, then x87 final sum.
//! - `Avx2PerChunk`: throughput-oriented AVX2/FMA chunk work, AVX2 final sum.
//! - `Avx2PerChunkX87Final`: SIMD broad phase, scalar x87 cleanup reduction.

use std::thread;

use crate::{avx2_dot, avx2_sum, x87_dot, x87_sum};

/// Multicore reduction policy for accumulation-heavy kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelReductionStrategy {
    /// Each worker computes its chunk entirely with x87, then the partials are
    /// reduced with x87 as well. This best matches "true FP80 per chunk".
    X87PerChunk,
    /// Each worker computes its chunk with AVX2/FMA, then the partials are
    /// reduced with the AVX2/f64 path.
    Avx2PerChunk,
    /// Each worker computes its chunk with AVX2/FMA, but the cross-chunk merge
    /// is performed with x87. This is a hybrid "broad phase + cleanup" mode.
    Avx2PerChunkX87Final,
}

impl ParallelReductionStrategy {
    /// Stable label for reports and CLI output.
    pub const fn label(self) -> &'static str {
        match self {
            Self::X87PerChunk => "x87_per_chunk",
            Self::Avx2PerChunk => "avx2_per_chunk",
            Self::Avx2PerChunkX87Final => "avx2_per_chunk_x87_final",
        }
    }
}

/// Chosen physical-core placement plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalCorePlan {
    core_ids: Vec<usize>,
}

impl PhysicalCorePlan {
    /// Build a pinned worker plan using one logical CPU per physical core.
    pub fn pinned(worker_limit: Option<usize>) -> Self {
        let mut core_ids = physical_core_ids();
        if let Some(limit) = worker_limit {
            core_ids.truncate(limit.max(1).min(core_ids.len()));
        }
        if core_ids.is_empty() {
            core_ids.push(0);
        }
        Self { core_ids }
    }

    /// Chosen logical CPU ids, one per physical core.
    pub fn core_ids(&self) -> &[usize] {
        &self.core_ids
    }

    /// Number of workers represented by the plan.
    pub fn worker_count(&self) -> usize {
        self.core_ids.len()
    }
}

/// Detect one logical CPU id per physical core via centralized oracle.
pub fn physical_core_ids() -> Vec<usize> {
    ::verified_core::topology::HardwareTopology::current()
        .physical_core_ids
        .clone()
}

/// Parallel sum with pinned workers and deterministic cross-chunk reduction.
pub fn parallel_sum(
    a: &[f64],
    strategy: ParallelReductionStrategy,
    plan: &PhysicalCorePlan,
) -> f64 {
    if a.is_empty() {
        return 0.0;
    }

    let workers = plan.worker_count().min(a.len());
    let bounds = split_bounds(a.len(), workers);
    let partials = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(bounds.len());
        for (worker_idx, (start, end)) in bounds.iter().copied().enumerate() {
            let core_id = plan.core_ids[worker_idx];
            let chunk = &a[start..end];
            handles.push(scope.spawn(move || {
                pin_current_thread(core_id);
                match strategy {
                    ParallelReductionStrategy::X87PerChunk => x87_sum(chunk),
                    ParallelReductionStrategy::Avx2PerChunk
                    | ParallelReductionStrategy::Avx2PerChunkX87Final => avx2_sum(chunk),
                }
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().expect("parallel_sum worker panicked"))
            .collect::<Vec<_>>()
    });

    finalize_partials(strategy, &partials)
}

/// Parallel dot product with pinned workers and deterministic cross-chunk reduction.
pub fn parallel_dot(
    a: &[f64],
    b: &[f64],
    strategy: ParallelReductionStrategy,
    plan: &PhysicalCorePlan,
) -> f64 {
    assert_eq!(a.len(), b.len(), "parallel_dot: slice length mismatch");
    if a.is_empty() {
        return 0.0;
    }

    let workers = plan.worker_count().min(a.len());
    let bounds = split_bounds(a.len(), workers);
    let partials = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(bounds.len());
        for (worker_idx, (start, end)) in bounds.iter().copied().enumerate() {
            let core_id = plan.core_ids[worker_idx];
            let a_chunk = &a[start..end];
            let b_chunk = &b[start..end];
            handles.push(scope.spawn(move || {
                pin_current_thread(core_id);
                match strategy {
                    ParallelReductionStrategy::X87PerChunk => x87_dot(a_chunk, b_chunk),
                    ParallelReductionStrategy::Avx2PerChunk
                    | ParallelReductionStrategy::Avx2PerChunkX87Final => avx2_dot(a_chunk, b_chunk),
                }
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().expect("parallel_dot worker panicked"))
            .collect::<Vec<_>>()
    });

    finalize_partials(strategy, &partials)
}

fn finalize_partials(strategy: ParallelReductionStrategy, partials: &[f64]) -> f64 {
    match strategy {
        ParallelReductionStrategy::X87PerChunk
        | ParallelReductionStrategy::Avx2PerChunkX87Final => x87_sum(partials),
        ParallelReductionStrategy::Avx2PerChunk => avx2_sum(partials),
    }
}

fn split_bounds(len: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1).min(len.max(1));
    let base = len / parts;
    let remainder = len % parts;
    let mut bounds = Vec::with_capacity(parts);
    let mut start = 0usize;
    for idx in 0..parts {
        let extra = usize::from(idx < remainder);
        let end = start + base + extra;
        bounds.push((start, end));
        start = end;
    }
    bounds
}

fn pin_current_thread(core_id: usize) {
    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_core_plan_nonempty() {
        let plan = PhysicalCorePlan::pinned(Some(2));
        assert!(!plan.core_ids().is_empty());
    }

    #[test]
    fn test_split_bounds_cover_input() {
        let bounds = split_bounds(10, 3);
        assert_eq!(bounds, vec![(0, 4), (4, 7), (7, 10)]);
    }

    #[test]
    fn test_parallel_sum_close_to_serial() {
        let a: Vec<f64> = (0..4096)
            .map(|i| {
                if i % 2 == 0 {
                    1e8
                } else {
                    -(1e8 + (i as f64) * 1e-6)
                }
            })
            .collect();
        let plan = PhysicalCorePlan::pinned(Some(4));
        let serial = x87_sum(&a);

        for strategy in [
            ParallelReductionStrategy::X87PerChunk,
            ParallelReductionStrategy::Avx2PerChunk,
            ParallelReductionStrategy::Avx2PerChunkX87Final,
        ] {
            let got = parallel_sum(&a, strategy, &plan);
            assert!(
                (got - serial).abs() < 1e-4,
                "strategy={} got={got} serial={serial}",
                strategy.label()
            );
        }
    }

    #[test]
    fn test_parallel_dot_close_to_serial() {
        let a: Vec<f64> = (0..4096).map(|i| 1e6 + (i as f64) * 1e-3).collect();
        let b: Vec<f64> = (0..4096)
            .map(|i| {
                if i % 2 == 0 {
                    1e6
                } else {
                    -(1e6 - (i as f64) * 1e-9)
                }
            })
            .collect();
        let plan = PhysicalCorePlan::pinned(Some(4));
        let serial = x87_dot(&a, &b);

        for strategy in [
            ParallelReductionStrategy::X87PerChunk,
            ParallelReductionStrategy::Avx2PerChunk,
            ParallelReductionStrategy::Avx2PerChunkX87Final,
        ] {
            let got = parallel_dot(&a, &b, strategy, &plan);
            assert!(
                (got - serial).abs() < 1e-1,
                "strategy={} got={got} serial={serial}",
                strategy.label()
            );
        }
    }
}
