//! Test-only utilities for physical core pinning and Rayon pool setup.
//!
//! Ensures math-heavy tests pin to physical cores only (no SMT siblings),
//! preventing V-Cache thrashing on AMD Zen 3D and similar architectures.

use std::sync::Once;

use cd_kernel::physical_core_ids;

static INIT_POOL: Once = Once::new();

/// Initialize a Rayon global thread pool pinned to physical cores only.
///
/// Safe to call from any test -- `Once` guarantees single initialization.
/// Subsequent calls are no-ops. If topology detection fails, falls back
/// to a standard unpinned pool with `num_physical` threads.
pub fn init_physical_rayon_pool() {
    INIT_POOL.call_once(|| {
        let physical_ids = physical_core_ids();
        let n_threads = physical_ids.len().max(1);

        let pool_result = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .start_handler(move |idx| {
                if idx < physical_ids.len() {
                    let core_id = core_affinity::CoreId {
                        id: physical_ids[idx],
                    };
                    core_affinity::set_for_current(core_id);
                }
            })
            .build_global();

        if let Err(e) = pool_result {
            // Global pool already initialized (e.g., by another test in the same binary).
            eprintln!("Note: Rayon global pool already set: {e}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_physical_cores() {
        let ids = physical_core_ids();
        assert!(!ids.is_empty(), "Should detect at least one physical core");
        // IDs should be sorted and unique
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "Core IDs should be sorted and unique");
        }
    }

    #[test]
    fn test_init_pool_idempotent() {
        init_physical_rayon_pool();
        init_physical_rayon_pool(); // second call is no-op
    }
}
