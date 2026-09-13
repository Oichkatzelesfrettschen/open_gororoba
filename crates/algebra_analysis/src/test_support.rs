//! Test-only utilities for process-visible Rayon pool setup.

use std::sync::Once;

static INIT_POOL: Once = Once::new();

/// Initialize a Rayon global thread pool with every process-visible CPU.
///
/// Safe to call from any test -- `Once` guarantees single initialization.
/// Subsequent calls are no-ops. CPU detection fails closed.
pub fn init_process_visible_rayon_pool() {
    INIT_POOL.call_once(|| {
        let worker_count = std::thread::available_parallelism()
            .expect("process-visible CPU detection must succeed")
            .get();

        let pool_result = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build_global();

        if let Err(error) = pool_result {
            // Global pool already initialized (e.g., by another test in the same binary).
            eprintln!("Note: Rayon global pool already set: {error}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_process_visible_cpus() {
        assert!(std::thread::available_parallelism().is_ok());
    }

    #[test]
    fn test_init_pool_idempotent() {
        init_process_visible_rayon_pool();
        init_process_visible_rayon_pool(); // second call is no-op
    }
}
