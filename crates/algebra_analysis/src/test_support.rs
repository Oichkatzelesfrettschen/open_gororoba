//! Test-only utilities for physical core pinning and Rayon pool setup.
//!
//! Ensures math-heavy tests pin to physical cores only (no SMT siblings),
//! preventing V-Cache thrashing on AMD Zen 3D and similar architectures.

use std::sync::Once;

static INIT_POOL: Once = Once::new();

/// Initialize a Rayon global thread pool pinned to physical cores only.
///
/// Safe to call from any test -- `Once` guarantees single initialization.
/// Subsequent calls are no-ops. If topology detection fails, falls back
/// to a standard unpinned pool with `num_physical` threads.
pub fn init_physical_rayon_pool() {
    INIT_POOL.call_once(|| {
        let physical_ids = detect_physical_core_ids();
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

/// Detect physical core IDs by reading Linux sysfs topology.
///
/// Reads `/sys/devices/system/cpu/cpuN/topology/core_id` and
/// `physical_package_id` for each online CPU. Groups logical CPUs by their
/// physical core. Returns the lowest logical CPU ID from each group.
///
/// Falls back to `step_by(2)` heuristic if sysfs is unavailable.
fn detect_physical_core_ids() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Some(ids) = detect_physical_core_ids_sysfs() {
            return ids;
        }
    }

    // Fallback: heuristic for common SMT layouts.
    match core_affinity::get_core_ids() {
        Some(ids) if !ids.is_empty() => {
            let all: Vec<usize> = ids.iter().map(|c| c.id).collect();
            if all.len().is_multiple_of(2) && all.len() > 1 {
                all.iter().step_by(2).copied().collect()
            } else {
                all
            }
        }
        _ => vec![0],
    }
}

/// Linux-specific: read sysfs topology to find one logical CPU per physical core.
#[cfg(target_os = "linux")]
fn detect_physical_core_ids_sysfs() -> Option<Vec<usize>> {
    use std::collections::BTreeMap;
    use std::fs;

    let online = fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let online_cpus = parse_cpu_range_list(online.trim());
    if online_cpus.is_empty() {
        return None;
    }

    // Group CPUs by (package_id, core_id) = unique physical core.
    let mut core_groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();

    for cpu_id in &online_cpus {
        let pkg_path = format!(
            "/sys/devices/system/cpu/cpu{cpu_id}/topology/physical_package_id",
        );
        let core_path = format!(
            "/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id",
        );

        let pkg_id = fs::read_to_string(&pkg_path)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let core_id = fs::read_to_string(&core_path)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(*cpu_id);

        core_groups
            .entry((pkg_id, core_id))
            .or_default()
            .push(*cpu_id);
    }

    let mut physical_ids: Vec<usize> = core_groups
        .values()
        .filter_map(|cpus| cpus.iter().min().copied())
        .collect();
    physical_ids.sort_unstable();

    if physical_ids.is_empty() {
        None
    } else {
        Some(physical_ids)
    }
}

/// Parse Linux CPU range list format: "0-5,8,10-12" -> [0,1,2,3,4,5,8,10,11,12]
#[cfg(target_os = "linux")]
fn parse_cpu_range_list(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (a.parse::<usize>(), b.parse::<usize>()) {
                result.extend(lo..=hi);
            }
        } else if let Ok(n) = part.parse::<usize>() {
            result.push(n);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_physical_cores() {
        let ids = detect_physical_core_ids();
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

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_range_list() {
        assert_eq!(parse_cpu_range_list("0-5"), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(parse_cpu_range_list("0-2,4,6-7"), vec![0, 1, 2, 4, 6, 7]);
        assert_eq!(parse_cpu_range_list("3"), vec![3]);
    }
}
