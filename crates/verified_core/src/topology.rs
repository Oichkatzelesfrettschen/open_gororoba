//! Hardware Topology Discovery and Oracle.
//!
//! Provides a unified, thread-safe, and lazily evaluated oracle for host hardware
//! capabilities, explicitly targeting:
//! - SMT-Evasion (identifying pure physical cores for pinned math threads).
//! - V-Cache/L3 Detection (identifying optimal streaming block sizes for memory-bound loops).

use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Ordered list of logical OS thread IDs that map directly to unique physical cores.
    /// By binding threads to these IDs, we evade OS-level SMT (Hyperthreading) oversubscription.
    pub physical_core_ids: Vec<usize>,

    /// The total detected Level 3 Cache size in bytes.
    /// Falls back to a conservative 16MB if detection fails.
    pub l3_cache_bytes: usize,

    /// Estimated safe working set size (90% of L3 cache to leave room for OS/background tasks).
    pub l3_safe_working_set_bytes: usize,
}

impl HardwareTopology {
    /// Returns the global singleton HardwareTopology instance, computing it once on first access.
    pub fn current() -> &'static HardwareTopology {
        static TOPOLOGY: OnceLock<HardwareTopology> = OnceLock::new();
        TOPOLOGY.get_or_init(Self::detect)
    }

    /// Explicitly initialize and bind a global Rayon thread pool to the detected physical cores.
    /// This prevents thread migrations and SMT thrashing for parallel iterators.
    pub fn init_pinned_rayon_pool() -> Result<(), rayon::ThreadPoolBuildError> {
        let topo = Self::current();
        let cores = topo.physical_core_ids.clone();

        eprintln!(
            "HardwareTopology: Detected {} physical cores. L3 Cache: {:.1} MB (Safe Working Set: {:.1} MB)",
            cores.len(),
            topo.l3_cache_bytes as f64 / 1024.0 / 1024.0,
            topo.l3_safe_working_set_bytes as f64 / 1024.0 / 1024.0
        );

        rayon::ThreadPoolBuilder::new()
            .num_threads(cores.len())
            .start_handler(move |idx| {
                if let Some(&core_id) = cores.get(idx) {
                    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                }
            })
            .build_global()
    }
    fn detect() -> Self {
        let l3_cache_bytes = Self::detect_l3_cache().unwrap_or(16 * 1024 * 1024);
        let physical_core_ids = Self::detect_physical_cores();

        HardwareTopology {
            physical_core_ids,
            l3_cache_bytes,
            l3_safe_working_set_bytes: (l3_cache_bytes as f64 * 0.90) as usize,
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_l3_cache() -> Option<usize> {
        let cpuid = raw_cpuid::CpuId::new();
        if let Some(cparams) = cpuid.get_cache_parameters() {
            for cache in cparams {
                if cache.level() == 3 {
                    // Cache size = sets * associativity * coherency_line_size * physical_line_partitions
                    let size = cache.sets()
                        * cache.associativity()
                        * cache.coherency_line_size()
                        * cache.physical_line_partitions();
                    return Some(size);
                }
            }
        }
        None
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect_l3_cache() -> Option<usize> {
        None
    }

    fn detect_physical_cores() -> Vec<usize> {
        #[cfg(target_os = "linux")]
        if let Some(ids) = Self::detect_physical_core_ids_sysfs() {
            return ids;
        }

        // Cross-platform fallback via core_affinity
        match core_affinity::get_core_ids() {
            Some(ids) if !ids.is_empty() => {
                let all: Vec<usize> = ids.iter().map(|c| c.id).collect();
                // Naive SMT detection: If we have an even number of logical cores > 1,
                // assume the OS enumerates them as alternating physical/logical pairs
                // or blocks (like Windows/Linux default APIC mapping).
                // Grabbing every other core isolates the primary thread.
                if all.len().is_multiple_of(2) && all.len() > 1 {
                    all.into_iter().step_by(2).collect()
                } else {
                    all
                }
            }
            _ => vec![0],
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_physical_core_ids_sysfs() -> Option<Vec<usize>> {
        use std::{collections::BTreeMap, fs};

        let online = fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
        let online_cpus = Self::parse_cpu_range_list(online.trim());
        if online_cpus.is_empty() {
            return None;
        }

        let mut core_groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for cpu_id in &online_cpus {
            let pkg_path = format!(
                "/sys/devices/system/cpu/cpu{}/topology/physical_package_id",
                cpu_id
            );
            let core_path = format!("/sys/devices/system/cpu/cpu{}/topology/core_id", cpu_id);

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

        // Select the first (primary) logical CPU ID for each unique physical core package/core_id
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
}
