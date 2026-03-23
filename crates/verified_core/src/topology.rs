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

    /// Detected cache sizes in bytes per level (L1d, L2, L3, L4 if present).
    /// Falls back to conservative defaults if detection fails.
    pub cache_hierarchy: CacheHierarchy,

    /// The total detected Level 3 Cache size in bytes.
    /// Falls back to a conservative 16MB if detection fails.
    pub l3_cache_bytes: usize,

    /// Estimated safe working set size (90% of L3 cache to leave room for OS/background tasks).
    pub l3_safe_working_set_bytes: usize,
}

/// Detected cache sizes for all levels.
///
/// Used by the CD sign table to auto-select representation:
/// - dim <= sqrt(L1d * 8): bit-packed SignTable (1 bit/entry, fits L1)
/// - dim <= sqrt(L2): i8 SignTableI8 (1 byte/entry, fits L2)
/// - dim <= sqrt(L3): precomputed f64 rows (8 bytes/entry, fits L3)
///
/// # Detection
///
/// Uses CPUID leaf 4 (deterministic cache parameters) on x86_64.
/// Falls back to conservative defaults on other architectures.
#[derive(Debug, Clone)]
pub struct CacheHierarchy {
    /// L1 data cache size in bytes (typically 32-48KB).
    pub l1d_bytes: usize,
    /// L2 cache size in bytes (typically 256KB-1MB).
    pub l2_bytes: usize,
    /// L3 cache size in bytes (typically 8-96MB).
    pub l3_bytes: usize,
    /// L4 cache size in bytes (0 if not present).
    pub l4_bytes: usize,
    /// Cache line size in bytes (typically 64).
    pub line_size: usize,
}

impl CacheHierarchy {
    /// Conservative defaults for when detection fails.
    pub fn defaults() -> Self {
        Self {
            l1d_bytes: 32 * 1024,       // 32 KB
            l2_bytes: 512 * 1024,       // 512 KB
            l3_bytes: 16 * 1024 * 1024, // 16 MB
            l4_bytes: 0,
            line_size: 64,
        }
    }

    /// Maximum CD dimension whose sign table fits in the given cache level.
    ///
    /// For bit-packed: dim^2 / 8 bytes. Max dim = sqrt(cache_bytes * 8).
    /// For i8: dim^2 bytes. Max dim = sqrt(cache_bytes).
    pub fn max_dim_bitpacked(&self, level: usize) -> usize {
        let bytes = match level {
            1 => self.l1d_bytes,
            2 => self.l2_bytes,
            3 => self.l3_bytes,
            _ => self.l3_bytes,
        };
        ((bytes * 8) as f64).sqrt() as usize
    }

    pub fn max_dim_i8(&self, level: usize) -> usize {
        let bytes = match level {
            1 => self.l1d_bytes,
            2 => self.l2_bytes,
            3 => self.l3_bytes,
            _ => self.l3_bytes,
        };
        (bytes as f64).sqrt() as usize
    }
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
        let cache_hierarchy = Self::detect_cache_hierarchy();
        let l3_cache_bytes = cache_hierarchy.l3_bytes;
        let physical_core_ids = Self::detect_physical_cores();

        HardwareTopology {
            physical_core_ids,
            cache_hierarchy,
            l3_cache_bytes,
            l3_safe_working_set_bytes: (l3_cache_bytes as f64 * 0.90) as usize,
        }
    }

    /// Detect all cache levels via CPUID.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_cache_hierarchy() -> CacheHierarchy {
        let cpuid = raw_cpuid::CpuId::new();
        let mut hier = CacheHierarchy::defaults();

        if let Some(cparams) = cpuid.get_cache_parameters() {
            for cache in cparams {
                let size = cache.sets()
                    * cache.associativity()
                    * cache.coherency_line_size()
                    * cache.physical_line_partitions();
                let line_size = cache.coherency_line_size();

                match cache.level() {
                    1 => {
                        // Only count data caches (not instruction)
                        if cache.cache_type() == raw_cpuid::CacheType::Data
                            || cache.cache_type() == raw_cpuid::CacheType::Unified
                        {
                            hier.l1d_bytes = size;
                            hier.line_size = line_size;
                        }
                    }
                    2 => { hier.l2_bytes = size; }
                    3 => { hier.l3_bytes = size; }
                    4 => { hier.l4_bytes = size; }
                    _ => {}
                }
            }
        }

        hier
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect_cache_hierarchy() -> CacheHierarchy {
        CacheHierarchy::defaults()
    }

    // detect_l3_cache removed: superseded by detect_cache_hierarchy (P7).

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
