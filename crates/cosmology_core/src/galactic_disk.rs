//! Galactic Disk Coupling Analysis for AVT Resonances.
//!
//! Compares resonance strengths (cross-correlation variance drop) between
//! pulsars inside and outside the Galactic disk to determine if signals are
//! ISM-induced (non-associative scattering) or universal vacuum properties.
//!
//! Migrated from src/scripts/analysis/analyze_disk_coupling.py.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Results of a Galactic latitude-dependent AVT sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskCouplingComparison {
    pub dimension: usize,
    pub disk_drop_pct: f64,
    pub halo_drop_pct: f64,
    pub delta_pct: f64,
}

/// Analysis result for Galactic disk coupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CouplingResult {
    DiskCouplingConfirmed,
    IsotropicCoupling,
    Inconclusive,
}

/// Compute the comparison matrix between Disk (|b| < 20) and Halo (|b| > 20) sweeps.
pub fn compare_disk_halo_coupling(
    disk_data: &[(usize, f64)], // (dim, drop_pct)
    halo_data: &[(usize, f64)],
) -> Vec<DiskCouplingComparison> {
    let mut results = Vec::new();
    for (d, dd) in disk_data {
        if let Some((_, hd)) = halo_data.iter().find(|(dim, _)| dim == d) {
            results.push(DiskCouplingComparison {
                dimension: *d,
                disk_drop_pct: *dd,
                halo_drop_pct: *hd,
                delta_pct: dd - hd,
            });
        }
    }
    results
}

/// Interpret the coupling results based on maximal drops.
pub fn interpret_coupling(disk_max: f64, halo_max: f64) -> CouplingResult {
    if disk_max > halo_max + 2.0 {
        CouplingResult::DiskCouplingConfirmed
    } else if (disk_max - halo_max).abs() < 1.0 {
        CouplingResult::IsotropicCoupling
    } else {
        CouplingResult::Inconclusive
    }
}

/// Helper to load sweep data from CSV.
pub fn load_sweep_csv<P: AsRef<Path>>(path: P) -> Result<Vec<(usize, f64)>, String> {
    let mut reader = csv::Reader::from_path(path).map_err(|e| e.to_string())?;
    let mut data = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let dim: usize = record
            .get(0)
            .ok_or("Missing dimension")?
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let drop: f64 = record
            .get(1)
            .ok_or("Missing drop_pct")?
            .parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        data.push((dim, drop));
    }
    Ok(data)
}
