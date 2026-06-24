//! HDF5 export layer for computed data.
//!
//! Provides functions to export lattice data, motif census results,
//! GPU sweep results, and registry summaries to HDF5 format.
//!
//! Feature-gated behind `hdf5-export`. Requires system libhdf5 at link time.
//!
//! # System Requirements
//!
//! Uses `hdf5-metno` 0.12 (maintained fork of `hdf5-rust`) which supports
//! HDF5 1.8.4 through 2.0.x.  The original `hdf5` 0.8.1 panicked on HDF5 2.0
//! with `Invalid H5_VERSION: "2.0.0"`; the metno fork fixes this.
//!
//! On Arch Linux / CachyOS: `pacman -S hdf5`
//! On Ubuntu / Debian: `apt install libhdf5-dev`

use std::{collections::BTreeMap, path::Path};

use crate::quality::{RhoQualityThresholds, RhoTraceQuality};
use gororoba_contracts::WarpRingExperiment;
use hdf5::File as H5File;

pub const REQUIRED_TRACE_CHANNELS: [&str; 3] = ["rho_mean", "enstrophy", "algebra_norm"];

#[derive(Debug, Clone, Default)]
pub struct SimulationTraceBundle {
    pub time: Vec<f64>,
    pub channels: BTreeMap<String, Vec<f64>>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct SpectralSummarySeries {
    pub time: Vec<f64>,
    pub k_peak: Vec<f64>,
    pub power_peak: Vec<f64>,
    pub total_power: Vec<f64>,
    pub slope: Vec<f64>,
    pub triad_count: Vec<f64>,
    pub triad_clustering: Vec<f64>,
}

fn ensure_simulation_group(file: &H5File) -> hdf5::Result<hdf5::Group> {
    match file.group("simulation") {
        Ok(g) => Ok(g),
        Err(_) => file.create_group("simulation"),
    }
}

fn ensure_child_group(parent: &hdf5::Group, name: &str) -> hdf5::Result<hdf5::Group> {
    match parent.group(name) {
        Ok(g) => Ok(g),
        Err(_) => parent.create_group(name),
    }
}

fn write_or_replace_f64_dataset(group: &hdf5::Group, name: &str, data: &[f64]) -> hdf5::Result<()> {
    if group.link_exists(name) {
        group.unlink(name)?;
    }
    group
        .new_dataset::<f64>()
        .shape([data.len()])
        .create(name)?
        .write(data)?;
    Ok(())
}

fn write_or_replace_str_dataset(
    group: &hdf5::Group,
    name: &str,
    values: &[String],
) -> hdf5::Result<()> {
    if group.link_exists(name) {
        group.unlink(name)?;
    }
    let dataset = group
        .new_dataset::<hdf5::types::VarLenUnicode>()
        .shape([values.len()])
        .create(name)?;
    let encoded: Vec<hdf5::types::VarLenUnicode> =
        values.iter().map(|s| s.parse().unwrap()).collect();
    dataset.write(&encoded)?;
    Ok(())
}

fn clear_group_members(group: &hdf5::Group) -> hdf5::Result<()> {
    for name in group.member_names()? {
        group.unlink(&name)?;
    }
    Ok(())
}

/// Export a Warp Ring Experiment contract to HDF5.
///
/// Serializes the contract metadata as attributes and groups.
pub fn export_experiment_contract(path: &Path, contract: &WarpRingExperiment) -> hdf5::Result<()> {
    let file = H5File::create(path)?; // Overwrite if exists, it's a new artifact
    let group = file.create_group("experiment")?;

    // Metadata
    group
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("id")?
        .write_scalar(
            &contract
                .experiment_id
                .parse::<hdf5::types::VarLenUnicode>()
                .unwrap(),
        )?;

    // Config
    let config_grp = group.create_group("config")?;
    config_grp
        .new_attr::<usize>()
        .create("resolution")?
        .write_scalar(&contract.config.resolution)?;
    config_grp
        .new_attr::<usize>()
        .create("steps")?
        .write_scalar(&contract.config.steps)?;
    config_grp
        .new_attr::<f64>()
        .create("tau")?
        .write_scalar(&contract.config.tau)?;
    config_grp
        .new_attr::<f64>()
        .create("coupling_lambda")?
        .write_scalar(&contract.config.coupling_lambda)?;
    config_grp
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("forcing_type")?
        .write_scalar(
            &contract
                .config
                .forcing_type
                .parse::<hdf5::types::VarLenUnicode>()
                .unwrap(),
        )?;

    // Results
    let res_grp = group.create_group("results")?;
    res_grp
        .new_attr::<f64>()
        .create("final_enstrophy")?
        .write_scalar(&contract.results.final_enstrophy)?;
    res_grp
        .new_attr::<f64>()
        .create("mean_density")?
        .write_scalar(&contract.results.mean_density)?;
    res_grp
        .new_attr::<f64>()
        .create("execution_time_s")?
        .write_scalar(&contract.results.execution_time_s)?;

    Ok(())
}

/// Export Cayley-Dickson lattice embedding data to HDF5.
///
/// Writes lattice points as a 2D dataset [n_points x n_coords] under
/// `/lattice/dim{dim}/points`.
pub fn export_lattice_data(
    path: &Path,
    dim: usize,
    lattice_points: &[Vec<i8>],
) -> hdf5::Result<()> {
    let file = H5File::append(path)?;
    let group_name = format!("lattice/dim{dim}");
    let group = file.create_group(&group_name)?;

    if lattice_points.is_empty() {
        return Ok(());
    }

    let n_coords = lattice_points[0].len();
    let flat: Vec<i8> = lattice_points
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();

    let dataset = group
        .new_dataset::<i8>()
        .shape([lattice_points.len(), n_coords])
        .create("points")?;
    dataset.write_raw(&flat)?;

    // Metadata
    dataset
        .new_attr::<usize>()
        .shape(())
        .create("dim")?
        .write_scalar(&dim)?;
    dataset
        .new_attr::<usize>()
        .shape(())
        .create("n_points")?
        .write_scalar(&lattice_points.len())?;
    dataset
        .new_attr::<usize>()
        .shape(())
        .create("n_coords")?
        .write_scalar(&n_coords)?;

    Ok(())
}

/// Export motif census data to HDF5.
///
/// Writes component data under `/motifs/dim{dim}/`.
pub fn export_motif_census(
    path: &Path,
    dim: usize,
    n_components: usize,
    nodes_per_component: usize,
    n_motif_classes: usize,
) -> hdf5::Result<()> {
    let file = H5File::append(path)?;
    let group_name = format!("motifs/dim{dim}");
    let group = file.create_group(&group_name)?;

    group
        .new_attr::<usize>()
        .shape(())
        .create("n_components")?
        .write_scalar(&n_components)?;
    group
        .new_attr::<usize>()
        .shape(())
        .create("nodes_per_component")?
        .write_scalar(&nodes_per_component)?;
    group
        .new_attr::<usize>()
        .shape(())
        .create("n_motif_classes")?
        .write_scalar(&n_motif_classes)?;

    Ok(())
}

/// Export a full simulation trace to HDF5.
///
/// Saves fluid density, enstrophy, and algebraic field norms over time.
pub fn export_simulation_trace(
    path: &Path,
    time: &[f64],
    rho_mean: &[f64],
    enstrophy: &[f64],
    algebra_norm: &[f64],
) -> hdf5::Result<()> {
    let mut channels = BTreeMap::new();
    channels.insert("rho_mean".to_string(), rho_mean.to_vec());
    channels.insert("enstrophy".to_string(), enstrophy.to_vec());
    channels.insert("algebra_norm".to_string(), algebra_norm.to_vec());
    let bundle = SimulationTraceBundle {
        time: time.to_vec(),
        channels,
        metadata: BTreeMap::new(),
    };
    export_simulation_trace_bundle(path, &bundle)
}

/// Export a trace bundle with required and optional channels.
///
/// Required channels are:
/// - `rho_mean`
/// - `enstrophy`
/// - `algebra_norm`
///
/// Additional channels are persisted under `/simulation/trace/<name>`.
/// Metadata key/value pairs are stored under `/simulation/trace_meta/{keys,values}`.
pub fn export_simulation_trace_bundle(
    path: &Path,
    bundle: &SimulationTraceBundle,
) -> hdf5::Result<()> {
    let n = bundle.time.len();
    for required in REQUIRED_TRACE_CHANNELS {
        let values = bundle.channels.get(required).ok_or_else(|| {
            hdf5::Error::from(format!("missing required trace channel '{required}'"))
        })?;
        if values.len() != n {
            return Err(hdf5::Error::from(format!(
                "trace channel '{required}' length mismatch: expected {n}, got {}",
                values.len()
            )));
        }
    }
    for (name, values) in &bundle.channels {
        if values.len() != n {
            return Err(hdf5::Error::from(format!(
                "trace channel '{name}' length mismatch: expected {n}, got {}",
                values.len()
            )));
        }
    }

    let file = H5File::append(path)?;
    let simulation = ensure_simulation_group(&file)?;
    let trace_group = ensure_child_group(&simulation, "trace")?;
    clear_group_members(&trace_group)?;
    write_or_replace_f64_dataset(&trace_group, "time", &bundle.time)?;
    for (name, values) in &bundle.channels {
        write_or_replace_f64_dataset(&trace_group, name, values)?;
    }

    let trace_meta_group = ensure_child_group(&simulation, "trace_meta")?;
    clear_group_members(&trace_meta_group)?;
    if !bundle.metadata.is_empty() {
        let mut keys = Vec::with_capacity(bundle.metadata.len());
        let mut values = Vec::with_capacity(bundle.metadata.len());
        for (k, v) in &bundle.metadata {
            keys.push(k.clone());
            values.push(v.clone());
        }
        write_or_replace_str_dataset(&trace_meta_group, "keys", &keys)?;
        write_or_replace_str_dataset(&trace_meta_group, "values", &values)?;
    }

    Ok(())
}

/// Export compact spectral summaries under `/simulation/spectral`.
pub fn export_simulation_spectral_summary(
    path: &Path,
    summary: &SpectralSummarySeries,
) -> hdf5::Result<()> {
    let n = summary.time.len();
    let checks = [
        ("k_peak", summary.k_peak.len()),
        ("power_peak", summary.power_peak.len()),
        ("total_power", summary.total_power.len()),
        ("slope", summary.slope.len()),
        ("triad_count", summary.triad_count.len()),
        ("triad_clustering", summary.triad_clustering.len()),
    ];
    for (name, len) in checks {
        if len != n {
            return Err(hdf5::Error::from(format!(
                "spectral channel '{name}' length mismatch: expected {n}, got {len}"
            )));
        }
    }

    let file = H5File::append(path)?;
    let simulation = ensure_simulation_group(&file)?;
    let spectral = ensure_child_group(&simulation, "spectral")?;
    write_or_replace_f64_dataset(&spectral, "time", &summary.time)?;
    write_or_replace_f64_dataset(&spectral, "k_peak", &summary.k_peak)?;
    write_or_replace_f64_dataset(&spectral, "power_peak", &summary.power_peak)?;
    write_or_replace_f64_dataset(&spectral, "total_power", &summary.total_power)?;
    write_or_replace_f64_dataset(&spectral, "slope", &summary.slope)?;
    write_or_replace_f64_dataset(&spectral, "triad_count", &summary.triad_count)?;
    write_or_replace_f64_dataset(&spectral, "triad_clustering", &summary.triad_clustering)?;
    Ok(())
}

/// Read `/simulation/trace/rho_mean` from an HDF5 artifact.
pub fn read_rho_mean_trace(path: &Path) -> hdf5::Result<Vec<f64>> {
    let file = H5File::open(path)?;
    let dataset = file.dataset("simulation/trace/rho_mean")?;
    dataset.read_raw()
}

/// Read a named `/simulation/trace/<name>` f64 vector from an HDF5 artifact.
pub fn read_simulation_trace_component(path: &Path, name: &str) -> hdf5::Result<Vec<f64>> {
    let file = H5File::open(path)?;
    let dataset = file.dataset(&format!("simulation/trace/{name}"))?;
    dataset.read_raw()
}

/// Read a named `/simulation/spectral/<name>` f64 vector from an HDF5 artifact.
pub fn read_simulation_spectral_component(path: &Path, name: &str) -> hdf5::Result<Vec<f64>> {
    let file = H5File::open(path)?;
    let dataset = file.dataset(&format!("simulation/spectral/{name}"))?;
    dataset.read_raw()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericDatasetScanStatus {
    Checked,
    NonNumericSkipped,
    UnsupportedNumericLayout,
}

#[derive(Debug, Clone)]
pub struct NumericDatasetScanEntry {
    pub path: String,
    pub dtype: String,
    pub element_count: usize,
    pub finite_count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub status: NumericDatasetScanStatus,
}

#[derive(Debug, Clone)]
pub struct Hdf5NumericScanReport {
    pub datasets_total: usize,
    pub numeric_checked: usize,
    pub non_numeric_skipped: usize,
    pub unsupported_numeric_layouts: usize,
    pub datasets_with_non_finite: usize,
    pub entries: Vec<NumericDatasetScanEntry>,
}

impl Hdf5NumericScanReport {
    pub fn has_failures(&self) -> bool {
        self.datasets_with_non_finite > 0 || self.unsupported_numeric_layouts > 0
    }
}

fn descriptor_is_numeric(desc: &hdf5::types::TypeDescriptor) -> bool {
    use hdf5::types::TypeDescriptor;
    match desc {
        TypeDescriptor::Integer(_)
        | TypeDescriptor::Unsigned(_)
        | TypeDescriptor::Float(_)
        | TypeDescriptor::Boolean
        | TypeDescriptor::Enum(_) => true,
        TypeDescriptor::FixedArray(inner, _) | TypeDescriptor::VarLenArray(inner) => {
            descriptor_is_numeric(inner)
        }
        TypeDescriptor::Compound(compound) => {
            compound.fields.iter().any(|f| descriptor_is_numeric(&f.ty))
        }
        TypeDescriptor::FixedAscii(_)
        | TypeDescriptor::FixedUnicode(_)
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode
        | TypeDescriptor::Reference(_) => false,
    }
}

fn analyze_dataset(path: &str, dataset: &hdf5::Dataset) -> hdf5::Result<NumericDatasetScanEntry> {
    use hdf5::types::{FloatSize, IntSize, TypeDescriptor};

    let descriptor = dataset.dtype()?.to_descriptor()?;
    let dtype = descriptor.to_string();

    let mk_checked =
        |element_count: usize, finite_count: usize, nan_count: usize, inf_count: usize| {
            NumericDatasetScanEntry {
                path: path.to_string(),
                dtype: dtype.clone(),
                element_count,
                finite_count,
                nan_count,
                inf_count,
                status: NumericDatasetScanStatus::Checked,
            }
        };

    let mk_skipped = |status: NumericDatasetScanStatus| NumericDatasetScanEntry {
        path: path.to_string(),
        dtype: dtype.clone(),
        element_count: 0,
        finite_count: 0,
        nan_count: 0,
        inf_count: 0,
        status,
    };

    match descriptor {
        TypeDescriptor::Float(FloatSize::U4) => {
            let values: Vec<f32> = dataset.read_raw()?;
            let mut finite_count = 0usize;
            let mut nan_count = 0usize;
            let mut inf_count = 0usize;
            for &v in &values {
                if v.is_finite() {
                    finite_count += 1;
                } else if v.is_nan() {
                    nan_count += 1;
                } else {
                    inf_count += 1;
                }
            }
            Ok(mk_checked(values.len(), finite_count, nan_count, inf_count))
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            let values: Vec<f64> = dataset.read_raw()?;
            let mut finite_count = 0usize;
            let mut nan_count = 0usize;
            let mut inf_count = 0usize;
            for &v in &values {
                if v.is_finite() {
                    finite_count += 1;
                } else if v.is_nan() {
                    nan_count += 1;
                } else {
                    inf_count += 1;
                }
            }
            Ok(mk_checked(values.len(), finite_count, nan_count, inf_count))
        }
        TypeDescriptor::Integer(IntSize::U1) => {
            let values: Vec<i8> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            let values: Vec<i16> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Integer(IntSize::U4) => {
            let values: Vec<i32> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            let values: Vec<i64> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            let values: Vec<u8> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            let values: Vec<u16> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            let values: Vec<u32> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            let values: Vec<u64> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Boolean => {
            let values: Vec<bool> = dataset.read_raw()?;
            let n = values.len();
            Ok(mk_checked(n, n, 0, 0))
        }
        TypeDescriptor::Enum(enum_ty) => {
            let n = match (enum_ty.signed, enum_ty.size) {
                (true, IntSize::U1) => dataset.read_raw::<i8>()?.len(),
                (true, IntSize::U2) => dataset.read_raw::<i16>()?.len(),
                (true, IntSize::U4) => dataset.read_raw::<i32>()?.len(),
                (true, IntSize::U8) => dataset.read_raw::<i64>()?.len(),
                (false, IntSize::U1) => dataset.read_raw::<u8>()?.len(),
                (false, IntSize::U2) => dataset.read_raw::<u16>()?.len(),
                (false, IntSize::U4) => dataset.read_raw::<u32>()?.len(),
                (false, IntSize::U8) => dataset.read_raw::<u64>()?.len(),
            };
            Ok(mk_checked(n, n, 0, 0))
        }
        other => {
            if descriptor_is_numeric(&other) {
                Ok(mk_skipped(
                    NumericDatasetScanStatus::UnsupportedNumericLayout,
                ))
            } else {
                Ok(mk_skipped(NumericDatasetScanStatus::NonNumericSkipped))
            }
        }
    }
}

fn scan_group_recursive(
    group: &hdf5::Group,
    prefix: &str,
    report: &mut Hdf5NumericScanReport,
) -> hdf5::Result<()> {
    for name in group.member_names()? {
        let path = if prefix.is_empty() {
            format!("/{}", name)
        } else {
            format!("{}/{}", prefix, name)
        };

        if let Ok(sub_group) = group.group(&name) {
            scan_group_recursive(&sub_group, &path, report)?;
            continue;
        }
        if let Ok(dataset) = group.dataset(&name) {
            report.datasets_total += 1;
            let entry = analyze_dataset(&path, &dataset)?;
            match entry.status {
                NumericDatasetScanStatus::Checked => {
                    report.numeric_checked += 1;
                    if entry.nan_count > 0 || entry.inf_count > 0 {
                        report.datasets_with_non_finite += 1;
                    }
                }
                NumericDatasetScanStatus::NonNumericSkipped => {
                    report.non_numeric_skipped += 1;
                }
                NumericDatasetScanStatus::UnsupportedNumericLayout => {
                    report.unsupported_numeric_layouts += 1;
                }
            }
            report.entries.push(entry);
        }
    }
    Ok(())
}

/// Recursively scan every dataset in an HDF5 file and validate numeric datasets for finite values.
///
/// Numeric layouts that cannot be decoded losslessly by this scanner are flagged as
/// `UnsupportedNumericLayout` so callers can fail closed instead of silently skipping.
pub fn scan_hdf5_numeric_datasets(path: &Path) -> hdf5::Result<Hdf5NumericScanReport> {
    let file = H5File::open(path)?;
    let root = file.group("/")?;
    let mut report = Hdf5NumericScanReport {
        datasets_total: 0,
        numeric_checked: 0,
        non_numeric_skipped: 0,
        unsupported_numeric_layouts: 0,
        datasets_with_non_finite: 0,
        entries: Vec::new(),
    };
    scan_group_recursive(&root, "", &mut report)?;
    Ok(report)
}

/// Persist rho trace quality metrics under `/simulation/trace_quality`.
pub fn export_rho_quality_metrics(
    path: &Path,
    quality: &RhoTraceQuality,
    thresholds: RhoQualityThresholds,
) -> hdf5::Result<()> {
    let file = H5File::append(path)?;
    let simulation = match file.group("simulation") {
        Ok(g) => g,
        Err(_) => file.create_group("simulation")?,
    };
    let quality_group = match simulation.group("trace_quality") {
        Ok(g) => g,
        Err(_) => simulation.create_group("trace_quality")?,
    };

    quality_group
        .new_attr::<usize>()
        .create("sample_count")?
        .write_scalar(&quality.sample_count)?;
    quality_group
        .new_attr::<usize>()
        .create("finite_count")?
        .write_scalar(&quality.finite_count)?;
    quality_group
        .new_attr::<usize>()
        .create("nan_count")?
        .write_scalar(&quality.nan_count)?;
    quality_group
        .new_attr::<usize>()
        .create("inf_count")?
        .write_scalar(&quality.inf_count)?;
    quality_group
        .new_attr::<f64>()
        .create("min")?
        .write_scalar(&quality.min)?;
    quality_group
        .new_attr::<f64>()
        .create("max")?
        .write_scalar(&quality.max)?;
    quality_group
        .new_attr::<f64>()
        .create("mean")?
        .write_scalar(&quality.mean)?;
    quality_group
        .new_attr::<f64>()
        .create("std_dev")?
        .write_scalar(&quality.std_dev)?;
    quality_group
        .new_attr::<f64>()
        .create("final_value")?
        .write_scalar(&quality.final_value)?;
    quality_group
        .new_attr::<f64>()
        .create("abs_drift_final")?
        .write_scalar(&quality.abs_drift_final)?;
    quality_group
        .new_attr::<f64>()
        .create("max_abs_drift_from_one")?
        .write_scalar(&quality.max_abs_drift_from_one)?;
    quality_group
        .new_attr::<f64>()
        .create("threshold_max_abs_drift_final")?
        .write_scalar(&thresholds.max_abs_drift_final)?;
    quality_group
        .new_attr::<f64>()
        .create("threshold_max_std_dev")?
        .write_scalar(&thresholds.max_std_dev)?;

    Ok(())
}

/// Export a claims summary table to HDF5.
///
/// Writes claim IDs, statuses as variable-length string datasets under
/// `/registry/claims/`.
pub fn export_claims_summary(path: &Path, ids: &[String], statuses: &[String]) -> hdf5::Result<()> {
    if ids.len() != statuses.len() {
        return Err(hdf5::Error::from(format!(
            "claim summary length mismatch: ids has {}, statuses has {}",
            ids.len(),
            statuses.len()
        )));
    }

    let file = H5File::append(path)?;
    let group = file.create_group("registry/claims")?;

    // HDF5 variable-length strings
    let id_ds = group
        .new_dataset::<hdf5::types::VarLenUnicode>()
        .shape([ids.len()])
        .create("id")?;
    let id_data: Vec<hdf5::types::VarLenUnicode> = ids.iter().map(|s| s.parse().unwrap()).collect();
    id_ds.write(&id_data)?;

    let status_ds = group
        .new_dataset::<hdf5::types::VarLenUnicode>()
        .shape([statuses.len()])
        .create("status")?;
    let status_data: Vec<hdf5::types::VarLenUnicode> =
        statuses.iter().map(|s| s.parse().unwrap()).collect();
    status_ds.write(&status_data)?;

    group
        .new_attr::<usize>()
        .shape(())
        .create("count")?
        .write_scalar(&ids.len())?;

    Ok(())
}

/// Export a full 3D vector field to HDF5.
///
/// Shape: [nx, ny, nz, 3]
pub fn export_field_3d(
    path: &std::path::Path,
    name: &str,
    nx: usize,
    ny: usize,
    nz: usize,
    field: &[f64], // Flattened [nx * ny * nz * 3]
) -> hdf5::Result<()> {
    // Create new file for the snapshot (overwrite if exists)
    let file = H5File::create(path)?;
    let dataset = file
        .new_dataset::<f64>()
        .shape([nx, ny, nz, 3])
        .create(name)?;
    dataset.write_raw(field)?;
    Ok(())
}
pub fn read_lattice_data(path: &Path, dim: usize) -> hdf5::Result<Vec<Vec<i8>>> {
    let file = H5File::open(path)?;
    let group_name = format!("lattice/dim{dim}");
    let dataset = file.dataset(&format!("{group_name}/points"))?;

    let shape = dataset.shape();
    let n_points = shape[0];
    let n_coords = shape[1];

    let flat: Vec<i8> = dataset.read_raw()?;
    let points: Vec<Vec<i8>> = flat
        .chunks_exact(n_coords)
        .map(|chunk| chunk.to_vec())
        .collect();

    assert_eq!(points.len(), n_points);
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        let points = vec![
            vec![1i8, 0, -1, 0, 0, 0, 0, 1],
            vec![-1, -1, 0, 0, 1, 0, 0, -1],
            vec![0, 1, 1, -1, 0, 0, -1, 0],
        ];

        export_lattice_data(&path, 256, &points).unwrap();
        let read_back = read_lattice_data(&path, 256).unwrap();

        assert_eq!(points, read_back);
    }

    #[test]
    fn test_claims_summary_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_claims.h5");

        let ids = vec!["C-001".to_string(), "C-002".to_string()];
        let statuses = vec!["Verified".to_string(), "Refuted".to_string()];

        export_claims_summary(&path, &ids, &statuses).unwrap();

        // Verify we can open the file and read back
        let file = H5File::open(&path).unwrap();
        let group = file.group("registry/claims").unwrap();
        let count: usize = group.attr("count").unwrap().read_scalar().unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_claims_summary_rejects_column_length_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_claims_mismatch.h5");

        let ids = vec!["C-001".to_string(), "C-002".to_string()];
        let statuses = vec!["Verified".to_string()];

        let err = export_claims_summary(&path, &ids, &statuses).unwrap_err();
        assert!(
            err.to_string().contains("claim summary length mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_trace_reexport_removes_stale_optional_channels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_trace_stale.h5");

        let mut first_channels = BTreeMap::new();
        first_channels.insert("rho_mean".to_string(), vec![1.0, 1.1]);
        first_channels.insert("enstrophy".to_string(), vec![0.1, 0.2]);
        first_channels.insert("algebra_norm".to_string(), vec![2.0, 2.1]);
        first_channels.insert("obsolete_channel".to_string(), vec![9.0, 9.1]);

        let mut metadata = BTreeMap::new();
        metadata.insert("run".to_string(), "old".to_string());

        export_simulation_trace_bundle(
            &path,
            &SimulationTraceBundle {
                time: vec![0.0, 1.0],
                channels: first_channels,
                metadata,
            },
        )
        .unwrap();

        let mut second_channels = BTreeMap::new();
        second_channels.insert("rho_mean".to_string(), vec![3.0]);
        second_channels.insert("enstrophy".to_string(), vec![0.3]);
        second_channels.insert("algebra_norm".to_string(), vec![4.0]);

        export_simulation_trace_bundle(
            &path,
            &SimulationTraceBundle {
                time: vec![2.0],
                channels: second_channels,
                metadata: BTreeMap::new(),
            },
        )
        .unwrap();

        let file = H5File::open(&path).unwrap();
        let trace = file.group("simulation/trace").unwrap();
        assert!(!trace.link_exists("obsolete_channel"));
        let rho_mean: Vec<f64> = trace.dataset("rho_mean").unwrap().read_raw().unwrap();
        assert_eq!(rho_mean, vec![3.0]);

        let trace_meta = file.group("simulation/trace_meta").unwrap();
        assert!(!trace_meta.link_exists("keys"));
        assert!(!trace_meta.link_exists("values"));
    }
}
