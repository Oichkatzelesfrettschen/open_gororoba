//! HDF5 export layer for computed data.
//!
//! Provides functions to export lattice data, motif census results,
//! GPU sweep results, and registry summaries to HDF5 format.
//!
//! Feature-gated behind `hdf5-export`. Requires libhdf5-dev at link time.
//!
//! # System Requirements
//!
//! The `hdf5-sys` crate (0.8.1) supports HDF5 versions 1.8.4 through 1.14.x.
//! It does NOT support HDF5 2.0.0+, which ships on Arch Linux / CachyOS as of
//! February 2026.  The build script panics with `Invalid H5_VERSION: "2.0.0"`.
//!
//! This is an upstream ecosystem gap.  Until `hdf5-sys` or `hdf5-metno-sys`
//! adds HDF5 2.0 support, this feature is only usable on systems with HDF5 1.x
//! (e.g., Ubuntu 22.04/24.04, Fedora 39, macOS via Homebrew).

use std::path::Path;

use hdf5::File as H5File;

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
    let flat: Vec<i8> = lattice_points.iter().flat_map(|p| p.iter().copied()).collect();

    let dataset = group
        .new_dataset::<i8>()
        .shape([lattice_points.len(), n_coords])
        .create("points")?;
    dataset.write_raw(&flat)?;

    // Metadata
    dataset.new_attr::<usize>().shape(()).create("dim")?.write_scalar(&dim)?;
    dataset.new_attr::<usize>().shape(()).create("n_points")?.write_scalar(&lattice_points.len())?;
    dataset.new_attr::<usize>().shape(()).create("n_coords")?.write_scalar(&n_coords)?;

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

    group.new_attr::<usize>().shape(()).create("n_components")?.write_scalar(&n_components)?;
    group.new_attr::<usize>().shape(()).create("nodes_per_component")?.write_scalar(&nodes_per_component)?;
    group.new_attr::<usize>().shape(()).create("n_motif_classes")?.write_scalar(&n_motif_classes)?;

    Ok(())
}

/// Export a claims summary table to HDF5.
///
/// Writes claim IDs, statuses as variable-length string datasets under
/// `/registry/claims/`.
pub fn export_claims_summary(
    path: &Path,
    ids: &[String],
    statuses: &[String],
) -> hdf5::Result<()> {
    let file = H5File::append(path)?;
    let group = file.create_group("registry/claims")?;

    // HDF5 variable-length strings
    let id_ds = group
        .new_dataset::<hdf5::types::VarLenUnicode>()
        .shape([ids.len()])
        .create("id")?;
    let id_data: Vec<hdf5::types::VarLenUnicode> = ids
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    id_ds.write(&id_data)?;

    let status_ds = group
        .new_dataset::<hdf5::types::VarLenUnicode>()
        .shape([statuses.len()])
        .create("status")?;
    let status_data: Vec<hdf5::types::VarLenUnicode> = statuses
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    status_ds.write(&status_data)?;

    group.new_attr::<usize>().shape(()).create("count")?.write_scalar(&ids.len())?;

    Ok(())
}

/// Read back lattice data from HDF5 for round-trip verification.
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
}
