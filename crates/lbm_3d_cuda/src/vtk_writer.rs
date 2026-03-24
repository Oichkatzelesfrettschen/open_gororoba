//! VTK file writers for ParaView visualization of LBM simulation data.
//!
//! Pure Rust implementation -- no external VTK library dependency.
//! Generates standard ASCII format files directly loadable by ParaView.
//!
//! # Supported Formats
//!
//! | Format | Extension | Use Case |
//! |--------|-----------|----------|
//! | VTK Legacy STRUCTURED_POINTS | `.vtk` | 3D density/velocity field snapshots |
//! | VTP XML PolyData | `.vtp` | Particle trajectories (points + scalars) |
//! | PVD Collection | `.pvd` | Time-series animation index for ParaView |
//!
//! # Usage
//!
//! ```rust,ignore
//! // Write a density field snapshot
//! write_vtk_structured_points("field_000.vtk", 64, 64, 64,
//!     [0.0; 3], [1.0; 3], &rho, Some(&velocity))?;
//!
//! // Write particle trajectories
//! write_vtp_trajectories("particles_000.vtp", &trajectory_points)?;
//!
//! // Create time-series index
//! write_pvd_collection("simulation.pvd", &[(0.0, "field_000.vtk"), (0.01, "field_001.vtk")])?;
//! ```
//!
//! # ParaView Workflow
//!
//! 1. Open the `.pvd` file in ParaView for animated time-series
//! 2. Or open individual `.vtk` files for single snapshots
//! 3. Apply "Contour" filter for isosurfaces of density
//! 4. Apply "Glyph" filter for velocity arrows
//! 5. Use "Temporal Particles" filter on `.vtp` files for trajectory animation

use std::io::Write;

/// Grid parameters for VTK export.
#[derive(Debug, Clone)]
pub struct VtkGrid {
    /// Grid dimensions.
    pub dims: (usize, usize, usize),
    /// World-space origin.
    pub origin: [f64; 3],
    /// Cell spacing.
    pub spacing: [f64; 3],
}

/// Write a VTK Legacy STRUCTURED_POINTS file for a 3D scalar + vector field.
///
/// Creates a `.vtk` file loadable by ParaView as a structured grid.
/// For time series, name files as `field_000.vtk`, `field_001.vtk`, etc.
#[allow(clippy::too_many_arguments)]
pub fn write_vtk_structured_points(
    path: &str,
    grid: &VtkGrid,
    rho: &[f32],
    velocity: Option<&[f32]>, // SoA: [ux(N), uy(N), uz(N)] or None
) -> std::io::Result<()> {
    let (nx, ny, nz) = grid.dims;
    let origin = grid.origin;
    let spacing = grid.spacing;
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    let n = nx * ny * nz;
    assert_eq!(rho.len(), n);

    // Header
    writeln!(f, "# vtk DataFile Version 3.0")?;
    writeln!(f, "LBM simulation output")?;
    writeln!(f, "ASCII")?;
    writeln!(f, "DATASET STRUCTURED_POINTS")?;
    writeln!(f, "DIMENSIONS {} {} {}", nx, ny, nz)?;
    writeln!(f, "ORIGIN {} {} {}", origin[0], origin[1], origin[2])?;
    writeln!(f, "SPACING {} {} {}", spacing[0], spacing[1], spacing[2])?;
    writeln!(f, "POINT_DATA {n}")?;

    // Density scalar
    writeln!(f, "SCALARS rho float 1")?;
    writeln!(f, "LOOKUP_TABLE default")?;
    for val in rho {
        writeln!(f, "{val:.6}")?;
    }

    // Velocity vector (optional)
    if let Some(vel) = velocity {
        assert!(vel.len() >= 3 * n);
        writeln!(f, "VECTORS velocity float")?;
        for i in 0..n {
            writeln!(
                f,
                "{:.6e} {:.6e} {:.6e}",
                vel[i],         // ux
                vel[n + i],     // uy
                vel[2 * n + i], // uz
            )?;
        }
    }

    Ok(())
}

/// Write a VTP XML PolyData file for particle trajectories.
///
/// Each particle's trajectory is a polyline connecting its positions over time.
/// Creates a `.vtp` file loadable by ParaView's "Temporal Particles" filter.
/// A trajectory point for VTP export.
#[derive(Debug, Clone, Copy)]
pub struct VtpPoint {
    /// Simulation step.
    pub step: u64,
    /// Particle ID.
    pub pid: u32,
    /// Position (x, y, z).
    pub pos: [f32; 3],
    /// Velocity (vx, vy, vz).
    pub vel: [f32; 3],
}

pub fn write_vtp_trajectories(path: &str, trajectories: &[VtpPoint]) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    let n_points = trajectories.len();

    writeln!(f, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        f,
        r#"<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">"#
    )?;
    writeln!(f, r#"  <PolyData>"#)?;
    writeln!(
        f,
        r#"    <Piece NumberOfPoints="{n_points}" NumberOfVerts="{n_points}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">"#
    )?;

    // Points
    writeln!(f, r#"      <Points>"#)?;
    writeln!(
        f,
        r#"        <DataArray type="Float32" NumberOfComponents="3" format="ascii">"#
    )?;
    for pt in trajectories {
        writeln!(
            f,
            "          {:.6} {:.6} {:.6}",
            pt.pos[0], pt.pos[1], pt.pos[2]
        )?;
    }
    writeln!(f, r#"        </DataArray>"#)?;
    writeln!(f, r#"      </Points>"#)?;

    // Vertices (one per point for particle visualization)
    writeln!(f, r#"      <Verts>"#)?;
    writeln!(
        f,
        r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#
    )?;
    for i in 0..n_points {
        write!(f, "          {i}")?;
        if (i + 1) % 20 == 0 {
            writeln!(f)?;
        }
    }
    writeln!(f)?;
    writeln!(f, r#"        </DataArray>"#)?;
    writeln!(
        f,
        r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#
    )?;
    for i in 1..=n_points {
        write!(f, "          {i}")?;
        if i % 20 == 0 {
            writeln!(f)?;
        }
    }
    writeln!(f)?;
    writeln!(f, r#"        </DataArray>"#)?;
    writeln!(f, r#"      </Verts>"#)?;

    // Point data: velocity magnitude, time step, particle ID
    writeln!(f, r#"      <PointData Scalars="velocity_magnitude">"#)?;
    writeln!(
        f,
        r#"        <DataArray type="Float32" Name="velocity_magnitude" format="ascii">"#
    )?;
    for pt in trajectories {
        let vmag = (pt.vel[0] * pt.vel[0] + pt.vel[1] * pt.vel[1] + pt.vel[2] * pt.vel[2]).sqrt();
        writeln!(f, "          {vmag:.6e}")?;
    }
    writeln!(f, r#"        </DataArray>"#)?;
    writeln!(
        f,
        r#"        <DataArray type="Int32" Name="particle_id" format="ascii">"#
    )?;
    for pt in trajectories {
        writeln!(f, "          {}", pt.pid)?;
    }
    writeln!(f, r#"        </DataArray>"#)?;
    writeln!(
        f,
        r#"        <DataArray type="Int32" Name="time_step" format="ascii">"#
    )?;
    for pt in trajectories {
        writeln!(f, "          {}", pt.step)?;
    }
    writeln!(f, r#"        </DataArray>"#)?;

    // Velocity vectors
    writeln!(
        f,
        r#"        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">"#
    )?;
    for pt in trajectories {
        writeln!(
            f,
            "          {:.6e} {:.6e} {:.6e}",
            pt.vel[0], pt.vel[1], pt.vel[2]
        )?;
    }
    writeln!(f, r#"        </DataArray>"#)?;

    writeln!(f, r#"      </PointData>"#)?;
    writeln!(f, r#"    </Piece>"#)?;
    writeln!(f, r#"  </PolyData>"#)?;
    writeln!(f, r#"</VTKFile>"#)?;

    Ok(())
}

/// Write a ParaView Data (.pvd) collection file for time-series VTP files.
///
/// Creates a `.pvd` file that references a sequence of `.vtp` files,
/// allowing ParaView to animate the particle trajectories.
pub fn write_pvd_collection(
    path: &str,
    vtp_files: &[(f64, &str)], // (timestep, filename)
) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;

    writeln!(f, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        f,
        r#"<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">"#
    )?;
    writeln!(f, r#"  <Collection>"#)?;
    for (time, filename) in vtp_files {
        writeln!(f, r#"    <DataSet timestep="{time}" file="{filename}"/>"#)?;
    }
    writeln!(f, r#"  </Collection>"#)?;
    writeln!(f, r#"</VTKFile>"#)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_vtk_structured() {
        let path = "/tmp/test_lbm_field.vtk";
        let grid = VtkGrid {
            dims: (2, 2, 2),
            origin: [0.0; 3],
            spacing: [1.0; 3],
        };
        let rho = vec![1.0f32; 8];
        write_vtk_structured_points(path, &grid, &rho, None).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("STRUCTURED_POINTS"));
        assert!(content.contains("DIMENSIONS 2 2 2"));
        assert!(content.contains("SCALARS rho float"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_vtp_trajectories() {
        let path = "/tmp/test_particles.vtp";
        let traj = vec![
            VtpPoint {
                step: 0,
                pid: 0,
                pos: [0.0, 0.0, 0.0],
                vel: [0.1, 0.0, 0.0],
            },
            VtpPoint {
                step: 0,
                pid: 1,
                pos: [1.0, 0.0, 0.0],
                vel: [0.0, 0.1, 0.0],
            },
            VtpPoint {
                step: 1,
                pid: 0,
                pos: [0.1, 0.0, 0.0],
                vel: [0.1, 0.0, 0.0],
            },
            VtpPoint {
                step: 1,
                pid: 1,
                pos: [1.0, 0.1, 0.0],
                vel: [0.0, 0.1, 0.0],
            },
        ];
        write_vtp_trajectories(path, &traj).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("PolyData"));
        assert!(content.contains("NumberOfPoints=\"4\""));
        assert!(content.contains("velocity_magnitude"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_pvd_collection() {
        let path = "/tmp/test_collection.pvd";
        let files = vec![(0.0, "particles_000.vtp"), (0.001, "particles_001.vtp")];
        write_pvd_collection(path, &files).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("Collection"));
        assert!(content.contains("timestep=\"0\""));
        assert!(content.contains("particles_001.vtp"));
        std::fs::remove_file(path).ok();
    }
}
