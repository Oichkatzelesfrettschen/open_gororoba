//! Bridge between lattice_filtration and LBM velocity fields.
//!
//! Converts 3D LBM velocity fields into inputs suitable for the
//! filtration collision storm simulator. This enables Thesis 2
//! (particle mass from filtration) to receive velocity data from
//! Thesis 1 (frustration-viscosity coupling via LBM).

use crate::basis_index::project_to_lattice;
use crate::patricia_trie::PatriciaIndex;
use crate::survival_spectrum::{classify_latency_law, radial_bins, LatencyLaw, SpectrumBin};

/// Summary of a velocity-field-to-filtration conversion.
#[derive(Debug, Clone)]
pub struct FiltrationFromVelocity {
    /// Number of nonzero velocity cells processed
    pub n_active_cells: usize,
    /// Total grid cells (nx * ny * nz)
    pub n_total_cells: usize,
    /// Radial bins from the converted field
    pub spectrum_bins: Vec<SpectrumBin>,
    /// Classified latency law of the converted field
    pub latency_law: LatencyLaw,
    /// PatriciaIndex containing all encoded lattice keys
    pub trie: PatriciaIndex,
}

/// Convert a 3D velocity field into filtration inputs.
///
/// Each velocity vector [ux, uy, uz] at grid point (x, y, z) is mapped to
/// a 16D sedenion sample by interleaving coordinates and velocity components,
/// then projected to an 8D lattice vector via `project_to_lattice`. The
/// lattice keys are inserted into a PatriciaIndex for survival depth analysis.
///
/// The radius is the Euclidean norm of the 8D lattice vector, and the
/// "latency" is the velocity magnitude (serves as a proxy for collision time).
///
/// # Arguments
/// * `velocity_field` - Flat array of [ux, uy, uz] triples, one per grid point
/// * `nx`, `ny`, `nz` - Grid dimensions
/// * `scale` - Projection scale factor (controls lattice resolution)
/// * `n_bins` - Number of radial bins for spectrum analysis
pub fn filtration_from_velocity_field(
    velocity_field: &[[f64; 3]],
    nx: usize,
    ny: usize,
    nz: usize,
    scale: f64,
    n_bins: usize,
) -> FiltrationFromVelocity {
    let n_total = nx * ny * nz;
    assert_eq!(
        velocity_field.len(),
        n_total,
        "velocity_field length must match nx*ny*nz"
    );

    let mut trie = PatriciaIndex::new();
    let mut samples = Vec::with_capacity(n_total);
    let mut n_active = 0usize;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let vel = velocity_field[idx];
                let vel_mag = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();

                if vel_mag < 1e-12 {
                    continue;
                }
                n_active += 1;

                // Build a 16D sample by interleaving spatial and velocity info.
                // Low 8 dims: normalized coordinates + velocity components
                // High 8 dims: velocity gradients proxy (velocity * position)
                let xn = x as f64 / nx.max(1) as f64;
                let yn = y as f64 / ny.max(1) as f64;
                let zn = z as f64 / nz.max(1) as f64;

                let mut sample = [0.0f64; 16];
                sample[0] = xn;
                sample[1] = yn;
                sample[2] = zn;
                sample[3] = vel[0];
                sample[4] = vel[1];
                sample[5] = vel[2];
                sample[6] = vel_mag;
                sample[7] = xn * vel[0]; // vorticity proxy
                sample[8] = yn * vel[1];
                sample[9] = zn * vel[2];
                sample[10] = vel[0] * vel[1];
                sample[11] = vel[1] * vel[2];
                sample[12] = vel[0] * vel[2];
                sample[13] = vel_mag * xn;
                sample[14] = vel_mag * yn;
                sample[15] = vel_mag * zn;

                let lattice_key = project_to_lattice(&sample, scale);

                // Compute 8D radius
                let radius_sq: f64 = lattice_key.iter().map(|&k| (k as f64).powi(2)).sum();
                let radius = radius_sq.sqrt().max(1.0);

                // Encode first lattice component as trie key
                let trie_key = lattice_key[0] as u64;
                trie.insert(trie_key.wrapping_mul(31).wrapping_add(lattice_key[1] as u64));

                // (radius, latency=vel_mag) pair for spectrum analysis
                samples.push((radius, vel_mag));
            }
        }
    }

    let spectrum_bins = radial_bins(&samples, n_bins);
    let latency_law = classify_latency_law(&samples);

    FiltrationFromVelocity {
        n_active_cells: n_active,
        n_total_cells: n_total,
        spectrum_bins,
        latency_law,
        trie,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_velocity_field(nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
        let n = nx * ny * nz;
        let mut field = vec![[0.0; 3]; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + nx * (y + ny * z);
                    let yn = y as f64 / ny as f64;
                    // Parabolic-ish profile in y
                    field[idx] = [yn * (1.0 - yn), 0.0, 0.0];
                }
            }
        }
        field
    }

    #[test]
    fn test_filtration_from_velocity_counts() {
        let (nx, ny, nz) = (4, 8, 4);
        let field = make_simple_velocity_field(nx, ny, nz);
        let result = filtration_from_velocity_field(&field, nx, ny, nz, 10.0, 8);
        assert_eq!(result.n_total_cells, 4 * 8 * 4);
        assert!(result.n_active_cells > 0);
        assert!(result.n_active_cells <= result.n_total_cells);
    }

    #[test]
    fn test_filtration_spectrum_bins_nonempty() {
        let (nx, ny, nz) = (4, 8, 4);
        let field = make_simple_velocity_field(nx, ny, nz);
        let result = filtration_from_velocity_field(&field, nx, ny, nz, 10.0, 8);
        assert!(
            !result.spectrum_bins.is_empty(),
            "Should produce spectrum bins"
        );
    }

    #[test]
    fn test_filtration_zero_velocity_field() {
        let field = vec![[0.0; 3]; 64];
        let result = filtration_from_velocity_field(&field, 4, 4, 4, 10.0, 4);
        assert_eq!(result.n_active_cells, 0);
        assert!(result.spectrum_bins.is_empty());
    }

    #[test]
    fn test_filtration_latency_law_determined() {
        let (nx, ny, nz) = (8, 8, 8);
        let field = make_simple_velocity_field(nx, ny, nz);
        let result = filtration_from_velocity_field(&field, nx, ny, nz, 10.0, 10);
        // Just verify it returns a valid variant
        match result.latency_law {
            LatencyLaw::InverseSquare
            | LatencyLaw::PowerLaw
            | LatencyLaw::Linear
            | LatencyLaw::Exponential
            | LatencyLaw::Uniform
            | LatencyLaw::Undetermined => {}
        }
    }

    #[test]
    #[should_panic(expected = "velocity_field length must match")]
    fn test_filtration_dimension_mismatch() {
        let field = vec![[1.0, 0.0, 0.0]; 10];
        let _ = filtration_from_velocity_field(&field, 4, 4, 4, 10.0, 4);
    }
}
