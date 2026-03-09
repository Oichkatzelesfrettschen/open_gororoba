//! CSV snapshot output for cosmic ray phase-space distributions.

use std::fmt::Write as FmtWrite;

use crate::solver::PteSolver;

/// Write a snapshot CSV string for a PTE solver state.
///
/// Columns: step, x, y, z, rigidity_gv, flux_J, modulation_phi
///
/// Only writes the mid-y, mid-z plane for compact output.
/// `phi_map` is a flat array of modulation potentials, length nx (one per x-slice).
pub fn write_snapshot_csv(solver: &PteSolver, phi_map: &[f64], step: usize) -> String {
    let mut out = String::new();
    writeln!(out, "step,x,y,z,rigidity_gv,flux_J,modulation_phi").unwrap();

    let y_mid = solver.ny / 2;
    let z_mid = solver.nz / 2;

    for x in 0..solver.nx {
        let phi = if x < phi_map.len() { phi_map[x] } else { 0.0 };
        let flux = solver.flux_at(x, y_mid, z_mid);
        for (p, &j) in flux.iter().enumerate() {
            let r_gv = solver.grid.rigidity(p);
            writeln!(out, "{step},{x},{y_mid},{z_mid},{r_gv:.6e},{j:.6e},{phi:.6e}").unwrap();
        }
    }
    out
}
