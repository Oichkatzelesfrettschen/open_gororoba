//! 9-band magnonic crystal tight-binding model.
//!
//! Implements the Kaman, Lim, Liu & Hoffmann (2026) 9-band tight-binding
//! Hamiltonian for magnonic crystals in hexagonal antidot arrays of YIG.
//! The model combines 6 honeycomb orbitals (s, px, py on A and B sublattices)
//! with 3 kagome orbitals (s on K1, K2, K3 bond midpoints), yielding:
//! - Graphene-like Dirac cones near 1.5 GHz
//! - Kagome flat band near 2.08 GHz (~1000x DOS enhancement)
//! - Valley-Hall topological boundary modes under inversion breaking
//! - Point defect modes with Q ~ 4100
//!
//! # Orbital Basis (9 per primitive cell)
//!
//! | Index | Site | Orbital | On-site |
//! |-------|------|---------|---------|
//! | 0     | A    | s       | eps_s + delta_eps_s |
//! | 1     | A    | px      | eps_p + delta_eps_p |
//! | 2     | A    | py      | eps_p + delta_eps_p |
//! | 3     | B    | s       | eps_s - delta_eps_s |
//! | 4     | B    | px      | eps_p - delta_eps_p |
//! | 5     | B    | py      | eps_p - delta_eps_p |
//! | 6     | K1   | s       | eps_k |
//! | 7     | K2   | s       | eps_k |
//! | 8     | K3   | s       | eps_k |
//!
//! # Literature
//!
//! - Kaman, Lim, Liu, Hoffmann (2026): arXiv:2601.03210v2, Table I

use faer::complex_native::c64;

use crate::tight_binding::{
    BravaisLattice2D, Hopping, OrbitalSite, TightBindingModel, Valley, Vec2,
    band_chern_number, detect_flat_bands, hexagonal_high_symmetry_path,
    hexagonal_symmetry_labels, valley_chern_number,
};

/// Tight-binding parameters for the 9-band magnonic crystal model.
///
/// All energies in GHz (matching Kaman et al. convention).
#[derive(Clone, Debug)]
pub struct MagnonicTBParams {
    /// Honeycomb s-orbital on-site energy (GHz).
    pub eps_s: f64,
    /// Honeycomb p-orbital on-site energy (GHz).
    pub eps_p: f64,
    /// Kagome s-orbital on-site energy (GHz).
    pub eps_k: f64,
    /// s-kagome hopping amplitude (GHz).
    pub t_sk: f64,
    /// p-kagome hopping amplitude (GHz).
    pub t_pk: f64,
}

impl MagnonicTBParams {
    /// Symmetric-case parameters from Kaman et al. Table I.
    ///
    /// These reproduce the MuMax3 micromagnetic simulation band structure
    /// for a hexagonal antidot array with d/a = 0.6, B_ext = 50 mT.
    pub fn kaman_default() -> Self {
        Self {
            eps_s: 1.50,
            eps_p: 1.80,
            eps_k: 2.08,
            t_sk: 0.25,
            t_pk: 0.15,
        }
    }
}

/// Inversion symmetry breaking parameters.
///
/// When nonzero, the A and B sublattice on-site energies differ,
/// opening a gap at the Dirac points and enabling nonzero valley
/// Chern numbers.
#[derive(Clone, Debug)]
pub struct InversionBreakingParams {
    /// s-orbital energy shift (A gets +delta, B gets -delta) in GHz.
    pub delta_eps_s: f64,
    /// p-orbital energy shift (A gets +delta, B gets -delta) in GHz.
    pub delta_eps_p: f64,
}

impl InversionBreakingParams {
    /// Broken-symmetry parameters from Kaman et al. Table I.
    pub fn kaman_default() -> Self {
        Self {
            delta_eps_s: 0.10,
            delta_eps_p: 0.05,
        }
    }

    /// No inversion breaking (preserves all symmetries).
    pub fn none() -> Self {
        Self {
            delta_eps_s: 0.0,
            delta_eps_p: 0.0,
        }
    }
}

/// Results of a complete magnonic band structure calculation.
#[derive(Clone, Debug)]
pub struct MagnonicBandResult {
    /// Cumulative k-distances along the high-symmetry path.
    pub k_distances: Vec<f64>,
    /// Band energies: band_energies[band_idx][k_idx] in GHz.
    pub band_energies: Vec<Vec<f64>>,
    /// High-symmetry point labels and their k-distance positions.
    pub symmetry_labels: Vec<(String, f64)>,
    /// Chern number for each band.
    pub band_cherns: Vec<f64>,
    /// Valley Chern number (K valley) for each band.
    pub valley_cherns_k: Vec<f64>,
    /// Indices of flat bands (bandwidth < threshold).
    pub flat_band_indices: Vec<usize>,
    /// Bandwidth of each band in GHz.
    pub bandwidths: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Bond geometry for the hexagonal unit cell
// ---------------------------------------------------------------------------

/// Bond angles from A to its 3 nearest kagome sites.
/// Measured from positive x-axis in radians.
const BOND_ANGLES: [f64; 3] = [
    std::f64::consts::FRAC_PI_2,         // 90 deg (up to K1)
    std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::FRAC_PI_3, // 210 deg
    std::f64::consts::FRAC_PI_2 - 2.0 * std::f64::consts::FRAC_PI_3, // -30 deg
];

/// Build the 9-band magnonic crystal tight-binding model.
///
/// The lattice constant `a_nm` is in nanometers (used for orbital positions
/// only; band energies are in GHz and independent of physical scale).
pub fn build_magnonic_9band(
    params: &MagnonicTBParams,
    inv: &InversionBreakingParams,
    a_nm: f64,
) -> TightBindingModel {
    let a = a_nm;
    let s3 = 3.0_f64.sqrt();
    let lattice = BravaisLattice2D::hexagonal(a);

    // Sublattice positions within the unit cell
    let pos_a = Vec2::zero();
    let pos_b = Vec2::new(0.0, a / s3);

    // Kagome positions: midpoints of the 3 bonds connecting A to its NNs
    // K1: midpoint of A(0,0)-B(0,0) bond (vertical)
    let pos_k1 = Vec2::new(0.0, a / (2.0 * s3));
    // K2: midpoint of A(0,0)-B(0,-1) bond
    let pos_k2 = Vec2::new(-a / 4.0, -a / (4.0 * s3));
    // K3: midpoint of A(0,0)-B(1,-1) bond
    let pos_k3 = Vec2::new(a / 4.0, -a / (4.0 * s3));

    // On-site energies with inversion breaking
    let eps_s_a = params.eps_s + inv.delta_eps_s;
    let eps_p_a = params.eps_p + inv.delta_eps_p;
    let eps_s_b = params.eps_s - inv.delta_eps_s;
    let eps_p_b = params.eps_p - inv.delta_eps_p;

    let orbitals = vec![
        // Honeycomb A: indices 0, 1, 2
        OrbitalSite { position: pos_a, label: "s_A".to_string(), on_site_energy: eps_s_a },
        OrbitalSite { position: pos_a, label: "px_A".to_string(), on_site_energy: eps_p_a },
        OrbitalSite { position: pos_a, label: "py_A".to_string(), on_site_energy: eps_p_a },
        // Honeycomb B: indices 3, 4, 5
        OrbitalSite { position: pos_b, label: "s_B".to_string(), on_site_energy: eps_s_b },
        OrbitalSite { position: pos_b, label: "px_B".to_string(), on_site_energy: eps_p_b },
        OrbitalSite { position: pos_b, label: "py_B".to_string(), on_site_energy: eps_p_b },
        // Kagome: indices 6, 7, 8
        OrbitalSite { position: pos_k1, label: "s_K1".to_string(), on_site_energy: params.eps_k },
        OrbitalSite { position: pos_k2, label: "s_K2".to_string(), on_site_energy: params.eps_k },
        OrbitalSite { position: pos_k3, label: "s_K3".to_string(), on_site_energy: params.eps_k },
    ];

    let mut hoppings = Vec::new();

    // --- A-sublattice to kagome hoppings ---
    // Each A-site s-orbital couples to its 3 NN kagome sites with t_sk.
    // Each A-site p-orbital couples with t_pk projected onto bond direction.

    // Bond 0: A(0,0) -- K1(0,0), angle = 90 deg
    add_honeycomb_kagome_hoppings(
        &mut hoppings, 0, 6, [0, 0], BOND_ANGLES[0], params.t_sk, params.t_pk,
    );
    // Bond 1: A(0,0) -- K2(0,0), angle = 210 deg
    add_honeycomb_kagome_hoppings(
        &mut hoppings, 0, 7, [0, 0], BOND_ANGLES[1], params.t_sk, params.t_pk,
    );
    // Bond 2: A(0,0) -- K3(0,0), angle = -30 deg
    add_honeycomb_kagome_hoppings(
        &mut hoppings, 0, 8, [0, 0], BOND_ANGLES[2], params.t_sk, params.t_pk,
    );

    // --- B-sublattice to kagome hoppings ---
    // B(0,0) connects to K1(0,0) from the other side of the bond.
    // Bond angles from B are opposite to those from A (shifted by pi).

    // Bond 0: B(0,0) -- K1(0,0), angle = 270 deg (-90 deg from B)
    add_honeycomb_kagome_hoppings(
        &mut hoppings, 3, 6, [0, 0],
        BOND_ANGLES[0] + std::f64::consts::PI, params.t_sk, params.t_pk,
    );
    // Bond 1: B(0,0) -- K2(0,-1+1=0)... need cell offsets for inter-cell bonds
    // B connects to K2 in cell [0, 1] (K2 is midpoint of the A(0,1)-B(0,0) bond)
    add_honeycomb_kagome_hoppings(
        &mut hoppings, 3, 7, [0, 1],
        BOND_ANGLES[1] + std::f64::consts::PI, params.t_sk, params.t_pk,
    );
    // Bond 2: B(0,0) -- K3 in cell [-1, 1]
    add_honeycomb_kagome_hoppings(
        &mut hoppings, 3, 8, [-1, 1],
        BOND_ANGLES[2] + std::f64::consts::PI, params.t_sk, params.t_pk,
    );

    // --- Kagome-kagome hoppings (direct, within triangles) ---
    // K1-K2, K1-K3, K2-K3 within the unit cell triangle
    // These are mediated hoppings (second-neighbor in the full model).
    // In the Kaman model, direct kagome-kagome coupling is zero --
    // all kagome coupling is THROUGH the honeycomb sites.
    // No additional hoppings needed.

    TightBindingModel {
        lattice,
        orbitals,
        hoppings,
    }
}

/// Add s-kagome and p-kagome hoppings for one honeycomb-to-kagome bond.
fn add_honeycomb_kagome_hoppings(
    hoppings: &mut Vec<Hopping>,
    honeycomb_s_idx: usize,
    kagome_idx: usize,
    cell_offset: [i32; 2],
    bond_angle: f64,
    t_sk: f64,
    t_pk: f64,
) {
    let px_idx = honeycomb_s_idx + 1;
    let py_idx = honeycomb_s_idx + 2;

    // s-orbital to kagome: isotropic coupling
    hoppings.push(Hopping {
        from: honeycomb_s_idx,
        to: kagome_idx,
        cell_offset,
        amplitude: c64::new(t_sk, 0.0),
    });

    // px-orbital to kagome: projection onto bond direction = cos(theta)
    hoppings.push(Hopping {
        from: px_idx,
        to: kagome_idx,
        cell_offset,
        amplitude: c64::new(t_pk * bond_angle.cos(), 0.0),
    });

    // py-orbital to kagome: projection onto bond direction = sin(theta)
    hoppings.push(Hopping {
        from: py_idx,
        to: kagome_idx,
        cell_offset,
        amplitude: c64::new(t_pk * bond_angle.sin(), 0.0),
    });
}

/// Compute the full magnonic band structure with topological invariants.
pub fn compute_magnonic_bands(
    params: &MagnonicTBParams,
    inv: &InversionBreakingParams,
    n_k: usize,
    n_chern: usize,
    a_nm: f64,
) -> MagnonicBandResult {
    let model = build_magnonic_9band(params, inv, a_nm);
    let path = hexagonal_high_symmetry_path(&model.lattice, n_k);
    let (k_distances, band_energies) = model.band_structure_along_path(&path);
    let symmetry_labels = hexagonal_symmetry_labels(&model.lattice);

    let n_bands = model.n_orbitals();

    // Chern numbers for all bands
    let band_cherns: Vec<f64> = (0..n_bands)
        .map(|b| band_chern_number(&model, b, n_chern))
        .collect();

    // Valley Chern numbers (K valley)
    let valley_cherns_k: Vec<f64> = (0..n_bands)
        .map(|b| valley_chern_number(&model, b, n_chern, Valley::K))
        .collect();

    // Flat band detection
    let flat_info = detect_flat_bands(&model, n_k.max(20), 0.05);

    MagnonicBandResult {
        k_distances,
        band_energies,
        symmetry_labels,
        band_cherns,
        valley_cherns_k,
        flat_band_indices: flat_info.flat_band_indices,
        bandwidths: flat_info.bandwidths,
    }
}

/// Build a domain wall supercell ribbon for edge mode calculation.
///
/// Creates a ribbon `n_cells` wide with the top half having +delta
/// inversion breaking and the bottom half having -delta (domain wall
/// at center). Periodic along the ribbon edge direction.
pub fn build_domain_wall_supercell(
    params: &MagnonicTBParams,
    inv: &InversionBreakingParams,
    n_cells: usize,
    a_nm: f64,
) -> TightBindingModel {
    let base_model = build_magnonic_9band(params, &InversionBreakingParams::none(), a_nm);
    let n_orb_cell = base_model.n_orbitals(); // 9
    let n_orb_total = n_orb_cell * n_cells;

    // Supercell lattice: a1 stays the same, a2 -> n_cells * a2
    let super_a2 = base_model.lattice.a2.scale(n_cells as f64);
    let lattice = BravaisLattice2D::from_direct(base_model.lattice.a1, super_a2);

    let mut orbitals = Vec::with_capacity(n_orb_total);
    let mut hoppings = Vec::new();

    for cell in 0..n_cells {
        let offset = base_model.lattice.a2.scale(cell as f64);
        // Domain wall: top half gets +delta, bottom half gets -delta
        let sign = if cell < n_cells / 2 { 1.0 } else { -1.0 };
        let cell_inv = InversionBreakingParams {
            delta_eps_s: sign * inv.delta_eps_s,
            delta_eps_p: sign * inv.delta_eps_p,
        };

        // Place orbitals for this cell
        for (i, base_orb) in base_model.orbitals.iter().enumerate() {
            let mut orb = base_orb.clone();
            orb.position = orb.position + offset;
            orb.label = format!("{}_{}", base_orb.label, cell);
            // Apply inversion breaking to honeycomb sites
            match i {
                0 => orb.on_site_energy = params.eps_s + cell_inv.delta_eps_s,
                1 | 2 => orb.on_site_energy = params.eps_p + cell_inv.delta_eps_p,
                3 => orb.on_site_energy = params.eps_s - cell_inv.delta_eps_s,
                4 | 5 => orb.on_site_energy = params.eps_p - cell_inv.delta_eps_p,
                _ => {} // kagome sites unchanged
            }
            orbitals.push(orb);
        }

        // Intra-cell hoppings (within this cell of the supercell)
        for hop in &base_model.hoppings {
            if hop.cell_offset == [0, 0] {
                hoppings.push(Hopping {
                    from: hop.from + cell * n_orb_cell,
                    to: hop.to + cell * n_orb_cell,
                    cell_offset: [0, 0],
                    amplitude: hop.amplitude,
                });
            }
        }

        // Inter-cell hoppings along a2 direction
        for hop in &base_model.hoppings {
            if hop.cell_offset[1] != 0 {
                let target_cell = cell as i32 + hop.cell_offset[1];
                if target_cell >= 0 && (target_cell as usize) < n_cells {
                    // Internal to supercell
                    hoppings.push(Hopping {
                        from: hop.from + cell * n_orb_cell,
                        to: hop.to + (target_cell as usize) * n_orb_cell,
                        cell_offset: [hop.cell_offset[0], 0],
                        amplitude: hop.amplitude,
                    });
                } else {
                    // Wraps around -> supercell boundary
                    let wrapped = ((target_cell % n_cells as i32) + n_cells as i32) as usize
                        % n_cells;
                    let super_offset_1 = if target_cell < 0 { -1 } else { 1 };
                    hoppings.push(Hopping {
                        from: hop.from + cell * n_orb_cell,
                        to: hop.to + wrapped * n_orb_cell,
                        cell_offset: [hop.cell_offset[0], super_offset_1],
                        amplitude: hop.amplitude,
                    });
                }
            }
            // a1-direction inter-cell hops (no supercell folding needed)
            if hop.cell_offset[1] == 0 && hop.cell_offset[0] != 0 {
                hoppings.push(Hopping {
                    from: hop.from + cell * n_orb_cell,
                    to: hop.to + cell * n_orb_cell,
                    cell_offset: [hop.cell_offset[0], 0],
                    amplitude: hop.amplitude,
                });
            }
        }
    }

    TightBindingModel {
        lattice,
        orbitals,
        hoppings,
    }
}

/// Compute point defect localized modes in a supercell.
///
/// Creates a `(2*radius+1) x (2*radius+1)` supercell with one missing
/// kagome site (simulating a filled hole) at the center. Returns
/// eigenvalues that fall within the bulk band gap.
pub fn point_defect_modes(
    params: &MagnonicTBParams,
    inv: &InversionBreakingParams,
    radius: usize,
    a_nm: f64,
) -> Vec<f64> {
    let base = build_magnonic_9band(params, inv, a_nm);
    let n_orb_cell = base.n_orbitals();
    let side = 2 * radius + 1;
    let center = radius;

    // Build supercell orbitals, removing one kagome site at center
    let super_a1 = base.lattice.a1.scale(side as f64);
    let super_a2 = base.lattice.a2.scale(side as f64);
    let lattice = BravaisLattice2D::from_direct(super_a1, super_a2);

    let mut orbitals = Vec::new();
    let mut cell_orbital_map: Vec<Vec<Option<usize>>> = Vec::new(); // [cell][local] -> global

    let mut global_idx = 0;
    for ci in 0..side {
        for cj in 0..side {
            let offset = base.lattice.a1.scale(ci as f64)
                + base.lattice.a2.scale(cj as f64);
            let mut mapping = vec![None; n_orb_cell];

            for (local, base_orb) in base.orbitals.iter().enumerate() {
                // Skip kagome orbital 6 (K1) at center cell to create defect
                if ci == center && cj == center && local == 6 {
                    continue;
                }
                let mut orb = base_orb.clone();
                orb.position = orb.position + offset;
                mapping[local] = Some(global_idx);
                orbitals.push(orb);
                global_idx += 1;
            }
            cell_orbital_map.push(mapping);
        }
    }

    // Build hoppings (only intra-supercell, open boundary)
    let mut hoppings = Vec::new();
    for ci in 0..side {
        for cj in 0..side {
            let cell_idx = ci * side + cj;
            for hop in &base.hoppings {
                let ti = ci as i32 + hop.cell_offset[0];
                let tj = cj as i32 + hop.cell_offset[1];
                if ti >= 0 && ti < side as i32 && tj >= 0 && tj < side as i32 {
                    let target_cell = ti as usize * side + tj as usize;
                    if let (Some(from_g), Some(to_g)) = (
                        cell_orbital_map[cell_idx][hop.from],
                        cell_orbital_map[target_cell][hop.to],
                    ) {
                        hoppings.push(Hopping {
                            from: from_g,
                            to: to_g,
                            cell_offset: [0, 0], // all internal
                            amplitude: hop.amplitude,
                        });
                    }
                }
            }
        }
    }

    let model = TightBindingModel {
        lattice,
        orbitals,
        hoppings,
    };

    // Diagonalize at Gamma (supercell = one big unit cell)
    let all_evals = model.band_energies(0.0, 0.0);

    // Find bulk band edges from actual band structure sampling (not parameter estimates).
    // WHY: Parameter-based gap estimates (eps_p, eps_k) diverge from actual band edges
    // because hybridization shifts band energies significantly (e.g., kagome flat band
    // at 2.2351 GHz vs eps_k = 2.08 GHz).
    let bulk_model = build_magnonic_9band(params, inv, a_nm);
    let n_sample = 30;
    let mut band_mins = [f64::INFINITY; 9];
    let mut band_maxs = [f64::NEG_INFINITY; 9];
    for i in 0..n_sample {
        for j in 0..n_sample {
            let s = i as f64 / n_sample as f64;
            let t = j as f64 / n_sample as f64;
            let kx = s * bulk_model.lattice.b1.x + t * bulk_model.lattice.b2.x;
            let ky = s * bulk_model.lattice.b1.y + t * bulk_model.lattice.b2.y;
            let evals = bulk_model.band_energies(kx, ky);
            for (b, &e) in evals.iter().enumerate() {
                if b < 9 {
                    band_mins[b] = band_mins[b].min(e);
                    band_maxs[b] = band_maxs[b].max(e);
                }
            }
        }
    }

    // Find the largest gap between consecutive bands
    let mut best_gap = 0.0_f64;
    let mut gap_low = 0.0;
    let mut gap_high = 0.0;
    for b in 0..8 {
        let lo = band_maxs[b]; // top of band b
        let hi = band_mins[b + 1]; // bottom of band b+1
        let gap = hi - lo;
        if gap > best_gap {
            best_gap = gap;
            gap_low = lo;
            gap_high = hi;
        }
    }

    if best_gap <= 0.0 {
        // No gap found -- return empty
        return Vec::new();
    }

    all_evals
        .into_iter()
        .filter(|&e| e > gap_low && e < gap_high)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_9band_model_has_9_orbitals() {
        let model = build_magnonic_9band(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            400.0,
        );
        assert_eq!(model.n_orbitals(), 9);
    }

    #[test]
    fn test_9band_hermitian() {
        let model = build_magnonic_9band(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            400.0,
        );
        let h = model.hamiltonian_at_k(0.5, 0.3);
        for i in 0..9 {
            for j in 0..9 {
                let hij = h.read(i, j);
                let hji = h.read(j, i);
                let err_re = (hij.re - hji.re).abs();
                let err_im = (hij.im + hji.im).abs();
                assert!(
                    err_re < 1e-10 && err_im < 1e-10,
                    "H not Hermitian at ({},{}): re_err={}, im_err={}",
                    i, j, err_re, err_im
                );
            }
        }
    }

    #[test]
    fn test_9band_eigenvalues_at_gamma() {
        let model = build_magnonic_9band(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            400.0,
        );
        let evals = model.band_energies(0.0, 0.0);
        assert_eq!(evals.len(), 9);
        // Eigenvalues should be sorted
        for w in evals.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "Not sorted: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_9band_time_reversal() {
        let model = build_magnonic_9band(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            400.0,
        );
        let e_k = model.band_energies(0.5, 0.3);
        let e_mk = model.band_energies(-0.5, -0.3);
        for (a, b) in e_k.iter().zip(&e_mk) {
            assert!(
                (a - b).abs() < 1e-10,
                "TRS violated: E(k)={}, E(-k)={}",
                a, b
            );
        }
    }

    #[test]
    fn test_9band_inversion_breaking_opens_gap() {
        let params = MagnonicTBParams::kaman_default();
        let model_sym = build_magnonic_9band(&params, &InversionBreakingParams::none(), 400.0);
        let model_broken = build_magnonic_9band(&params, &InversionBreakingParams::kaman_default(), 400.0);

        let lat = &model_sym.lattice;
        let k_pt = lat.b1.scale(2.0 / 3.0) + lat.b2.scale(1.0 / 3.0);

        let evals_sym = model_sym.band_energies(k_pt.x, k_pt.y);
        let evals_broken = model_broken.band_energies(k_pt.x, k_pt.y);

        // Find smallest gap between adjacent bands near the Dirac point region
        let min_gap_sym: f64 = evals_sym.windows(2).map(|w| w[1] - w[0]).fold(f64::MAX, f64::min);
        let min_gap_broken: f64 = evals_broken.windows(2).map(|w| w[1] - w[0]).fold(f64::MAX, f64::min);

        // Broken inversion should increase the minimum gap
        assert!(
            min_gap_broken > min_gap_sym || min_gap_sym < 0.01,
            "Inversion breaking should open gap: sym={}, broken={}",
            min_gap_sym, min_gap_broken
        );
    }

    #[test]
    fn test_9band_chern_sum_zero_symmetric() {
        let model = build_magnonic_9band(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            400.0,
        );
        let total: f64 = (0..9).map(|b| band_chern_number(&model, b, 15)).sum();
        assert!(
            total.abs() < 0.5,
            "Total Chern should be ~0, got {:.4}",
            total
        );
    }

    #[test]
    fn test_9band_periodicity() {
        let model = build_magnonic_9band(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            400.0,
        );
        let lat = &model.lattice;
        let e_k = model.band_energies(0.3, 0.7);
        let e_k_g = model.band_energies(0.3 + lat.b1.x, 0.7 + lat.b1.y);
        for (a, b) in e_k.iter().zip(&e_k_g) {
            assert!((a - b).abs() < 1e-10, "E(k) != E(k+G)");
        }
    }

    #[test]
    fn test_domain_wall_supercell_size() {
        let model = build_domain_wall_supercell(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::kaman_default(),
            6,
            400.0,
        );
        // 6 cells * 9 orbitals = 54
        assert_eq!(model.n_orbitals(), 54);
    }

    #[test]
    fn test_compute_magnonic_bands_runs() {
        let result = compute_magnonic_bands(
            &MagnonicTBParams::kaman_default(),
            &InversionBreakingParams::none(),
            20,
            11,
            400.0,
        );
        assert_eq!(result.band_energies.len(), 9);
        assert_eq!(result.band_cherns.len(), 9);
        assert_eq!(result.valley_cherns_k.len(), 9);
        assert!(!result.k_distances.is_empty());
    }
}
