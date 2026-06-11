//! Generic 2D tight-binding framework with topological invariants.
//!
//! Provides a model-independent tight-binding engine for computing band
//! structures, Berry curvature, Chern numbers, and valley Chern numbers
//! on arbitrary 2D Bravais lattices. Generalizes the FHS (Fukui-Hatsugai-
//! Suzuki) algorithm from the square-lattice Harper-Hofstadter model
//! (see `harper_chern.rs`) to hexagonal, kagome, and hybrid lattices.
//!
//! # Physics Background
//!
//! The Bloch Hamiltonian for a periodic 2D system with n_orb orbitals per
//! unit cell is:
//!
//!   H_{ij}(k) = eps_i delta_{ij} + sum_R t_{ij}(R) exp(i k.R)
//!
//! where R = n1*a1 + n2*a2 are Bravais lattice vectors and t_{ij}(R) are
//! hopping amplitudes. The Hermitian conjugate is added automatically.
//!
//! # Literature
//!
//! - Fukui, Hatsugai, Suzuki (2005): JPSJ 74, 1674 (FHS algorithm)
//! - Kaman, Lim, Liu, Hoffmann (2026): arXiv:2601.03210v2 (magnonic crystals)

use faer::{Mat, Side, c64};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Vec2: minimal 2D vector for k-space and real-space operations
// ---------------------------------------------------------------------------

/// Minimal 2D vector for lattice vectors, reciprocal vectors, and orbital
/// positions. Deliberately lightweight (no nalgebra dependency).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

// ---------------------------------------------------------------------------
// BravaisLattice2D
// ---------------------------------------------------------------------------

/// 2D Bravais lattice with direct and reciprocal vectors.
/// Reciprocal vectors satisfy a_i . b_j = 2*pi * delta_{ij}.
#[derive(Clone, Debug)]
pub struct BravaisLattice2D {
    pub a1: Vec2,
    pub a2: Vec2,
    pub b1: Vec2,
    pub b2: Vec2,
}

impl BravaisLattice2D {
    /// Construct from direct vectors, computing reciprocal vectors automatically.
    pub fn from_direct(a1: Vec2, a2: Vec2) -> Self {
        let det = a1.x * a2.y - a1.y * a2.x;
        assert!(
            det.abs() > 1e-15,
            "Lattice vectors must be linearly independent"
        );
        let inv_det = 2.0 * PI / det;
        let b1 = Vec2::new(a2.y * inv_det, -a2.x * inv_det);
        let b2 = Vec2::new(-a1.y * inv_det, a1.x * inv_det);
        Self { a1, a2, b1, b2 }
    }

    /// Hexagonal lattice: a1 = a*(1,0), a2 = a*(1/2, sqrt(3)/2).
    pub fn hexagonal(a: f64) -> Self {
        let s3 = 3.0_f64.sqrt();
        let a1 = Vec2::new(a, 0.0);
        let a2 = Vec2::new(a * 0.5, a * s3 * 0.5);
        Self::from_direct(a1, a2)
    }

    /// Square lattice: a1 = a*(1,0), a2 = a*(0,1).
    pub fn square(a: f64) -> Self {
        Self::from_direct(Vec2::new(a, 0.0), Vec2::new(0.0, a))
    }
}

// ---------------------------------------------------------------------------
// Tight-binding model types
// ---------------------------------------------------------------------------

/// Single orbital site within the unit cell.
#[derive(Clone, Debug)]
pub struct OrbitalSite {
    /// Position within the unit cell (metadata, not used in H(k)).
    pub position: Vec2,
    /// Label for identification (e.g. "s_A", "px_B").
    pub label: String,
    /// On-site energy (diagonal of H).
    pub on_site_energy: f64,
}

/// Hopping between orbitals. Specifies one direction only; the Hermitian
/// conjugate `H[to,from] += conj(amp)*exp(-ik.R)` is added automatically.
#[derive(Clone, Debug)]
pub struct Hopping {
    /// Source orbital index (in cell 0).
    pub from: usize,
    /// Target orbital index (in cell at `cell_offset`).
    pub to: usize,
    /// Bravais lattice offset: `R = cell_offset[0]*a1 + cell_offset[1]*a2`.
    pub cell_offset: [i32; 2],
    /// Complex hopping amplitude t_{from,to}(R).
    pub amplitude: c64,
}

/// Complete tight-binding model: lattice + orbitals + hoppings.
#[derive(Clone, Debug)]
pub struct TightBindingModel {
    pub lattice: BravaisLattice2D,
    pub orbitals: Vec<OrbitalSite>,
    pub hoppings: Vec<Hopping>,
}

/// Valley label for hexagonal Brillouin zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Valley {
    /// K point at (2*b1 + b2)/3.
    K,
    /// K' point at (b1 + 2*b2)/3.
    KPrime,
}

/// Result of flat band analysis.
#[derive(Clone, Debug)]
pub struct FlatBandInfo {
    /// Indices of bands with bandwidth below threshold.
    pub flat_band_indices: Vec<usize>,
    /// Bandwidth of every band (max - min over BZ).
    pub bandwidths: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Complex arithmetic helpers (following harper_chern.rs convention)
// ---------------------------------------------------------------------------

#[inline]
fn cmul(a: c64, b: c64) -> c64 {
    c64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn cadd(a: c64, b: c64) -> c64 {
    c64::new(a.re + b.re, a.im + b.im)
}

#[inline]
fn cconj(a: c64) -> c64 {
    c64::new(a.re, -a.im)
}

// ---------------------------------------------------------------------------
// Eigendecomposition (adapted from harper_chern.rs)
// ---------------------------------------------------------------------------

/// Diagonalize Hermitian matrix, returning sorted (eigenvalues, eigenvectors).
fn diagonalize(h: &Mat<c64>) -> (Vec<f64>, Mat<c64>) {
    let n = h.nrows();
    let eig = h.self_adjoint_eigen(Side::Lower).unwrap();

    let s_diag = eig.S();
    let eigenvalues_raw: Vec<f64> = (0..n).map(|i| s_diag.column_vector()[i].re).collect();

    let mut indexed: Vec<(usize, f64)> = eigenvalues_raw
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, e)| *e).collect();

    let u = eig.U();
    let mut eigenvectors = Mat::<c64>::zeros(n, n);
    for (new_col, &(orig_col, _)) in indexed.iter().enumerate() {
        for row in 0..n {
            eigenvectors[(row, new_col)] = u[(row, orig_col)];
        }
    }

    (eigenvalues, eigenvectors)
}

// ---------------------------------------------------------------------------
// TightBindingModel implementation
// ---------------------------------------------------------------------------

impl TightBindingModel {
    /// Number of orbitals per unit cell.
    pub fn n_orbitals(&self) -> usize {
        self.orbitals.len()
    }

    /// Build the Bloch Hamiltonian H(k) as an n_orb x n_orb Hermitian matrix.
    pub fn hamiltonian_at_k(&self, kx: f64, ky: f64) -> Mat<c64> {
        let n = self.n_orbitals();
        let mut h = Mat::<c64>::zeros(n, n);

        // Diagonal: on-site energies
        for (i, orb) in self.orbitals.iter().enumerate() {
            h[(i, i)] = c64::new(orb.on_site_energy, 0.0);
        }

        // Off-diagonal: hoppings + automatic Hermitian conjugate
        let k = Vec2::new(kx, ky);
        for hop in &self.hoppings {
            let r = self.lattice.a1.scale(hop.cell_offset[0] as f64)
                + self.lattice.a2.scale(hop.cell_offset[1] as f64);
            let phase_angle = k.dot(r);
            let phase = c64::new(phase_angle.cos(), phase_angle.sin());

            // Forward: H[from, to] += amplitude * exp(i k.R)
            let val = cmul(hop.amplitude, phase);
            let prev = h[(hop.from, hop.to)];
            h[(hop.from, hop.to)] = cadd(prev, val);

            // Hermitian conjugate: H[to, from] += conj(val)
            let val_hc = cconj(val);
            let prev_hc = h[(hop.to, hop.from)];
            h[(hop.to, hop.from)] = cadd(prev_hc, val_hc);
        }

        h
    }

    /// Compute sorted band energies at a single k-point.
    pub fn band_energies(&self, kx: f64, ky: f64) -> Vec<f64> {
        let h = self.hamiltonian_at_k(kx, ky);
        let (evals, _) = diagonalize(&h);
        evals
    }

    /// Compute band structure along a k-path.
    ///
    /// Returns (k_distances, band_energies) where `band_energies[band][k_idx]`.
    pub fn band_structure_along_path(&self, path: &[(f64, f64)]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n_bands = self.n_orbitals();
        let n_k = path.len();

        let mut k_dists = Vec::with_capacity(n_k);
        let mut bands = vec![Vec::with_capacity(n_k); n_bands];

        let mut cumulative = 0.0;
        for (idx, &(kx, ky)) in path.iter().enumerate() {
            if idx > 0 {
                let (kx0, ky0) = path[idx - 1];
                cumulative += ((kx - kx0).powi(2) + (ky - ky0).powi(2)).sqrt();
            }
            k_dists.push(cumulative);

            let evals = self.band_energies(kx, ky);
            for (band, &e) in evals.iter().enumerate() {
                bands[band].push(e);
            }
        }

        (k_dists, bands)
    }
}

// ---------------------------------------------------------------------------
// Brillouin zone paths
// ---------------------------------------------------------------------------

/// High-symmetry path Gamma -> M -> K -> Gamma for hexagonal lattice.
/// Returns n points per segment (3*n + 1 total).
pub fn hexagonal_high_symmetry_path(lattice: &BravaisLattice2D, n: usize) -> Vec<(f64, f64)> {
    let gamma = Vec2::zero();
    let m = lattice.b1.scale(0.5);
    let k_pt = lattice.b1.scale(2.0 / 3.0) + lattice.b2.scale(1.0 / 3.0);

    let mut path = Vec::with_capacity(3 * n + 1);

    let lerp =
        |a: Vec2, b: Vec2, t: f64| -> (f64, f64) { (a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)) };

    // Gamma -> M (n points, excluding M)
    for i in 0..n {
        path.push(lerp(gamma, m, i as f64 / n as f64));
    }
    // M -> K (n points, excluding K)
    for i in 0..n {
        path.push(lerp(m, k_pt, i as f64 / n as f64));
    }
    // K -> Gamma (n+1 points, including both endpoints)
    for i in 0..=n {
        path.push(lerp(k_pt, gamma, i as f64 / n as f64));
    }

    path
}

/// Symmetry point labels with k-distance positions for hexagonal path.
pub fn hexagonal_symmetry_labels(lattice: &BravaisLattice2D) -> Vec<(String, f64)> {
    let m = lattice.b1.scale(0.5);
    let k_pt = lattice.b1.scale(2.0 / 3.0) + lattice.b2.scale(1.0 / 3.0);

    let d_gm = m.length();
    let d_mk = (k_pt - m).length();
    let d_kg = k_pt.length();

    vec![
        ("Gamma".to_string(), 0.0),
        ("M".to_string(), d_gm),
        ("K".to_string(), d_gm + d_mk),
        ("Gamma".to_string(), d_gm + d_mk + d_kg),
    ]
}

// ---------------------------------------------------------------------------
// FHS Berry curvature and Chern numbers
// ---------------------------------------------------------------------------

/// Compute FHS Berry curvature for a single band over the full BZ.
///
/// The BZ is the parallelogram spanned by b1, b2, discretized into
/// n_grid x n_grid plaquettes.
pub fn fhs_berry_curvature(model: &TightBindingModel, band: usize, n_grid: usize) -> Vec<Vec<f64>> {
    let n_orb = model.n_orbitals();
    let b1 = model.lattice.b1;
    let b2 = model.lattice.b2;

    // Build eigenvector grid over parallelogram BZ
    let mut evecs: Vec<Vec<Mat<c64>>> = Vec::with_capacity(n_grid);
    for i in 0..n_grid {
        let mut row = Vec::with_capacity(n_grid);
        for j in 0..n_grid {
            let s = i as f64 / n_grid as f64;
            let t = j as f64 / n_grid as f64;
            let k = b1.scale(s) + b2.scale(t);
            let h = model.hamiltonian_at_k(k.x, k.y);
            let (_, evec) = diagonalize(&h);
            row.push(evec);
        }
        evecs.push(row);
    }

    // U(1) link variable: <psi(k1)|psi(k2)> / |<...>|
    let link = |i1: usize, j1: usize, i2: usize, j2: usize| -> c64 {
        let v1 = &evecs[i1][j1];
        let v2 = &evecs[i2][j2];
        let mut overlap = c64::new(0.0, 0.0);
        for orb in 0..n_orb {
            let a = v1[(orb, band)];
            let b = v2[(orb, band)];
            overlap = c64::new(
                overlap.re + a.re * b.re + a.im * b.im,
                overlap.im + a.re * b.im - a.im * b.re,
            );
        }
        let norm = (overlap.re * overlap.re + overlap.im * overlap.im).sqrt();
        if norm > 1e-15 {
            c64::new(overlap.re / norm, overlap.im / norm)
        } else {
            c64::new(1.0, 0.0)
        }
    };

    let mut curvature = vec![vec![0.0; n_grid]; n_grid];
    for (i, curv_row) in curvature.iter_mut().enumerate() {
        for (j, curv_ij) in curv_row.iter_mut().enumerate() {
            let i_next = (i + 1) % n_grid;
            let j_next = (j + 1) % n_grid;

            let u_x = link(i, j, i_next, j);
            let u_y = link(i_next, j, i_next, j_next);
            let u_x_inv = cconj(link(i, j_next, i_next, j_next));
            let u_y_inv = cconj(link(i, j, i, j_next));

            let p1 = cmul(u_x, u_y);
            let p2 = cmul(p1, u_x_inv);
            let plaquette = cmul(p2, u_y_inv);

            *curv_ij = plaquette.im.atan2(plaquette.re);
        }
    }

    curvature
}

/// Compute Chern number for a single band (should be integer).
///
/// Uses Kahan compensated summation to guarantee O(eps) accumulation error
/// instead of O(n^2 * eps), ensuring correct integer rounding for large n_grid.
pub fn band_chern_number(model: &TightBindingModel, band: usize, n_grid: usize) -> f64 {
    let curvature = fhs_berry_curvature(model, band, n_grid);
    let mut f_sum = crate::kahan::KahanSum::new();
    for row in &curvature {
        for &f_ij in row {
            f_sum.add(f_ij);
        }
    }
    f_sum.total() / (2.0 * PI)
}

/// Compute valley Chern number by integrating Berry curvature over half BZ.
///
/// The parallelogram BZ is split by the diagonal s = t in fractional
/// coordinates: K valley has s > t, K' valley has s < t.
pub fn valley_chern_number(
    model: &TightBindingModel,
    band: usize,
    n_grid: usize,
    valley: Valley,
) -> f64 {
    let curvature = fhs_berry_curvature(model, band, n_grid);
    let mut f_sum = crate::kahan::KahanSum::new();
    for (i, row) in curvature.iter().enumerate() {
        for (j, &f_ij) in row.iter().enumerate() {
            let s = (i as f64 + 0.5) / n_grid as f64;
            let t = (j as f64 + 0.5) / n_grid as f64;
            let in_valley = match valley {
                Valley::K => s > t,
                Valley::KPrime => s < t,
            };
            if in_valley {
                f_sum.add(f_ij);
            }
        }
    }
    f_sum.total() / (2.0 * PI)
}

/// Detect flat bands: bands with bandwidth below threshold.
pub fn detect_flat_bands(model: &TightBindingModel, n_k: usize, threshold: f64) -> FlatBandInfo {
    let n_bands = model.n_orbitals();
    let b1 = model.lattice.b1;
    let b2 = model.lattice.b2;

    let mut band_min = vec![f64::MAX; n_bands];
    let mut band_max = vec![f64::MIN; n_bands];

    for i in 0..n_k {
        for j in 0..n_k {
            let s = i as f64 / n_k as f64;
            let t = j as f64 / n_k as f64;
            let k = b1.scale(s) + b2.scale(t);
            let evals = model.band_energies(k.x, k.y);
            for (band, &e) in evals.iter().enumerate() {
                band_min[band] = band_min[band].min(e);
                band_max[band] = band_max[band].max(e);
            }
        }
    }

    let bandwidths: Vec<f64> = band_min
        .iter()
        .zip(&band_max)
        .map(|(lo, hi)| hi - lo)
        .collect();
    let flat_band_indices: Vec<usize> = bandwidths
        .iter()
        .enumerate()
        .filter(|&(_, &bw)| bw < threshold)
        .map(|(i, _)| i)
        .collect();

    FlatBandInfo {
        flat_band_indices,
        bandwidths,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // --- Vec2 ---

    #[test]
    fn test_vec2_add_sub() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, -1.0);
        let c = a + b;
        assert!((c.x - 4.0).abs() < TOL);
        assert!((c.y - 1.0).abs() < TOL);
        let d = a - b;
        assert!((d.x + 2.0).abs() < TOL);
        assert!((d.y - 3.0).abs() < TOL);
    }

    #[test]
    fn test_vec2_dot_length() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.length() - 5.0).abs() < TOL);
        assert!((a.dot(Vec2::new(1.0, 0.0)) - 3.0).abs() < TOL);
    }

    // --- BravaisLattice2D ---

    #[test]
    fn test_hexagonal_reciprocal_orthogonality() {
        let lat = BravaisLattice2D::hexagonal(1.0);
        assert!((lat.a1.dot(lat.b1) - 2.0 * PI).abs() < TOL, "a1.b1");
        assert!(lat.a1.dot(lat.b2).abs() < TOL, "a1.b2");
        assert!(lat.a2.dot(lat.b1).abs() < TOL, "a2.b1");
        assert!((lat.a2.dot(lat.b2) - 2.0 * PI).abs() < TOL, "a2.b2");
    }

    #[test]
    fn test_square_reciprocal_orthogonality() {
        let lat = BravaisLattice2D::square(2.5);
        assert!((lat.a1.dot(lat.b1) - 2.0 * PI).abs() < TOL);
        assert!(lat.a1.dot(lat.b2).abs() < TOL);
        assert!(lat.a2.dot(lat.b1).abs() < TOL);
        assert!((lat.a2.dot(lat.b2) - 2.0 * PI).abs() < TOL);
    }

    #[test]
    fn test_hexagonal_reciprocal_magnitudes_equal() {
        let lat = BravaisLattice2D::hexagonal(1.0);
        assert!(
            (lat.b1.length() - lat.b2.length()).abs() < TOL,
            "|b1| != |b2|"
        );
    }

    // --- Graphene (canonical 2-band test case) ---

    fn graphene_model(t: f64) -> TightBindingModel {
        let lat = BravaisLattice2D::hexagonal(1.0);
        let s3 = 3.0_f64.sqrt();
        let orbitals = vec![
            OrbitalSite {
                position: Vec2::zero(),
                label: "A".to_string(),
                on_site_energy: 0.0,
            },
            OrbitalSite {
                position: Vec2::new(0.0, 1.0 / s3),
                label: "B".to_string(),
                on_site_energy: 0.0,
            },
        ];
        let hoppings = vec![
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [0, 0],
                amplitude: c64::new(t, 0.0),
            },
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [0, -1],
                amplitude: c64::new(t, 0.0),
            },
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [1, -1],
                amplitude: c64::new(t, 0.0),
            },
        ];
        TightBindingModel {
            lattice: lat,
            orbitals,
            hoppings,
        }
    }

    #[test]
    fn test_graphene_hermitian() {
        let model = graphene_model(-1.0);
        let h = model.hamiltonian_at_k(0.5, 0.3);
        for i in 0..2 {
            for j in 0..2 {
                let hij = h[(i, j)];
                let hji = h[(j, i)];
                assert!((hij.re - hji.re).abs() < TOL, "H not Hermitian (re)");
                assert!((hij.im + hji.im).abs() < TOL, "H not Hermitian (im)");
            }
        }
    }

    #[test]
    fn test_graphene_dirac_cone_at_k() {
        let model = graphene_model(-1.0);
        let lat = &model.lattice;
        let k_pt = lat.b1.scale(2.0 / 3.0) + lat.b2.scale(1.0 / 3.0);
        let evals = model.band_energies(k_pt.x, k_pt.y);
        assert!(
            (evals[0] - evals[1]).abs() < 0.01,
            "Bands should touch at K: {} vs {}",
            evals[0],
            evals[1]
        );
        assert!(
            evals[0].abs() < 0.01,
            "Dirac point at E=0, got {}",
            evals[0]
        );
    }

    #[test]
    fn test_graphene_periodicity() {
        let model = graphene_model(-1.0);
        let lat = &model.lattice;
        let e_k = model.band_energies(0.7, 0.4);
        let e_k_g1 = model.band_energies(0.7 + lat.b1.x, 0.4 + lat.b1.y);
        for (a, b) in e_k.iter().zip(&e_k_g1) {
            assert!((a - b).abs() < TOL, "E(k) != E(k+b1)");
        }
        let e_k_g2 = model.band_energies(0.7 + lat.b2.x, 0.4 + lat.b2.y);
        for (a, b) in e_k.iter().zip(&e_k_g2) {
            assert!((a - b).abs() < TOL, "E(k) != E(k+b2)");
        }
    }

    #[test]
    fn test_graphene_time_reversal() {
        let model = graphene_model(-1.0);
        let e_k = model.band_energies(0.5, 0.3);
        let e_mk = model.band_energies(-0.5, -0.3);
        for (a, b) in e_k.iter().zip(&e_mk) {
            assert!((a - b).abs() < TOL, "TRS: E(k)={}, E(-k)={}", a, b);
        }
    }

    #[test]
    fn test_graphene_chern_zero() {
        let model = graphene_model(-1.0);
        let c0 = band_chern_number(&model, 0, 21);
        let c1 = band_chern_number(&model, 1, 21);
        assert!(c0.abs() < 0.1, "Band 0 Chern ~0, got {:.4}", c0);
        assert!((c0 + c1).abs() < 0.1, "Sum ~0, got {:.4}", c0 + c1);
    }

    #[test]
    fn test_graphene_valley_chern_antisymmetric() {
        let model = graphene_model(-1.0);
        let vcn_k = valley_chern_number(&model, 0, 21, Valley::K);
        let vcn_kp = valley_chern_number(&model, 0, 21, Valley::KPrime);
        assert!(
            (vcn_k + vcn_kp).abs() < 0.15,
            "VCN(K)+VCN(K') ~ 0, got {:.4}+{:.4}",
            vcn_k,
            vcn_kp
        );
    }

    #[test]
    fn test_band_structure_along_path() {
        let model = graphene_model(-1.0);
        let path = hexagonal_high_symmetry_path(&model.lattice, 10);
        let (k_dists, bands) = model.band_structure_along_path(&path);
        assert_eq!(bands.len(), 2);
        assert_eq!(k_dists.len(), path.len());
        assert!((k_dists[0]).abs() < TOL);
        for w in k_dists.windows(2) {
            assert!(w[1] >= w[0] - TOL, "k-distances not monotonic");
        }
    }

    // --- Square lattice (1-band cosine dispersion) ---

    #[test]
    fn test_square_cosine_band() {
        let lat = BravaisLattice2D::square(1.0);
        let orbitals = vec![OrbitalSite {
            position: Vec2::zero(),
            label: "s".to_string(),
            on_site_energy: 0.0,
        }];
        let hoppings = vec![
            Hopping {
                from: 0,
                to: 0,
                cell_offset: [1, 0],
                amplitude: c64::new(-1.0, 0.0),
            },
            Hopping {
                from: 0,
                to: 0,
                cell_offset: [0, 1],
                amplitude: c64::new(-1.0, 0.0),
            },
        ];
        let model = TightBindingModel {
            lattice: lat,
            orbitals,
            hoppings,
        };
        let (kx, ky) = (0.5, 0.8);
        let evals = model.band_energies(kx, ky);
        let expected = -2.0 * kx.cos() - 2.0 * ky.cos();
        assert!(
            (evals[0] - expected).abs() < TOL,
            "E={}, expected {}",
            evals[0],
            expected
        );
    }

    // --- Kagome lattice (flat band detection) ---

    fn kagome_model(t: f64) -> TightBindingModel {
        let lat = BravaisLattice2D::hexagonal(1.0);
        let s3 = 3.0_f64.sqrt();
        let orbitals = vec![
            OrbitalSite {
                position: Vec2::zero(),
                label: "A".to_string(),
                on_site_energy: 0.0,
            },
            OrbitalSite {
                position: Vec2::new(0.5, 0.0),
                label: "B".to_string(),
                on_site_energy: 0.0,
            },
            OrbitalSite {
                position: Vec2::new(0.25, s3 / 4.0),
                label: "C".to_string(),
                on_site_energy: 0.0,
            },
        ];
        let amp = c64::new(t, 0.0);
        let hoppings = vec![
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [0, 0],
                amplitude: amp,
            },
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [-1, 0],
                amplitude: amp,
            },
            Hopping {
                from: 0,
                to: 2,
                cell_offset: [0, 0],
                amplitude: amp,
            },
            Hopping {
                from: 0,
                to: 2,
                cell_offset: [0, -1],
                amplitude: amp,
            },
            Hopping {
                from: 1,
                to: 2,
                cell_offset: [0, 0],
                amplitude: amp,
            },
            Hopping {
                from: 1,
                to: 2,
                cell_offset: [1, -1],
                amplitude: amp,
            },
        ];
        TightBindingModel {
            lattice: lat,
            orbitals,
            hoppings,
        }
    }

    #[test]
    fn test_kagome_flat_band() {
        let model = kagome_model(1.0);
        let info = detect_flat_bands(&model, 20, 0.01);
        assert!(
            !info.flat_band_indices.is_empty(),
            "Kagome should have flat band, bandwidths: {:?}",
            info.bandwidths
        );
    }

    #[test]
    fn test_kagome_flat_band_energy() {
        let model = kagome_model(1.0);
        // Flat band at E = -2t = -2.0 at all k-points
        let e_gamma = model.band_energies(0.0, 0.0);
        let lat = &model.lattice;
        let k_m = lat.b1.scale(0.5);
        let e_m = model.band_energies(k_m.x, k_m.y);
        // Lowest eigenvalue should be -2 at both points
        assert!(
            (e_gamma[0] - (-2.0)).abs() < TOL,
            "Flat band at Gamma: {}",
            e_gamma[0]
        );
        assert!((e_m[0] - (-2.0)).abs() < TOL, "Flat band at M: {}", e_m[0]);
    }

    #[test]
    fn test_hexagonal_path_endpoints() {
        let lat = BravaisLattice2D::hexagonal(1.0);
        let path = hexagonal_high_symmetry_path(&lat, 10);
        assert!(path[0].0.abs() < TOL && path[0].1.abs() < TOL);
        let last = path.last().unwrap();
        assert!(last.0.abs() < TOL && last.1.abs() < TOL);
    }
}
