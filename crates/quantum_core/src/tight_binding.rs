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
    try_diagonalize(h).expect("Hermitian eigendecomposition failed")
}

fn try_diagonalize(h: &Mat<c64>) -> Option<(Vec<f64>, Mat<c64>)> {
    let n = h.nrows();
    let eig = h.self_adjoint_eigen(Side::Lower).ok()?;

    let s_diag = eig.S();
    let eigenvalues_raw: Vec<f64> = (0..n).map(|i| s_diag.column_vector()[i].re).collect();

    let mut indexed: Vec<(usize, f64)> = eigenvalues_raw
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let eigenvalues: Vec<f64> = indexed.iter().map(|(_, e)| *e).collect();

    let u = eig.U();
    let mut eigenvectors = Mat::<c64>::zeros(n, n);
    for (new_col, &(orig_col, _)) in indexed.iter().enumerate() {
        for row in 0..n {
            eigenvectors[(row, new_col)] = u[(row, orig_col)];
        }
    }

    Some((eigenvalues, eigenvectors))
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

/// Numerical admission thresholds, in Hamiltonian energy units for gaps.
#[derive(Clone, Copy, Debug)]
pub struct TopologyAdmission {
    pub minimum_gap: f64,
    pub minimum_link_determinant: f64,
}

impl Default for TopologyAdmission {
    fn default() -> Self {
        Self {
            minimum_gap: 1e-10,
            minimum_link_determinant: 1e-12,
        }
    }
}

/// A failed sampled topology predicate; an error never represents zero curvature.
#[derive(Clone, Debug, PartialEq)]
pub enum TopologyError {
    InvalidInput,
    DiagonalizationFailure {
        point: [usize; 2],
    },
    NonFiniteHamiltonian {
        point: [usize; 2],
    },
    SampledGap {
        point: [usize; 2],
        gap: f64,
    },
    SingularLink {
        point: [usize; 2],
        axis: usize,
        determinant_norm: f64,
    },
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for TopologyError {}

/// FHS determinant curvature for a subspace admitted on a finite grid.
/// Sampled isolation does not establish a gap between sampled points, and
/// integer sums alone do not establish resolution convergence.
#[derive(Clone, Debug)]
pub struct SampledTopology {
    pub curvature: Vec<Vec<f64>>,
    pub bands: std::ops::Range<usize>,
    /// Absent when the selected subspace contains every orbital.
    pub minimum_sampled_gap: Option<f64>,
    pub minimum_link_determinant: f64,
}

impl SampledTopology {
    pub fn chern_number(&self) -> f64 {
        let mut sum = crate::kahan::KahanSum::new();
        for value in self.curvature.iter().flatten() {
            sum.add(*value);
        }
        sum.total() / (2.0 * PI)
    }

    /// Fractional-coordinate half-zone integral, not a quantized invariant.
    pub fn valley_chern_number(&self, valley: Valley) -> f64 {
        let mut sum = crate::kahan::KahanSum::new();
        for (row_index, row) in self.curvature.iter().enumerate() {
            for (column_index, value) in row.iter().enumerate() {
                let selected = match valley {
                    Valley::K => row_index > column_index,
                    Valley::KPrime => row_index < column_index,
                };
                if selected {
                    sum.add(*value);
                }
            }
        }
        sum.total() / (2.0 * PI)
    }
}

fn overlap_determinant(left: &Mat<c64>, right: &Mat<c64>, bands: std::ops::Range<usize>) -> c64 {
    let count = bands.len();
    let mut overlap = Mat::from_fn(count, count, |row, column| {
        let mut sum = c64::new(0.0, 0.0);
        for orbital in 0..left.nrows() {
            sum +=
                cconj(left[(orbital, bands.start + row)]) * right[(orbital, bands.start + column)];
        }
        sum
    });
    let mut determinant = c64::new(1.0, 0.0);
    for column in 0..count {
        let pivot = (column..count)
            .max_by(|&left_row, &right_row| {
                overlap[(left_row, column)]
                    .norm()
                    .total_cmp(&overlap[(right_row, column)].norm())
            })
            .expect("nonempty pivot range");
        if overlap[(pivot, column)].norm() == 0.0 {
            return c64::new(0.0, 0.0);
        }
        if pivot != column {
            for index in 0..count {
                let temporary = overlap[(pivot, index)];
                overlap[(pivot, index)] = overlap[(column, index)];
                overlap[(column, index)] = temporary;
            }
            determinant = -determinant;
        }
        let diagonal = overlap[(column, column)];
        determinant *= diagonal;
        for row in column + 1..count {
            let factor = overlap[(row, column)] / diagonal;
            for index in column + 1..count {
                let subtract = factor * overlap[(column, index)];
                overlap[(row, index)] -= subtract;
            }
        }
    }
    determinant
}

fn topology_from_frames(
    frames: &[Vec<Mat<c64>>],
    bands: std::ops::Range<usize>,
    admission: TopologyAdmission,
    minimum_sampled_gap: Option<f64>,
) -> Result<SampledTopology, TopologyError> {
    let grid = frames.len();
    let mut links = vec![vec![[c64::new(0.0, 0.0); 2]; grid]; grid];
    let mut minimum_link = f64::INFINITY;
    for row in 0..grid {
        for column in 0..grid {
            for (axis, link) in links[row][column].iter_mut().enumerate() {
                let neighbor = if axis == 0 {
                    [(row + 1) % grid, column]
                } else {
                    [row, (column + 1) % grid]
                };
                let determinant = overlap_determinant(
                    &frames[row][column],
                    &frames[neighbor[0]][neighbor[1]],
                    bands.clone(),
                );
                let norm = determinant.norm();
                if !norm.is_finite() || norm <= admission.minimum_link_determinant {
                    return Err(TopologyError::SingularLink {
                        point: [row, column],
                        axis,
                        determinant_norm: norm,
                    });
                }
                minimum_link = minimum_link.min(norm);
                *link = determinant / norm;
            }
        }
    }
    let curvature = (0..grid)
        .map(|row| {
            (0..grid)
                .map(|column| {
                    let plaquette = links[row][column][0]
                        * links[(row + 1) % grid][column][1]
                        * cconj(links[row][(column + 1) % grid][0])
                        * cconj(links[row][column][1]);
                    plaquette.im.atan2(plaquette.re)
                })
                .collect()
        })
        .collect();
    Ok(SampledTopology {
        curvature,
        bands,
        minimum_sampled_gap,
        minimum_link_determinant: minimum_link,
    })
}

fn admits_bloch_grid(model: &TightBindingModel) -> bool {
    let lattice = &model.lattice;
    let direct = [lattice.a1, lattice.a2];
    let reciprocal = [lattice.b1, lattice.b2];
    if direct
        .iter()
        .chain(&reciprocal)
        .any(|vector| !vector.x.is_finite() || !vector.y.is_finite())
        || model
            .hoppings
            .iter()
            .any(|hopping| hopping.from >= model.n_orbitals() || hopping.to >= model.n_orbitals())
    {
        return false;
    }
    for vectors in [direct, reciprocal] {
        let determinant = vectors[0].x * vectors[1].y - vectors[0].y * vectors[1].x;
        if !determinant.is_finite() || determinant == 0.0 {
            return false;
        }
    }
    // Public lattice fields must describe the same reciprocal cell. The
    // tolerance covers rounding in the two dot-product terms, not a fit.
    for (direct_index, direct_vector) in direct.iter().enumerate() {
        for (reciprocal_index, reciprocal_vector) in reciprocal.iter().enumerate() {
            let horizontal = direct_vector.x * reciprocal_vector.x;
            let vertical = direct_vector.y * reciprocal_vector.y;
            let expected = if direct_index == reciprocal_index {
                2.0 * PI
            } else {
                0.0
            };
            let scale = (horizontal.abs() + vertical.abs()).max(2.0 * PI);
            let actual = horizontal + vertical;
            if !scale.is_finite()
                || !actual.is_finite()
                || (actual - expected).abs() > 64.0 * f64::EPSILON * scale
            {
                return false;
            }
        }
    }
    true
}

/// Admit a contiguous isolated band or separated subspace on a finite BZ grid.
/// Internal degeneracies are allowed; each sampled external gap must exceed
/// the declared absolute tolerance. Determinant links reject singular overlaps.
pub fn checked_subspace_topology(
    model: &TightBindingModel,
    bands: std::ops::Range<usize>,
    n_grid: usize,
    admission: TopologyAdmission,
) -> Result<SampledTopology, TopologyError> {
    if !admits_bloch_grid(model)
        || n_grid < 2
        || bands.is_empty()
        || bands.end > model.n_orbitals()
        || !admission.minimum_gap.is_finite()
        || admission.minimum_gap < 0.0
        || !admission.minimum_link_determinant.is_finite()
        || admission.minimum_link_determinant <= 0.0
    {
        return Err(TopologyError::InvalidInput);
    }
    let mut minimum_gap = f64::INFINITY;
    let mut frames = Vec::with_capacity(n_grid);
    for row in 0..n_grid {
        let mut frame_row = Vec::with_capacity(n_grid);
        for column in 0..n_grid {
            let momentum = model.lattice.b1.scale(row as f64 / n_grid as f64)
                + model.lattice.b2.scale(column as f64 / n_grid as f64);
            let hamiltonian = model.hamiltonian_at_k(momentum.x, momentum.y);
            if (0..hamiltonian.nrows()).any(|row| {
                (0..hamiltonian.ncols()).any(|column| {
                    let value = hamiltonian[(row, column)];
                    !value.re.is_finite() || !value.im.is_finite()
                })
            }) {
                return Err(TopologyError::NonFiniteHamiltonian {
                    point: [row, column],
                });
            }
            let (energies, frame) =
                try_diagonalize(&hamiltonian).ok_or(TopologyError::DiagonalizationFailure {
                    point: [row, column],
                })?;
            for boundary in [bands.start, bands.end] {
                if boundary == 0 || boundary == energies.len() {
                    continue;
                }
                let gap = energies[boundary] - energies[boundary - 1];
                if !gap.is_finite() || gap <= admission.minimum_gap {
                    return Err(TopologyError::SampledGap {
                        point: [row, column],
                        gap,
                    });
                }
                minimum_gap = minimum_gap.min(gap);
            }
            frame_row.push(frame);
        }
        frames.push(frame_row);
    }
    topology_from_frames(
        &frames,
        bands,
        admission,
        minimum_gap.is_finite().then_some(minimum_gap),
    )
}

/// Checked single-band curvature with default absolute tolerances.
/// Panics on failed sampled admission; use `checked_subspace_topology` to retain errors.
pub fn fhs_berry_curvature(model: &TightBindingModel, band: usize, n_grid: usize) -> Vec<Vec<f64>> {
    checked_subspace_topology(
        model,
        band..band.saturating_add(1),
        n_grid,
        TopologyAdmission::default(),
    )
    .expect("single-band sampled topology admission failed")
    .curvature
}

/// Checked finite-grid single-band sum; a successful sum is not a convergence proof.
/// Panics on failed admission; use `checked_subspace_topology` for scientific reports.
pub fn band_chern_number(model: &TightBindingModel, band: usize, n_grid: usize) -> f64 {
    checked_subspace_topology(
        model,
        band..band.saturating_add(1),
        n_grid,
        TopologyAdmission::default(),
    )
    .expect("single-band sampled topology admission failed")
    .chern_number()
}

/// Checked half-zone integral with default tolerances; not a quantized invariant.
/// Panics on failed admission; use `checked_subspace_topology` for scientific reports.
pub fn valley_chern_number(
    model: &TightBindingModel,
    band: usize,
    n_grid: usize,
    valley: Valley,
) -> f64 {
    checked_subspace_topology(
        model,
        band..band.saturating_add(1),
        n_grid,
        TopologyAdmission::default(),
    )
    .expect("single-band sampled topology admission failed")
    .valley_chern_number(valley)
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
    fn gapless_graphene_rejects_individual_band_but_admits_full_subspace() {
        let model = graphene_model(-1.0);
        assert!(matches!(
            checked_subspace_topology(&model, 0..1, 21, TopologyAdmission::default()),
            Err(TopologyError::SampledGap { .. })
        ));
        let topology =
            checked_subspace_topology(&model, 0..2, 21, TopologyAdmission::default()).unwrap();
        assert!(topology.chern_number().abs() < 1e-12);
        assert_eq!(topology.minimum_sampled_gap, None);
    }

    #[test]
    fn gapped_graphene_admits_isolated_bands() {
        let mut model = graphene_model(-1.0);
        model.orbitals[0].on_site_energy = 0.3;
        model.orbitals[1].on_site_energy = -0.3;
        for band in 0..2 {
            let topology =
                checked_subspace_topology(&model, band..band + 1, 21, TopologyAdmission::default())
                    .unwrap();
            assert!(topology.minimum_sampled_gap.unwrap() >= 0.6 - 1e-12);
            assert!(topology.chern_number().abs() < 1e-12);
        }
    }

    #[test]
    fn separated_degenerate_pair_admits_only_subspace_invariant() {
        let mut model = graphene_model(0.0);
        let mut separated = model.orbitals[0].clone();
        separated.on_site_energy = 2.0;
        model.orbitals.push(separated);
        assert!(matches!(
            checked_subspace_topology(&model, 0..1, 3, TopologyAdmission::default()),
            Err(TopologyError::SampledGap { .. })
        ));
        let topology =
            checked_subspace_topology(&model, 0..2, 3, TopologyAdmission::default()).unwrap();
        assert_eq!(topology.minimum_sampled_gap, Some(2.0));
        assert!(topology.chern_number().abs() < 1e-12);
    }

    #[test]
    fn singular_overlap_rejects_instead_of_substituting_unity() {
        let identity = Mat::identity(2, 2);
        let swapped = Mat::from_fn(2, 2, |row, column| c64::new(f64::from(row != column), 0.0));
        let frames = vec![
            vec![identity.clone(), swapped],
            vec![identity.clone(), identity],
        ];
        assert!(matches!(
            topology_from_frames(&frames, 0..1, TopologyAdmission::default(), Some(1.0)),
            Err(TopologyError::SingularLink { .. })
        ));
    }

    #[test]
    fn subspace_curvature_is_invariant_under_local_unitary_rotations() {
        let grid = 15;
        // A two-band Chern insulator plus an inert occupied orbital supplies
        // nonzero curvature; rotations mix both occupied eigenvectors.
        let baseline: Vec<Vec<_>> = (0..grid)
            .map(|row| {
                (0..grid)
                    .map(|column| {
                        let momentum_x = 2.0 * PI * row as f64 / grid as f64;
                        let momentum_y = 2.0 * PI * column as f64 / grid as f64;
                        let mass = -1.0 + momentum_x.cos() + momentum_y.cos();
                        let mut hamiltonian = Mat::zeros(3, 3);
                        hamiltonian[(0, 0)] = c64::new(mass, 0.0);
                        hamiltonian[(1, 1)] = c64::new(-mass, 0.0);
                        hamiltonian[(0, 1)] = c64::new(momentum_x.sin(), -momentum_y.sin());
                        hamiltonian[(1, 0)] = cconj(hamiltonian[(0, 1)]);
                        hamiltonian[(2, 2)] = c64::new(-5.0, 0.0);
                        diagonalize(&hamiltonian).1
                    })
                    .collect()
            })
            .collect();
        let frames: Vec<Vec<_>> = baseline
            .iter()
            .enumerate()
            .map(|(row, entries)| {
                entries
                    .iter()
                    .enumerate()
                    .map(|(column, frame)| {
                        let angle = 0.13 * (row + 2 * column) as f64;
                        let phase_angle = 0.31 * (2 * row + column) as f64;
                        let phase = c64::new(phase_angle.cos(), phase_angle.sin());
                        let mut rotated = frame.clone();
                        for orbital in 0..3 {
                            rotated[(orbital, 0)] = phase
                                * (frame[(orbital, 0)] * angle.cos()
                                    - frame[(orbital, 1)] * angle.sin());
                            rotated[(orbital, 1)] = frame[(orbital, 0)] * angle.sin()
                                + frame[(orbital, 1)] * angle.cos();
                        }
                        rotated
                    })
                    .collect()
            })
            .collect();
        let rotated =
            topology_from_frames(&frames, 0..2, TopologyAdmission::default(), Some(1.0)).unwrap();
        let reference =
            topology_from_frames(&baseline, 0..2, TopologyAdmission::default(), Some(1.0)).unwrap();
        assert!((reference.chern_number().abs() - 1.0).abs() < 1e-12);
        for (actual, expected) in rotated
            .curvature
            .iter()
            .flatten()
            .zip(reference.curvature.iter().flatten())
        {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn topology_rejects_hopping_endpoints_before_matrix_indexing() {
        let model = graphene_model(-1.0);
        for endpoint in [0, 1] {
            let mut invalid = model.clone();
            if endpoint == 0 {
                invalid.hoppings[0].from = invalid.n_orbitals();
            } else {
                invalid.hoppings[0].to = invalid.n_orbitals();
            }
            assert_eq!(
                checked_subspace_topology(&invalid, 0..1, 3, TopologyAdmission::default())
                    .unwrap_err(),
                TopologyError::InvalidInput
            );
        }
    }

    #[test]
    fn topology_rejects_invalid_geometry_even_for_momentum_independent_hamiltonians() {
        let mut model = graphene_model(0.0);
        model.hoppings.clear();
        model.orbitals[0].on_site_energy = -1.0;
        model.orbitals[1].on_site_energy = 1.0;
        assert!(checked_subspace_topology(&model, 0..1, 3, TopologyAdmission::default()).is_ok());
        let mut invalid_lattices = Vec::new();
        let mut nonfinite = model.lattice.clone();
        nonfinite.b1.x = f64::NAN;
        invalid_lattices.push(nonfinite);
        let mut degenerate_reciprocal = model.lattice.clone();
        degenerate_reciprocal.b2 = degenerate_reciprocal.b1;
        invalid_lattices.push(degenerate_reciprocal);
        let mut degenerate_direct = model.lattice.clone();
        degenerate_direct.a2 = degenerate_direct.a1;
        invalid_lattices.push(degenerate_direct);
        let mut inconsistent_reciprocal = model.lattice.clone();
        inconsistent_reciprocal.b1 = inconsistent_reciprocal.b1.scale(2.0);
        invalid_lattices.push(inconsistent_reciprocal);
        for lattice in invalid_lattices {
            model.lattice = lattice;
            assert_eq!(
                checked_subspace_topology(&model, 0..1, 3, TopologyAdmission::default())
                    .unwrap_err(),
                TopologyError::InvalidInput
            );
        }
    }

    #[test]
    fn topology_rejects_invalid_grid_and_nonfinite_hamiltonian() {
        let mut model = graphene_model(-1.0);
        assert_eq!(
            checked_subspace_topology(&model, 0..1, 0, TopologyAdmission::default()).unwrap_err(),
            TopologyError::InvalidInput
        );
        model.orbitals[0].on_site_energy = f64::NAN;
        assert!(matches!(
            checked_subspace_topology(&model, 0..1, 3, TopologyAdmission::default()),
            Err(TopologyError::NonFiniteHamiltonian { .. })
        ));
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
