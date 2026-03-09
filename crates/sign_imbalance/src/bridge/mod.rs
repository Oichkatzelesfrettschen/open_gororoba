//! Bridge between signed graph imbalance and LBM 3D fluid dynamics.
//!
//! Implements the imbalance-viscosity coupling principle (Thesis 1):
//! Fluid viscosity in spacetime emerges from algebraic imbalance density
//! in Cayley-Dickson signed graphs. Local viscosity nu(x,y,z) is derived from
//! per-cell imbalance index via spatial evolution of Sedenion field.

use crate::signed_graph::SignedGraph;

#[derive(Clone, Debug)]
struct TriangleSignCache {
    dim: usize,
    triangles: Vec<(usize, usize, usize, bool)>,
}

fn build_triangle_sign_cache(dim: usize) -> TriangleSignCache {
    use gororoba_algebra::construction::cayley_dickson::cd_basis_mul_sign;

    let mut psi_flat = vec![0i32; dim * dim];
    for i in 0..dim {
        for j in (i + 1)..dim {
            let sign = cd_basis_mul_sign(dim, i, j);
            psi_flat[i * dim + j] = sign;
            psi_flat[j * dim + i] = sign;
        }
    }

    let mut triangles = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            for k in (j + 1)..dim {
                let product = psi_flat[i * dim + j] * psi_flat[j * dim + k] * psi_flat[i * dim + k];
                triangles.push((i, j, k, product == -1));
            }
        }
    }

    TriangleSignCache { dim, triangles }
}

fn weighted_imbalance_density(
    weights: &[f64],
    cache: &TriangleSignCache,
    fallback_value: f64,
) -> f64 {
    let magnitude_sum: f64 = weights.iter().take(cache.dim).sum();
    if magnitude_sum < 1e-10 {
        return fallback_value;
    }

    let mut weighted_frustrated = 0.0f64;
    let mut weighted_total = 0.0f64;
    for &(i, j, k, is_frustrated) in &cache.triangles {
        let w = weights[i] * weights[j] * weights[k];
        weighted_total += w;
        if is_frustrated {
            weighted_frustrated += w;
        }
    }

    if weighted_total < 1e-30 {
        fallback_value
    } else {
        weighted_frustrated / weighted_total
    }
}

/// Spatial Sedenion field abstraction.
///
/// Represents evolution of 16-dimensional Sedenion algebra elements
/// across a 3D lattice grid. Used to compute local imbalance density
/// at each grid point via signed-graph projection.
#[derive(Clone, Debug)]
pub struct SedenionField {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Sedenion values: shape (nx * ny * nz, 16)
    pub data: Vec<[f64; 16]>,
}

impl SedenionField {
    /// Create uniform Sedenion field (all elements = basis e_0).
    pub fn uniform(nx: usize, ny: usize, nz: usize) -> Self {
        let mut data = vec![[0.0; 16]; nx * ny * nz];
        // Initialize to e_0 (scalar part = 1.0)
        for sedenion in data.iter_mut() {
            sedenion[0] = 1.0;
        }
        Self { nx, ny, nz, data }
    }

    /// Linear index from (x, y, z) coordinates.
    pub fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Get Sedenion at grid point.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &[f64; 16] {
        &self.data[self.linearize(x, y, z)]
    }

    /// Get mutable Sedenion at grid point.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut [f64; 16] {
        let idx = self.linearize(x, y, z);
        &mut self.data[idx]
    }

    /// Compute local imbalance density at each grid point.
    ///
    /// Uses the Cayley-Dickson psi matrix as the base signed graph, with
    /// Sedenion component magnitudes as vertex weights. The weighted
    /// triangle imbalance is:
    ///
    ///   F(x) = sum_T w_i*w_j*w_k * frustrated(T) / sum_T w_i*w_j*w_k
    ///
    /// where w_i = |sedenion[i]| and frustrated(T) = 1 iff psi(i,j)*psi(j,k)*psi(i,k) = -1.
    ///
    /// At uniform weights, this reduces to the global CD imbalance (~3/8).
    /// Spatially varying Sedenion fields produce spatially varying imbalance
    /// because different components dominate at different grid points,
    /// weighting different triangles.
    pub fn local_imbalance_density(&self, dim: usize) -> Vec<f64> {
        let cache = build_triangle_sign_cache(dim);

        self.data
            .iter()
            .map(|sedenion| {
                let weights: Vec<f64> = sedenion[0..dim].iter().map(|x| x.abs()).collect();
                weighted_imbalance_density(&weights, &cache, 0.375)
            })
            .collect()
    }
}

impl SedenionField {
    /// Compute local imbalance density in parallel using rayon.
    ///
    /// Identical results to `local_imbalance_density` but uses rayon's
    /// par_iter() for ~Nx speedup on multi-core machines.
    pub fn local_imbalance_density_par(&self, dim: usize) -> Vec<f64> {
        use rayon::prelude::*;

        let cache = build_triangle_sign_cache(dim);

        self.data
            .par_iter()
            .map(|sedenion| {
                let weights: Vec<f64> = sedenion[0..dim].iter().map(|x| x.abs()).collect();
                weighted_imbalance_density(&weights, &cache, 0.375)
            })
            .collect()
    }

    /// Compute the local associator norm field.
    ///
    /// At each grid point, computes the mean associator norm from a representative
    /// sample of sedenion triples (a, b, c) formed by the local element and its
    /// neighbors. The associator is:
    ///
    ///   [a, b, c] = (a * b) * c - a * (b * c)
    ///
    /// For sedenions (dim=16), the associator is generically nonzero. The norm
    /// `||[a,b,c]||` measures the local non-associativity strength.
    ///
    /// Uses CD basis multiplication from `gororoba_algebra` for genuine associators.
    pub fn local_associator_norm_field(&self, dim: usize) -> Vec<f64> {
        use gororoba_algebra::construction::cayley_dickson::cd_multiply;

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n_nodes = nx * ny * nz;
        let mut norms = vec![0.0; n_nodes];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    let a = &self.data[idx][..dim];

                    // Use 6 face-neighbors as triple partners (b, c)
                    let neighbors = [
                        ((x + 1) % nx, y, z),
                        ((x + nx - 1) % nx, y, z),
                        (x, (y + 1) % ny, z),
                        (x, (y + ny - 1) % ny, z),
                        (x, y, (z + 1) % nz),
                        (x, y, (z + nz - 1) % nz),
                    ];

                    let mut norm_sum = 0.0;
                    let mut count = 0;

                    // For each pair of distinct neighbors, compute associator
                    for i in 0..neighbors.len() {
                        let (bx, by, bz) = neighbors[i];
                        let b_idx = bz * (nx * ny) + by * nx + bx;
                        let b = &self.data[b_idx][..dim];

                        let j = (i + 1) % neighbors.len();
                        let (cx, cy, cz) = neighbors[j];
                        let c_idx = cz * (nx * ny) + cy * nx + cx;
                        let c = &self.data[c_idx][..dim];

                        // Associator [a, b, c] = (a*b)*c - a*(b*c)
                        let ab = cd_multiply(a, b);
                        let bc = cd_multiply(b, c);
                        let ab_c = cd_multiply(&ab, c);
                        let a_bc = cd_multiply(a, &bc);

                        let assoc_norm_sq: f64 = ab_c
                            .iter()
                            .zip(a_bc.iter())
                            .map(|(x, y)| (x - y).powi(2))
                            .sum();

                        norm_sum += assoc_norm_sq.sqrt();
                        count += 1;
                    }

                    norms[idx] = if count > 0 {
                        norm_sum / count as f64
                    } else {
                        0.0
                    };
                }
            }
        }

        norms
    }
}

/// Imbalance-Viscosity coupling bridge.
///
/// Transforms signed-graph imbalance density into spatially-varying
/// kinematic viscosity field for LBM simulation.
#[derive(Clone, Debug)]
pub struct ImbalanceViscosityBridge {
    pub dim: usize,
    pub signed_graph: SignedGraph,
}

impl ImbalanceViscosityBridge {
    /// Create bridge from Cayley-Dickson dimension.
    pub fn new(dim: usize) -> Self {
        use gororoba_algebra::construction::cayley_dickson::cd_basis_mul_sign;

        let signed_graph = SignedGraph::from_psi_matrix(dim, |i, j| cd_basis_mul_sign(dim, i, j));

        Self { dim, signed_graph }
    }

    /// Convert local imbalance density to kinematic viscosity.
    ///
    /// Viscosity relation:
    /// nu(x) = nu_base * exp(-lambda * (F(x) - 3/8)^2)
    ///
    /// where F(x) is local imbalance density and 3/8 is the imbalance attractor.
    ///
    /// # Arguments
    /// * `imbalance_field` - Local imbalance density [0,1] at each grid point
    /// * `nu_base` - Base kinematic viscosity (e.g., 1/3 for inviscid limit)
    /// * `lambda` - Coupling strength (typical: 1.0-2.0)
    pub fn imbalance_to_viscosity(
        &self,
        imbalance_field: &[f64],
        nu_base: f64,
        lambda: f64,
    ) -> Vec<f64> {
        const IMBALANCE_ATTRACTOR: f64 = 3.0 / 8.0;

        imbalance_field
            .iter()
            .map(|&f| {
                let deviation = (f - IMBALANCE_ATTRACTOR).powi(2);
                nu_base * (-lambda * deviation).exp()
            })
            .collect()
    }

    /// Full pipeline: Sedenion field -> Imbalance -> Viscosity.
    ///
    /// Computes spatially-varying viscosity field from Sedenion field evolution.
    pub fn compute_viscosity_field(
        &self,
        sedenion_field: &SedenionField,
        nu_base: f64,
        lambda: f64,
    ) -> Vec<f64> {
        let imbalance = sedenion_field.local_imbalance_density(self.dim);
        self.imbalance_to_viscosity(&imbalance, nu_base, lambda)
    }

    /// Convert imbalance field to viscosity using a specified coupling model.
    ///
    /// This is the multi-model generalization of `imbalance_to_viscosity`.
    /// Different models make different physical assumptions about how algebraic
    /// imbalance modulates fluid viscosity:
    ///
    /// - **Exponential**: Gaussian decay from imbalance attractor (original model)
    /// - **Linear**: First-order perturbation theory around vacuum
    /// - **PowerLaw**: Scale-free coupling (fractal-like)
    /// - **Sigmoid**: Phase-transition model with sharp crossover
    /// - **Constant**: Control model (imbalance has no effect)
    pub fn imbalance_to_viscosity_model(
        &self,
        imbalance_field: &[f64],
        model: &ViscosityCouplingModel,
    ) -> Vec<f64> {
        imbalance_field.iter().map(|&f| model.compute(f)).collect()
    }
}

/// Imbalance attractor index for sedenions (~3/8 = 0.375).
///
/// This is the equilibrium imbalance for uniform-weight sedenion fields.
/// All coupling models are centered on this value.
pub const IMBALANCE_ATTRACTOR: f64 = 3.0 / 8.0;

/// Coupling model for converting imbalance density to kinematic viscosity.
///
/// Each model represents a different physical hypothesis about how algebraic
/// imbalance in Cayley-Dickson signed graphs modulates fluid viscosity.
/// The experiment-lab runs all models in parallel to discriminate between
/// hypotheses using correlation strength.
#[derive(Debug, Clone)]
pub enum ViscosityCouplingModel {
    /// Gaussian decay: nu = nu_base * exp(-lambda * (F - F0)^2)
    ///
    /// Physical interpretation: viscosity peaks at imbalance attractor and
    /// decays symmetrically for deviations. Quadratic sensitivity means
    /// small deviations have little effect, large deviations strongly reduce
    /// viscosity. This is the natural "perturbative" model.
    Exponential { nu_base: f64, lambda: f64 },

    /// Linear coupling: nu = nu_base * (1 + alpha * (F - F0))
    ///
    /// Physical interpretation: first-order Taylor expansion around vacuum.
    /// Positive alpha means frustrated regions are more viscous (dissipative).
    /// Negative alpha means frustrated regions flow more freely.
    /// Simplest possible coupling; serves as baseline.
    Linear { nu_base: f64, alpha: f64 },

    /// Power-law coupling: nu = nu_base * (|F - F0| + eps)^n
    ///
    /// Physical interpretation: scale-free (fractal) relationship between
    /// imbalance deviation and viscosity. No preferred scale means the
    /// coupling has self-similar structure across imbalance magnitudes.
    /// n > 1 = superlinear sensitivity; n < 1 = sublinear (saturating).
    PowerLaw { nu_base: f64, n: f64 },

    /// Sigmoid (logistic) coupling: nu = nu_low + (nu_high - nu_low) / (1 + exp(-k*(F - F_crit)))
    ///
    /// Physical interpretation: sharp phase transition at critical imbalance.
    /// Below F_crit, viscosity is nu_low; above, it jumps to nu_high.
    /// Steepness k controls how sharp the transition is.
    /// Models a imbalance-driven "phase boundary" in the fluid.
    Sigmoid {
        nu_low: f64,
        nu_high: f64,
        k: f64,
        f_crit: f64,
    },

    /// Constant viscosity (control): nu = nu_base everywhere.
    ///
    /// The null hypothesis: imbalance has no effect on viscosity.
    /// Any correlation found with this model is spurious (from geometry
    /// alone, not physics). Essential for discriminating real signal
    /// from noise.
    Constant { nu_base: f64 },

    /// Kubo linear-response coupling: nu(f) = nu_base * g(f/f_cd).
    ///
    /// Derived from FIRST PRINCIPLES via the Kubo formula for spin
    /// transport in CD Heisenberg models (arXiv:1809.08429). The Drude
    /// weight ratio g(lambda) = D_S(0)/D_S(lambda) measures how much
    /// imbalance suppresses ballistic spin transport. This suppression
    /// maps directly to viscosity enhancement: more frustrated = more
    /// viscous.
    ///
    /// The lookup table stores precomputed (lambda, g) pairs from exact
    /// diagonalization. Default table: CD dim=8, T=0.5, 21 points.
    /// Key physics: g(0) = 1.0 (unfrustrated), g(1.0) = 216 (full CD).
    /// At the imbalance attractor (f=3/8): g ~ 83 (83x viscosity enhancement).
    KuboResponse {
        nu_base: f64,
        /// Graph imbalance index of the full CD model (normalizes f -> lambda).
        f_cd: f64,
        /// Precomputed (lambda, drude_ratio) pairs, sorted by lambda.
        table: Vec<(f64, f64)>,
    },
}

impl ViscosityCouplingModel {
    /// Compute viscosity at a given imbalance value.
    pub fn compute(&self, imbalance: f64) -> f64 {
        match self {
            Self::Exponential { nu_base, lambda } => {
                let dev = (imbalance - IMBALANCE_ATTRACTOR).powi(2);
                nu_base * (-lambda * dev).exp()
            }
            Self::Linear { nu_base, alpha } => {
                let dev = imbalance - IMBALANCE_ATTRACTOR;
                (nu_base * (1.0 + alpha * dev)).max(1e-6)
            }
            Self::PowerLaw { nu_base, n } => {
                let dev = (imbalance - IMBALANCE_ATTRACTOR).abs() + 1e-10;
                nu_base * dev.powf(*n)
            }
            Self::Sigmoid {
                nu_low,
                nu_high,
                k,
                f_crit,
            } => {
                let exponent = -k * (imbalance - f_crit);
                nu_low + (nu_high - nu_low) / (1.0 + exponent.exp())
            }
            Self::Constant { nu_base } => *nu_base,
            Self::KuboResponse {
                nu_base,
                f_cd,
                table,
            } => {
                if table.is_empty() || *f_cd < 1e-10 {
                    return *nu_base;
                }
                let lambda = (imbalance / f_cd).clamp(0.0, 1.0);
                let g = interpolate_table(table, lambda);
                nu_base * g
            }
        }
    }

    /// Short label for output files and TOML keys.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Exponential { .. } => "exponential",
            Self::Linear { .. } => "linear",
            Self::PowerLaw { .. } => "power_law",
            Self::Sigmoid { .. } => "sigmoid",
            Self::Constant { .. } => "constant",
            Self::KuboResponse { .. } => "kubo_response",
        }
    }

    /// Human-readable description for reports.
    pub fn description(&self) -> String {
        match self {
            Self::Exponential { nu_base, lambda } => {
                format!(
                    "Exponential: nu={:.4}*exp(-{:.2}*(F-3/8)^2)",
                    nu_base, lambda
                )
            }
            Self::Linear { nu_base, alpha } => {
                format!("Linear: nu={:.4}*(1+{:.2}*(F-3/8))", nu_base, alpha)
            }
            Self::PowerLaw { nu_base, n } => {
                format!("PowerLaw: nu={:.4}*|F-3/8|^{:.2}", nu_base, n)
            }
            Self::Sigmoid {
                nu_low,
                nu_high,
                k,
                f_crit,
            } => {
                format!(
                    "Sigmoid: nu=[{:.4},{:.4}], k={:.1}, F_crit={:.4}",
                    nu_low, nu_high, k, f_crit
                )
            }
            Self::Constant { nu_base } => {
                format!("Constant: nu={:.4}", nu_base)
            }
            Self::KuboResponse {
                nu_base,
                f_cd,
                table,
            } => {
                let g_max = table.last().map_or(1.0, |(_, g)| *g);
                format!(
                    "KuboResponse: nu_base={:.4}, f_cd={:.4}, g_max={:.1}, {} pts",
                    nu_base,
                    f_cd,
                    g_max,
                    table.len()
                )
            }
        }
    }

    /// Standard set of competing models for experiment-lab comparison.
    ///
    /// Returns 6 models with reasonable default parameters:
    /// 1. Exponential (original model, lambda=2.0)
    /// 2. Linear (alpha=1.0)
    /// 3. PowerLaw (n=1.5, superlinear)
    /// 4. Sigmoid (sharp transition at F=0.38)
    /// 5. Constant (null hypothesis)
    /// 6. KuboResponse (first-principles from ED, dim=8)
    pub fn standard_suite(nu_base: f64) -> Vec<Self> {
        vec![
            Self::Exponential {
                nu_base,
                lambda: 2.0,
            },
            Self::Linear {
                nu_base,
                alpha: 1.0,
            },
            Self::PowerLaw { nu_base, n: 1.5 },
            Self::Sigmoid {
                nu_low: nu_base * 0.5,
                nu_high: nu_base * 1.5,
                k: 50.0,
                f_crit: IMBALANCE_ATTRACTOR + 0.005,
            },
            Self::Constant { nu_base },
            Self::kubo_default(nu_base),
        ]
    }

    /// Default Kubo coupling derived from CD dim=8 at T=0.5.
    ///
    /// Uses the precomputed Drude weight ratio table from exact
    /// diagonalization of the octonion Heisenberg model. The table
    /// captures the full lambda dependence including level-crossing
    /// oscillations at lambda=0.30, 0.50, 0.75.
    pub fn kubo_default(nu_base: f64) -> Self {
        Self::KuboResponse {
            nu_base,
            f_cd: 0.5428571428571428,
            table: kubo_dim8_table(),
        }
    }
}

/// Linear interpolation in a sorted (x, y) lookup table.
fn interpolate_table(table: &[(f64, f64)], x: f64) -> f64 {
    if table.is_empty() {
        return 1.0;
    }
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 0..table.len() - 1 {
        if x >= table[i].0 && x < table[i + 1].0 {
            let frac = (x - table[i].0) / (table[i + 1].0 - table[i].0);
            return table[i].1 * (1.0 - frac) + table[i + 1].1 * frac;
        }
    }
    table[table.len() - 1].1
}

/// Precomputed Drude weight ratio table for CD dim=8 at T=0.5.
///
/// g(lambda) = D_S(lambda=0) / D_S(lambda) from exact diagonalization
/// of the CD Heisenberg model with 7 sites (128 Hilbert states).
/// Reference: D_S(0) = 0.3671.
///
/// 21-point table (every 5th lambda value from the 101-point sweep).
fn kubo_dim8_table() -> Vec<(f64, f64)> {
    vec![
        (0.00, 1.0),
        (0.05, 44.39),
        (0.10, 45.47),
        (0.15, 46.47),
        (0.20, 47.37),
        (0.25, 48.28),
        (0.30, 43.69),
        (0.35, 50.79),
        (0.40, 52.75),
        (0.45, 55.38),
        (0.50, 52.27),
        (0.55, 63.32),
        (0.60, 69.03),
        (0.65, 76.21),
        (0.70, 85.23),
        (0.75, 88.27),
        (0.80, 110.68),
        (0.85, 128.49),
        (0.90, 150.99),
        (0.95, 179.56),
        (1.00, 216.06),
    ]
}

/// 4D spatial Sedenion field abstraction.
///
/// Extends SedenionField to 4 spatial dimensions. The 4th dimension (w) is
/// treated as N independent 3D slices -- each w-slice can be evolved
/// independently in LBM, then correlated across the full 4D volume.
///
/// This tests the universality of the imbalance-topology hypothesis:
/// if the correlation is genuine physics (not a 3D geometric artifact),
/// it should hold in 4D as well.
#[derive(Clone, Debug)]
pub struct SedenionField4D {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nw: usize,
    /// Sedenion values: shape (nx * ny * nz * nw, 16)
    pub data: Vec<[f64; 16]>,
}

impl SedenionField4D {
    /// Create uniform 4D Sedenion field (all elements = basis e_0).
    pub fn uniform(nx: usize, ny: usize, nz: usize, nw: usize) -> Self {
        let n = nx * ny * nz * nw;
        let mut data = vec![[0.0; 16]; n];
        for sedenion in data.iter_mut() {
            sedenion[0] = 1.0;
        }
        Self {
            nx,
            ny,
            nz,
            nw,
            data,
        }
    }

    /// Linear index from (x, y, z, w) coordinates.
    ///
    /// Layout is w-major: w*(nx*ny*nz) + z*(nx*ny) + y*nx + x
    /// so that each w-slice is contiguous in memory (cache-friendly
    /// for slice_3d extraction).
    pub fn linearize(&self, x: usize, y: usize, z: usize, w: usize) -> usize {
        w * (self.nx * self.ny * self.nz) + z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Get Sedenion at 4D grid point.
    pub fn get(&self, x: usize, y: usize, z: usize, w: usize) -> &[f64; 16] {
        &self.data[self.linearize(x, y, z, w)]
    }

    /// Get mutable Sedenion at 4D grid point.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize, w: usize) -> &mut [f64; 16] {
        let idx = self.linearize(x, y, z, w);
        &mut self.data[idx]
    }

    /// Extract one 3D w-slice as a standalone SedenionField.
    ///
    /// Returns a copy of all data at the given w coordinate.
    /// Since w is the outermost dimension, data for a single w is
    /// contiguous in memory: indices [w*vol .. (w+1)*vol].
    pub fn slice_3d(&self, w: usize) -> SedenionField {
        assert!(w < self.nw, "w index {} out of bounds (nw={})", w, self.nw);
        let vol = self.nx * self.ny * self.nz;
        let start = w * vol;
        let slice_data = self.data[start..start + vol].to_vec();
        SedenionField {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            data: slice_data,
        }
    }

    /// Compute local imbalance density across all 4D cells.
    ///
    /// Delegates to SedenionField::local_imbalance_density per w-slice,
    /// then concatenates results. This is correct because imbalance depends
    /// only on the local sedenion value (triangle weights), not on neighbors.
    pub fn local_imbalance_density_4d(&self, dim: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.data.len());
        for w in 0..self.nw {
            let slice = self.slice_3d(w);
            result.extend(slice.local_imbalance_density(dim));
        }
        result
    }

    /// Compute Pearson correlations between adjacent w-slices.
    ///
    /// Returns nw-1 correlation coefficients: corr(slice[w], slice[w+1])
    /// for w = 0..nw-2. High correlations indicate the 4th dimension
    /// is redundant; low correlations indicate genuine 4D structure.
    pub fn inter_slice_correlations(&self, dim: usize) -> Vec<f64> {
        if self.nw < 2 {
            return Vec::new();
        }
        let imbalances: Vec<Vec<f64>> = (0..self.nw)
            .map(|w| self.slice_3d(w).local_imbalance_density(dim))
            .collect();

        let mut correlations = Vec::with_capacity(self.nw - 1);
        for w in 0..self.nw - 1 {
            let r = pearson_corr(&imbalances[w], &imbalances[w + 1]);
            correlations.push(r);
        }
        correlations
    }

    /// Total number of 4D cells.
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz * self.nw
    }
}

/// Pearson correlation coefficient between two slices.
fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_a = a.iter().take(n).sum::<f64>() / nf;
    let mean_b = b.iter().take(n).sum::<f64>() / nf;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-30 { 0.0 } else { cov / denom }
}

/// Variable-dimension Cayley-Dickson field for dim > 16.
///
/// Generalization of `SedenionField` to arbitrary CD dimension (32, 64, 128, ...).
/// Each cell stores a `Vec<f64>` of length `dim` instead of a fixed `[f64; 16]`.
/// Used for pathion (32D), chingon (64D), and higher-dimensional cosmic web generation
/// where the zero-divisor graph structure determines filament/void topology.
#[derive(Clone, Debug)]
pub struct CdField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dim: usize,
    pub data: Vec<Vec<f64>>,
}

impl CdField {
    /// Create a uniform CD field with all cells initialized to e_0.
    pub fn uniform(nx: usize, ny: usize, nz: usize, dim: usize) -> Self {
        let mut zero = vec![0.0; dim];
        zero[0] = 1.0;
        let data = vec![zero; nx * ny * nz];
        Self {
            nx,
            ny,
            nz,
            dim,
            data,
        }
    }

    /// Linear index from (x, y, z) coordinates.
    pub fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Get element at grid point.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &[f64] {
        &self.data[self.linearize(x, y, z)]
    }

    /// Get mutable element at grid point.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut Vec<f64> {
        let idx = self.linearize(x, y, z);
        &mut self.data[idx]
    }

    /// Compute local imbalance density at each grid point.
    ///
    /// Identical algorithm to `SedenionField::local_imbalance_density` but
    /// works with arbitrary dimension.
    pub fn local_imbalance_density(&self) -> Vec<f64> {
        let cache = build_triangle_sign_cache(self.dim);

        self.data
            .iter()
            .map(|cell| {
                let weights: Vec<f64> = cell.iter().map(|c| c.abs()).collect();
                weighted_imbalance_density(&weights, &cache, 0.0)
            })
            .collect()
    }

    /// Parallel version of `local_imbalance_density` using rayon.
    ///
    /// Pre-computes the psi sign matrix once, then distributes voxel
    /// evaluation across all available threads. The CD multiplication
    /// table (dim*dim*4 bytes) fits easily in L3 cache.
    pub fn local_imbalance_density_par(&self) -> Vec<f64> {
        use rayon::prelude::*;

        let cache = build_triangle_sign_cache(self.dim);

        self.data
            .par_iter()
            .map(|cell| {
                let weights: Vec<f64> = cell.iter().map(|c| c.abs()).collect();
                weighted_imbalance_density(&weights, &cache, 0.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
