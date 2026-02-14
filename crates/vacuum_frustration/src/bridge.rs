//! Bridge between signed graph frustration and LBM 3D fluid dynamics.
//!
//! Implements the frustration-viscosity coupling principle (Thesis 1):
//! Fluid viscosity in spacetime emerges from algebraic frustration density
//! in Cayley-Dickson signed graphs. Local viscosity nu(x,y,z) is derived from
//! per-cell frustration index via spatial evolution of Sedenion field.

use crate::signed_graph::SignedGraph;

/// Spatial Sedenion field abstraction.
///
/// Represents evolution of 16-dimensional Sedenion algebra elements
/// across a 3D lattice grid. Used to compute local frustration density
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

    /// Compute local frustration density at each grid point.
    ///
    /// Uses the Cayley-Dickson psi matrix as the base signed graph, with
    /// Sedenion component magnitudes as vertex weights. The weighted
    /// triangle frustration is:
    ///
    ///   F(x) = sum_T w_i*w_j*w_k * frustrated(T) / sum_T w_i*w_j*w_k
    ///
    /// where w_i = |sedenion[i]| and frustrated(T) = 1 iff psi(i,j)*psi(j,k)*psi(i,k) = -1.
    ///
    /// At uniform weights, this reduces to the global CD frustration (~3/8).
    /// Spatially varying Sedenion fields produce spatially varying frustration
    /// because different components dominate at different grid points,
    /// weighting different triangles.
    pub fn local_frustration_density(&self, dim: usize) -> Vec<f64> {
        use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;

        // Precompute psi sign matrix (dim x dim) once for all cells.
        // Use a flat vec indexed by (i * dim + j) to avoid clippy range-loop lint.
        let mut psi_flat = vec![0i32; dim * dim];
        for i in 0..dim {
            for j in (i + 1)..dim {
                let sign = cd_basis_mul_sign(dim, i, j);
                psi_flat[i * dim + j] = sign;
                psi_flat[j * dim + i] = sign;
            }
        }

        // Precompute which triangles are frustrated (topology-independent).
        let mut triangles: Vec<(usize, usize, usize, bool)> = Vec::new();
        for i in 0..dim {
            for j in (i + 1)..dim {
                for k in (j + 1)..dim {
                    let product = psi_flat[i * dim + j]
                        * psi_flat[j * dim + k]
                        * psi_flat[i * dim + k];
                    triangles.push((i, j, k, product == -1));
                }
            }
        }

        self.data
            .iter()
            .map(|sedenion| {
                let magnitude_sum: f64 = sedenion[0..dim].iter().map(|x| x.abs()).sum();

                if magnitude_sum < 1e-10 {
                    return 0.375;
                }

                let weights: Vec<f64> =
                    sedenion[0..dim].iter().map(|x| x.abs()).collect();

                let mut weighted_frustrated = 0.0f64;
                let mut weighted_total = 0.0f64;

                for &(i, j, k, is_frustrated) in &triangles {
                    let w = weights[i] * weights[j] * weights[k];
                    weighted_total += w;
                    if is_frustrated {
                        weighted_frustrated += w;
                    }
                }

                if weighted_total < 1e-30 {
                    0.375
                } else {
                    weighted_frustrated / weighted_total
                }
            })
            .collect()
    }
}

impl SedenionField {
    /// Compute local frustration density in parallel using rayon.
    ///
    /// Identical results to `local_frustration_density` but uses rayon's
    /// par_iter() for ~Nx speedup on multi-core machines.
    pub fn local_frustration_density_par(&self, dim: usize) -> Vec<f64> {
        use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;
        use rayon::prelude::*;

        // Precompute psi sign matrix
        let mut psi_flat = vec![0i32; dim * dim];
        for i in 0..dim {
            for j in (i + 1)..dim {
                let sign = cd_basis_mul_sign(dim, i, j);
                psi_flat[i * dim + j] = sign;
                psi_flat[j * dim + i] = sign;
            }
        }

        // Precompute frustrated triangles
        let triangles: Vec<(usize, usize, usize, bool)> = {
            let mut tris = Vec::new();
            for i in 0..dim {
                for j in (i + 1)..dim {
                    for k in (j + 1)..dim {
                        let product = psi_flat[i * dim + j]
                            * psi_flat[j * dim + k]
                            * psi_flat[i * dim + k];
                        tris.push((i, j, k, product == -1));
                    }
                }
            }
            tris
        };

        self.data
            .par_iter()
            .map(|sedenion| {
                let magnitude_sum: f64 = sedenion[0..dim].iter().map(|x| x.abs()).sum();

                if magnitude_sum < 1e-10 {
                    return 0.375;
                }

                let weights: Vec<f64> =
                    sedenion[0..dim].iter().map(|x| x.abs()).collect();

                let mut weighted_frustrated = 0.0f64;
                let mut weighted_total = 0.0f64;

                for &(i, j, k, is_frustrated) in &triangles {
                    let w = weights[i] * weights[j] * weights[k];
                    weighted_total += w;
                    if is_frustrated {
                        weighted_frustrated += w;
                    }
                }

                if weighted_total < 1e-30 {
                    0.375
                } else {
                    weighted_frustrated / weighted_total
                }
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
    /// Uses CD basis multiplication from `algebra_core` for genuine associators.
    pub fn local_associator_norm_field(&self, dim: usize) -> Vec<f64> {
        use algebra_core::construction::cayley_dickson::cd_multiply;

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

/// Frustration-Viscosity coupling bridge.
///
/// Transforms signed-graph frustration density into spatially-varying
/// kinematic viscosity field for LBM simulation.
#[derive(Clone, Debug)]
pub struct FrustrationViscosityBridge {
    pub dim: usize,
    pub signed_graph: SignedGraph,
}

impl FrustrationViscosityBridge {
    /// Create bridge from Cayley-Dickson dimension.
    pub fn new(dim: usize) -> Self {
        use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;

        let signed_graph = SignedGraph::from_psi_matrix(dim, |i, j| cd_basis_mul_sign(dim, i, j));

        Self { dim, signed_graph }
    }

    /// Convert local frustration density to kinematic viscosity.
    ///
    /// Viscosity relation:
    /// nu(x) = nu_base * exp(-lambda * (F(x) - 3/8)^2)
    ///
    /// where F(x) is local frustration density and 3/8 is the vacuum attractor.
    ///
    /// # Arguments
    /// * `frustration_field` - Local frustration density [0,1] at each grid point
    /// * `nu_base` - Base kinematic viscosity (e.g., 1/3 for inviscid limit)
    /// * `lambda` - Coupling strength (typical: 1.0-2.0)
    pub fn frustration_to_viscosity(
        &self,
        frustration_field: &[f64],
        nu_base: f64,
        lambda: f64,
    ) -> Vec<f64> {
        const VACUUM_ATTRACTOR: f64 = 3.0 / 8.0;

        frustration_field
            .iter()
            .map(|&f| {
                let deviation = (f - VACUUM_ATTRACTOR).powi(2);
                nu_base * (-lambda * deviation).exp()
            })
            .collect()
    }

    /// Full pipeline: Sedenion field -> Frustration -> Viscosity.
    ///
    /// Computes spatially-varying viscosity field from Sedenion field evolution.
    pub fn compute_viscosity_field(
        &self,
        sedenion_field: &SedenionField,
        nu_base: f64,
        lambda: f64,
    ) -> Vec<f64> {
        let frustration = sedenion_field.local_frustration_density(self.dim);
        self.frustration_to_viscosity(&frustration, nu_base, lambda)
    }

    /// Convert frustration field to viscosity using a specified coupling model.
    ///
    /// This is the multi-model generalization of `frustration_to_viscosity`.
    /// Different models make different physical assumptions about how algebraic
    /// frustration modulates fluid viscosity:
    ///
    /// - **Exponential**: Gaussian decay from vacuum attractor (original model)
    /// - **Linear**: First-order perturbation theory around vacuum
    /// - **PowerLaw**: Scale-free coupling (fractal-like)
    /// - **Sigmoid**: Phase-transition model with sharp crossover
    /// - **Constant**: Control model (frustration has no effect)
    pub fn frustration_to_viscosity_model(
        &self,
        frustration_field: &[f64],
        model: &ViscosityCouplingModel,
    ) -> Vec<f64> {
        frustration_field
            .iter()
            .map(|&f| model.compute(f))
            .collect()
    }
}

/// Vacuum attractor frustration index for sedenions (~3/8 = 0.375).
///
/// This is the equilibrium frustration for uniform-weight sedenion fields.
/// All coupling models are centered on this value.
pub const VACUUM_ATTRACTOR: f64 = 3.0 / 8.0;

/// Coupling model for converting frustration density to kinematic viscosity.
///
/// Each model represents a different physical hypothesis about how algebraic
/// frustration in Cayley-Dickson signed graphs modulates fluid viscosity.
/// The experiment-lab runs all models in parallel to discriminate between
/// hypotheses using correlation strength.
#[derive(Debug, Clone)]
pub enum ViscosityCouplingModel {
    /// Gaussian decay: nu = nu_base * exp(-lambda * (F - F0)^2)
    ///
    /// Physical interpretation: viscosity peaks at vacuum attractor and
    /// decays symmetrically for deviations. Quadratic sensitivity means
    /// small deviations have little effect, large deviations strongly reduce
    /// viscosity. This is the natural "perturbative" model.
    Exponential {
        nu_base: f64,
        lambda: f64,
    },

    /// Linear coupling: nu = nu_base * (1 + alpha * (F - F0))
    ///
    /// Physical interpretation: first-order Taylor expansion around vacuum.
    /// Positive alpha means frustrated regions are more viscous (dissipative).
    /// Negative alpha means frustrated regions flow more freely.
    /// Simplest possible coupling; serves as baseline.
    Linear {
        nu_base: f64,
        alpha: f64,
    },

    /// Power-law coupling: nu = nu_base * (|F - F0| + eps)^n
    ///
    /// Physical interpretation: scale-free (fractal) relationship between
    /// frustration deviation and viscosity. No preferred scale means the
    /// coupling has self-similar structure across frustration magnitudes.
    /// n > 1 = superlinear sensitivity; n < 1 = sublinear (saturating).
    PowerLaw {
        nu_base: f64,
        n: f64,
    },

    /// Sigmoid (logistic) coupling: nu = nu_low + (nu_high - nu_low) / (1 + exp(-k*(F - F_crit)))
    ///
    /// Physical interpretation: sharp phase transition at critical frustration.
    /// Below F_crit, viscosity is nu_low; above, it jumps to nu_high.
    /// Steepness k controls how sharp the transition is.
    /// Models a frustration-driven "phase boundary" in the fluid.
    Sigmoid {
        nu_low: f64,
        nu_high: f64,
        k: f64,
        f_crit: f64,
    },

    /// Constant viscosity (control): nu = nu_base everywhere.
    ///
    /// The null hypothesis: frustration has no effect on viscosity.
    /// Any correlation found with this model is spurious (from geometry
    /// alone, not physics). Essential for discriminating real signal
    /// from noise.
    Constant {
        nu_base: f64,
    },

    /// Kubo linear-response coupling: nu(f) = nu_base * g(f/f_cd).
    ///
    /// Derived from FIRST PRINCIPLES via the Kubo formula for spin
    /// transport in CD Heisenberg models (arXiv:1809.08429). The Drude
    /// weight ratio g(lambda) = D_S(0)/D_S(lambda) measures how much
    /// frustration suppresses ballistic spin transport. This suppression
    /// maps directly to viscosity enhancement: more frustrated = more
    /// viscous.
    ///
    /// The lookup table stores precomputed (lambda, g) pairs from exact
    /// diagonalization. Default table: CD dim=8, T=0.5, 21 points.
    /// Key physics: g(0) = 1.0 (unfrustrated), g(1.0) = 216 (full CD).
    /// At the vacuum attractor (f=3/8): g ~ 83 (83x viscosity enhancement).
    KuboResponse {
        nu_base: f64,
        /// Graph frustration index of the full CD model (normalizes f -> lambda).
        f_cd: f64,
        /// Precomputed (lambda, drude_ratio) pairs, sorted by lambda.
        table: Vec<(f64, f64)>,
    },
}

impl ViscosityCouplingModel {
    /// Compute viscosity at a given frustration value.
    pub fn compute(&self, frustration: f64) -> f64 {
        match self {
            Self::Exponential { nu_base, lambda } => {
                let dev = (frustration - VACUUM_ATTRACTOR).powi(2);
                nu_base * (-lambda * dev).exp()
            }
            Self::Linear { nu_base, alpha } => {
                let dev = frustration - VACUUM_ATTRACTOR;
                (nu_base * (1.0 + alpha * dev)).max(1e-6)
            }
            Self::PowerLaw { nu_base, n } => {
                let dev = (frustration - VACUUM_ATTRACTOR).abs() + 1e-10;
                nu_base * dev.powf(*n)
            }
            Self::Sigmoid {
                nu_low,
                nu_high,
                k,
                f_crit,
            } => {
                let exponent = -k * (frustration - f_crit);
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
                let lambda = (frustration / f_cd).clamp(0.0, 1.0);
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
                format!("Exponential: nu={:.4}*exp(-{:.2}*(F-3/8)^2)", nu_base, lambda)
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
                    nu_base, f_cd, g_max, table.len()
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
            Self::PowerLaw {
                nu_base,
                n: 1.5,
            },
            Self::Sigmoid {
                nu_low: nu_base * 0.5,
                nu_high: nu_base * 1.5,
                k: 50.0,
                f_crit: VACUUM_ATTRACTOR + 0.005,
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
/// This tests the universality of the frustration-topology hypothesis:
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

    /// Compute local frustration density across all 4D cells.
    ///
    /// Delegates to SedenionField::local_frustration_density per w-slice,
    /// then concatenates results. This is correct because frustration depends
    /// only on the local sedenion value (triangle weights), not on neighbors.
    pub fn local_frustration_density_4d(&self, dim: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.data.len());
        for w in 0..self.nw {
            let slice = self.slice_3d(w);
            result.extend(slice.local_frustration_density(dim));
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
        let frustrations: Vec<Vec<f64>> = (0..self.nw)
            .map(|w| self.slice_3d(w).local_frustration_density(dim))
            .collect();

        let mut correlations = Vec::with_capacity(self.nw - 1);
        for w in 0..self.nw - 1 {
            let r = pearson_corr(&frustrations[w], &frustrations[w + 1]);
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
    if denom < 1e-30 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_field_creation() {
        let field = SedenionField::uniform(8, 8, 4);
        assert_eq!(field.data.len(), 8 * 8 * 4);
        // Check that e_0 basis is initialized
        assert!((field.get(0, 0, 0)[0] - 1.0).abs() < 1e-14);
        assert!((field.get(7, 7, 3)[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sedenion_field_linearization() {
        let field = SedenionField::uniform(4, 4, 4);
        // Check linearization consistency
        let idx1 = field.linearize(2, 1, 0);
        let idx2 = field.linearize(2, 1, 0);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_frustration_viscosity_bridge_creation() {
        let bridge = FrustrationViscosityBridge::new(16);
        assert_eq!(bridge.dim, 16);
        assert_eq!(bridge.signed_graph.dim, 16);
    }

    #[test]
    fn test_frustration_to_viscosity_vacuum() {
        let bridge = FrustrationViscosityBridge::new(16);
        let frustration = vec![3.0 / 8.0; 100]; // All vacuum
        let viscosity = bridge.frustration_to_viscosity(&frustration, 1.0 / 3.0, 1.0);

        // At vacuum attractor, nu should be nu_base
        for &nu in viscosity.iter() {
            assert!(
                (nu - 1.0 / 3.0).abs() < 1e-10,
                "Expected nu={}, got {}",
                1.0 / 3.0,
                nu
            );
        }
    }

    #[test]
    fn test_frustration_to_viscosity_variance() {
        let bridge = FrustrationViscosityBridge::new(16);
        let frustration = vec![0.2, 0.375, 0.8]; // Varying frustration around vacuum (3/8)
        let viscosity = bridge.frustration_to_viscosity(&frustration, 1.0 / 3.0, 1.0);

        // At vacuum attractor (0.375), viscosity should equal nu_base
        // Away from attractor, viscosity should decrease (exponential decay)
        assert!(
            viscosity[0] < viscosity[1],
            "Viscosity at 0.2 should be less than at 0.375"
        );
        assert!(
            viscosity[1].abs() - (1.0 / 3.0) < 1e-10,
            "Viscosity at vacuum should equal nu_base"
        );
        assert!(
            viscosity[2] < viscosity[1],
            "Viscosity at 0.8 should be less than at 0.375"
        );
    }

    #[test]
    fn test_full_pipeline_uniform_field() {
        let bridge = FrustrationViscosityBridge::new(16);
        let field = SedenionField::uniform(8, 8, 4);
        let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

        // Uniform field should produce roughly uniform viscosity
        assert_eq!(viscosity.len(), 8 * 8 * 4);
        for &nu in viscosity.iter() {
            assert!(nu > 0.0, "Viscosity must be positive");
            assert!(nu < 1.0, "Viscosity should be reasonable");
        }
    }

    #[test]
    fn test_full_pipeline_diverse_field() {
        let bridge = FrustrationViscosityBridge::new(16);
        let mut field = SedenionField::uniform(4, 4, 4);

        // Create varied Sedenion field by modulating multiple components
        // This creates edge variation that affects frustration density
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let sedenion = field.get_mut(x, y, z);
                    // Vary components based on position to create diverse edges
                    let scale = ((x + y + z) as f64) / 12.0;
                    sedenion[0] = 1.0 + 0.3 * scale;
                    sedenion[1] = 0.5 * scale;
                    sedenion[2] = 0.3 * (1.0 - scale);
                    sedenion[3] = 0.4 * scale.sin();
                }
            }
        }

        let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

        // Viscosity should be positive and reasonable (may not vary significantly
        // depending on frustration distribution, but should all be valid)
        for &nu in viscosity.iter() {
            assert!(nu > 0.0, "Viscosity must be positive");
            assert!(nu < 1.0, "Viscosity should be reasonable");
        }
    }

    #[test]
    fn test_viscosity_positivity() {
        let bridge = FrustrationViscosityBridge::new(16);
        let mut field = SedenionField::uniform(4, 4, 4);

        // Extreme frustration values
        field.data[0] = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];

        let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

        // All viscosities must be positive and finite
        for &nu in viscosity.iter() {
            assert!(nu > 0.0, "Viscosity must be positive");
            assert!(nu.is_finite(), "Viscosity must be finite");
        }
    }

    #[test]
    fn test_sedenion_field_mutation() {
        let mut field = SedenionField::uniform(2, 2, 2);
        field.get_mut(0, 0, 0)[5] = 0.5;
        assert!((field.get(0, 0, 0)[5] - 0.5).abs() < 1e-14);
        assert!((field.get(0, 0, 1)[5]).abs() < 1e-14); // Other points unchanged
    }

    #[test]
    fn test_frustration_density_par_matches_sequential() {
        let mut field = SedenionField::uniform(4, 4, 4);
        // Add some variation
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let s = field.get_mut(x, y, z);
                    s[1] = 0.3 * (x as f64) / 4.0;
                    s[5] = 0.2 * (y as f64) / 4.0;
                }
            }
        }

        let sequential = field.local_frustration_density(16);
        let parallel = field.local_frustration_density_par(16);

        assert_eq!(sequential.len(), parallel.len());
        for (s, p) in sequential.iter().zip(parallel.iter()) {
            assert!(
                (s - p).abs() < 1e-14,
                "Parallel result differs: {} vs {}",
                s,
                p
            );
        }
    }

    #[test]
    fn test_frustration_density_par_uniform() {
        let field = SedenionField::uniform(8, 8, 4);
        let par_result = field.local_frustration_density_par(16);
        let seq_result = field.local_frustration_density(16);
        for (s, p) in seq_result.iter().zip(par_result.iter()) {
            assert!((s - p).abs() < 1e-14);
        }
    }

    #[test]
    fn test_associator_norm_uniform_field() {
        // Uniform e_0 field: all products are (1)(1)(1)=1 (real),
        // so associator is zero (reals are associative).
        let field = SedenionField::uniform(4, 4, 4);
        let norms = field.local_associator_norm_field(16);
        assert_eq!(norms.len(), 4 * 4 * 4);
        for &n in &norms {
            assert!(n.abs() < 1e-10, "Uniform e_0 should have zero associator");
        }
    }

    #[test]
    fn test_associator_norm_varied_field() {
        // Create a field with multiple nonzero basis elements to trigger
        // non-associativity
        let mut field = SedenionField::uniform(4, 4, 4);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let s = field.get_mut(x, y, z);
                    let i = (x + y + z) % 16;
                    s[i] = 0.8;
                    s[(i + 3) % 16] = 0.5;
                    s[(i + 7) % 16] = 0.3;
                }
            }
        }

        let norms = field.local_associator_norm_field(16);

        // At least some points should have nonzero associator norm
        let max_norm = norms.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_norm > 1e-4,
            "Varied sedenion field should have nonzero associators: max={}",
            max_norm
        );

        // All norms should be finite and non-negative
        for &n in &norms {
            assert!(n.is_finite());
            assert!(n >= 0.0);
        }
    }

    #[test]
    fn test_associator_norm_field_length() {
        let field = SedenionField::uniform(8, 6, 4);
        let norms = field.local_associator_norm_field(16);
        assert_eq!(norms.len(), 8 * 6 * 4);
    }

    // ---- ViscosityCouplingModel tests ----

    #[test]
    fn test_exponential_model_at_vacuum() {
        let model = ViscosityCouplingModel::Exponential {
            nu_base: 1.0 / 3.0,
            lambda: 2.0,
        };
        let nu = model.compute(VACUUM_ATTRACTOR);
        assert!(
            (nu - 1.0 / 3.0).abs() < 1e-10,
            "At vacuum attractor, exponential model should return nu_base"
        );
    }

    #[test]
    fn test_exponential_model_away_from_vacuum() {
        let model = ViscosityCouplingModel::Exponential {
            nu_base: 1.0 / 3.0,
            lambda: 2.0,
        };
        let nu_low = model.compute(0.1);
        let nu_vac = model.compute(VACUUM_ATTRACTOR);
        assert!(
            nu_low < nu_vac,
            "Away from attractor, viscosity should decrease"
        );
    }

    #[test]
    fn test_linear_model_positive_alpha() {
        let model = ViscosityCouplingModel::Linear {
            nu_base: 0.1,
            alpha: 1.0,
        };
        let nu_low = model.compute(0.2); // F < 3/8
        let nu_high = model.compute(0.5); // F > 3/8
        assert!(
            nu_high > nu_low,
            "Positive alpha: higher frustration -> higher viscosity"
        );
    }

    #[test]
    fn test_power_law_model_superlinear() {
        let model = ViscosityCouplingModel::PowerLaw {
            nu_base: 0.1,
            n: 2.0,
        };
        let nu_close = model.compute(VACUUM_ATTRACTOR + 0.01);
        let nu_far = model.compute(VACUUM_ATTRACTOR + 0.1);
        // Superlinear: far deviation grows faster than linearly
        let ratio_dev = 0.1 / 0.01;
        let ratio_nu = nu_far / nu_close;
        assert!(
            ratio_nu > ratio_dev,
            "Superlinear (n=2): nu ratio ({:.2}) should exceed deviation ratio ({:.2})",
            ratio_nu,
            ratio_dev
        );
    }

    #[test]
    fn test_sigmoid_model_transition() {
        let model = ViscosityCouplingModel::Sigmoid {
            nu_low: 0.05,
            nu_high: 0.5,
            k: 100.0,
            f_crit: 0.38,
        };
        let nu_below = model.compute(0.30);
        let nu_above = model.compute(0.45);
        assert!(
            (nu_below - 0.05).abs() < 0.01,
            "Below F_crit, sigmoid should approach nu_low: got {}",
            nu_below
        );
        assert!(
            (nu_above - 0.5).abs() < 0.01,
            "Above F_crit, sigmoid should approach nu_high: got {}",
            nu_above
        );
    }

    #[test]
    fn test_constant_model_invariance() {
        let model = ViscosityCouplingModel::Constant { nu_base: 0.1 };
        for f in [0.0, 0.2, VACUUM_ATTRACTOR, 0.5, 1.0] {
            assert!(
                (model.compute(f) - 0.1).abs() < 1e-14,
                "Constant model should return nu_base at all frustrations"
            );
        }
    }

    #[test]
    fn test_all_models_finite_positive() {
        let suite = ViscosityCouplingModel::standard_suite(1.0 / 3.0);
        for model in &suite {
            for &f in &[0.0, 0.1, 0.2, VACUUM_ATTRACTOR, 0.5, 0.8, 1.0] {
                let nu = model.compute(f);
                assert!(
                    nu.is_finite(),
                    "{}: nu not finite at F={}",
                    model.label(),
                    f
                );
                assert!(
                    nu > 0.0,
                    "{}: nu not positive at F={}: got {}",
                    model.label(),
                    f,
                    nu
                );
            }
        }
    }

    #[test]
    fn test_standard_suite_has_six_models() {
        let suite = ViscosityCouplingModel::standard_suite(0.1);
        assert_eq!(suite.len(), 6);

        let labels: Vec<&str> = suite.iter().map(|m| m.label()).collect();
        assert!(labels.contains(&"exponential"));
        assert!(labels.contains(&"linear"));
        assert!(labels.contains(&"power_law"));
        assert!(labels.contains(&"sigmoid"));
        assert!(labels.contains(&"constant"));
        assert!(labels.contains(&"kubo_response"));
    }

    #[test]
    fn test_bridge_multi_model_integration() {
        let bridge = FrustrationViscosityBridge::new(16);
        let frustration = vec![0.3, VACUUM_ATTRACTOR, 0.45];
        let model = ViscosityCouplingModel::Exponential {
            nu_base: 1.0 / 3.0,
            lambda: 1.0,
        };
        let viscosity = bridge.frustration_to_viscosity_model(&frustration, &model);
        assert_eq!(viscosity.len(), 3);
        for &nu in &viscosity {
            assert!(nu.is_finite() && nu > 0.0);
        }
    }

    #[test]
    fn test_model_descriptions_non_empty() {
        let suite = ViscosityCouplingModel::standard_suite(0.1);
        for model in &suite {
            assert!(!model.description().is_empty());
            assert!(!model.label().is_empty());
        }
    }

    // ---- KuboResponse model tests ----

    #[test]
    fn test_kubo_response_at_zero_frustration() {
        let model = ViscosityCouplingModel::kubo_default(1.0 / 3.0);
        let nu = model.compute(0.0);
        // At f=0, lambda=0, g=1.0, so nu = nu_base
        assert!(
            (nu - 1.0 / 3.0).abs() < 1e-10,
            "At zero frustration, KuboResponse should return nu_base, got {}",
            nu
        );
    }

    #[test]
    fn test_kubo_response_monotonic_near_vacuum() {
        let model = ViscosityCouplingModel::kubo_default(0.1);
        // Near the vacuum attractor, viscosity should increase with frustration
        let nu_low = model.compute(0.30);
        let nu_vac = model.compute(0.375);
        let nu_high = model.compute(0.45);
        assert!(
            nu_low < nu_vac,
            "nu should increase toward vacuum: nu(0.30)={} > nu(0.375)={}",
            nu_low, nu_vac
        );
        assert!(
            nu_vac < nu_high,
            "nu should increase beyond vacuum: nu(0.375)={} > nu(0.45)={}",
            nu_vac, nu_high
        );
    }

    #[test]
    fn test_kubo_response_enhancement_ratio() {
        let model = ViscosityCouplingModel::kubo_default(1.0);
        let nu_zero = model.compute(0.0);
        let nu_full = model.compute(0.5429); // full CD frustration
        let ratio = nu_full / nu_zero;
        // Full CD: g(1.0) = 216, so ratio should be ~216
        assert!(
            ratio > 200.0 && ratio < 230.0,
            "Full CD enhancement ratio should be ~216, got {}",
            ratio
        );
    }

    #[test]
    fn test_kubo_response_vacuum_attractor_enhancement() {
        let model = ViscosityCouplingModel::kubo_default(1.0);
        let nu_zero = model.compute(0.0);
        let nu_vac = model.compute(VACUUM_ATTRACTOR);
        let ratio = nu_vac / nu_zero;
        // At vacuum: f=0.375, lambda=0.375/0.5429=0.691, g~83
        assert!(
            ratio > 60.0 && ratio < 110.0,
            "Vacuum attractor enhancement should be ~83, got {}",
            ratio
        );
    }

    #[test]
    fn test_kubo_response_all_finite() {
        let model = ViscosityCouplingModel::kubo_default(0.1);
        for i in 0..100 {
            let f = i as f64 * 0.01;
            let nu = model.compute(f);
            assert!(
                nu.is_finite() && nu > 0.0,
                "KuboResponse not finite/positive at f={}: nu={}",
                f, nu
            );
        }
    }

    #[test]
    fn test_interpolate_table_boundary() {
        let table = vec![(0.0, 1.0), (0.5, 50.0), (1.0, 100.0)];
        assert!((interpolate_table(&table, -0.1) - 1.0).abs() < 1e-10);
        assert!((interpolate_table(&table, 0.0) - 1.0).abs() < 1e-10);
        assert!((interpolate_table(&table, 0.25) - 25.5).abs() < 1e-10);
        assert!((interpolate_table(&table, 0.5) - 50.0).abs() < 1e-10);
        assert!((interpolate_table(&table, 1.0) - 100.0).abs() < 1e-10);
        assert!((interpolate_table(&table, 1.5) - 100.0).abs() < 1e-10);
    }

    // ---- SedenionField4D tests ----

    #[test]
    fn test_sedenion_field_4d_creation() {
        let field = SedenionField4D::uniform(4, 4, 4, 3);
        assert_eq!(field.data.len(), 4 * 4 * 4 * 3);
        assert_eq!(field.n_cells(), 192);
        // Check e_0 basis is initialized
        assert!((field.get(0, 0, 0, 0)[0] - 1.0).abs() < 1e-14);
        assert!((field.get(3, 3, 3, 2)[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sedenion_field_4d_linearize() {
        let field = SedenionField4D::uniform(4, 4, 4, 3);
        // w-major layout: w=0 occupies [0..64), w=1 occupies [64..128)
        assert_eq!(field.linearize(0, 0, 0, 0), 0);
        assert_eq!(field.linearize(0, 0, 0, 1), 4 * 4 * 4);
        assert_eq!(field.linearize(1, 0, 0, 0), 1);
        assert_eq!(field.linearize(0, 1, 0, 0), 4);
        assert_eq!(field.linearize(0, 0, 1, 0), 4 * 4);
        // Round-trip consistency
        let idx1 = field.linearize(2, 3, 1, 2);
        let idx2 = field.linearize(2, 3, 1, 2);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_sedenion_field_4d_slice_isolation() {
        let mut field = SedenionField4D::uniform(4, 4, 4, 3);
        // Modify w=1 slice only
        field.get_mut(2, 1, 0, 1)[5] = 0.99;

        let slice0 = field.slice_3d(0);
        let slice1 = field.slice_3d(1);
        let slice2 = field.slice_3d(2);

        // w=0 and w=2 should be unaffected
        assert!(slice0.get(2, 1, 0)[5].abs() < 1e-14);
        assert!(slice2.get(2, 1, 0)[5].abs() < 1e-14);
        // w=1 should have the modification
        assert!((slice1.get(2, 1, 0)[5] - 0.99).abs() < 1e-14);

        // Slice dimensions match
        assert_eq!(slice0.nx, 4);
        assert_eq!(slice0.ny, 4);
        assert_eq!(slice0.nz, 4);
        assert_eq!(slice0.data.len(), 64);
    }

    #[test]
    fn test_sedenion_field_4d_frustration_length() {
        let field = SedenionField4D::uniform(4, 4, 4, 2);
        let frustration = field.local_frustration_density_4d(16);
        assert_eq!(frustration.len(), 4 * 4 * 4 * 2);
        // Uniform field -> all frustration values equal (vacuum attractor)
        for &f in &frustration {
            assert!(f.is_finite());
            assert!(f > 0.0);
        }
    }

    #[test]
    fn test_sedenion_field_4d_inter_slice_uniform() {
        let field = SedenionField4D::uniform(4, 4, 4, 3);
        let corrs = field.inter_slice_correlations(16);
        // 3 slices -> 2 correlations
        assert_eq!(corrs.len(), 2);
        // Uniform field: all slices identical, so correlation should be
        // undefined (zero variance). Our pearson_corr returns 0.0 for zero variance.
        for &r in &corrs {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_sedenion_field_4d_varied_inter_slice() {
        let mut field = SedenionField4D::uniform(4, 4, 4, 3);
        // Create distinct variation in each w-slice
        for w in 0..3 {
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let s = field.get_mut(x, y, z, w);
                        let xn = x as f64 / 4.0;
                        let wn = w as f64 / 3.0;
                        s[1] = 0.3 * xn * (1.0 + wn);
                        s[3] = 0.2 * (wn + 0.1);
                        s[5] = 0.1 * (y as f64 / 4.0) * (1.0 - wn);
                    }
                }
            }
        }

        let corrs = field.inter_slice_correlations(16);
        assert_eq!(corrs.len(), 2);
        for &r in &corrs {
            assert!(r.is_finite());
            // With variation, correlations should be in [-1, 1]
            assert!(r >= -1.0 - 1e-10 && r <= 1.0 + 1e-10);
        }
    }
}
