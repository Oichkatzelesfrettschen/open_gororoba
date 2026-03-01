// Tensor loops use multi-index access across several arrays; iterators obscure the math.
#![allow(clippy::needless_range_loop)]

//! ADM 3+1 decomposition of spacetime.
//!
//! The Arnowitt-Deser-Misner (ADM) formalism foliates a 4D Lorentzian
//! spacetime (M, g) into a family of spacelike hypersurfaces {Sigma_t}
//! parameterized by a global time function t. The 4-metric decomposes as:
//!
//!   ds^2 = -alpha^2 dt^2 + gamma_{ij}(dx^i + beta^i dt)(dx^j + beta^j dt)
//!
//! where:
//! - alpha: lapse function (proper time between slices)
//! - beta^i: shift vector (coordinate drift along the slice)
//! - gamma_{ij}: induced 3-metric on Sigma_t
//!
//! The extrinsic curvature K_{ij} encodes how Sigma_t bends in 4D:
//!   K_{ij} = -(1/2alpha)(d_t gamma_{ij} - D_i beta_j - D_j beta_i)
//!
//! # Constraint equations
//!
//! The Hamiltonian and momentum constraints must be satisfied on each slice:
//! - Hamiltonian: R^(3) + K^2 - K_{ij}K^{ij} = 16 pi rho
//! - Momentum:    D_j(K^{ij} - gamma^{ij} K) = 8 pi S^i
//!
//! # References
//! - Arnowitt, Deser, Misner (1962): The Dynamics of General Relativity
//! - York (1979): Kinematics and dynamics of general relativity
//! - Gourgoulhon (2012): 3+1 Formalism in General Relativity
//! - Baumgarte & Shapiro (2010): Numerical Relativity

use crate::metric::{DIM, MetricComponents, SpacetimeMetric, T};
use crate::spatial::{
    SDIM, SpatialChristoffel, SpatialMetric, SpatialSymmetricTensor, SpatialVector,
    invert_3x3_symmetric, raise_index_vector, raise_symmetric_tensor, spatial_determinant,
    trace_symmetric,
};

// ============================================================================
// Primary data structures
// ============================================================================

/// Result of the ADM 3+1 decomposition of a spacetime metric.
#[derive(Debug, Clone)]
pub struct AdmDecomposition {
    /// Lapse function alpha (positive).
    pub lapse: f64,
    /// Shift vector beta^i (contravariant).
    pub shift: SpatialVector,
    /// Induced 3-metric gamma_{ij}.
    pub spatial_metric: SpatialMetric,
    /// Inverse spatial metric gamma^{ij}.
    pub spatial_metric_inv: SpatialMetric,
    /// Determinant of the spatial metric det(gamma).
    pub det_gamma: f64,
}

/// Extrinsic curvature data on a hypersurface.
#[derive(Debug, Clone)]
pub struct ExtrinsicCurvatureData {
    /// Extrinsic curvature K_{ij}.
    pub k_ij: SpatialSymmetricTensor,
    /// York time (expansion scalar) theta = -K = -gamma^{ij} K_{ij}.
    /// Positive = volume expanding, negative = contracting.
    pub york_time: f64,
    /// Trace-free part of K_{ij}: A_{ij} = K_{ij} - (1/3) gamma_{ij} K.
    pub tracefree_k: SpatialSymmetricTensor,
}

/// Hamiltonian and momentum constraint residuals.
#[derive(Debug, Clone)]
pub struct AdmConstraints {
    /// Hamiltonian constraint residual:
    /// H = R^(3) + K^2 - K_{ij}K^{ij} - 16 pi rho.
    /// Should be zero for valid initial data.
    pub hamiltonian_residual: f64,
    /// Momentum constraint residual vector:
    /// M^i = D_j(K^{ij} - gamma^{ij}K) - 8 pi S^i.
    pub momentum_residual: SpatialVector,
    /// Energy density rho = n^a n^b T_{ab} projected from the stress-energy.
    pub energy_density: f64,
    /// Momentum density S^i = -gamma^{ia} n^b T_{ab}.
    pub momentum_density: SpatialVector,
}

// ============================================================================
// Decomposition and reconstruction
// ============================================================================

/// Decompose a 4D metric g_{mu nu} into ADM variables.
///
/// Given the covariant metric components:
///   g_{00} = -alpha^2 + beta_k beta^k
///   g_{0i} = beta_i (covariant shift)
///   g_{ij} = gamma_{ij}
///
/// we extract:
///   gamma_{ij} = g_{ij}
///   beta^i = gamma^{ij} g_{0j}
///   alpha = sqrt(beta^i beta_i - g_{00})
///
/// The sign convention follows Baumgarte & Shapiro (2010).
pub fn decompose_metric(g: &MetricComponents) -> AdmDecomposition {
    // Extract spatial metric
    let mut gamma = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            gamma[i][j] = g[i + 1][j + 1];
        }
    }

    let gamma_inv = invert_3x3_symmetric(&gamma);
    let det_gamma = spatial_determinant(&gamma);

    // Extract covariant shift: beta_i = g_{0i}
    let mut beta_lower = [0.0; SDIM];
    beta_lower[..SDIM].copy_from_slice(&g[T][1..(SDIM + 1)]);

    // Raise to get contravariant shift: beta^i = gamma^{ij} beta_j
    let shift = raise_index_vector(&gamma_inv, &beta_lower);

    // Lapse: alpha^2 = beta^i beta_i - g_{00}
    let mut beta_sq = 0.0;
    for i in 0..SDIM {
        beta_sq += shift[i] * beta_lower[i];
    }
    let alpha_sq = beta_sq - g[T][T];
    assert!(
        alpha_sq > 0.0,
        "lapse^2 must be positive (got {alpha_sq}); metric may not have Lorentzian signature"
    );
    let lapse = alpha_sq.sqrt();

    AdmDecomposition {
        lapse,
        shift,
        spatial_metric: gamma,
        spatial_metric_inv: gamma_inv,
        det_gamma,
    }
}

/// Reconstruct the full 4D metric from ADM variables.
///
/// g_{00} = -alpha^2 + beta_k beta^k
/// g_{0i} = beta_i = gamma_{ij} beta^j
/// g_{ij} = gamma_{ij}
pub fn reconstruct_metric(adm: &AdmDecomposition) -> MetricComponents {
    let mut g = [[0.0; DIM]; DIM];

    // g_{ij} = gamma_{ij}
    for i in 0..SDIM {
        for j in 0..SDIM {
            g[i + 1][j + 1] = adm.spatial_metric[i][j];
        }
    }

    // beta_i = gamma_{ij} beta^j
    let mut beta_lower = [0.0; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            beta_lower[i] += adm.spatial_metric[i][j] * adm.shift[j];
        }
    }

    // g_{0i} = beta_i
    for i in 0..SDIM {
        g[T][i + 1] = beta_lower[i];
        g[i + 1][T] = beta_lower[i];
    }

    // g_{00} = -alpha^2 + beta_k beta^k
    let mut beta_sq = 0.0;
    for i in 0..SDIM {
        beta_sq += adm.shift[i] * beta_lower[i];
    }
    g[T][T] = -adm.lapse * adm.lapse + beta_sq;

    g
}

// ============================================================================
// Extrinsic curvature
// ============================================================================

/// Compute extrinsic curvature K_{ij} via numerical differentiation.
///
/// K_{ij} = -(1/2alpha)(d_t gamma_{ij} - D_i beta_j - D_j beta_i)
///
/// For stationary spacetimes (d_t gamma = 0), this reduces to:
///   K_{ij} = (1/2alpha)(D_i beta_j + D_j beta_i)
///
/// We compute this by finite-differencing the full 4D metric's Christoffel
/// symbols, projecting onto the hypersurface. Specifically:
///   K_{ij} = -alpha * Gamma^0_{ij}    (from g, for lapse alpha)
/// or equivalently via the 4D Christoffels from the spacetime metric.
///
/// Uses the identity K_{ij} = -(1/2) L_n gamma_{ij} where n is the unit normal,
/// computed via K_{ij} = -(gamma_{ia} gamma_{jb} nabla_a n_b + gamma_{ia} gamma_{jb} nabla_b n_a)/2
/// which reduces to K_{ij} = -alpha * (Gamma^0_{ij} evaluated in the 4D metric).
pub fn extrinsic_curvature<M: SpacetimeMetric>(
    metric: &M,
    x: &[f64; DIM],
) -> ExtrinsicCurvatureData {
    // Get the ADM decomposition at this point
    let g = metric.metric_components(x);
    let adm = decompose_metric(&g);

    // K_{ij} = -(1/2alpha)(d_t gamma_{ij} - D_i beta_j - D_j beta_i)
    // Using the identity: K_{ij} = -alpha * Gamma^0_{(ij)} + beta^k Gamma^0_{k(ij)}/alpha
    // More directly: from the 4D metric, K_{ij} = -(gamma^a_i gamma^b_j nabla_a n_b)
    // For coordinates where n_mu = (-alpha, 0, 0, 0):
    // K_{ij} = (1/(2*alpha)) * (-d_t g_{ij} + nabla_i beta_j + nabla_j beta_i)
    // Since Gamma^0_{ij} = (g^{0a}/2)(g_{ai,j} + g_{aj,i} - g_{ij,a}), and using
    // the 4D inverse metric, we get:
    // K_{ij} = (1/(2*alpha))(beta^k d_k gamma_{ij} + gamma_{kj} d_i beta^k + gamma_{ik} d_j beta^k - d_t gamma_{ij})
    //
    // A cleaner approach: K_{ij} relates to 4D Christoffels as
    //   K_{ij} = -alpha * Gamma^0_{(i+1)(j+1)}  (Baumgarte & Shapiro 7.75)
    //            ... but with a correction from the inverse metric.
    //
    // Most directly: K_{ij} = -(1/(2*alpha))(g_{(i+1)(j+1),0} - 2*alpha*Gamma^0_{(i+1)(j+1)})
    //
    // For the standard form: K_{ij} = (1/(2 alpha)) * (-partial_t gamma_{ij} + D_i beta_j + D_j beta_i)
    // where D is the spatial covariant derivative.
    //
    // We use: K_{ij} = -(1/(2 alpha))(partial_t gamma_{ij} - D_i beta_j - D_j beta_i)
    // For a stationary metric (partial_t gamma = 0):
    //   K_{ij} = (1/(2 alpha))(D_i beta_j + D_j beta_i)

    // Compute spatial Christoffel symbols from gamma_{ij}
    let spatial_gamma = spatial_christoffel_numerical(metric, x);

    // beta_j = gamma_{jk} beta^k (covariant shift)
    let mut beta_lower = [0.0; SDIM];
    for j in 0..SDIM {
        for k in 0..SDIM {
            beta_lower[j] += adm.spatial_metric[j][k] * adm.shift[k];
        }
    }

    // D_i beta_j = partial_i beta_j - Gamma^k_{ij} beta_k
    // partial_i beta_j via finite differences on the shift
    let mut d_beta = [[0.0; SDIM]; SDIM]; // d_beta[i][j] = D_i beta_j
    for i in 0..SDIM {
        // Finite difference for partial_i beta_lower[j]
        let h = adaptive_spatial_step(x[i + 1], i);
        let mut x_plus = *x;
        let mut x_minus = *x;
        x_plus[i + 1] += h;
        x_minus[i + 1] -= h;

        let g_plus = metric.metric_components(&x_plus);
        let g_minus = metric.metric_components(&x_minus);
        let adm_plus = decompose_metric(&g_plus);
        let adm_minus = decompose_metric(&g_minus);

        let mut beta_low_plus = [0.0; SDIM];
        let mut beta_low_minus = [0.0; SDIM];
        for j in 0..SDIM {
            for k in 0..SDIM {
                beta_low_plus[j] += adm_plus.spatial_metric[j][k] * adm_plus.shift[k];
                beta_low_minus[j] += adm_minus.spatial_metric[j][k] * adm_minus.shift[k];
            }
        }

        for j in 0..SDIM {
            let partial_i_beta_j = (beta_low_plus[j] - beta_low_minus[j]) / (2.0 * h);
            let mut christoffel_term = 0.0;
            for k in 0..SDIM {
                christoffel_term += spatial_gamma[k][i][j] * beta_lower[k];
            }
            d_beta[i][j] = partial_i_beta_j - christoffel_term;
        }
    }

    // Time derivative of gamma via finite differences
    let h_t = 1e-6;
    let mut x_tp = *x;
    let mut x_tm = *x;
    x_tp[T] += h_t;
    x_tm[T] -= h_t;
    let g_tp = metric.metric_components(&x_tp);
    let g_tm = metric.metric_components(&x_tm);

    // K_{ij} = -(1/(2 alpha))(dt_gamma_{ij} - D_i beta_j - D_j beta_i)
    let mut k_ij = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in i..SDIM {
            let dt_gamma = (g_tp[i + 1][j + 1] - g_tm[i + 1][j + 1]) / (2.0 * h_t);
            k_ij[i][j] =
                -(1.0 / (2.0 * adm.lapse)) * (dt_gamma - d_beta[i][j] - d_beta[j][i]);
            k_ij[j][i] = k_ij[i][j]; // symmetric
        }
    }

    // York time: theta = -K = -gamma^{ij} K_{ij}
    let trace_k = trace_symmetric(&adm.spatial_metric_inv, &k_ij);
    let york_time = -trace_k;

    // Trace-free part: A_{ij} = K_{ij} - (1/3) gamma_{ij} K
    let mut tracefree_k = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            tracefree_k[i][j] = k_ij[i][j] - (1.0 / 3.0) * adm.spatial_metric[i][j] * trace_k;
        }
    }

    ExtrinsicCurvatureData {
        k_ij,
        york_time,
        tracefree_k,
    }
}

/// Compute York time theta = -K from the extrinsic curvature trace.
///
/// Positive theta = expanding volume element, negative = contracting.
/// For de Sitter: theta = 3H (Hubble rate).
pub fn york_time(gamma_inv: &SpatialMetric, k_ij: &SpatialSymmetricTensor) -> f64 {
    -trace_symmetric(gamma_inv, k_ij)
}

// ============================================================================
// Constraint equations
// ============================================================================

/// Compute the Hamiltonian constraint residual at a point.
///
/// H = R^(3) + K^2 - K_{ij}K^{ij} - 16 pi rho
///
/// For vacuum spacetimes (rho = 0), H should vanish identically.
/// A nonzero residual indicates either a numerical error or a matter source.
pub fn hamiltonian_constraint<M: SpacetimeMetric>(
    metric: &M,
    x: &[f64; DIM],
    energy_density: f64,
) -> f64 {
    let g = metric.metric_components(x);
    let adm = decompose_metric(&g);
    let ext = extrinsic_curvature(metric, x);

    let r3 = spatial_ricci_scalar(metric, x);

    let k_trace = trace_symmetric(&adm.spatial_metric_inv, &ext.k_ij);
    let k_upper = raise_symmetric_tensor(&adm.spatial_metric_inv, &ext.k_ij);
    let mut kij_kij = 0.0;
    for i in 0..SDIM {
        for j in 0..SDIM {
            kij_kij += k_upper[i][j] * ext.k_ij[i][j];
        }
    }

    r3 + k_trace * k_trace - kij_kij - 16.0 * std::f64::consts::PI * energy_density
}

/// Compute the 3D spatial Ricci scalar R^(3) from the spatial metric.
///
/// Uses numerical finite differences of the spatial metric, building
/// spatial Christoffel symbols and then their derivatives.
pub fn spatial_ricci_scalar<M: SpacetimeMetric>(metric: &M, x: &[f64; DIM]) -> f64 {
    let g = metric.metric_components(x);
    let adm = decompose_metric(&g);

    // Spatial Christoffels at the point
    let gamma_s = spatial_christoffel_numerical(metric, x);

    // Derivatives of spatial Christoffels
    let mut dgamma = [[[[0.0; SDIM]; SDIM]; SDIM]; SDIM];
    for d in 0..SDIM {
        let h = adaptive_spatial_step(x[d + 1], d);
        let mut x_plus = *x;
        let mut x_minus = *x;
        x_plus[d + 1] += h;
        x_minus[d + 1] -= h;

        let gamma_plus = spatial_christoffel_numerical(metric, &x_plus);
        let gamma_minus = spatial_christoffel_numerical(metric, &x_minus);

        for a in 0..SDIM {
            for b in 0..SDIM {
                for c in 0..SDIM {
                    dgamma[a][b][c][d] =
                        (gamma_plus[a][b][c] - gamma_minus[a][b][c]) / (2.0 * h);
                }
            }
        }
    }

    // Riemann^a_{bcd} = Gamma^a_{bd,c} - Gamma^a_{bc,d} + Gamma^a_{ec}Gamma^e_{bd} - Gamma^a_{ed}Gamma^e_{bc}
    // Ricci_{bd} = Riemann^a_{bad}
    // R^(3) = gamma^{bd} Ricci_{bd}

    let mut ricci_scalar = 0.0;
    for b in 0..SDIM {
        for d in 0..SDIM {
            let mut r_bd = 0.0;
            for a in 0..SDIM {
                r_bd += dgamma[a][b][d][a] - dgamma[a][b][a][d];
                for e in 0..SDIM {
                    r_bd +=
                        gamma_s[a][e][a] * gamma_s[e][b][d] - gamma_s[a][e][d] * gamma_s[e][b][a];
                }
            }
            ricci_scalar += adm.spatial_metric_inv[b][d] * r_bd;
        }
    }

    ricci_scalar
}

// ============================================================================
// Spatial Christoffel symbols
// ============================================================================

/// Compute spatial Christoffel symbols from the 3-metric numerically.
///
/// Gamma^i_{jk} = (1/2) gamma^{il} (gamma_{lj,k} + gamma_{lk,j} - gamma_{jk,l})
fn spatial_christoffel_numerical<M: SpacetimeMetric>(
    metric: &M,
    x: &[f64; DIM],
) -> SpatialChristoffel {
    let g = metric.metric_components(x);
    let adm = decompose_metric(&g);

    // Derivatives of the spatial metric
    let mut dg = [[[0.0; SDIM]; SDIM]; SDIM]; // dg[a][b][c] = d(gamma_{ab})/d(x^c)
    for c in 0..SDIM {
        let h = adaptive_spatial_step(x[c + 1], c);
        let mut x_plus = *x;
        let mut x_minus = *x;
        x_plus[c + 1] += h;
        x_minus[c + 1] -= h;

        let g_plus = metric.metric_components(&x_plus);
        let g_minus = metric.metric_components(&x_minus);

        for a in 0..SDIM {
            for b in 0..SDIM {
                dg[a][b][c] = (g_plus[a + 1][b + 1] - g_minus[a + 1][b + 1]) / (2.0 * h);
            }
        }
    }

    // Assemble spatial Christoffels
    let mut gamma_s = [[[0.0; SDIM]; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            for k in j..SDIM {
                let mut sum = 0.0;
                for l in 0..SDIM {
                    sum += adm.spatial_metric_inv[i][l]
                        * (dg[l][j][k] + dg[l][k][j] - dg[j][k][l]);
                }
                gamma_s[i][j][k] = 0.5 * sum;
                gamma_s[i][k][j] = gamma_s[i][j][k]; // symmetric in lower indices
            }
        }
    }

    gamma_s
}

/// Adaptive step size for spatial finite differences.
fn adaptive_spatial_step(x_val: f64, _coord_index: usize) -> f64 {
    let abs_x = x_val.abs();
    if abs_x > 1.0 {
        abs_x * 1e-7
    } else {
        1e-7
    }
}

// ============================================================================
// ADM decomposition from the inverse metric (alternative extraction)
// ============================================================================

/// Decompose using the inverse metric (useful when g^{mu nu} is known).
///
/// alpha = 1/sqrt(-g^{00})
/// beta^i = -g^{0i}/g^{00}
/// gamma^{ij} = g^{ij} - beta^i beta^j / alpha^2 = g^{ij} + g^{0i}g^{0j}/g^{00}
pub fn decompose_from_inverse(g_inv: &MetricComponents) -> AdmDecomposition {
    assert!(
        g_inv[T][T] < 0.0,
        "g^{{00}} must be negative for Lorentzian signature (got {})",
        g_inv[T][T]
    );

    let lapse = 1.0 / (-g_inv[T][T]).sqrt();

    let mut shift = [0.0; SDIM];
    for i in 0..SDIM {
        shift[i] = -g_inv[T][i + 1] / g_inv[T][T];
    }

    // gamma^{ij} = g^{ij} + g^{0i}g^{0j}/g^{00}
    let mut gamma_inv = [[0.0; SDIM]; SDIM];
    for i in 0..SDIM {
        for j in 0..SDIM {
            gamma_inv[i][j] =
                g_inv[i + 1][j + 1] + g_inv[T][i + 1] * g_inv[T][j + 1] / g_inv[T][T];
        }
    }

    // Need gamma_{ij} by inverting gamma^{ij}
    // But gamma^{ij} might be poorly conditioned if extracted this way.
    // Safer: reconstruct g_{mu nu}, then read off gamma_{ij}.
    // For now, invert gamma^{ij} directly.
    let spatial_metric = invert_3x3_symmetric(&gamma_inv);
    let det_gamma = spatial_determinant(&spatial_metric);

    AdmDecomposition {
        lapse,
        shift,
        spatial_metric,
        spatial_metric_inv: gamma_inv,
        det_gamma,
    }
}

// ============================================================================
// Painleve-Gullstrand coordinates (Schwarzschild test case)
// ============================================================================

/// Schwarzschild metric in Painleve-Gullstrand (rain) coordinates.
///
/// ds^2 = -dT^2 + (dr + sqrt(2M/r) dT)^2 + r^2 dOmega^2
///      = -(1-2M/r)dT^2 + 2*sqrt(2M/r) dT dr + dr^2 + r^2 dOmega^2
///
/// ADM variables:
/// - alpha = 1 (coordinate time = proper time for rain observers)
/// - beta^r = -sqrt(2M/r) (radial infall velocity)
/// - gamma_{ij} = flat (Euclidean) in spherical coords
///
/// This provides an exact test case for the ADM decomposition:
/// the lapse is unity and the shift is the Newtonian free-fall velocity.
pub struct PainleveGullstrand {
    /// Mass parameter (G = c = 1).
    pub mass: f64,
}

impl PainleveGullstrand {
    pub fn new(mass: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive");
        Self { mass }
    }
}

impl SpacetimeMetric for PainleveGullstrand {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let r = x[1];
        let theta = x[2];
        let sin2 = theta.sin().powi(2);

        let v = (2.0 * self.mass / r).sqrt(); // infall velocity

        let mut g = [[0.0; DIM]; DIM];
        // g_{TT} = -(1 - 2M/r) = -(1 - v^2)
        g[0][0] = -(1.0 - v * v);
        // g_{Tr} = g_{rT} = v (covariant shift = sqrt(2M/r))
        g[0][1] = v;
        g[1][0] = v;
        // g_{rr} = 1
        g[1][1] = 1.0;
        // g_{theta theta} = r^2
        g[2][2] = r * r;
        // g_{phi phi} = r^2 sin^2(theta)
        g[3][3] = r * r * sin2;

        g
    }
}

// ============================================================================
// Cartesian Minkowski (for warp metric testing)
// ============================================================================

/// Flat Minkowski spacetime in Cartesian coordinates [t, x, y, z].
///
/// ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
///
/// ADM: alpha = 1, beta = 0, gamma = delta_{ij}.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinkowskiCartesian;

impl SpacetimeMetric for MinkowskiCartesian {
    fn metric_components(&self, _x: &[f64; DIM]) -> MetricComponents {
        let mut g = [[0.0; DIM]; DIM];
        g[0][0] = -1.0;
        g[1][1] = 1.0;
        g[2][2] = 1.0;
        g[3][3] = 1.0;
        g
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::invert_4x4_symmetric;

    const TOL: f64 = 1e-10;
    const NUMERICAL_TOL: f64 = 1e-4;

    // -- Minkowski (Cartesian) --

    #[test]
    fn test_minkowski_decomposition() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 1.0, 1.0, 1.0];
        let g = mink.metric_components(&x);
        let adm = decompose_metric(&g);

        assert!(
            (adm.lapse - 1.0).abs() < TOL,
            "Minkowski lapse = {}",
            adm.lapse
        );
        for i in 0..SDIM {
            assert!(
                adm.shift[i].abs() < TOL,
                "Minkowski shift[{i}] = {}",
                adm.shift[i]
            );
        }
        for i in 0..SDIM {
            for j in 0..SDIM {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (adm.spatial_metric[i][j] - expected).abs() < TOL,
                    "gamma[{i}][{j}] = {}",
                    adm.spatial_metric[i][j]
                );
            }
        }
    }

    #[test]
    fn test_minkowski_extrinsic_curvature_zero() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 1.0, 1.0, 1.0];
        let ext = extrinsic_curvature(&mink, &x);

        for i in 0..SDIM {
            for j in 0..SDIM {
                assert!(
                    ext.k_ij[i][j].abs() < NUMERICAL_TOL,
                    "Minkowski K[{i}][{j}] = {}",
                    ext.k_ij[i][j]
                );
            }
        }
        assert!(
            ext.york_time.abs() < NUMERICAL_TOL,
            "Minkowski York time = {}",
            ext.york_time
        );
    }

    // -- Reconstruct round-trip --

    #[test]
    fn test_decompose_reconstruct_minkowski() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 1.0, 1.0, 1.0];
        let g = mink.metric_components(&x);
        let adm = decompose_metric(&g);
        let g_recon = reconstruct_metric(&adm);

        for mu in 0..DIM {
            for nu in 0..DIM {
                assert!(
                    (g[mu][nu] - g_recon[mu][nu]).abs() < TOL,
                    "g[{mu}][{nu}]: original={}, reconstructed={}",
                    g[mu][nu],
                    g_recon[mu][nu]
                );
            }
        }
    }

    #[test]
    fn test_decompose_reconstruct_painleve() {
        let pg = PainleveGullstrand::new(1.0);
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let g = pg.metric_components(&x);
        let adm = decompose_metric(&g);
        let g_recon = reconstruct_metric(&adm);

        for mu in 0..DIM {
            for nu in 0..DIM {
                assert!(
                    (g[mu][nu] - g_recon[mu][nu]).abs() < TOL,
                    "PG g[{mu}][{nu}]: original={}, reconstructed={}",
                    g[mu][nu],
                    g_recon[mu][nu]
                );
            }
        }
    }

    // -- Painleve-Gullstrand ADM variables --

    #[test]
    fn test_painleve_lapse_is_unity() {
        let pg = PainleveGullstrand::new(1.0);
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let g = pg.metric_components(&x);
        let adm = decompose_metric(&g);

        assert!(
            (adm.lapse - 1.0).abs() < TOL,
            "PG lapse at r=10: {}",
            adm.lapse
        );
    }

    #[test]
    fn test_painleve_shift_is_infall_velocity() {
        let pg = PainleveGullstrand::new(1.0);
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let g = pg.metric_components(&x);
        let adm = decompose_metric(&g);

        // In PG coords with spherical spatial coords, beta^r = sqrt(2M/r).
        // But the shift sign convention: g_{0r} = v = sqrt(2M/r),
        // and beta^r = gamma^{rr} g_{0r} = 1 * sqrt(2M/r)
        // (positive because rain observers fall inward and we define beta^r > 0 for the shift)
        let expected_beta_r = (2.0 / r).sqrt();
        assert!(
            (adm.shift[0] - expected_beta_r).abs() < TOL,
            "PG beta^r at r={r}: got {}, expected {expected_beta_r}",
            adm.shift[0]
        );
        assert!(adm.shift[1].abs() < TOL, "PG beta^theta should be 0");
        assert!(adm.shift[2].abs() < TOL, "PG beta^phi should be 0");
    }

    #[test]
    fn test_painleve_spatial_metric_is_flat() {
        let pg = PainleveGullstrand::new(1.0);
        let r = 10.0;
        let theta = std::f64::consts::FRAC_PI_2;
        let x = [0.0, r, theta, 0.0];
        let g = pg.metric_components(&x);
        let adm = decompose_metric(&g);

        // gamma_{rr} = 1, gamma_{theta theta} = r^2, gamma_{phi phi} = r^2 sin^2(theta)
        assert!(
            (adm.spatial_metric[0][0] - 1.0).abs() < TOL,
            "gamma_rr = {}",
            adm.spatial_metric[0][0]
        );
        assert!(
            (adm.spatial_metric[1][1] - r * r).abs() < TOL,
            "gamma_tt = {}",
            adm.spatial_metric[1][1]
        );
        let sin2 = theta.sin().powi(2);
        assert!(
            (adm.spatial_metric[2][2] - r * r * sin2).abs() < TOL,
            "gamma_pp = {}",
            adm.spatial_metric[2][2]
        );
    }

    // -- Inverse decomposition --

    #[test]
    fn test_decompose_from_inverse_minkowski() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 1.0, 1.0, 1.0];
        let g_inv = mink.inverse_metric(&x);
        let adm = decompose_from_inverse(&g_inv);

        assert!(
            (adm.lapse - 1.0).abs() < TOL,
            "inverse decomp lapse = {}",
            adm.lapse
        );
        for i in 0..SDIM {
            assert!(adm.shift[i].abs() < TOL);
        }
    }

    #[test]
    fn test_decompose_from_inverse_painleve() {
        let pg = PainleveGullstrand::new(1.0);
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let g = pg.metric_components(&x);
        let g_inv = invert_4x4_symmetric(&g);
        let adm_inv = decompose_from_inverse(&g_inv);
        let adm_direct = decompose_metric(&g);

        assert!(
            (adm_inv.lapse - adm_direct.lapse).abs() < 1e-6,
            "lapse mismatch: {} vs {}",
            adm_inv.lapse,
            adm_direct.lapse
        );
        for i in 0..SDIM {
            assert!(
                (adm_inv.shift[i] - adm_direct.shift[i]).abs() < 1e-6,
                "shift[{i}] mismatch: {} vs {}",
                adm_inv.shift[i],
                adm_direct.shift[i]
            );
        }
    }

    // -- Hamiltonian constraint --

    #[test]
    fn test_hamiltonian_constraint_minkowski() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 5.0, 3.0, 2.0];
        let h = hamiltonian_constraint(&mink, &x, 0.0);
        assert!(
            h.abs() < 0.01,
            "Minkowski Hamiltonian constraint = {h} (expected ~0)"
        );
    }

    // -- York time --

    #[test]
    fn test_york_time_function() {
        let gamma_inv = crate::spatial::SPATIAL_IDENTITY;
        let k = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let theta = york_time(&gamma_inv, &k);
        // theta = -tr(K) = -(1+2+3) = -6
        assert!(
            (theta - (-6.0)).abs() < TOL,
            "York time = {theta}, expected -6"
        );
    }

    // -- Spatial Ricci scalar --

    #[test]
    fn test_spatial_ricci_scalar_flat() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 5.0, 3.0, 2.0];
        let r3 = spatial_ricci_scalar(&mink, &x);
        assert!(
            r3.abs() < 0.01,
            "Flat space R^(3) = {r3} (expected 0)"
        );
    }

    // -- Determinant --

    #[test]
    fn test_adm_det_gamma_minkowski() {
        let mink = MinkowskiCartesian;
        let x = [0.0, 1.0, 1.0, 1.0];
        let g = mink.metric_components(&x);
        let adm = decompose_metric(&g);
        assert!(
            (adm.det_gamma - 1.0).abs() < TOL,
            "Minkowski det(gamma) = {}",
            adm.det_gamma
        );
    }

    #[test]
    fn test_adm_det_gamma_painleve() {
        let pg = PainleveGullstrand::new(1.0);
        let r = 10.0;
        let theta = std::f64::consts::FRAC_PI_2;
        let x = [0.0, r, theta, 0.0];
        let g = pg.metric_components(&x);
        let adm = decompose_metric(&g);

        // PG spatial metric is flat spherical: det = r^4 sin^2(theta)
        let expected = r.powi(4) * theta.sin().powi(2);
        assert!(
            (adm.det_gamma - expected).abs() / expected < 1e-6,
            "PG det(gamma) = {}, expected {expected}",
            adm.det_gamma
        );
    }

    // -- Extrinsic curvature trace-free part --

    #[test]
    fn test_tracefree_k_has_zero_trace() {
        let pg = PainleveGullstrand::new(1.0);
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let g = pg.metric_components(&x);
        let adm = decompose_metric(&g);
        let ext = extrinsic_curvature(&pg, &x);

        let tr_tf = trace_symmetric(&adm.spatial_metric_inv, &ext.tracefree_k);
        assert!(
            tr_tf.abs() < NUMERICAL_TOL,
            "trace of trace-free K = {tr_tf} (expected ~0)"
        );
    }
}
