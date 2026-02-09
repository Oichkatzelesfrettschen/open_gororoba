//! Generic spacetime metric trait and Christoffel symbol computation.
//!
//! Defines the `SpacetimeMetric` trait that all specific metrics (Schwarzschild,
//! Kerr, Kerr-Newman, FLRW) implement. Provides generic numerical computation
//! of Christoffel symbols, Riemann tensor, Ricci tensor, and scalar curvature
//! invariants via finite differencing of metric components.
//!
//! For known metrics with closed-form Christoffel symbols (Schwarzschild, Kerr),
//! the `christoffel()` method is overridden with exact expressions. The generic
//! finite-difference computation serves as:
//! 1. A verification tool for closed-form expressions
//! 2. A fallback for metrics without known closed forms
//!
//! # Conventions
//! - Signature: (-,+,+,+)
//! - Coordinates: [t, r, theta, phi] (Boyer-Lindquist for Kerr family)
//! - Natural units: G = c = 1, mass M as parameter
//!
//! # Literature
//! - Misner, Thorne, Wheeler (1973): Gravitation, Ch. 8-14
//! - Wald (1984): General Relativity, Ch. 3
//! - Carroll (2004): Spacetime and Geometry, Ch. 3

/// Coordinate indices for 4D spacetime.
pub const T: usize = 0;
pub const R: usize = 1;
pub const THETA: usize = 2;
pub const PHI: usize = 3;
pub const DIM: usize = 4;

/// The covariant metric tensor g_{mu nu} at a point, stored as a symmetric 4x4.
pub type MetricComponents = [[f64; DIM]; DIM];

/// Christoffel symbols Gamma^alpha_{mu nu}, stored as [alpha][mu][nu].
/// Symmetric in lower indices: Gamma^a_{mn} = Gamma^a_{nm}.
pub type ChristoffelComponents = [[[f64; DIM]; DIM]; DIM];

/// Riemann tensor R^alpha_{beta mu nu}, stored as [alpha][beta][mu][nu].
pub type RiemannComponents = [[[[f64; DIM]; DIM]; DIM]; DIM];

/// Ricci tensor R_{mu nu}, stored as [mu][nu].
pub type RicciComponents = [[f64; DIM]; DIM];

/// Result of computing curvature tensors at a single spacetime point.
#[derive(Debug, Clone)]
pub struct CurvatureResult {
    /// Covariant metric g_{mu nu}
    pub g: MetricComponents,
    /// Inverse metric g^{mu nu}
    pub g_inv: MetricComponents,
    /// Christoffel symbols Gamma^alpha_{mu nu}
    pub christoffel: ChristoffelComponents,
    /// Riemann tensor R^alpha_{beta mu nu}
    pub riemann: RiemannComponents,
    /// Ricci tensor R_{mu nu}
    pub ricci: RicciComponents,
    /// Ricci scalar R = g^{mu nu} R_{mu nu}
    pub ricci_scalar: f64,
    /// Kretschner scalar K = R_{abcd} R^{abcd}
    pub kretschner: f64,
}

/// Trait for spacetime metrics. Implement this to define a new spacetime.
///
/// The only required method is `metric_components()`, which returns the
/// covariant metric tensor g_{mu nu}(x) at a given coordinate point.
/// All curvature tensors are then computed automatically via finite
/// differencing (or overridden with closed-form expressions).
pub trait SpacetimeMetric {
    /// Compute the covariant metric tensor g_{mu nu} at coordinates x.
    ///
    /// Coordinates are [t, r, theta, phi] in the natural coordinate
    /// system for this metric.
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents;

    /// Compute Christoffel symbols Gamma^alpha_{mu nu} at coordinates x.
    ///
    /// Default implementation uses centered finite differences of the
    /// metric components. Override for closed-form expressions.
    fn christoffel(&self, x: &[f64; DIM]) -> ChristoffelComponents {
        christoffel_numerical(self, x)
    }

    /// Compute the inverse metric g^{mu nu} at coordinates x.
    ///
    /// Default implementation inverts the 4x4 metric matrix.
    fn inverse_metric(&self, x: &[f64; DIM]) -> MetricComponents {
        invert_4x4_symmetric(&self.metric_components(x))
    }

    /// Event horizon radius (if applicable). Returns None for spacetimes
    /// without horizons (e.g. FLRW).
    fn event_horizon_radius(&self) -> Option<f64> {
        None
    }

    /// ISCO (Innermost Stable Circular Orbit) radius, if applicable.
    fn isco_radius(&self) -> Option<f64> {
        None
    }

    /// Photon sphere radius, if applicable.
    fn photon_sphere_radius(&self) -> Option<f64> {
        None
    }
}

/// Compute Christoffel symbols numerically via centered finite differences.
///
/// Gamma^alpha_{mu nu} = (1/2) g^{alpha beta} (g_{beta mu,nu} + g_{beta nu,mu} - g_{mu nu,beta})
///
/// where g_{ab,c} = dg_{ab}/dx^c is approximated by:
///   g_{ab,c} ~ (g_{ab}(x + h*e_c) - g_{ab}(x - h*e_c)) / (2h)
///
/// The step size h is adaptive: proportional to |x^c| for large coordinates,
/// with a floor to avoid division by zero.
#[allow(clippy::needless_range_loop)]
pub fn christoffel_numerical<M: SpacetimeMetric + ?Sized>(
    metric: &M,
    x: &[f64; DIM],
) -> ChristoffelComponents {
    let g_inv = metric.inverse_metric(x);

    // Compute metric derivatives dg_{ab}/dx^c via centered differences
    let mut dg = [[[0.0; DIM]; DIM]; DIM]; // dg[a][b][c] = dg_{ab}/dx^c

    for c in 0..DIM {
        let h = step_size(x[c], c);
        let mut x_plus = *x;
        let mut x_minus = *x;
        x_plus[c] += h;
        x_minus[c] -= h;

        let g_plus = metric.metric_components(&x_plus);
        let g_minus = metric.metric_components(&x_minus);

        for a in 0..DIM {
            for b in 0..DIM {
                dg[a][b][c] = (g_plus[a][b] - g_minus[a][b]) / (2.0 * h);
            }
        }
    }

    // Assemble Christoffel symbols
    let mut gamma = [[[0.0; DIM]; DIM]; DIM];
    for alpha in 0..DIM {
        for mu in 0..DIM {
            for nu in mu..DIM {
                let mut sum = 0.0;
                for beta in 0..DIM {
                    // g_{beta mu,nu} + g_{beta nu,mu} - g_{mu nu,beta}
                    sum += g_inv[alpha][beta]
                        * (dg[beta][mu][nu] + dg[beta][nu][mu] - dg[mu][nu][beta]);
                }
                gamma[alpha][mu][nu] = 0.5 * sum;
                gamma[alpha][nu][mu] = gamma[alpha][mu][nu]; // symmetric
            }
        }
    }

    gamma
}

/// Compute the Riemann tensor numerically from Christoffel symbols.
///
/// R^alpha_{beta mu nu} = Gamma^alpha_{beta nu,mu} - Gamma^alpha_{beta mu,nu}
///                        + Gamma^alpha_{sigma mu} Gamma^sigma_{beta nu}
///                        - Gamma^alpha_{sigma nu} Gamma^sigma_{beta mu}
#[allow(clippy::needless_range_loop)]
pub fn riemann_from_christoffel<M: SpacetimeMetric + ?Sized>(
    metric: &M,
    x: &[f64; DIM],
) -> RiemannComponents {
    let gamma = metric.christoffel(x);

    // Christoffel derivatives via finite differences
    let mut dgamma = [[[[0.0; DIM]; DIM]; DIM]; DIM]; // dgamma[a][b][c][d] = Gamma^a_{bc,d}

    for d in 0..DIM {
        let h = step_size(x[d], d);
        let mut x_plus = *x;
        let mut x_minus = *x;
        x_plus[d] += h;
        x_minus[d] -= h;

        let gamma_plus = metric.christoffel(&x_plus);
        let gamma_minus = metric.christoffel(&x_minus);

        for a in 0..DIM {
            for b in 0..DIM {
                for c in 0..DIM {
                    dgamma[a][b][c][d] = (gamma_plus[a][b][c] - gamma_minus[a][b][c]) / (2.0 * h);
                }
            }
        }
    }

    // Assemble Riemann tensor
    let mut riemann = [[[[0.0; DIM]; DIM]; DIM]; DIM];
    for alpha in 0..DIM {
        for beta in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    let mut val = dgamma[alpha][beta][nu][mu] - dgamma[alpha][beta][mu][nu];
                    for sigma in 0..DIM {
                        val += gamma[alpha][sigma][mu] * gamma[sigma][beta][nu]
                            - gamma[alpha][sigma][nu] * gamma[sigma][beta][mu];
                    }
                    riemann[alpha][beta][mu][nu] = val;
                }
            }
        }
    }

    riemann
}

/// Compute Ricci tensor from Riemann tensor by contraction.
///
/// R_{mu nu} = R^alpha_{mu alpha nu}
#[allow(clippy::needless_range_loop)]
pub fn ricci_from_riemann(riemann: &RiemannComponents) -> RicciComponents {
    let mut ricci = [[0.0; DIM]; DIM];
    for mu in 0..DIM {
        for nu in 0..DIM {
            let mut sum = 0.0;
            for alpha in 0..DIM {
                sum += riemann[alpha][mu][alpha][nu];
            }
            ricci[mu][nu] = sum;
        }
    }
    ricci
}

/// Compute Ricci scalar from Ricci tensor and inverse metric.
///
/// R = g^{mu nu} R_{mu nu}
#[allow(clippy::needless_range_loop)]
pub fn ricci_scalar(g_inv: &MetricComponents, ricci: &RicciComponents) -> f64 {
    let mut r = 0.0;
    for mu in 0..DIM {
        for nu in 0..DIM {
            r += g_inv[mu][nu] * ricci[mu][nu];
        }
    }
    r
}

/// Compute Kretschner scalar K = R_{abcd} R^{abcd}.
///
/// This requires lowering all indices of the Riemann tensor using the metric,
/// then contracting. For Schwarzschild: K = 48 M^2 / r^6.
#[allow(clippy::needless_range_loop)]
pub fn kretschner_scalar(
    g: &MetricComponents,
    g_inv: &MetricComponents,
    riemann: &RiemannComponents,
) -> f64 {
    // Lower the first index: R_{alpha beta mu nu} = g_{alpha sigma} R^sigma_{beta mu nu}
    let mut r_lower = [[[[0.0; DIM]; DIM]; DIM]; DIM];
    for alpha in 0..DIM {
        for beta in 0..DIM {
            for mu in 0..DIM {
                for nu in 0..DIM {
                    let mut sum = 0.0;
                    for sigma in 0..DIM {
                        sum += g[alpha][sigma] * riemann[sigma][beta][mu][nu];
                    }
                    r_lower[alpha][beta][mu][nu] = sum;
                }
            }
        }
    }

    // Raise all indices of the second copy: R^{abcd} = g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh}
    // Then contract: K = R_{abcd} R^{abcd}
    // Equivalently: K = sum_{a,b,c,d} R_{abcd} * (g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh})
    let mut k = 0.0;
    for a in 0..DIM {
        for b in 0..DIM {
            for c in 0..DIM {
                for d in 0..DIM {
                    // Raise all indices of r_lower
                    let mut r_upper = 0.0;
                    for e in 0..DIM {
                        for f in 0..DIM {
                            for gg in 0..DIM {
                                for h in 0..DIM {
                                    r_upper += g_inv[a][e]
                                        * g_inv[b][f]
                                        * g_inv[c][gg]
                                        * g_inv[d][h]
                                        * r_lower[e][f][gg][h];
                                }
                            }
                        }
                    }
                    k += r_lower[a][b][c][d] * r_upper;
                }
            }
        }
    }
    k
}

/// Compute all curvature tensors at a spacetime point.
pub fn full_curvature<M: SpacetimeMetric + ?Sized>(metric: &M, x: &[f64; DIM]) -> CurvatureResult {
    let g = metric.metric_components(x);
    let g_inv = metric.inverse_metric(x);
    let christoffel = metric.christoffel(x);
    let riemann = riemann_from_christoffel(metric, x);
    let ricci = ricci_from_riemann(&riemann);
    let r_scalar = ricci_scalar(&g_inv, &ricci);
    let kretschner = kretschner_scalar(&g, &g_inv, &riemann);

    CurvatureResult {
        g,
        g_inv,
        christoffel,
        riemann,
        ricci,
        ricci_scalar: r_scalar,
        kretschner,
    }
}

// -- Internal helpers --

/// Adaptive step size for finite differencing.
///
/// For large coordinates (r >> 1), use a proportional step.
/// For small coordinates or angles, use an absolute step.
/// Special handling for angular coordinates theta (index 2).
fn step_size(x_val: f64, coord_index: usize) -> f64 {
    let abs_x = x_val.abs();
    match coord_index {
        // Time: rarely varied, use absolute step
        T => 1e-6,
        // Radial: proportional for large r, absolute floor for small r
        R => {
            if abs_x > 1.0 {
                abs_x * 1e-7
            } else {
                1e-7
            }
        }
        // Theta: keep away from poles (0 and pi)
        THETA => {
            let safe = abs_x.min(std::f64::consts::PI - abs_x);
            if safe > 0.01 {
                1e-6
            } else {
                safe * 0.01_f64.max(1e-8)
            }
        }
        // Phi: periodic, absolute step
        PHI => 1e-6,
        _ => 1e-7,
    }
}

/// Invert a 4x4 symmetric matrix using cofactor expansion.
///
/// For GR metrics, the matrix is always invertible (det != 0) away
/// from coordinate singularities. Panics if det = 0.
#[allow(clippy::needless_range_loop)]
pub fn invert_4x4_symmetric(m: &MetricComponents) -> MetricComponents {
    // Use nalgebra for the inversion to avoid a 100-line cofactor expansion
    let mat = nalgebra::Matrix4::new(
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    );

    let inv = mat
        .try_inverse()
        .expect("metric tensor must be invertible (singular metric at coordinate singularity?)");

    let mut result = [[0.0; DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            result[i][j] = inv[(i, j)];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat Minkowski metric for testing.
    struct Minkowski;

    impl SpacetimeMetric for Minkowski {
        fn metric_components(&self, _x: &[f64; DIM]) -> MetricComponents {
            let mut g = [[0.0; DIM]; DIM];
            g[T][T] = -1.0;
            g[R][R] = 1.0;
            g[THETA][THETA] = 1.0;
            g[PHI][PHI] = 1.0;
            g
        }
    }

    #[test]
    fn test_minkowski_inverse() {
        let m = Minkowski;
        let x = [0.0, 1.0, 1.0, 0.0];
        let g_inv = m.inverse_metric(&x);
        assert!((g_inv[T][T] - (-1.0)).abs() < 1e-14);
        assert!((g_inv[R][R] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_minkowski_christoffel_zero() {
        let m = Minkowski;
        let x = [0.0, 5.0, 1.0, 0.5];
        let gamma = m.christoffel(&x);
        for a in 0..DIM {
            for b in 0..DIM {
                for c in 0..DIM {
                    assert!(
                        gamma[a][b][c].abs() < 1e-8,
                        "Minkowski Gamma^{}_{}{} = {} (expected 0)",
                        a,
                        b,
                        c,
                        gamma[a][b][c]
                    );
                }
            }
        }
    }

    #[test]
    fn test_minkowski_riemann_zero() {
        let m = Minkowski;
        let x = [0.0, 5.0, 1.0, 0.5];
        let riemann = riemann_from_christoffel(&m, &x);
        for a in 0..DIM {
            for b in 0..DIM {
                for c in 0..DIM {
                    for d in 0..DIM {
                        assert!(
                            riemann[a][b][c][d].abs() < 1e-4,
                            "Minkowski R^{}_{}{}{} = {} (expected 0)",
                            a,
                            b,
                            c,
                            d,
                            riemann[a][b][c][d]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_minkowski_ricci_zero() {
        let m = Minkowski;
        let x = [0.0, 5.0, 1.0, 0.5];
        let riemann = riemann_from_christoffel(&m, &x);
        let ricci = ricci_from_riemann(&riemann);
        for mu in 0..DIM {
            for nu in 0..DIM {
                assert!(
                    ricci[mu][nu].abs() < 1e-4,
                    "Minkowski R_{}{} = {} (expected 0)",
                    mu,
                    nu,
                    ricci[mu][nu]
                );
            }
        }
    }

    #[test]
    fn test_minkowski_kretschner_zero() {
        let m = Minkowski;
        let x = [0.0, 5.0, 1.0, 0.5];
        let g = m.metric_components(&x);
        let g_inv = m.inverse_metric(&x);
        let riemann = riemann_from_christoffel(&m, &x);
        let k = kretschner_scalar(&g, &g_inv, &riemann);
        assert!(k.abs() < 1e-4, "Minkowski Kretschner = {} (expected 0)", k);
    }

    #[test]
    fn test_invert_4x4_identity() {
        let mut m = [[0.0; DIM]; DIM];
        m[0][0] = -1.0;
        m[1][1] = 1.0;
        m[2][2] = 1.0;
        m[3][3] = 1.0;
        let inv = invert_4x4_symmetric(&m);
        assert!((inv[0][0] - (-1.0)).abs() < 1e-14);
        assert!((inv[1][1] - 1.0).abs() < 1e-14);
    }
}
