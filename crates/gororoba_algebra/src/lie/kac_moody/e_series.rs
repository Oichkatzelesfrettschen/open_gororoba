//! E-series infinite-dimensional Kac-Moody algebras.
//!
//! - [`E9RootSystem`] -- `E_9 = E_8^{(1)}` affine extension. Real roots
//!   `alpha + n*delta` for `alpha` an E8 root; imaginary roots `n*delta`.
//! - [`E10RootSystem`] -- hyperbolic Lorentzian extension, conjectured to
//!   encode M-theory symmetries (Damour-Henneaux-Nicolai).
//! - [`E11RootSystem`] -- proposed hidden symmetry of D=11 supergravity (West).
//! - [`ESeriesRootSystem`] -- a unified enum spanning `E_8` through `E_11`.
//!
//! All four use Cartan matrices from [`super::e8_cartan`], [`super::e9_cartan`],
//! [`super::e10_cartan`], and [`super::e11_cartan`] respectively, sharing the
//! branch-at-node-4 numbering with [`crate::lie::e8::root_system`].

use super::*;

/// E9 = E8^{(1)} root system (affine E8).
///
/// The root system consists of:
/// - Real roots: alpha + n*delta for alpha in E8 roots, n in Z
/// - Imaginary roots: n*delta for n != 0
///   where delta is the null root (minimal imaginary root).
#[derive(Debug, Clone)]
pub struct E9RootSystem {
    /// E8 simple roots (finite part)
    pub e8_simple_roots: Vec<KacMoodyRoot>,
    /// The null root delta
    pub delta: KacMoodyRoot,
    /// Cartan matrix
    pub cartan: GeneralizedCartanMatrix,
}

impl E9RootSystem {
    /// Create the E9 root system.
    pub fn new() -> Self {
        // E8 simple roots in R^8
        let e8_simple = vec![
            vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            vec![-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        ];

        let e8_simple_roots: Vec<KacMoodyRoot> =
            e8_simple.into_iter().map(KacMoodyRoot::real).collect();

        // Delta is the null root corresponding to the affine extension
        // It represents the imaginary direction
        let delta = KacMoodyRoot {
            finite_part: vec![0.0; 8],
            level: 1,
            lorentz_coords: vec![],
            root_type: RootType::Null,
        };

        Self {
            e8_simple_roots,
            delta,
            cartan: e9_cartan(),
        }
    }

    /// Get the affine simple root (alpha_8 in our numbering).
    /// alpha_8 = delta - theta where theta is the highest root of E8.
    ///
    /// In our 0-indexed root-vector convention:
    ///   theta = 2*alpha_0 + 3*alpha_1 + 4*alpha_2 + 5*alpha_3
    ///         + 6*alpha_4 + 3*alpha_5 + 4*alpha_6 + 2*alpha_7
    ///         = (1, 0, 0, 0, 0, 0, 0, -1)
    pub fn affine_simple_root(&self) -> KacMoodyRoot {
        // Exact highest root: theta = (1, 0, 0, 0, 0, 0, 0, -1)
        // Verified: |theta|^2 = 2, <theta, alpha_0> = 1, <theta, alpha_i> = 0 for i > 0.
        let theta = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let neg_theta: Vec<f64> = theta.iter().map(|x| -x).collect();

        KacMoodyRoot::affine(neg_theta, 1)
    }

    /// Generate real roots up to a maximum level.
    pub fn real_roots_up_to_level(&self, max_level: u32) -> Vec<KacMoodyRoot> {
        let mut roots = Vec::new();

        // Level 0: E8 roots (240 of them)
        // We just generate a sample for now
        for root in &self.e8_simple_roots {
            roots.push(root.clone());
            // Also add negative roots
            let neg: Vec<f64> = root.finite_part.iter().map(|x| -x).collect();
            roots.push(KacMoodyRoot::real(neg));
        }

        // Higher levels: alpha + n*delta
        for level in 1..=max_level {
            for base_root in &self.e8_simple_roots {
                let mut pos = base_root.clone();
                pos.level = level as i32;
                pos.root_type = RootType::Real;
                roots.push(pos);

                let mut neg = base_root.clone();
                neg.finite_part = neg.finite_part.iter().map(|x| -x).collect();
                neg.level = -(level as i32);
                neg.root_type = RootType::Real;
                roots.push(neg);
            }
        }

        roots
    }

    /// Count of imaginary roots at level n (multiplicity).
    /// For E9, imaginary root multiplicity at level n is 8 (dimension of E8 Cartan).
    pub fn imaginary_root_multiplicity(&self, level: i32) -> usize {
        if level == 0 {
            0 // No imaginary roots at level 0
        } else {
            8 // Cartan subalgebra dimension
        }
    }
}

impl Default for E9RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// E10 root system (hyperbolic, Lorentzian).
///
/// E10 is the over-extended E8, conjectured to be a symmetry of M-theory.
/// Its root lattice has signature (9, 1) - Lorentzian.
#[derive(Debug, Clone)]
pub struct E10RootSystem {
    /// E9 embedded in E10
    pub e9_roots: E9RootSystem,
    /// The hyperbolic extension direction
    pub hyperbolic_coord: f64,
    /// Cartan matrix
    pub cartan: GeneralizedCartanMatrix,
}

impl E10RootSystem {
    /// Create the E10 root system.
    pub fn new() -> Self {
        Self {
            e9_roots: E9RootSystem::new(),
            hyperbolic_coord: 0.0,
            cartan: e10_cartan(),
        }
    }

    /// The signature of the E10 root lattice: (9, 1).
    pub fn signature() -> (usize, usize) {
        (9, 1) // 9 positive, 1 negative eigenvalue
    }

    /// Inner product for E10 (Lorentzian).
    /// Standard realization uses 8D finite space + 2D Lorentzian extension (+1, -1).
    pub fn inner_product(&self, a: &KacMoodyRoot, b: &KacMoodyRoot) -> f64 {
        let mut result = 0.0;

        // Finite part: positive definite (E8 inner product)
        for (x, y) in a.finite_part.iter().zip(b.finite_part.iter()) {
            result += x * y;
        }

        // Lorentzian extension (2D): signature (+1, -1)
        if a.lorentz_coords.len() >= 2 && b.lorentz_coords.len() >= 2 {
            result += a.lorentz_coords[0] * b.lorentz_coords[0]; // Spacelike
            result -= a.lorentz_coords[1] * b.lorentz_coords[1]; // Timelike
        } else if !a.lorentz_coords.is_empty() && !b.lorentz_coords.is_empty() {
            // Fallback for 1D: assume timelike if it's the only one
            result -= a.lorentz_coords[0] * b.lorentz_coords[0];
        }

        // Ignore level for E10 since we use lorentz_coords instead for consistency
        result
    }

    /// Generate simple roots for E10.
    pub fn simple_roots(&self) -> Vec<KacMoodyRoot> {
        let mut roots = Vec::with_capacity(10);

        // E8 simple roots: (alpha_i, 0, 0)
        let e8_roots = &self.e9_roots.e8_simple_roots;
        for root in e8_roots {
            let mut r = root.clone();
            r.lorentz_coords = vec![0.0, 0.0];
            roots.push(r);
        }

        // Exact Highest Root theta for E8
        // Using the property that dot(theta, alpha_i) >= 0 and theta is the highest.
        // For standard E8 basis, theta = (1, 1, 0, 0, 0, 0, 0, 0) in some conventions.
        // Let's use the one that makes dot(theta, alpha_i) = 1 for the affine connection.
        // In our e8_cartan, node 0 connects to node 1. So we want dot(theta, alpha_0) = 1.
        let mut theta_vec = [0.0; 8];
        // Sum simple roots weighted by Coxeter labels to obtain highest root.
        // Our Dynkin diagram: 0--1--2--3--4(--5)(--6--7), branching at node 4.
        // Coxeter labels for this ordering: [2, 3, 4, 5, 6, 3, 4, 2].
        // Yields theta = [1, 0, 0, 0, 0, 0, 0, -1] with |theta|^2 = 2.
        let labels = [2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 2.0];
        for (i, &label) in labels.iter().enumerate() {
            for (tv, &fp) in theta_vec.iter_mut().zip(e8_roots[i].finite_part.iter()) {
                *tv += label * fp;
            }
        }

        // E9 affine root (alpha_0): (-theta, 1, 1)
        roots.push(KacMoodyRoot {
            finite_part: theta_vec.iter().map(|x| -x).collect(),
            level: 0,
            lorentz_coords: vec![1.0, 1.0],
            root_type: RootType::Real,
        });

        // E10 hyperbolic extension (alpha_9): (0, -1.5, -0.5)
        roots.push(KacMoodyRoot {
            finite_part: vec![0.0; 8],
            level: 0,
            lorentz_coords: vec![-1.5, -0.5],
            root_type: RootType::Real,
        });

        roots
    }

    /// Check if a root is timelike, spacelike, or null.
    pub fn causal_type(&self, root: &KacMoodyRoot) -> &'static str {
        let norm_sq = self.inner_product(root, root);
        if norm_sq > 1e-10 {
            "spacelike"
        } else if norm_sq < -1e-10 {
            "timelike"
        } else {
            "null"
        }
    }
}

impl Default for E10RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// E11 root system (very extended E8).
///
/// E11 is proposed as a hidden symmetry of 11D supergravity (West 2001).
/// Its structure is even more complex than E10.
#[derive(Debug, Clone)]
pub struct E11RootSystem {
    /// E10 embedded in E11
    pub e10_base: E10RootSystem,
    /// Cartan matrix
    pub cartan: GeneralizedCartanMatrix,
}

impl E11RootSystem {
    /// Create the E11 root system.
    pub fn new() -> Self {
        Self {
            e10_base: E10RootSystem::new(),
            cartan: e11_cartan(),
        }
    }

    /// The signature of E11.
    /// E11 has an even more indefinite signature than E10.
    pub fn signature() -> (usize, usize) {
        (9, 2) // Rough estimate - actual structure is more complex
    }

    /// Connection to 11D supergravity (informational).
    pub fn supergravity_connection() -> &'static str {
        "E11 is conjectured to be a symmetry of 11D supergravity. \
         Its level decomposition under GL(11) gives the graviton, \
         3-form, 6-form, and dual graviton representations. \
         Reference: West (2001), Class. Quantum Grav. 18, 4443."
    }

    /// Decomposition levels relevant to M-theory.
    /// Returns (level, representation dimension, interpretation).
    pub fn mtheory_level_decomposition() -> Vec<(i32, &'static str, &'static str)> {
        vec![
            (0, "SO(1,10)", "Spacetime Lorentz group"),
            (1, "A_abc", "3-form potential (M2-brane)"),
            (2, "A_abcdef", "6-form potential (M5-brane)"),
            (3, "h_a,bcdefghi", "Dual graviton"),
            // Higher levels continue...
        ]
    }
}

impl Default for E11RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified interface for E-series root systems.
#[derive(Debug, Clone)]
pub enum ESeriesRootSystem {
    E8(Box<crate::lie::e8::root_system::E8Lattice>),
    E9(E9RootSystem),
    E10(E10RootSystem),
    E11(E11RootSystem),
}

impl ESeriesRootSystem {
    /// Get the rank of the root system.
    pub fn rank(&self) -> usize {
        match self {
            ESeriesRootSystem::E8(_) => 8,
            ESeriesRootSystem::E9(_) => 9,
            ESeriesRootSystem::E10(_) => 10,
            ESeriesRootSystem::E11(_) => 11,
        }
    }

    /// Get the classification.
    pub fn classification(&self) -> KacMoodyType {
        match self {
            ESeriesRootSystem::E8(_) => KacMoodyType::Finite,
            ESeriesRootSystem::E9(_) => KacMoodyType::Affine,
            ESeriesRootSystem::E10(_) => KacMoodyType::Hyperbolic,
            ESeriesRootSystem::E11(_) => KacMoodyType::Indefinite,
        }
    }

    /// Is the Weyl group finite?
    pub fn has_finite_weyl_group(&self) -> bool {
        matches!(self, ESeriesRootSystem::E8(_))
    }

    /// Physics applications.
    pub fn physics_applications(&self) -> Vec<&'static str> {
        match self {
            ESeriesRootSystem::E8(_) => vec![
                "Heterotic string theory (E8 x E8)",
                "Grand unified theories",
                "Moonshine connections to Monster group",
            ],
            ESeriesRootSystem::E9(_) => vec![
                "2D conformal field theory",
                "Affine Lie algebras in string theory",
                "WZW models",
            ],
            ESeriesRootSystem::E10(_) => vec![
                "M-theory hidden symmetry (Damour-Henneaux-Nicolai)",
                "Cosmological billiards near spacelike singularities",
                "BKL dynamics of general relativity",
            ],
            ESeriesRootSystem::E11(_) => vec![
                "11D supergravity symmetry (West)",
                "M-theory duality unification",
                "Form field democracy",
            ],
        }
    }
}
