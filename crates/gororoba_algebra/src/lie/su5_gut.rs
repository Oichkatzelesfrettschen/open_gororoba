//! SU(5) Grand Unified Theory generator decomposition and quark sector.
//!
//! Classifies the 24 generators of SU(5) under SU(3)_C x SU(2)_L x U(1)_Y,
//! constructs the quark sector (5-bar and 10 representations), and runs 11
//! independent algebraic checks.
//!
//! References:
//!   Georgi, Glashow. PRL 32, 438 (1974).
//!   Georgi. "Lie Algebras in Particle Physics." 2nd ed. (1999).
//!   Langacker. Phys. Rep. 72, 185 (1981).
//!   Slansky. Phys. Rep. 79, 1 (1981).

use nalgebra::SMatrix;
use num_complex::Complex;
use serde::Serialize;

/// 5x5 complex matrix type for SU(5) generators in the fundamental representation.
pub type Mat5 = SMatrix<Complex<f64>, 5, 5>;

/// 10x10 complex matrix type for the antisymmetric tensor representation.
pub type Mat10 = SMatrix<Complex<f64>, 10, 10>;

const TOL: f64 = 1e-12;

// -------------------------------------------------------------------------
// Enums
// -------------------------------------------------------------------------

/// Which Standard Model gauge sector a generator belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GeneratorSector {
    SU3,
    SU2,
    U1,
    Leptoquark,
}

/// Whether a generator is symmetric, antisymmetric, or diagonal (Cartan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GenType {
    Symmetric,
    Antisymmetric,
    Diagonal,
}

/// Standard Model representation label.
#[derive(Debug, Clone, Serialize)]
pub struct SmRep {
    pub dim_su3: &'static str,
    pub dim_su2: u8,
    pub hypercharge: &'static str,
}

/// Which SU(5) representation a fermion sits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FermionRep {
    FiveBar,
    Ten,
}

/// Quantum numbers of a fermion.
#[derive(Debug, Clone, Serialize)]
pub struct QuantumNumbers {
    pub t3: f64,
    pub y: f64,
    pub q: f64,
    pub color: &'static str,
}

// -------------------------------------------------------------------------
// Public data types
// -------------------------------------------------------------------------

/// Classification of a single SU(5) generator.
#[derive(Debug, Clone)]
pub struct GeneratorInfo {
    pub index: usize,
    pub sector: GeneratorSector,
    pub generator_type: GenType,
    pub indices: (usize, usize),
    pub sm_representation: SmRep,
    pub physical_name: &'static str,
    pub matrix: Mat5,
}

/// Assignment of a fermion to an SU(5) representation.
#[derive(Debug, Clone, Serialize)]
pub struct FermionAssignment {
    pub name: &'static str,
    pub representation: FermionRep,
    pub position: &'static [usize],
    pub quantum_numbers: QuantumNumbers,
}

/// Result of a single algebraic verification check.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationResult {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

// -------------------------------------------------------------------------
// Helper functions (private)
// -------------------------------------------------------------------------

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn cr(re: f64) -> Complex<f64> {
    Complex::new(re, 0.0)
}

/// Elementary matrix E_{jk}: 1 at (j,k), 0 elsewhere.
fn eij(j: usize, k: usize) -> Mat5 {
    let mut m = Mat5::zeros();
    m[(j, k)] = c(1.0, 0.0);
    m
}

/// (E_{jk} + E_{kj}) / 2 -- off-diagonal symmetric generator.
fn symmetric_generator(j: usize, k: usize) -> Mat5 {
    (eij(j, k) + eij(k, j)) * cr(0.5)
}

/// -i(E_{jk} - E_{kj}) / 2 -- off-diagonal antisymmetric generator.
fn antisymmetric_generator(j: usize, k: usize) -> Mat5 {
    (eij(j, k) - eij(k, j)) * c(0.0, -0.5)
}

/// Generalized Cartan generator h_l (l=1,...,n-1).
///
/// h_l = sqrt(1/(2*l*(l+1))) * diag(1,...,1,-l,0,...,0)
/// where the first l entries are 1, entry l is -l, and the rest are 0.
fn cartan_generator(l: usize) -> Mat5 {
    let mut m = Mat5::zeros();
    let lf = l as f64;
    let norm = (1.0 / (2.0 * lf * (lf + 1.0))).sqrt();
    for i in 0..l {
        m[(i, i)] = cr(norm);
    }
    m[(l, l)] = cr(-lf * norm);
    m
}

/// U(1)_Y generator: (1/(2*sqrt(15))) * diag(-2,-2,-2,3,3).
fn hypercharge_generator() -> Mat5 {
    let norm = 1.0 / (2.0 * 15.0_f64.sqrt());
    let diag = [-2.0, -2.0, -2.0, 3.0, 3.0];
    let mut m = Mat5::zeros();
    for i in 0..5 {
        m[(i, i)] = cr(diag[i] * norm);
    }
    m
}

/// `[A, B]` = AB - BA.
fn commutator(a: &Mat5, b: &Mat5) -> Mat5 {
    a * b - b * a
}

/// {A, B} = AB + BA for 5x5 matrices.
fn anticommutator5(a: &Mat5, b: &Mat5) -> Mat5 {
    a * b + b * a
}

/// {A, B} = AB + BA for 10x10 matrices.
fn anticommutator10(a: &Mat10, b: &Mat10) -> Mat10 {
    a * b + b * a
}

/// Compute structure constants f^{abc} = -2i Tr(`[T^a,T^b]` T^c).
pub(crate) fn structure_constants(generators: &[Mat5; 24]) -> Box<[[[f64; 24]; 24]; 24]> {
    let mut f = Box::new([[[0.0_f64; 24]; 24]; 24]);
    for a in 0..24 {
        for b in (a + 1)..24 {
            let comm = commutator(&generators[a], &generators[b]);
            for ci in 0..24 {
                let val = (c(0.0, -2.0) * (comm * generators[ci]).trace()).re;
                f[a][b][ci] = val;
                f[b][a][ci] = -val;
            }
        }
    }
    f
}

fn is_zero(m: &Mat5) -> bool {
    m.iter().all(|v| v.norm() < TOL)
}

// -------------------------------------------------------------------------
// Generator construction
// -------------------------------------------------------------------------

/// Construct all 24 SU(5) generators as 5x5 complex matrices.
///
/// Ordered by sector:
///   0-5:   SU(3) off-diagonal (pairs (0,1),(0,2),(1,2) x {sym,antisym})
///   6-7:   SU(3) diagonal (Cartan h_1, h_2)
///   8-9:   SU(2) off-diagonal (pair (3,4) x {sym,antisym})
///   10:    SU(2) diagonal -- (1/2) diag(0,0,0,1,-1)
///   11:    U(1)_Y -- (1/(2*sqrt(15))) diag(-2,-2,-2,3,3)
///   12-17: Leptoquark symmetric (i in {0,1,2}, j in {3,4})
///   18-23: Leptoquark antisymmetric (same pairs)
pub fn build_generators() -> [Mat5; 24] {
    let mut gens = [Mat5::zeros(); 24];
    let mut idx = 0;

    // SU(3) off-diagonal: indices within {0,1,2}
    let su3_pairs = [(0, 1), (0, 2), (1, 2)];
    for &(j, k) in &su3_pairs {
        gens[idx] = symmetric_generator(j, k);
        idx += 1;
        gens[idx] = antisymmetric_generator(j, k);
        idx += 1;
    }

    // SU(3) diagonal: Cartan h_1, h_2
    gens[idx] = cartan_generator(1);
    idx += 1;
    gens[idx] = cartan_generator(2);
    idx += 1;

    // SU(2) off-diagonal: pair (3,4)
    gens[idx] = symmetric_generator(3, 4);
    idx += 1;
    gens[idx] = antisymmetric_generator(3, 4);
    idx += 1;

    // SU(2) diagonal: D_3 = (1/2) diag(0,0,0,1,-1)
    let mut d3 = Mat5::zeros();
    d3[(3, 3)] = cr(0.5);
    d3[(4, 4)] = cr(-0.5);
    gens[idx] = d3;
    idx += 1;

    // U(1)_Y
    gens[idx] = hypercharge_generator();
    idx += 1;

    // Leptoquark symmetric: one index in {0,1,2}, other in {3,4}
    let lq_pairs = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)];
    for &(i, j) in &lq_pairs {
        gens[idx] = symmetric_generator(i, j);
        idx += 1;
    }

    // Leptoquark antisymmetric: same pairs
    for &(i, j) in &lq_pairs {
        gens[idx] = antisymmetric_generator(i, j);
        idx += 1;
    }

    assert_eq!(idx, 24);

    // Verify orthonormality: Tr(T^a T^b) = (1/2) delta_{ab}
    for a in 0..24 {
        for b in 0..24 {
            let tr = (gens[a] * gens[b]).trace();
            let expected = if a == b { 0.5 } else { 0.0 };
            assert!(
                (tr - cr(expected)).norm() < TOL,
                "Normalization failed: Tr(T[{a}] T[{b}]) = {tr}, expected {expected}"
            );
        }
    }

    gens
}

// -------------------------------------------------------------------------
// Generator classification
// -------------------------------------------------------------------------

/// Static catalog entry: (sector, gen_type, indices, sm_rep, name).
struct CatalogEntry {
    sector: GeneratorSector,
    gen_type: GenType,
    indices: (usize, usize),
    sm_rep: SmRep,
    name: &'static str,
}

const SU3_NAMES: [&str; 8] = [
    "gluon_1", "gluon_2", "gluon_3", "gluon_4", "gluon_5", "gluon_6", "gluon_7", "gluon_8",
];
const SU2_NAMES: [&str; 3] = ["W_1", "W_2", "W_3"];
const U1_NAMES: [&str; 1] = ["B"];
const LQ_SYM_NAMES: [&str; 6] = [
    "X_boson_1",
    "X_boson_2",
    "X_boson_3",
    "X_boson_4",
    "X_boson_5",
    "X_boson_6",
];
const LQ_ANTI_NAMES: [&str; 6] = [
    "Y_boson_1",
    "Y_boson_2",
    "Y_boson_3",
    "Y_boson_4",
    "Y_boson_5",
    "Y_boson_6",
];

fn build_catalog() -> Vec<CatalogEntry> {
    let mut cat = Vec::with_capacity(24);
    let su3_pairs = [(0usize, 1usize), (0, 2), (1, 2)];

    // SU(3) off-diagonal
    for (pi, &(j, k)) in su3_pairs.iter().enumerate() {
        cat.push(CatalogEntry {
            sector: GeneratorSector::SU3,
            gen_type: GenType::Symmetric,
            indices: (j, k),
            sm_rep: SmRep {
                dim_su3: "8",
                dim_su2: 1,
                hypercharge: "0",
            },
            name: SU3_NAMES[2 * pi],
        });
        cat.push(CatalogEntry {
            sector: GeneratorSector::SU3,
            gen_type: GenType::Antisymmetric,
            indices: (j, k),
            sm_rep: SmRep {
                dim_su3: "8",
                dim_su2: 1,
                hypercharge: "0",
            },
            name: SU3_NAMES[2 * pi + 1],
        });
    }

    // SU(3) diagonal
    cat.push(CatalogEntry {
        sector: GeneratorSector::SU3,
        gen_type: GenType::Diagonal,
        indices: (0, 0),
        sm_rep: SmRep {
            dim_su3: "8",
            dim_su2: 1,
            hypercharge: "0",
        },
        name: SU3_NAMES[6],
    });
    cat.push(CatalogEntry {
        sector: GeneratorSector::SU3,
        gen_type: GenType::Diagonal,
        indices: (1, 1),
        sm_rep: SmRep {
            dim_su3: "8",
            dim_su2: 1,
            hypercharge: "0",
        },
        name: SU3_NAMES[7],
    });

    // SU(2) off-diagonal
    cat.push(CatalogEntry {
        sector: GeneratorSector::SU2,
        gen_type: GenType::Symmetric,
        indices: (3, 4),
        sm_rep: SmRep {
            dim_su3: "1",
            dim_su2: 3,
            hypercharge: "0",
        },
        name: SU2_NAMES[0],
    });
    cat.push(CatalogEntry {
        sector: GeneratorSector::SU2,
        gen_type: GenType::Antisymmetric,
        indices: (3, 4),
        sm_rep: SmRep {
            dim_su3: "1",
            dim_su2: 3,
            hypercharge: "0",
        },
        name: SU2_NAMES[1],
    });

    // SU(2) diagonal
    cat.push(CatalogEntry {
        sector: GeneratorSector::SU2,
        gen_type: GenType::Diagonal,
        indices: (3, 3),
        sm_rep: SmRep {
            dim_su3: "1",
            dim_su2: 3,
            hypercharge: "0",
        },
        name: SU2_NAMES[2],
    });

    // U(1)_Y
    cat.push(CatalogEntry {
        sector: GeneratorSector::U1,
        gen_type: GenType::Diagonal,
        indices: (0, 0),
        sm_rep: SmRep {
            dim_su3: "1",
            dim_su2: 1,
            hypercharge: "0",
        },
        name: U1_NAMES[0],
    });

    // Leptoquark symmetric
    let lq_pairs: [(usize, usize); 6] = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)];
    for (pi, &(i, j)) in lq_pairs.iter().enumerate() {
        cat.push(CatalogEntry {
            sector: GeneratorSector::Leptoquark,
            gen_type: GenType::Symmetric,
            indices: (i, j),
            sm_rep: SmRep {
                dim_su3: "3",
                dim_su2: 2,
                hypercharge: "5/6",
            },
            name: LQ_SYM_NAMES[pi],
        });
    }

    // Leptoquark antisymmetric
    for (pi, &(i, j)) in lq_pairs.iter().enumerate() {
        cat.push(CatalogEntry {
            sector: GeneratorSector::Leptoquark,
            gen_type: GenType::Antisymmetric,
            indices: (i, j),
            sm_rep: SmRep {
                dim_su3: "3*",
                dim_su2: 2,
                hypercharge: "-5/6",
            },
            name: LQ_ANTI_NAMES[pi],
        });
    }

    assert_eq!(cat.len(), 24);
    cat
}

/// Classify a single SU(5) generator by its Standard Model quantum numbers.
pub fn classify_generator(index: usize, generators: &[Mat5; 24]) -> GeneratorInfo {
    assert!(index < 24, "Generator index must be 0-23, got {index}");
    let cat = build_catalog();
    let e = &cat[index];
    GeneratorInfo {
        index,
        sector: e.sector,
        generator_type: e.gen_type,
        indices: e.indices,
        sm_representation: SmRep {
            dim_su3: e.sm_rep.dim_su3,
            dim_su2: e.sm_rep.dim_su2,
            hypercharge: e.sm_rep.hypercharge,
        },
        physical_name: e.name,
        matrix: generators[index],
    }
}

/// Classify all 24 SU(5) generators.
pub fn classify_all(generators: &[Mat5; 24]) -> Vec<GeneratorInfo> {
    (0..24).map(|i| classify_generator(i, generators)).collect()
}

// -------------------------------------------------------------------------
// Quark sector: 5-bar and 10 representations
// -------------------------------------------------------------------------

/// Construct the fermion content of one generation in SU(5).
pub fn build_quark_sector() -> Vec<FermionAssignment> {
    let mut fermions = Vec::with_capacity(15);

    // --- 5-bar representation ---
    // Indices 0,1,2 = d_R^c in colors red, green, blue (3*,1,1/3)
    static POS_0: [usize; 1] = [0];
    static POS_1: [usize; 1] = [1];
    static POS_2: [usize; 1] = [2];
    static POS_3: [usize; 1] = [3];
    static POS_4: [usize; 1] = [4];

    let colors: [(&str, &'static [usize]); 3] =
        [("red", &POS_0), ("green", &POS_1), ("blue", &POS_2)];
    static NAMES_DBAR: [&str; 3] = ["d_R_bar_red", "d_R_bar_green", "d_R_bar_blue"];
    for (ci, &(color, pos)) in colors.iter().enumerate() {
        fermions.push(FermionAssignment {
            name: NAMES_DBAR[ci],
            representation: FermionRep::FiveBar,
            position: pos,
            quantum_numbers: QuantumNumbers {
                t3: 0.0,
                y: 1.0 / 3.0,
                q: 1.0 / 3.0,
                color,
            },
        });
    }

    // Index 3 = nu_L, index 4 = e_L (1,2,-1/2)
    fermions.push(FermionAssignment {
        name: "nu_L",
        representation: FermionRep::FiveBar,
        position: &POS_3,
        quantum_numbers: QuantumNumbers {
            t3: 0.5,
            y: -0.5,
            q: 0.0,
            color: "singlet",
        },
    });
    fermions.push(FermionAssignment {
        name: "e_L",
        representation: FermionRep::FiveBar,
        position: &POS_4,
        quantum_numbers: QuantumNumbers {
            t3: -0.5,
            y: -0.5,
            q: -1.0,
            color: "singlet",
        },
    });

    // --- 10 representation ---
    // u_R^c: antisymmetric color tensor (3*,1,-2/3)
    static POS_01: [usize; 2] = [0, 1];
    static POS_02: [usize; 2] = [0, 2];
    static POS_12: [usize; 2] = [1, 2];
    static POS_03: [usize; 2] = [0, 3];
    static POS_04: [usize; 2] = [0, 4];
    static POS_13: [usize; 2] = [1, 3];
    static POS_14: [usize; 2] = [1, 4];
    static POS_23: [usize; 2] = [2, 3];
    static POS_24: [usize; 2] = [2, 4];
    static POS_34: [usize; 2] = [3, 4];

    fermions.push(FermionAssignment {
        name: "u_R_bar_blue",
        representation: FermionRep::Ten,
        position: &POS_01,
        quantum_numbers: QuantumNumbers {
            t3: 0.0,
            y: -2.0 / 3.0,
            q: -2.0 / 3.0,
            color: "blue",
        },
    });
    fermions.push(FermionAssignment {
        name: "u_R_bar_green",
        representation: FermionRep::Ten,
        position: &POS_02,
        quantum_numbers: QuantumNumbers {
            t3: 0.0,
            y: -2.0 / 3.0,
            q: -2.0 / 3.0,
            color: "green",
        },
    });
    fermions.push(FermionAssignment {
        name: "u_R_bar_red",
        representation: FermionRep::Ten,
        position: &POS_12,
        quantum_numbers: QuantumNumbers {
            t3: 0.0,
            y: -2.0 / 3.0,
            q: -2.0 / 3.0,
            color: "red",
        },
    });

    // Q_L: quark doublets (3,2,1/6)
    struct QLEntry {
        name: &'static str,
        position: &'static [usize],
        color: &'static str,
        t3: f64,
        q: f64,
    }
    let ql = [
        QLEntry {
            name: "u_L_red",
            position: &POS_03,
            color: "red",
            t3: 0.5,
            q: 2.0 / 3.0,
        },
        QLEntry {
            name: "d_L_red",
            position: &POS_04,
            color: "red",
            t3: -0.5,
            q: -1.0 / 3.0,
        },
        QLEntry {
            name: "u_L_green",
            position: &POS_13,
            color: "green",
            t3: 0.5,
            q: 2.0 / 3.0,
        },
        QLEntry {
            name: "d_L_green",
            position: &POS_14,
            color: "green",
            t3: -0.5,
            q: -1.0 / 3.0,
        },
        QLEntry {
            name: "u_L_blue",
            position: &POS_23,
            color: "blue",
            t3: 0.5,
            q: 2.0 / 3.0,
        },
        QLEntry {
            name: "d_L_blue",
            position: &POS_24,
            color: "blue",
            t3: -0.5,
            q: -1.0 / 3.0,
        },
    ];
    for e in &ql {
        fermions.push(FermionAssignment {
            name: e.name,
            representation: FermionRep::Ten,
            position: e.position,
            quantum_numbers: QuantumNumbers {
                t3: e.t3,
                y: 1.0 / 6.0,
                q: e.q,
                color: e.color,
            },
        });
    }

    // e_R^c: positron (1,1,1)
    fermions.push(FermionAssignment {
        name: "e_R_bar",
        representation: FermionRep::Ten,
        position: &POS_34,
        quantum_numbers: QuantumNumbers {
            t3: 0.0,
            y: 1.0,
            q: 1.0,
            color: "singlet",
        },
    });

    fermions
}

// -------------------------------------------------------------------------
// Representation matrices
// -------------------------------------------------------------------------

/// Construct the 10-dimensional representation matrices for each generator.
///
/// The antisymmetric tensor representation acts on the 10 independent
/// upper-triangular components of a 5x5 antisymmetric tensor:
///   (TV)_{ab} = T_{ac} V_{cb} + T_{bc} V_{ac}
///
/// Index map: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
pub fn build_ten_rep_matrices(generators: &[Mat5; 24]) -> [Mat10; 24] {
    // Build index map: pair -> sequential index
    let mut pairs = Vec::with_capacity(10);
    for i in 0..5 {
        for j in (i + 1)..5 {
            pairs.push((i, j));
        }
    }
    assert_eq!(pairs.len(), 10);

    let mut rep10 = [Mat10::zeros(); 24];
    for (gi, gmat) in generators.iter().enumerate() {
        let mut mat = Mat10::zeros();
        for (ab_idx, &(a, b)) in pairs.iter().enumerate() {
            for (cd_idx, &(ci, d)) in pairs.iter().enumerate() {
                // M_{(ab),(cd)} = T_{ac} delta_{bd} - T_{ad} delta_{bc}
                //               + T_{bd} delta_{ac} - T_{bc} delta_{ad}
                let mut val = Complex::new(0.0, 0.0);
                if b == d {
                    val += gmat[(a, ci)];
                }
                if b == ci {
                    val -= gmat[(a, d)];
                }
                if a == ci {
                    val += gmat[(b, d)];
                }
                if a == d {
                    val -= gmat[(b, ci)];
                }
                mat[(ab_idx, cd_idx)] = val;
            }
        }
        rep10[gi] = mat;
    }
    rep10
}

/// Construct the 5-bar (conjugate fundamental) representation matrices.
///
/// For the conjugate fundamental: T^a_{5*} = -(T^a)^T
pub fn build_five_bar_rep_matrices(generators: &[Mat5; 24]) -> [Mat5; 24] {
    let mut rep5bar = [Mat5::zeros(); 24];
    for (i, gmat) in generators.iter().enumerate() {
        rep5bar[i] = -gmat.transpose();
    }
    rep5bar
}

// -------------------------------------------------------------------------
// Verification suite (11 checks)
// -------------------------------------------------------------------------

/// Check 1: Tr(T^a T^b) = (1/2) delta_{ab}.
fn check_normalization(generators: &[Mat5; 24]) -> VerificationResult {
    let mut max_err = 0.0_f64;
    for a in 0..24 {
        for b in 0..24 {
            let tr = (generators[a] * generators[b]).trace();
            let expected = if a == b { 0.5 } else { 0.0 };
            let err = (tr - cr(expected)).norm();
            max_err = max_err.max(err);
        }
    }
    VerificationResult {
        name: "Normalization",
        passed: max_err < TOL,
        detail: format!("Tr(T^a T^b) = (1/2) delta_ab; max error = {max_err:.2e}"),
    }
}

/// Check 2: T^a = (T^a)^dagger.
fn check_hermiticity(generators: &[Mat5; 24]) -> VerificationResult {
    let mut max_err = 0.0_f64;
    for gm in generators {
        let diff = *gm - gm.adjoint();
        for v in diff.iter() {
            max_err = max_err.max(v.norm());
        }
    }
    VerificationResult {
        name: "Hermiticity",
        passed: max_err < TOL,
        detail: format!("T^a = (T^a)^dag; max error = {max_err:.2e}"),
    }
}

/// Check 3: Tr(T^a) = 0.
fn check_tracelessness(generators: &[Mat5; 24]) -> VerificationResult {
    let mut max_err = 0.0_f64;
    for gm in generators {
        let err = gm.trace().norm();
        max_err = max_err.max(err);
    }
    VerificationResult {
        name: "Tracelessness",
        passed: max_err < TOL,
        detail: format!("Tr(T^a) = 0; max trace = {max_err:.2e}"),
    }
}

/// Check 4: `[T^a,T^b]` = i f^{abc} T^c with f totally antisymmetric and Jacobi identity.
fn check_commutation(generators: &[Mat5; 24]) -> VerificationResult {
    let f = structure_constants(generators);

    // Verify closure: [T^a, T^b] should equal i * sum_c f^{abc} T^c
    let mut max_err = 0.0_f64;
    for a in 0..24 {
        for b in (a + 1)..24 {
            let comm = commutator(&generators[a], &generators[b]);
            let mut reconstructed = Mat5::zeros();
            for ci in 0..24 {
                reconstructed += generators[ci] * c(0.0, f[a][b][ci]);
            }
            for v in (comm - reconstructed).iter() {
                max_err = max_err.max(v.norm());
            }
        }
    }

    // Total antisymmetry of f
    let mut antisym_err = 0.0_f64;
    for a in 0..24 {
        for b in 0..24 {
            for ci in 0..24 {
                let err1 = (f[a][b][ci] + f[b][a][ci]).abs();
                let err2 = (f[a][b][ci] + f[a][ci][b]).abs();
                antisym_err = antisym_err.max(err1).max(err2);
            }
        }
    }

    // Jacobi identity (sampled with step=3)
    let mut jacobi_err = 0.0_f64;
    let step = 3;
    for a in (0..24).step_by(step) {
        for b in (0..24).step_by(step) {
            for ci in (0..24).step_by(step) {
                for d in 0..24 {
                    let mut s = 0.0_f64;
                    for e in 0..24 {
                        s += f[a][b][e] * f[e][ci][d]
                            + f[b][ci][e] * f[e][a][d]
                            + f[ci][a][e] * f[e][b][d];
                    }
                    jacobi_err = jacobi_err.max(s.abs());
                }
            }
        }
    }

    let passed = max_err < TOL && antisym_err < TOL && jacobi_err < TOL;
    VerificationResult {
        name: "Commutation",
        passed,
        detail: format!(
            "Lie algebra closure max error = {max_err:.2e}; \
             antisymmetry = {antisym_err:.2e}; Jacobi = {jacobi_err:.2e}"
        ),
    }
}

/// Check 5: `[SU(3), SU(3)]` is contained in SU(3).
fn check_su3_closure(generators: &[Mat5; 24]) -> VerificationResult {
    let su3_idx: Vec<usize> = (0..8).collect();
    let non_su3_idx: Vec<usize> = (8..24).collect();
    let mut max_leak = 0.0_f64;
    for &a in &su3_idx {
        for &b in &su3_idx {
            let comm = commutator(&generators[a], &generators[b]);
            for &ci in &non_su3_idx {
                let proj = (cr(2.0) * (comm * generators[ci]).trace()).norm();
                max_leak = max_leak.max(proj);
            }
        }
    }
    VerificationResult {
        name: "SU(3) closure",
        passed: max_leak < TOL,
        detail: format!("[SU3,SU3] leakage to non-SU3 = {max_leak:.2e}"),
    }
}

/// Check 6: `[SU(2), SU(2)]` is contained in SU(2).
fn check_su2_closure(generators: &[Mat5; 24]) -> VerificationResult {
    let su2_idx = [8, 9, 10];
    let non_su2_idx: Vec<usize> = (0..24).filter(|i| !su2_idx.contains(i)).collect();
    let mut max_leak = 0.0_f64;
    for &a in &su2_idx {
        for &b in &su2_idx {
            let comm = commutator(&generators[a], &generators[b]);
            for &ci in &non_su2_idx {
                let proj = (cr(2.0) * (comm * generators[ci]).trace()).norm();
                max_leak = max_leak.max(proj);
            }
        }
    }
    VerificationResult {
        name: "SU(2) closure",
        passed: max_leak < TOL,
        detail: format!("[SU2,SU2] leakage to non-SU2 = {max_leak:.2e}"),
    }
}

/// Check 7: `[U(1), SU(3)+SU(2)]` = 0.
fn check_u1_commutation(generators: &[Mat5; 24]) -> VerificationResult {
    let u1_gm = &generators[11];
    let mut max_err = 0.0_f64;
    for gm in &generators[..11] {
        let comm = commutator(u1_gm, gm);
        for v in comm.iter() {
            max_err = max_err.max(v.norm());
        }
    }
    VerificationResult {
        name: "U(1) commutation",
        passed: max_err < TOL,
        detail: format!("[U1, SU3+SU2] max = {max_err:.2e}"),
    }
}

/// Check 8: `[LQ, LQ]` generates SU(3)+SU(2)+U(1).
fn check_leptoquark_algebra(generators: &[Mat5; 24]) -> VerificationResult {
    let lq_idx: Vec<usize> = (12..24).collect();
    let mut has_su3 = false;
    let mut has_su2 = false;
    let mut has_u1 = false;
    let mut max_outside = 0.0_f64;

    for &a in &lq_idx {
        for &b in &lq_idx {
            let comm = commutator(&generators[a], &generators[b]);
            if is_zero(&comm) {
                continue;
            }

            for (ci, gm) in generators.iter().enumerate() {
                let proj = (cr(2.0) * (comm * *gm).trace()).norm();
                if proj > TOL {
                    if ci < 8 {
                        has_su3 = true;
                    } else if ci < 11 {
                        has_su2 = true;
                    } else if ci == 11 {
                        has_u1 = true;
                    }
                }
            }

            // Verify commutator is fully within the algebra
            let mut reconstructed = Mat5::zeros();
            for gm in generators.iter() {
                let coeff = cr(2.0) * (comm * *gm).trace();
                reconstructed += *gm * coeff;
            }
            for v in (comm - reconstructed).iter() {
                max_outside = max_outside.max(v.norm());
            }
        }
    }

    let passed = has_su3 && has_su2 && has_u1 && max_outside < TOL;
    VerificationResult {
        name: "Leptoquark algebra",
        passed,
        detail: format!(
            "[LQ,LQ] -> SU3={has_su3}, SU2={has_su2}, U1={has_u1}; \
             closure error = {max_outside:.2e}"
        ),
    }
}

/// Check 9: A(5-bar) + A(10) = 0 (gauge anomaly cancellation).
fn check_anomaly_cancellation(generators: &[Mat5; 24]) -> VerificationResult {
    // Method 1: group theory
    let a_5bar: i32 = -1;
    let a_10: i32 = 1; // N - 4 = 5 - 4 = 1
    let group_sum = a_5bar + a_10;

    // Method 2: numerical with cubic Casimir
    let rep5bar = build_five_bar_rep_matrices(generators);
    let rep10 = build_ten_rep_matrices(generators);

    let mut max_residual = 0.0_f64;
    for a in (0..24).step_by(4) {
        for b in (0..24).step_by(4) {
            for ci in (0..24).step_by(4) {
                let d_5bar =
                    cr(0.5) * (anticommutator5(&rep5bar[a], &rep5bar[b]) * rep5bar[ci]).trace();
                let d_10 = cr(0.5) * (anticommutator10(&rep10[a], &rep10[b]) * rep10[ci]).trace();
                let residual = (d_5bar + d_10).norm();
                max_residual = max_residual.max(residual);
            }
        }
    }

    let passed = group_sum == 0 && max_residual < TOL;
    VerificationResult {
        name: "Anomaly cancellation",
        passed,
        detail: format!(
            "Group theory: A(5*)={a_5bar} + A(10)={a_10} = {group_sum}; \
             numerical d^abc residual = {max_residual:.2e}"
        ),
    }
}

/// Check 10: sin^2(theta_W) = 3/8 at the GUT scale.
fn check_weinberg_angle(generators: &[Mat5; 24]) -> VerificationResult {
    let t3 = &generators[10];

    // Physical hypercharge: Y_phys = diag(-1/3, -1/3, -1/3, 1/2, 1/2)
    let mut y_phys = Mat5::zeros();
    for i in 0..3 {
        y_phys[(i, i)] = cr(-1.0 / 3.0);
    }
    y_phys[(3, 3)] = cr(0.5);
    y_phys[(4, 4)] = cr(0.5);

    let tr_y_phys_sq = (y_phys * y_phys).trace().re; // 5/6
    let tr_t3_phys_sq = (t3 * t3).trace().re; // 1/2

    // GUT normalization ratio
    let k = tr_t3_phys_sq / tr_y_phys_sq; // 3/5
    let sin2_tw = k / (1.0 + k); // 3/8
    let expected = 3.0 / 8.0;
    let err = (sin2_tw - expected).abs();

    VerificationResult {
        name: "Weinberg angle",
        passed: err < TOL,
        detail: format!(
            "sin^2(theta_W) = {sin2_tw:.6}  (expected 3/8 = {expected:.6}); \
             Tr(T3^2)={tr_t3_phys_sq:.4}, Tr(Y^2)={tr_y_phys_sq:.4}, \
             k=Tr(T3^2)/Tr(Y^2)={k:.4}"
        ),
    }
}

/// Check 11: Q = T_3 + Y for all fermions.
fn check_charge_quantization(fermions: &[FermionAssignment]) -> VerificationResult {
    let mut max_err = 0.0_f64;
    let mut failures = Vec::new();
    for f in fermions {
        let qn = &f.quantum_numbers;
        let computed_q = qn.t3 + qn.y;
        let err = (computed_q - qn.q).abs();
        if err > TOL {
            failures.push(format!(
                "{}: T3={} + Y={} = {} != Q={}",
                f.name, qn.t3, qn.y, computed_q, qn.q
            ));
        }
        max_err = max_err.max(err);
    }
    let detail = if failures.is_empty() {
        format!(
            "Q = T3 + Y verified for all {} fermions; max error = {max_err:.2e}",
            fermions.len()
        )
    } else {
        format!("FAILED for: {}", failures.join("; "))
    };
    VerificationResult {
        name: "Charge quantization",
        passed: max_err < TOL,
        detail,
    }
}

/// Run all 11 algebraic verification checks.
pub fn verify_algebra() -> Vec<VerificationResult> {
    let generators = build_generators();
    let fermions = build_quark_sector();

    vec![
        check_normalization(&generators),
        check_hermiticity(&generators),
        check_tracelessness(&generators),
        check_commutation(&generators),
        check_su3_closure(&generators),
        check_su2_closure(&generators),
        check_u1_commutation(&generators),
        check_leptoquark_algebra(&generators),
        check_anomaly_cancellation(&generators),
        check_weinberg_angle(&generators),
        check_charge_quantization(&fermions),
    ]
}

// -------------------------------------------------------------------------
// Output formatting (for CLI)
// -------------------------------------------------------------------------

/// Format an SM representation tuple for display.
pub fn format_sm_rep(rep: &SmRep) -> String {
    format!("({},{},{})", rep.dim_su3, rep.dim_su2, rep.hypercharge)
}

/// Format generator classification as an aligned table.
pub fn format_generator_table(infos: &[GeneratorInfo]) -> String {
    let mut lines = Vec::new();
    let hdr = format!(
        "{:>3}  {:<11}  {:<13}  {:<8}  {:<14}  {:<14}",
        "Idx", "Sector", "Type", "Indices", "SM Rep", "Name"
    );
    lines.push(hdr.clone());
    lines.push("-".repeat(hdr.len()));
    for g in infos {
        let idx_str = format!("({},{})", g.indices.0, g.indices.1);
        let rep_str = format_sm_rep(&g.sm_representation);
        let sector_str = format!("{:?}", g.sector);
        let type_str = format!("{:?}", g.generator_type);
        lines.push(format!(
            "{:>3}  {:<11}  {:<13}  {:<8}  {:<14}  {:<14}",
            g.index, sector_str, type_str, idx_str, rep_str, g.physical_name
        ));
    }
    lines.join("\n")
}

/// Format fermion assignments as aligned tables (5-bar and 10 separately).
pub fn format_fermion_table(fermions: &[FermionAssignment]) -> String {
    let mut lines = Vec::new();

    for (rep_name, rep_val) in [("5-bar", FermionRep::FiveBar), ("10", FermionRep::Ten)] {
        let subset: Vec<&FermionAssignment> = fermions
            .iter()
            .filter(|f| f.representation == rep_val)
            .collect();
        if subset.is_empty() {
            continue;
        }
        lines.push(format!("\n{rep_name} representation:"));
        let hdr = format!(
            "  {:<18}  {:<10}  {:>5}  {:>7}  {:>7}  {:<8}",
            "Name", "Position", "T3", "Y", "Q", "Color"
        );
        lines.push(hdr.clone());
        lines.push(format!("  {}", "-".repeat(hdr.len() - 2)));
        for f in &subset {
            let qn = &f.quantum_numbers;
            let pos_str = format!("{:?}", f.position);
            lines.push(format!(
                "  {:<18}  {:<10}  {:>5.2}  {:>7.4}  {:>7.4}  {:<8}",
                f.name, pos_str, qn.t3, qn.y, qn.q, qn.color
            ));
        }
    }

    lines.join("\n")
}

/// Format the 10-rep antisymmetric tensor matrix.
pub fn format_ten_tensor() -> String {
    let labels: [[&str; 5]; 5] = [
        ["0", "u_b^c", "-u_g^c", "u_r", "d_r"],
        ["-u_b^c", "0", "u_r^c", "u_g", "d_g"],
        ["u_g^c", "-u_r^c", "0", "u_b", "d_b"],
        ["-u_r", "-u_g", "-u_b", "0", "e^+"],
        ["-d_r", "-d_g", "-d_b", "-e^+", "0"],
    ];
    let mut lines = vec![
        "10-rep antisymmetric tensor (1/sqrt(2)) * psi^{ij}:".to_string(),
        String::new(),
    ];
    let col_w = 8;
    let mut header = "       ".to_string();
    for j in 0..5 {
        header.push_str(&format!("{j:>col_w$}"));
    }
    lines.push(header);
    for (i, row) in labels.iter().enumerate() {
        let mut s = format!("  [{i}]  ");
        for val in row {
            s.push_str(&format!("{val:>col_w$}"));
        }
        lines.push(s);
    }
    lines.join("\n")
}

/// Format verification results as an aligned table.
pub fn format_verification_table(results: &[VerificationResult]) -> String {
    let mut lines = Vec::new();
    let hdr = format!("{:>2}  {:<24}  {:<6}  {}", "#", "Check", "Result", "Detail");
    lines.push(hdr.clone());
    lines.push("-".repeat(hdr.len().max(80)));
    for (i, r) in results.iter().enumerate() {
        let status = if r.passed { "PASS" } else { "FAIL" };
        lines.push(format!(
            "{:>2}  {:<24}  {:<6}  {}",
            i + 1,
            r.name,
            status,
            r.detail
        ));
    }
    lines.join("\n")
}

/// Format a 5x5 complex matrix for verbose output.
pub fn format_matrix(m: &Mat5, label: &str) -> String {
    let mut lines = vec![format!("  {label}:")];
    for i in 0..5 {
        let mut row_parts = Vec::new();
        for j in 0..5 {
            let v = m[(i, j)];
            if v.im.abs() < TOL {
                row_parts.push(format!("{:>8.4}", v.re));
            } else if v.re.abs() < TOL {
                row_parts.push(format!("{:>7.4}i", v.im));
            } else {
                row_parts.push(format!("{:+.3}{:+.3}i", v.re, v.im));
            }
        }
        lines.push(format!("    [{}]", row_parts.join(" ")));
    }
    lines.join("\n")
}

/// Format the commutator closure table for verbose output.
pub fn format_commutator_table(generators: &[Mat5; 24]) -> String {
    let sector_ranges: [(&str, Vec<usize>); 4] = [
        ("SU(3)", (0..8).collect()),
        ("SU(2)", vec![8, 9, 10]),
        ("U(1)", vec![11]),
        ("LQ", (12..24).collect()),
    ];
    let mut lines = vec![
        "Commutator closure table:".to_string(),
        String::new(),
        format!("  {:<16} -> {:<30}", "[A, B]", "Result"),
        format!("  {}", "-".repeat(48)),
    ];

    for (s1i, (s1, idx1)) in sector_ranges.iter().enumerate() {
        for (s2, idx2) in &sector_ranges[s1i..] {
            let mut projections = [0.0_f64; 4];
            for &a in idx1 {
                for &b in idx2 {
                    let comm = commutator(&generators[a], &generators[b]);
                    if is_zero(&comm) {
                        continue;
                    }
                    for (si, (_, sidx)) in sector_ranges.iter().enumerate() {
                        for &ci in sidx {
                            let proj = (cr(2.0) * (comm * generators[ci]).trace()).norm();
                            projections[si] = projections[si].max(proj);
                        }
                    }
                }
            }

            let nonzero: Vec<&str> = sector_ranges
                .iter()
                .enumerate()
                .filter(|(si, _)| projections[*si] > TOL)
                .map(|(_, (name, _))| *name)
                .collect();
            let result = if nonzero.is_empty() {
                "0".to_string()
            } else {
                nonzero.join("+")
            };
            let label = format!("  [{s1}, {s2}]");
            let pad = " ".repeat(18usize.saturating_sub(label.len()));
            lines.push(format!("{label}{pad}-> {result}"));
        }
    }
    lines.join("\n")
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_11_checks_pass() {
        let results = verify_algebra();
        for r in &results {
            assert!(r.passed, "Check '{}' failed: {}", r.name, r.detail);
        }
        assert_eq!(results.len(), 11);
    }

    #[test]
    fn test_generator_count() {
        let gens = build_generators();
        assert_eq!(gens.len(), 24);
    }

    #[test]
    fn test_sector_counts() {
        let gens = build_generators();
        let infos = classify_all(&gens);
        let su3 = infos
            .iter()
            .filter(|g| g.sector == GeneratorSector::SU3)
            .count();
        let su2 = infos
            .iter()
            .filter(|g| g.sector == GeneratorSector::SU2)
            .count();
        let u1 = infos
            .iter()
            .filter(|g| g.sector == GeneratorSector::U1)
            .count();
        let lq = infos
            .iter()
            .filter(|g| g.sector == GeneratorSector::Leptoquark)
            .count();
        assert_eq!(su3, 8);
        assert_eq!(su2, 3);
        assert_eq!(u1, 1);
        assert_eq!(lq, 12);
        assert_eq!(su3 + su2 + u1 + lq, 24);
    }

    #[test]
    fn test_fermion_count() {
        let fermions = build_quark_sector();
        let five_bar = fermions
            .iter()
            .filter(|f| f.representation == FermionRep::FiveBar)
            .count();
        let ten = fermions
            .iter()
            .filter(|f| f.representation == FermionRep::Ten)
            .count();
        assert_eq!(five_bar, 5);
        assert_eq!(ten, 10);
        assert_eq!(fermions.len(), 15);
    }
}
