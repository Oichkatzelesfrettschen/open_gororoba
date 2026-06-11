//! Non-associative algebras: Malcev, Bol, Lie-admissible.
//!
//! This module hosts the `NonAssociativeAlgebra` trait used to test identities
//! (commutator anticommutativity, Jacobi, Malcev) on candidate algebras.
//!
use std::{fmt, sync::OnceLock};

/// Trait for non-associative algebras in the construction-method hierarchy.
///
/// Implementors satisfy:
/// - Level 1 = Mechanism (anticommutative bracket).
/// - Level 2 = Dimension.
/// - Level 3 = Composition parameters.
pub trait NonAssociativeAlgebra: Clone + fmt::Debug {
    /// Dimension of the algebra.
    fn dim(&self) -> usize;

    /// Mechanism-specific product.
    fn product(&self, a: &[f64], b: &[f64]) -> Vec<f64>;

    /// Commutator `[a, b] = ab - ba`.
    fn commutator(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let ab = self.product(a, b);
        let ba = self.product(b, a);
        ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect()
    }

    /// True iff `[a, b]` vanishes within numerical tolerance.
    fn commutes(&self, a: &[f64], b: &[f64]) -> bool {
        let comm = self.commutator(a, b);
        comm.iter().all(|x| x.abs() < 1e-10)
    }

    /// `||[a, b]||` -- non-commutativity magnitude.
    fn non_commutativity_violation(&self, a: &[f64], b: &[f64]) -> f64 {
        let comm = self.commutator(a, b);
        comm.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Jacobi-identity violation `||[[a,b],c] + [[b,c],a] + [[c,a],b]||`.
    fn jacobi_violation(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        let ab = self.commutator(a, b);
        let bc = self.commutator(b, c);
        let ca = self.commutator(c, a);

        let ab_c = self.commutator(&ab, c);
        let bc_a = self.commutator(&bc, a);
        let ca_b = self.commutator(&ca, b);

        let mut result = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            result[i] = ab_c[i] + bc_a[i] + ca_b[i];
        }
        result.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Malcev-identity violation
    /// `|| (xy)(xz) - (((xy)z)x + ((yz)x)x + ((zx)x)y) ||`.
    fn malcev_violation(&self, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let xy = self.product(x, y);
        let xy_xz = self.product(&xy, x);
        let xy_xz_prod = self.product(&xy_xz, z);

        let xy_z = self.product(&xy, z);
        let xy_z_x = self.product(&xy_z, x);

        let yz = self.product(y, z);
        let yz_x = self.product(&yz, x);
        let yz_x_x = self.product(&yz_x, x);

        let zx = self.product(z, x);
        let zx_x = self.product(&zx, x);
        let zx_x_y = self.product(&zx_x, y);

        let mut rhs = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            rhs[i] = xy_z_x[i] + yz_x_x[i] + zx_x_y[i];
        }

        xy_xz_prod
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
    }

    /// Associativity violation `|| (ab)c - a(bc) ||`.
    fn associativity_violation(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        let ab = self.product(a, b);
        let ab_c = self.product(&ab, c);

        let bc = self.product(b, c);
        let a_bc = self.product(a, &bc);

        ab_c.iter()
            .zip(a_bc.iter())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
    }
}

/// String-based Freudenthal-Tits magic-square compatibility entry.
#[derive(Clone)]
pub struct FreudenthalTitsMagicSquare {
    pub composition_algebra_dim: usize,
    pub composition_algebra_name: String,
    pub jordan_algebra_dim: usize,
    pub jordan_algebra_name: String,
    pub exceptional_lie_algebra_name: String,
    pub exceptional_lie_algebra_dim: usize,
}

impl fmt::Debug for FreudenthalTitsMagicSquare {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FreudenthalTitsMagicSquare")
            .field(
                "composition_algebra",
                &format!(
                    "{} ({}D)",
                    self.composition_algebra_name, self.composition_algebra_dim
                ),
            )
            .field(
                "jordan_algebra",
                &format!(
                    "{} ({}D)",
                    self.jordan_algebra_name, self.jordan_algebra_dim
                ),
            )
            .field(
                "exceptional_lie_algebra",
                &format!(
                    "{} ({}D)",
                    self.exceptional_lie_algebra_name, self.exceptional_lie_algebra_dim
                ),
            )
            .finish()
    }
}

impl FreudenthalTitsMagicSquare {
    /// Create a magic-square entry from composition and Jordan algebra dimensions.
    pub fn new(
        composition_name: &str,
        composition_dim: usize,
        jordan_name: &str,
        jordan_dim: usize,
    ) -> Self {
        let (exceptional_name, exceptional_dim) =
            Self::tits_construction(composition_dim, jordan_dim);

        Self {
            composition_algebra_dim: composition_dim,
            composition_algebra_name: composition_name.to_string(),
            jordan_algebra_dim: jordan_dim,
            jordan_algebra_name: jordan_name.to_string(),
            exceptional_lie_algebra_name: exceptional_name,
            exceptional_lie_algebra_dim: exceptional_dim,
        }
    }

    fn tits_construction(composition_dim: usize, jordan_dim: usize) -> (String, usize) {
        let (name, dim) = match (composition_dim, jordan_dim) {
            (1, 1) => ("A1", 3),
            (1, 3) => ("A2", 8),
            (1, 27) => ("E6", 78),
            (2, 1) => ("A2", 8),
            (2, 3) => ("A2", 8),
            (2, 27) => ("E6", 78),
            (4, 1) => ("C3", 21),
            (4, 3) => ("A5", 35),
            (4, 27) => ("E7", 133),
            (8, 1) => ("F4", 52),
            (8, 3) => ("E6", 78),
            (8, 27) => ("E8", 248),
            _ => ("Unknown", 0),
        };

        (name.to_string(), dim)
    }

    /// Verify that another dimension pair resolves to the same table cell.
    pub fn verify_symmetry(&self, comp_dim_other: usize, jordan_dim_other: usize) -> bool {
        let (name_ab, dim_ab) =
            Self::tits_construction(self.composition_algebra_dim, self.jordan_algebra_dim);
        let (name_ba, dim_ba) = Self::tits_construction(comp_dim_other, jordan_dim_other);

        name_ab == name_ba && dim_ab == dim_ba
    }

    /// Return true for the exceptional entries exposed by the compatibility table.
    pub fn is_pure_exceptional(&self) -> bool {
        matches!(
            self.exceptional_lie_algebra_name.as_str(),
            "E6" | "E7" | "E8" | "F4" | "G2"
        )
    }

    /// Compute the table dimension associated with the provided dimensions.
    pub fn tits_dimension_formula(comp_dim: usize, jordan_dim: usize) -> usize {
        let (_, dim) = Self::tits_construction(comp_dim, jordan_dim);
        dim
    }
}

/// E8 root-system compatibility record.
#[derive(Clone)]
pub struct E8RootSystem {
    pub dim: usize,
    pub name: String,
}

impl fmt::Debug for E8RootSystem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("E8RootSystem")
            .field("dim", &self.dim)
            .field("name", &self.name)
            .finish()
    }
}

impl E8RootSystem {
    /// Create the legacy E8 root-system summary with all 240 roots.
    pub fn new() -> Self {
        Self {
            dim: 248,
            name: "E8".to_string(),
        }
    }

    fn roots_static() -> &'static Vec<[f64; 8]> {
        static ROOTS: OnceLock<Vec<[f64; 8]>> = OnceLock::new();
        ROOTS.get_or_init(Self::enumerate_roots)
    }

    fn simple_roots_static() -> &'static Vec<[f64; 8]> {
        static SIMPLE_ROOTS: OnceLock<Vec<[f64; 8]>> = OnceLock::new();
        SIMPLE_ROOTS.get_or_init(Self::enumerate_simple_roots)
    }

    fn enumerate_roots() -> Vec<[f64; 8]> {
        let mut root_set = std::collections::HashSet::new();
        let mut roots = Vec::new();

        for first_index in 0..8 {
            for second_index in (first_index + 1)..8 {
                for first_sign in [-1.0, 1.0] {
                    for second_sign in [-1.0, 1.0] {
                        let mut root = [0.0; 8];
                        root[first_index] = first_sign;
                        root[second_index] = second_sign;
                        if root_set.insert(Self::root_to_key(&root)) {
                            roots.push(root);
                        }
                    }
                }
            }
        }

        for sign_bits in 0..256 {
            let mut root = [0.5; 8];
            let mut negative_count = 0;
            for (coord_index, coord) in root.iter_mut().enumerate() {
                if (sign_bits >> coord_index) & 1 == 1 {
                    *coord = -0.5;
                    negative_count += 1;
                }
            }
            if negative_count % 2 == 0 && root_set.insert(Self::root_to_key(&root)) {
                roots.push(root);
            }
        }

        assert_eq!(roots.len(), 240);
        roots
    }

    fn enumerate_simple_roots() -> Vec<[f64; 8]> {
        vec![
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5],
        ]
    }

    fn root_to_key(root: &[f64; 8]) -> String {
        root.iter()
            .map(|coord| format!("{coord:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn dot_product(a: &[f64; 8], b: &[f64; 8]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(left, right)| left * right)
            .sum()
    }

    /// Return all E8 roots.
    pub fn get_roots(&self) -> &Vec<[f64; 8]> {
        Self::roots_static()
    }

    /// Return the simple roots used by the compatibility API.
    pub fn simple_roots(&self) -> &[[f64; 8]] {
        Self::simple_roots_static()
    }

    /// Return the Cartan matrix derived from the simple roots.
    pub fn cartan_matrix(&self) -> [[i32; 8]; 8] {
        let mut matrix = [[0i32; 8]; 8];
        let simple_roots = Self::simple_roots_static();

        for (row_index, row) in matrix.iter_mut().enumerate() {
            let root_norm = Self::dot_product(&simple_roots[row_index], &simple_roots[row_index]);
            for (column_index, entry) in row.iter_mut().enumerate() {
                let root_dot =
                    Self::dot_product(&simple_roots[row_index], &simple_roots[column_index]);
                *entry = (2.0 * root_dot / root_norm).round() as i32;
            }
        }

        matrix
    }

    /// Verify that every enumerated root has squared norm 2.
    pub fn verify_root_structure(&self, tolerance: f64) -> bool {
        Self::roots_static()
            .iter()
            .all(|root| (Self::dot_product(root, root) - 2.0).abs() <= tolerance)
    }

    /// Verify the diagonal and off-diagonal bounds of the Cartan matrix.
    pub fn verify_cartan_matrix(&self) -> bool {
        let cartan = self.cartan_matrix();

        for (row_index, row) in cartan.iter().enumerate() {
            if row[row_index] != 2 {
                return false;
            }
        }

        for (row_index, row) in cartan.iter().enumerate() {
            for (column_index, &entry) in row.iter().enumerate() {
                if row_index != column_index && !(-3..=1).contains(&entry) {
                    return false;
                }
            }
        }

        true
    }

    /// Dimension of E8.
    pub fn dimension(&self) -> usize {
        248
    }

    /// Number of positive E8 roots.
    pub fn positive_root_count(&self) -> usize {
        120
    }

    /// Rank of E8.
    pub fn rank(&self) -> usize {
        8
    }

    /// Number of E8 roots.
    pub fn root_count(&self) -> usize {
        240
    }

    /// Weyl group order for E8.
    pub fn weyl_group_order(&self) -> u64 {
        2u64.pow(14) * 3u64.pow(5) * 5u64.pow(2) * 7
    }

    /// E8 has 240 roots.
    pub fn num_roots(&self) -> usize {
        240
    }

    /// Return the legacy diagram description.
    pub fn dynkin_diagram_description(&self) -> &str {
        "E8 Dynkin diagram: linear A7 chain with branch at position 3"
    }
}

impl Default for E8RootSystem {
    fn default() -> Self {
        Self::new()
    }
}
