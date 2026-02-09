//! Integer-exact M3 trilinear operation on Cayley-Dickson algebras.
//!
//! The M3 operation is defined as:
//!   M3(x,y,z) = p(h(xy)z) - p(xh(yz))
//! where p: S -> O is the projection (a,b) -> (a+b)/2 and h: S -> S is
//! the halving map (a,b) -> ((a-b)/2, -(a-b)/2), with S = O + O (sedenions)
//! viewed as pairs of octonions.
//!
//! When applied to octonion basis embeddings e_i -> (e_i, e_i) in S, the
//! result is an integer-valued vector in the octonion basis. This allows
//! exact classification into zero, scalar, or vector types.
//!
//! # Key Results
//! - Repeated indices: M3(i,i,k) = M3(i,j,i) = M3(i,j,j) = 0
//! - Fano line triples: M3 = +/-2 * e_0 (scalar)
//! - All other distinct triples: M3 = +/-2 * e_m (vector)
//!
//! # Literature
//! - de Marrais, Keenan: CD transfer and trilinear operations

const O_DIM: usize = 8;
const S_DIM: usize = 16;

/// Integer-only octonion multiplication table on the standard basis {e0=1, e1..e7}.
///
/// Encodes the Fano-plane structure via 7 oriented triples.
pub struct OctonionTable {
    table: [[i32; O_DIM]; O_DIM],
    sign: [[i32; O_DIM]; O_DIM],
    oriented_triples: [(usize, usize, usize); 7],
}

impl OctonionTable {
    /// Build the standard octonion multiplication table.
    pub fn new() -> Self {
        let mut table = [[0usize; O_DIM]; O_DIM];
        let mut sign = [[0i32; O_DIM]; O_DIM];

        // e_0 * e_i = e_i, e_i * e_0 = e_i
        for i in 0..O_DIM {
            table[0][i] = i;
            sign[0][i] = 1;
            table[i][0] = i;
            sign[i][0] = 1;
        }

        // e_i * e_i = -e_0  for i >= 1
        for i in 1..O_DIM {
            table[i][i] = 0;
            sign[i][i] = -1;
        }

        // Standard Fano-plane oriented triples
        let oriented: [(usize, usize, usize); 7] = [
            (1, 2, 3),
            (1, 4, 5),
            (1, 7, 6),
            (2, 4, 6),
            (2, 5, 7),
            (3, 4, 7),
            (3, 6, 5),
        ];

        for &(i, j, k) in &oriented {
            table[i][j] = k;
            sign[i][j] = 1;
            table[j][i] = k;
            sign[j][i] = -1;

            table[j][k] = i;
            sign[j][k] = 1;
            table[k][j] = i;
            sign[k][j] = -1;

            table[k][i] = j;
            sign[k][i] = 1;
            table[i][k] = j;
            sign[i][k] = -1;
        }

        // Convert table to i32 for consistent storage
        let mut t32 = [[0i32; O_DIM]; O_DIM];
        for i in 0..O_DIM {
            for j in 0..O_DIM {
                t32[i][j] = table[i][j] as i32;
            }
        }

        OctonionTable {
            table: t32,
            sign,
            oriented_triples: oriented,
        }
    }

    /// e_i * e_j = sign * e_{result_index}
    pub fn mul_basis(&self, i: usize, j: usize) -> (i32, usize) {
        (self.sign[i][j], self.table[i][j] as usize)
    }

    /// The 7 Fano-plane lines as unordered triples of indices.
    pub fn fano_lines(&self) -> Vec<[usize; 3]> {
        self.oriented_triples
            .iter()
            .map(|&(a, b, c)| {
                let mut t = [a, b, c];
                t.sort_unstable();
                t
            })
            .collect()
    }

    /// The 7 oriented triples encoding the Fano plane.
    pub fn oriented_triples(&self) -> &[(usize, usize, usize); 7] {
        &self.oriented_triples
    }
}

impl Default for OctonionTable {
    fn default() -> Self {
        Self::new()
    }
}

type OVec = [i32; O_DIM];
type SVec = [i32; S_DIM];

fn o_conj(v: &OVec) -> OVec {
    let mut out = *v;
    for x in out.iter_mut().skip(1) {
        *x = -*x;
    }
    out
}

fn o_mul(oct: &OctonionTable, a: &OVec, b: &OVec) -> OVec {
    let mut out = [0i32; O_DIM];
    for (ia, &va) in a.iter().enumerate() {
        if va == 0 {
            continue;
        }
        for (ib, &vb) in b.iter().enumerate() {
            if vb == 0 {
                continue;
            }
            let (s, k) = oct.mul_basis(ia, ib);
            out[k] += va * vb * s;
        }
    }
    out
}

fn s_mul(oct: &OctonionTable, x: &SVec, y: &SVec) -> SVec {
    // Cayley-Dickson: (a,b)*(c,d) = (ac - conj(d)b, da + b*conj(c))
    let a: OVec = x[..O_DIM].try_into().unwrap();
    let b: OVec = x[O_DIM..].try_into().unwrap();
    let c: OVec = y[..O_DIM].try_into().unwrap();
    let d: OVec = y[O_DIM..].try_into().unwrap();

    let ac = o_mul(oct, &a, &c);
    let d_conj_b = o_mul(oct, &o_conj(&d), &b);
    let mut re = [0i32; O_DIM];
    for i in 0..O_DIM {
        re[i] = ac[i] - d_conj_b[i];
    }

    let da = o_mul(oct, &d, &a);
    let b_conj_c = o_mul(oct, &b, &o_conj(&c));
    let mut im = [0i32; O_DIM];
    for i in 0..O_DIM {
        im[i] = da[i] + b_conj_c[i];
    }

    let mut out = [0i32; S_DIM];
    out[..O_DIM].copy_from_slice(&re);
    out[O_DIM..].copy_from_slice(&im);
    out
}

fn p_map(s: &SVec) -> OVec {
    let mut out = [0i32; O_DIM];
    for i in 0..O_DIM {
        let sum = s[i] + s[i + O_DIM];
        debug_assert!(
            sum % 2 == 0,
            "p-map requires even coordinates, got odd at index {i}"
        );
        out[i] = sum / 2;
    }
    out
}

fn h_map(s: &SVec) -> SVec {
    let mut out = [0i32; S_DIM];
    for i in 0..O_DIM {
        let diff = s[i] - s[i + O_DIM];
        debug_assert!(
            diff % 2 == 0,
            "h-map requires even coordinates, got odd at index {i}"
        );
        let half = diff / 2;
        out[i] = half;
        out[i + O_DIM] = -half;
    }
    out
}

/// Compute the M3 trilinear operation on octonion basis elements.
///
/// Given basis indices i, j, k in 0..7 (typically 1..7 for imaginary units),
/// computes M3(e_i, e_j, e_k) using integer-only arithmetic.
///
/// Returns an 8-element integer vector in the octonion basis.
pub fn compute_m3_octonion_basis(i: usize, j: usize, k: usize, oct: &OctonionTable) -> OVec {
    assert!(i < O_DIM && j < O_DIM && k < O_DIM);

    // Embed octonion basis into sedenion: e_m -> (e_m, e_m)
    let to_s_vec = |idx: usize| -> SVec {
        let mut v = [0i32; S_DIM];
        v[idx] = 1;
        v[idx + O_DIM] = 1;
        v
    };

    let x = to_s_vec(i);
    let y = to_s_vec(j);
    let z = to_s_vec(k);

    // term1 = p(h(xy) * z)
    let xy = s_mul(oct, &x, &y);
    let term1_s = s_mul(oct, &h_map(&xy), &z);
    let term1 = p_map(&term1_s);

    // term2 = p(x * h(yz))
    let yz = s_mul(oct, &y, &z);
    let term2_s = s_mul(oct, &x, &h_map(&yz));
    let term2 = p_map(&term2_s);

    let mut result = [0i32; O_DIM];
    for idx in 0..O_DIM {
        result[idx] = term1[idx] - term2[idx];
    }
    result
}

/// Classification of an M3 output vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum M3Classification {
    /// All components zero (repeated indices).
    Zero,
    /// Only the e_0 (scalar) component is nonzero.
    Scalar { value: i32 },
    /// Exactly one imaginary component e_m (m >= 1) is nonzero.
    Vector { index: usize, value: i32 },
    /// Multiple nonzero components (should not occur for basis inputs).
    Mixed,
}

/// Classify an M3 output vector.
pub fn classify_m3(o_vec: &OVec) -> M3Classification {
    if o_vec.iter().all(|&v| v == 0) {
        return M3Classification::Zero;
    }

    if o_vec[0] != 0 && o_vec[1..].iter().all(|&v| v == 0) {
        return M3Classification::Scalar { value: o_vec[0] };
    }

    if o_vec[0] == 0 {
        let nonzero: Vec<(usize, i32)> = o_vec
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, &v)| v != 0)
            .map(|(idx, &v)| (idx, v))
            .collect();
        if nonzero.len() == 1 {
            let (idx, val) = nonzero[0];
            return M3Classification::Vector {
                index: idx,
                value: val,
            };
        }
    }

    M3Classification::Mixed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m3_distinct_triples_split_42_scalar_168_vector() {
        let oct = OctonionTable::new();
        let mut zero = 0;
        let mut scalar = 0;
        let mut vector = 0;
        let mut mixed = 0;

        for i in 1..8 {
            for j in 1..8 {
                for k in 1..8 {
                    let unique: std::collections::HashSet<usize> = [i, j, k].into();
                    if unique.len() < 3 {
                        continue;
                    }
                    let res = compute_m3_octonion_basis(i, j, k, &oct);
                    match classify_m3(&res) {
                        M3Classification::Zero => zero += 1,
                        M3Classification::Scalar { .. } => scalar += 1,
                        M3Classification::Vector { .. } => vector += 1,
                        M3Classification::Mixed => mixed += 1,
                    }
                }
            }
        }

        assert_eq!(zero, 0, "No distinct triples should yield zero");
        assert_eq!(mixed, 0, "No distinct triples should yield mixed");
        assert_eq!(scalar, 42, "42 Fano-line permutations should be scalar");
        assert_eq!(vector, 168, "168 non-Fano triples should be vector");
        assert_eq!(scalar + vector, 7 * 6 * 5, "Total distinct triples = 210");
    }

    #[test]
    fn test_m3_scalar_outputs_are_exactly_fano_lines() {
        let oct = OctonionTable::new();
        let lines = oct.fano_lines();

        for i in 1..8 {
            for j in 1..8 {
                for k in 1..8 {
                    let unique: std::collections::HashSet<usize> = [i, j, k].into();
                    if unique.len() < 3 {
                        continue;
                    }
                    let res = compute_m3_octonion_basis(i, j, k, &oct);
                    let cls = classify_m3(&res);

                    let mut sorted = [i, j, k];
                    sorted.sort_unstable();
                    let is_fano = lines.contains(&sorted);

                    if is_fano {
                        match cls {
                            M3Classification::Scalar { value } => {
                                assert_eq!(value.unsigned_abs(), 2);
                            }
                            _ => panic!("Fano triple ({i},{j},{k}) should be scalar, got {cls:?}"),
                        }
                    } else {
                        match cls {
                            M3Classification::Vector { index, value } => {
                                assert!((1..=7).contains(&index));
                                assert_eq!(value.unsigned_abs(), 2);
                            }
                            _ => panic!("Non-Fano triple ({i},{j},{k}) should be vector, got {cls:?}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_m3_scalar_sign_flips_with_permutation_parity() {
        let oct = OctonionTable::new();

        for &(a, b, c) in oct.oriented_triples() {
            let base = compute_m3_octonion_basis(a, b, c, &oct);
            let base_val = match classify_m3(&base) {
                M3Classification::Scalar { value } => value,
                other => panic!("Expected scalar for ({a},{b},{c}), got {other:?}"),
            };
            assert_eq!(base_val.unsigned_abs(), 2);

            // All 6 permutations of (a, b, c)
            let perms = [
                (a, b, c, 1),
                (a, c, b, -1),
                (b, a, c, -1),
                (b, c, a, 1),
                (c, a, b, 1),
                (c, b, a, -1),
            ];

            for (pi, pj, pk, parity) in perms {
                let res = compute_m3_octonion_basis(pi, pj, pk, &oct);
                let val = match classify_m3(&res) {
                    M3Classification::Scalar { value } => value,
                    other => panic!("Expected scalar for ({pi},{pj},{pk}), got {other:?}"),
                };
                assert_eq!(
                    val,
                    base_val * parity,
                    "Parity mismatch for permutation ({pi},{pj},{pk}) of ({a},{b},{c})"
                );
            }
        }
    }

    #[test]
    fn test_m3_repeated_indices_are_zero() {
        let oct = OctonionTable::new();
        for i in 1..8 {
            for j in 1..8 {
                for k in 1..8 {
                    let unique: std::collections::HashSet<usize> = [i, j, k].into();
                    if unique.len() == 3 {
                        continue;
                    }
                    let res = compute_m3_octonion_basis(i, j, k, &oct);
                    assert_eq!(
                        classify_m3(&res),
                        M3Classification::Zero,
                        "Repeated indices ({i},{j},{k}) should give zero"
                    );
                }
            }
        }
    }
}
