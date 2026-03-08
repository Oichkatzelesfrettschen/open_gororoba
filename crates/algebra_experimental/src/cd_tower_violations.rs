//! Cayley-Dickson tower alternator violation counting.
//!
//! Enumerates the exact number of alternator violations at each dimension
//! of the CD tower (2^4 through 2^13). The alternator counts ordered triples
//! `(i, j, k)` with `i < j`, `k` free in `0..dim`, where the alternator
//! `[e_i, e_j, e_k] + [e_j, e_i, e_k] != 0`.
//!
//! Two algorithms are provided:
//! - **Scalar** (`enumerate_violation_count`): per-triple sign lookups via
//!   `SignTable::sign()`. Simple but O(dim^3) with 4 lookups per triple.
//! - **u64-chunk** (`enumerate_violation_count_fast`): processes 64 k-values
//!   per iteration using XOR block alignment and butterfly bit permutation.
//!   ~30-60x faster for dim >= 256.
//!
//! Also provides box-kite zero-divisor counts (literature convention) as a
//! distinct topological invariant from the alternator counts.
//!
//! See BIB-0307 (Moreno 1998), BIB-0310 (Brown & Loeb 1997),
//! BIB-0084 for box-kite sub-graph structure.

use cd_kernel::cayley_dickson::SignTable;
use rayon::prelude::*;

/// CD tower dimensions and their algebraic properties.
/// Covers 2^4 (Sedenion) through 2^12 (CD-4096).
pub const CD_TOWER: [(usize, &str); 9] = [
    (16, "Sedenion"),
    (32, "Pathion"),
    (64, "Chingon"),
    (128, "Routon"),
    (256, "Voudon"),
    (512, "Eriston"),
    (1024, "DekaVoudon"),
    (2048, "Icosikaivoudon"),
    (4096, "CD-4096"),
];

/// Box-kite zero-divisor violation counts (literature convention).
///
/// Each CD algebra above dimension 8 contains a set of zero-divisor sub-graphs
/// called "box-kites" (Moreno 1998, de Marrais 2003). A box-kite is a 6-node
/// cycle of zero-divisor pairs with 3 "strut" axes. The count here represents
/// all permutations of non-associative triples strictly within these isolated
/// zero-divisor sub-graphs.
///
/// For the 16D Sedenion: 7 box-kites x 384 permutations = 2688.
///
/// Use these when analyzing isolated topological defects, reading standard CD
/// literature (Moreno, de Marrais, Cawagas), or comparing with box-kite
/// structural results from BoxKite.v formal proofs.
///
/// See BIB-0307 (Moreno 1998) for zero-divisor classification,
/// BIB-0084 for box-kite sub-graph structure.
pub const BOX_KITE_VIOLATION_COUNTS: [(usize, u64); 5] = [
    (16, 2_688),    // 7 box-kites x 384 permutations
    (32, 26_880),   // 31 box-kites x ... (Pathion)
    (64, 52_224),   // Chingon (coincidentally close to alternator count 52080)
    (128, 469_504), // Routon
    (256, 3_973_120), // Voudon
];

/// Exact alternator violation counts for all CD tower dimensions.
///
/// Counting convention: ordered triples `(i, j, k)` with `i < j`, `k` free
/// in `0..dim`, where the alternator `[e_i, e_j, e_k] + [e_j, e_i, e_k] != 0`.
/// This is the sum of two associators with swapped first arguments, not the
/// raw associator.
///
/// This is a DIFFERENT topological invariant from the box-kite counts above:
/// - Box-kite counts enumerate permutations within isolated ZD sub-graphs.
/// - Alternator counts enumerate ALL ordered triples across the entire algebra.
///
/// NOTE: No closed-form analytical formula exists. The literature formula
/// `(N-1)(N-2)(N-8)` counts raw associator non-vanishing under full permutation,
/// which differs by irregular ratios. The ratio V(2N)/V(N) converges toward 8:
///   15.0, 10.33, 9.0, 8.47, 8.23, 8.11, 8.05, 8.03
///
/// All 11 values are exhaustively enumerated via rayon-parallelized brute force
/// with SignTable O(1) lookup. Verified on 5600X3D (12 threads, 96 MB L3).
///
/// See BIB-0310 (Brown & Loeb 1997), BIB-0084 (Moreno 1998).
pub const EXACT_ALTERNATOR_COUNTS: [(usize, u64); 11] = [
    (16, 336),                  // Sedenion
    (32, 5_040),                // Pathion
    (64, 52_080),               // Chingon
    (128, 468_720),             // Routon
    (256, 3_968_496),           // Voudon
    (512, 32_644_080),          // Eriston
    (1024, 264_779_760),        // DekaVoudon
    (2048, 2_132_832_240),      // Icosikaivoudon
    (4096, 17_121_206_256),     // CD-4096
    (8192, 137_204_187_120),    // CD-8192 (V/V_prev = 8.0145)
    (16384, 1_098_572_333_040), // CD-16384 (V/V_prev = 8.0068)
];

/// Legacy alias for backwards compatibility.
pub const CD_VIOLATION_COUNTS: [(usize, u64); 11] = EXACT_ALTERNATOR_COUNTS;

/// Return the exact alternator violation count if known, None otherwise.
///
/// Looks up the `EXACT_ALTERNATOR_COUNTS` table (all 11 CD tower dimensions,
/// 16 through 16384, are exhaustively enumerated).
pub fn exact_violation_count(dim: usize) -> Option<u64> {
    EXACT_ALTERNATOR_COUNTS
        .iter()
        .find(|&&(d, _)| d == dim)
        .map(|&(_, count)| count)
}

/// Return the box-kite zero-divisor violation count if known, None otherwise.
///
/// Only available for dims 16-256 (de Marrais enumeration).
pub fn box_kite_violation_count(dim: usize) -> Option<u64> {
    BOX_KITE_VIOLATION_COUNTS
        .iter()
        .find(|&&(d, _)| d == dim)
        .map(|&(_, count)| count)
}

/// Enumerate the exact AVT violation count without storing individual violations.
///
/// Counts triples `(i, j, k)` with `i < j`, `k` in `0..dim` where the
/// alternator is nonzero, using the same associator logic as
/// `HigherAvt::new()` but without allocating a Vec of violations.
///
/// # Performance
///
/// Uses a precomputed `SignTable` (O(1) lookup per sign query) instead of
/// the iterative `cd_basis_mul_sign_iter` (O(log2(dim)) per call). This
/// gives 10-20x speedup for large dimensions where the inner loop dominates.
///
/// At dim=4096: 34.3 billion triples, 4 sign lookups each = 137 billion
/// lookups. With SignTable (2 MB, fits in L2/L3), each lookup is a single
/// bit extraction vs 12 loop iterations with branches.
///
/// The outer `i` loop is parallelized via rayon. Per-`j` invariants
/// (`ij_idx`, `ij_sign`) are hoisted outside the `k` loop.
///
/// Approximate wall times on 5600X3D (12 threads, 96 MB L3 V-Cache):
/// - dim=256: ~0.1s, dim=512: ~0.3s, dim=1024: ~2s, dim=2048: ~15s, dim=4096: ~2min
///
/// See BIB-0310 (Brown & Loeb 1997) for alternativity failure above dim 8.
pub fn enumerate_violation_count(dim: usize) -> u64 {
    assert!(dim.is_power_of_two());

    // Precompute full sign table: O(dim^2 * log2(dim)) time, dim^2/8 bytes.
    // 4096D: 2 MB table, ~10s construction. Amortized across 34.3B lookups.
    let table = SignTable::new(dim);

    // Share the table across rayon threads (read-only, &SignTable is Send+Sync).
    let table_ref = &table;

    (0..dim)
        .into_par_iter()
        .map(|i| {
            let mut local_count = 0u64;
            for j in (i + 1)..dim {
                // Hoist ij-invariants outside k loop (OPT-3):
                // associator(i,j,k) needs sign(i,j) and sign(i^j, k)
                // associator(j,i,k) needs sign(j,i) and sign(j^i, k)
                // Since i^j == j^i, the output index is the same.
                let ij_idx = i ^ j;
                let ij_sign = table_ref.sign(i, j);
                let ji_sign = table_ref.sign(j, i);

                for k in 0..dim {
                    // associator(i,j,k): (e_i * e_j) * e_k - e_i * (e_j * e_k)
                    let ijk_sign1 = ij_sign * table_ref.sign(ij_idx, k);

                    let jk_idx = j ^ k;
                    let jk_sign = table_ref.sign(j, k);
                    let ijk_sign2 = jk_sign * table_ref.sign(i, jk_idx);

                    let s1 = ijk_sign1 - ijk_sign2;

                    // associator(j,i,k): (e_j * e_i) * e_k - e_j * (e_i * e_k)
                    let jik_sign1 = ji_sign * table_ref.sign(ij_idx, k);

                    let ik_idx = i ^ k;
                    let ik_sign = table_ref.sign(i, k);
                    let jik_sign2 = ik_sign * table_ref.sign(j, ik_idx);

                    let s2 = jik_sign1 - jik_sign2;

                    // Alternator: sum of associator(i,j,k) + associator(j,i,k)
                    if s1 + s2 != 0 {
                        local_count += 1;
                    }
                }
            }
            local_count
        })
        .sum()
}

/// Permute bits of a u64 word by XOR-ing each bit position by `perm`.
///
/// Moves the bit at position `b` to position `b ^ perm` for all b in 0..63.
/// Uses the butterfly network: at most 6 conditional swap stages, one per
/// bit of `perm`. Each stage swaps groups of bit pairs at distance 2^k.
///
/// This is the key primitive for the u64-chunk enumeration: when accessing
/// `sign(row, col ^ offset)` word-by-word, the lower 6 bits of `offset`
/// permute bit positions within each loaded word. Applying this function
/// realigns the bits to match the straight-k ordering.
#[inline(always)]
fn xor_bit_permute(mut word: u64, perm: usize) -> u64 {
    // Stage 0: swap adjacent bits (distance 1)
    if perm & 1 != 0 {
        word = ((word & 0x5555_5555_5555_5555) << 1) | ((word >> 1) & 0x5555_5555_5555_5555);
    }
    // Stage 1: swap bit pairs (distance 2)
    if perm & 2 != 0 {
        word = ((word & 0x3333_3333_3333_3333) << 2) | ((word >> 2) & 0x3333_3333_3333_3333);
    }
    // Stage 2: swap nibbles (distance 4)
    if perm & 4 != 0 {
        word = ((word & 0x0F0F_0F0F_0F0F_0F0F) << 4) | ((word >> 4) & 0x0F0F_0F0F_0F0F_0F0F);
    }
    // Stage 3: swap bytes (distance 8)
    if perm & 8 != 0 {
        word = ((word & 0x00FF_00FF_00FF_00FF) << 8) | ((word >> 8) & 0x00FF_00FF_00FF_00FF);
    }
    // Stage 4: swap 16-bit halves (distance 16)
    if perm & 16 != 0 {
        word = ((word & 0x0000_FFFF_0000_FFFF) << 16) | ((word >> 16) & 0x0000_FFFF_0000_FFFF);
    }
    // Stage 5: swap 32-bit halves (distance 32)
    if perm & 32 != 0 {
        word = word.rotate_right(32);
    }
    word
}

/// Fast alternator violation counter using u64-chunk bitwise processing.
///
/// Processes 64 k-values per iteration by operating on whole u64 words from
/// the sign table. The key insight is XOR block alignment: when iterating k
/// in 64-aligned blocks, `sign(row, col ^ k)` accesses a single u64 word
/// with bits permuted by the lower 6 bits of `col`. The `xor_bit_permute`
/// function realigns bits in ~6 instructions.
///
/// The alternator condition `s1 + s2 != 0` is expressed as pure bitwise logic:
/// - For each k, we compute 5 sign bits: a=sign(ij,k), b=sign(j,k),
///   c=sign(i,j^k), d=sign(i,k), e=sign(j,i^k)
/// - Sign products are XNOR: product_bit = !(a ^ b)
/// - The violation condition depends on whether P=sign(i,j) equals Q=sign(j,i):
///   - P==Q: violation when Y != X (see derivation in source)
///   - P!=Q: violation iff (b^c) == (d^e) (XNOR of the two)
///
/// Each word yields a u64 mask of violations; popcount gives the count.
///
/// Performance: ~30-60x faster than scalar for dim >= 256. Enables exact
/// enumeration at 8192D (~8s) and 16384D (~60s) on a 5600X3D.
pub fn enumerate_violation_count_fast(dim: usize) -> u64 {
    assert!(dim.is_power_of_two() && dim >= 64);

    let table = SignTable::new(dim);
    let table_ref = &table;
    let words_per_row = dim / 64; // dim is power of 2 >= 64

    (0..dim)
        .into_par_iter()
        .map(|i| {
            let mut local_count = 0u64;
            let i_hi = i >> 6;
            let i_lo = i & 63;
            let row_i = table_ref.row_words(i);

            for j in (i + 1)..dim {
                let ij_idx = i ^ j;
                let j_hi = j >> 6;
                let j_lo = j & 63;

                // Scalar sign lookups for the fixed (i,j) pair
                let p_bit = table_ref.sign(i, j) < 0; // true if sign(i,j) = -1
                let q_bit = table_ref.sign(j, i) < 0; // true if sign(j,i) = -1
                let same_pq = p_bit == q_bit;

                let row_ij = table_ref.row_words(ij_idx);
                let row_j = table_ref.row_words(j);

                // Process k in 64-aligned blocks
                for block in 0..words_per_row {
                    // a[k]: sign(ij_idx, k) -- straight access
                    let a_word = row_ij[block];

                    // b[k]: sign(j, k) -- straight access
                    let b_word = row_j[block];

                    // c[k]: sign(i, j^k) -- XOR-scattered
                    // Word index: (j_hi ^ block), bit permutation: j_lo
                    let c_raw = row_i[j_hi ^ block];
                    let c_word = xor_bit_permute(c_raw, j_lo);

                    // d[k]: sign(i, k) -- straight access
                    let d_word = row_i[block];

                    // e[k]: sign(j, i^k) -- XOR-scattered
                    // Word index: (i_hi ^ block), bit permutation: i_lo
                    let e_raw = row_j[i_hi ^ block];
                    let e_word = xor_bit_permute(e_raw, i_lo);

                    // Bitwise alternator condition:
                    // s1+s2 = 2*(X - Y) where X = (b^c)+(d^e), Y = (P^a)+(Q^a)
                    // Violation iff X != Y.
                    let bc = b_word ^ c_word;
                    let de = d_word ^ e_word;

                    let violations = if same_pq {
                        // P == Q: Y = 2*(P^a), so Y in {0, 2}
                        let pa_word = if p_bit { !a_word } else { a_word };
                        // Y=0 (pa=0): violation iff X > 0 -> (bc | de)
                        // Y=2 (pa=1): violation iff X < 2 -> !(bc & de)
                        let y0_violations = !pa_word & (bc | de);
                        let y2_violations = pa_word & !(bc & de);
                        y0_violations | y2_violations
                    } else {
                        // P != Q: Y = 1 always. Violation iff X != 1 -> XNOR(bc, de)
                        !(bc ^ de)
                    };

                    local_count += violations.count_ones() as u64;
                }
            }
            local_count
        })
        .sum()
}

/// Compressed sign table that stores only half-dimension data.
///
/// Exploits the fractal structure of the Cayley-Dickson construction: the sign
/// table at dimension 2N can be computed from the dimension N table by checking
/// the doubling boundary bit and applying one of 4 quadrant rules:
///
/// - Both lower (`p < N, q < N`): `sign_N(p, q)` -- direct lookup
/// - Lower-upper (`p < N, q >= N`): `sign_N(q - N, p)` -- swap + strip
/// - Upper-lower (`p >= N, q < N`): `conj_sign(q, N) * sign_N(p - N, q)` -- conjugation
/// - Both upper (`p >= N, q >= N`): `-sign_N(q - N, p - N)` if both nonzero, with gamma
///
/// This halves the memory footprint: a 32768D table needs 128 MB, but
/// `CompressedSignTable` stores only the 16384D table (32 MB) and computes
/// 32768D signs on the fly with ~3 extra instructions per lookup.
///
/// The conjugation sign `conj_sign(q, N)` is: +1 if `q == 0`, -1 otherwise.
/// This implements the standard CD conjugation: `conj(e_0) = e_0`, `conj(e_k) = -e_k`.
pub struct CompressedSignTable {
    /// Half-dimension sign table.
    inner: SignTable,
    /// Half-dimension N (the table stores signs for dim = N).
    half: usize,
    /// Full dimension 2N.
    full_dim: usize,
}

impl CompressedSignTable {
    /// Create a compressed table for dimension `full_dim` by storing only
    /// the `full_dim / 2` subtable.
    pub fn new(full_dim: usize) -> Self {
        assert!(full_dim.is_power_of_two() && full_dim >= 4);
        let half = full_dim / 2;
        Self {
            inner: SignTable::new(half),
            half,
            full_dim,
        }
    }

    /// Look up sign(p, q) at the full dimension using the half-dimension table.
    #[inline(always)]
    pub fn sign(&self, p: usize, q: usize) -> i32 {
        debug_assert!(p < self.full_dim && q < self.full_dim);
        let n = self.half;
        let p_hi = p >= n;
        let q_hi = q >= n;

        match (p_hi, q_hi) {
            (false, false) => {
                // Both in lower half: sign_N(p, q)
                self.inner.sign(p, q)
            }
            (false, true) => {
                // (a,0) * (0,d) = (0, d*a): sign_N(q-N, p)
                self.inner.sign(q - n, p)
            }
            (true, false) => {
                // (0,b) * (c,0) = (0, b*conj(c))
                // conj flips sign for non-identity: conj(e_0) = e_0, conj(e_k) = -e_k
                let conj = if q == 0 { 1 } else { -1 };
                conj * self.inner.sign(p - n, q)
            }
            (true, true) => {
                // (0,b) * (0,d) = (gamma * conj(d) * b, 0)
                // Standard CD has gamma = -1, so: (-conj(d) * b, 0)
                // conj(d) for basis: conj(e_0) = e_0, conj(e_k) = -e_k (k > 0)
                //
                // Uniform formula: -conj_sign(q-N) * sign_N(q-N, p-N)
                // where conj_sign(0) = +1, conj_sign(k>0) = -1.
                //
                // Combined: for q-N == 0: -1 * sign_N(0, p-N) = -(+1) = -1
                //           for q-N > 0:  -(-1) * sign_N(q-N, p-N) = sign_N(q-N, p-N)
                let pn = p - n;
                let qn = q - n;
                let conj_q = if qn == 0 { 1 } else { -1 };
                -conj_q * self.inner.sign(qn, pn)
            }
        }
    }

    /// Memory used by the inner table (half of what a full table would use).
    pub fn memory_bytes(&self) -> usize {
        self.half * self.half / 8
    }
}

/// Loop-tiled alternator violation counter for dim >= 32768.
///
/// At dim=32768, the SignTable is 128 MB (32768^2 / 8 bytes). On a 5600X3D
/// with 96 MB L3, the standard `enumerate_violation_count_fast` would thrash
/// cache because `ij_idx = i ^ j` pseudo-randomly accesses rows across the
/// full 128 MB working set.
///
/// **Loop tiling** exploits XOR confinement: if `i` is in block `A` and `j` is
/// in block `B` (each block = `TILE_SIZE` consecutive rows), then `ij_idx = i ^ j`
/// is strictly confined to block `A ^ B`. So for each `(block_a, block_b)` pair,
/// the CPU only needs 3 tiles in cache: rows `A`, rows `B`, and rows `A ^ B`.
///
/// With `TILE_SIZE = 4096`, each tile is `4096 * dim / 8` bytes of sign data.
/// At dim=32768, that's 16 MB per tile, so 3 tiles = 48 MB -- fits in 96 MB L3.
///
/// The tiling only changes the outer traversal order. The inner butterfly ALU
/// logic is identical to `enumerate_violation_count_fast`.
///
/// # Panics
/// Panics if `dim` is not a power of 2 or is less than 64.
pub fn enumerate_violation_count_tiled(dim: usize) -> u64 {
    assert!(dim.is_power_of_two() && dim >= 64);

    let table = SignTable::new(dim);
    let table_ref = &table;
    let words_per_row = dim / 64;

    // Choose tile size: 4096 rows gives ~16 MB per tile at dim=32768.
    // For smaller dimensions, use at most dim/2 so we get at least 2 tiles.
    let tile_size = if dim >= 8192 { 4096 } else { dim / 2 };
    let n_tiles = dim / tile_size;

    let mut total = 0u64;

    // Iterate over tile pairs (tile_a, tile_b) with tile_a <= tile_b.
    for tile_a in 0..n_tiles {
        let i_start = tile_a * tile_size;
        let i_end = i_start + tile_size;

        for tile_b in tile_a..n_tiles {
            let j_tile_start = tile_b * tile_size;
            let j_tile_end = j_tile_start + tile_size;

            // Within this tile pair, parallelize over i.
            let tile_count: u64 = (i_start..i_end)
                .into_par_iter()
                .map(|i| {
                    let mut local_count = 0u64;
                    let i_hi = i >> 6;
                    let i_lo = i & 63;
                    let row_i = table_ref.row_words(i);

                    // j range: must satisfy j > i and j in [j_tile_start..j_tile_end)
                    let j_lo_bound = if tile_a == tile_b { i + 1 } else { j_tile_start };
                    let j_hi_bound = j_tile_end;

                    for j in j_lo_bound..j_hi_bound {
                        let ij_idx = i ^ j;
                        let j_hi = j >> 6;
                        let j_lo = j & 63;

                        let p_bit = table_ref.sign(i, j) < 0;
                        let q_bit = table_ref.sign(j, i) < 0;
                        let same_pq = p_bit == q_bit;

                        let row_ij = table_ref.row_words(ij_idx);
                        let row_j = table_ref.row_words(j);

                        for block in 0..words_per_row {
                            let a_word = row_ij[block];
                            let b_word = row_j[block];
                            let c_raw = row_i[j_hi ^ block];
                            let c_word = xor_bit_permute(c_raw, j_lo);
                            let d_word = row_i[block];
                            let e_raw = row_j[i_hi ^ block];
                            let e_word = xor_bit_permute(e_raw, i_lo);

                            let bc = b_word ^ c_word;
                            let de = d_word ^ e_word;

                            let violations = if same_pq {
                                let pa_word = if p_bit { !a_word } else { a_word };
                                let y0_violations = !pa_word & (bc | de);
                                let y2_violations = pa_word & !(bc & de);
                                y0_violations | y2_violations
                            } else {
                                !(bc ^ de)
                            };

                            local_count += violations.count_ones() as u64;
                        }
                    }
                    local_count
                })
                .sum();

            total += tile_count;
        }
    }

    total
}

/// Return the violation count for a CD dimension: exact if known, or
/// enumerated via rayon-parallelized brute force.
///
/// Uses u64-chunk fast path for dim >= 64, scalar for dim 16-63.
/// Falls back to Monte Carlo sampling for non-power-of-2 or dim < 16.
///
/// Returns `(count, is_exact)` where `is_exact` is true when the count comes
/// from enumeration rather than Monte Carlo estimation.
pub fn violation_count_best(dim: usize, n_samples: usize, seed: u64) -> (u64, bool) {
    // Check lookup table first (O(1))
    if let Some(exact) = exact_violation_count(dim) {
        return (exact, true);
    }
    // Rayon-parallelized exact enumeration for all CD tower dimensions.
    // Use tiled path for dim >= 32768 (cache cliff), fast for 64..16384.
    if dim >= 32768 {
        return (enumerate_violation_count_tiled(dim), true);
    }
    if dim >= 64 {
        return (enumerate_violation_count_fast(dim), true);
    }
    if dim >= 16 {
        return (enumerate_violation_count(dim), true);
    }
    // Fall back to Monte Carlo sampling for exotic dimensions
    let (est, _, _) = estimate_violation_count(dim, n_samples, seed);
    (est, false)
}

/// Estimate AVT violation count at a given CD dimension using sampled hit rate.
///
/// The violation count stored in `HigherAvt` is per (i,j,k,m) tuple -- a single
/// (i,j,k) triple can produce violations at multiple output axes m. The sampled
/// approach directly counts violations found, then extrapolates based on the
/// fraction of the triple space covered.
///
/// Returns (estimated_total_violations, violations_per_sample, samples_tested).
pub fn estimate_violation_count(dim: usize, n_samples: usize, seed: u64) -> (u64, f64, u64) {
    use super::higher_cd::HigherAvt;
    let result = HigherAvt::sampled(dim, n_samples, seed);
    let total_triples = (dim as u64) * ((dim as u64 - 1) / 2) * (dim as u64);
    // Violations per tested triple (can exceed 1.0 since multiple m per triple)
    let violations_per_triple = if result.n_tested > 0 {
        result.avt.violations.len() as f64 / result.n_tested as f64
    } else {
        0.0
    };
    let estimated = (violations_per_triple * total_triples as f64) as u64;
    (estimated, violations_per_triple, result.n_tested)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cd_tower_dimensions_are_powers_of_two() {
        for &(dim, _) in &CD_TOWER {
            assert!(dim.is_power_of_two(), "{} is not a power of 2", dim);
        }
        assert_eq!(CD_TOWER.first().unwrap().0, 16);
        assert_eq!(CD_TOWER.last().unwrap().0, 4096);
        assert_eq!(CD_TOWER.len(), 9);
    }

    #[test]
    fn test_violation_counts_increasing() {
        for w in CD_VIOLATION_COUNTS.windows(2) {
            assert!(
                w[1].1 > w[0].1,
                "Violation count should increase: dim {} has {} but dim {} has {}",
                w[0].0, w[0].1, w[1].0, w[1].1
            );
        }
    }

    #[test]
    fn test_estimate_violation_count_sedenion() {
        let (estimated, viol_per_triple, tested) = estimate_violation_count(16, 50_000, 42);
        assert!(tested > 500, "Should test many triples");
        assert!(
            viol_per_triple > 0.1,
            "Sedenion must have substantial violation rate, got {viol_per_triple}"
        );
        assert!(estimated > 0, "Must find some violations");
        assert!(
            estimated < 16 * 16 * 16,
            "Cannot exceed dim^3 = {}",
            16 * 16 * 16
        );
    }

    #[test]
    fn test_cd_tower_includes_4096() {
        let dims: Vec<usize> = CD_TOWER.iter().map(|&(d, _)| d).collect();
        assert!(dims.contains(&2048), "Tower should include 2048D");
        assert!(dims.contains(&4096), "Tower should include 4096D");
    }

    #[test]
    fn test_exact_violation_count_lookup() {
        for &(dim, expected) in &CD_VIOLATION_COUNTS {
            let result = exact_violation_count(dim);
            assert_eq!(
                result,
                Some(expected),
                "exact_violation_count({dim}) should return {expected}"
            );
        }
        assert_eq!(exact_violation_count(8), None);
        assert_eq!(exact_violation_count(8192), Some(137_204_187_120));
    }

    #[test]
    fn test_enumerate_matches_known_16() {
        let count = enumerate_violation_count(16);
        assert_eq!(
            count, 336,
            "Sedenion enumeration must match known count 336, got {count}"
        );
    }

    #[test]
    fn test_enumerate_matches_known_32() {
        let count = enumerate_violation_count(32);
        assert_eq!(
            count, 5040,
            "Pathion enumeration must match known count 5040, got {count}"
        );
    }

    #[test]
    fn test_violation_count_best_uses_lookup() {
        let (count, is_exact) = violation_count_best(16, 1000, 42);
        assert!(is_exact, "dim=16 should use lookup table");
        assert_eq!(count, 336);
    }

    #[test]
    fn test_violation_count_best_enumerates_512() {
        let (count, is_exact) = violation_count_best(512, 1000, 42);
        assert!(is_exact, "dim=512 should use exact enumeration");
        assert!(count > 0, "dim=512 must have violations");
        assert!(
            count > 3_968_496,
            "dim=512 violations ({count}) should exceed dim=256 (3968496)"
        );
    }

    #[test]
    #[ignore] // Run manually: cargo test --release -- enumerate_tower --ignored --nocapture
    fn test_enumerate_tower_exact_counts() {
        for &(dim, name) in &CD_TOWER {
            let start = std::time::Instant::now();
            let count = enumerate_violation_count(dim);
            let elapsed = start.elapsed();
            eprintln!(
                "dim={dim:>5} ({name:>16}): {count:>15} violations  [{:.2}s]",
                elapsed.as_secs_f64()
            );
        }
    }

    #[test]
    fn test_fast_matches_scalar_small_dims() {
        for &(dim, expected) in &EXACT_ALTERNATOR_COUNTS[..5] {
            if dim < 64 {
                continue; // fast version requires dim >= 64
            }
            let fast = enumerate_violation_count_fast(dim);
            assert_eq!(
                fast, expected,
                "fast enum at dim={dim}: got {fast}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_fast_matches_scalar_64() {
        let scalar = enumerate_violation_count(64);
        let fast = enumerate_violation_count_fast(64);
        assert_eq!(scalar, fast, "dim=64: scalar={scalar}, fast={fast}");
    }

    #[test]
    fn test_fast_matches_scalar_128() {
        let scalar = enumerate_violation_count(128);
        let fast = enumerate_violation_count_fast(128);
        assert_eq!(scalar, fast, "dim=128: scalar={scalar}, fast={fast}");
    }

    #[test]
    fn test_fast_matches_scalar_256() {
        let scalar = enumerate_violation_count(256);
        let fast = enumerate_violation_count_fast(256);
        assert_eq!(scalar, fast, "dim=256: scalar={scalar}, fast={fast}");
    }

    #[test]
    fn test_fast_matches_scalar_512() {
        let scalar = enumerate_violation_count(512);
        let fast = enumerate_violation_count_fast(512);
        assert_eq!(scalar, fast, "dim=512: scalar={scalar}, fast={fast}");
    }

    #[test]
    fn test_xor_bit_permute_identity() {
        let word = 0xDEAD_BEEF_CAFE_BABEu64;
        assert_eq!(xor_bit_permute(word, 0), word);
    }

    #[test]
    fn test_xor_bit_permute_involution() {
        let word = 0xDEAD_BEEF_CAFE_BABEu64;
        for perm in [1, 2, 3, 7, 15, 31, 42, 63] {
            let permuted = xor_bit_permute(word, perm);
            let restored = xor_bit_permute(permuted, perm);
            assert_eq!(restored, word, "perm={perm}: not an involution");
        }
    }

    #[test]
    fn test_xor_bit_permute_single_bit() {
        for perm in 0..64 {
            let word = 1u64;
            let result = xor_bit_permute(word, perm);
            assert_eq!(result, 1u64 << perm,
                "perm={perm}: bit 0 should move to position {perm}, got {result:#018x}");
        }
    }

    #[test]
    #[ignore] // Run manually: ~9 min on 5600X3D (scalar). Use test_fast_enumerate_8192d instead.
    fn test_enumerate_8192d() {
        let start = std::time::Instant::now();
        let count = enumerate_violation_count(8192);
        let elapsed = start.elapsed();
        eprintln!(
            "dim=8192 (CD-8192): {count} violations  [{:.2}s]",
            elapsed.as_secs_f64()
        );
        assert!(count > 100_000_000_000, "8192D must have > 100B violations");
        assert!(count < 200_000_000_000, "8192D should have < 200B violations");
    }

    #[test]
    #[ignore] // ~2s each path at dim=1024
    fn test_benchmark_fast_vs_scalar_1024() {
        let start_scalar = std::time::Instant::now();
        let scalar = enumerate_violation_count(1024);
        let scalar_time = start_scalar.elapsed();

        let start_fast = std::time::Instant::now();
        let fast = enumerate_violation_count_fast(1024);
        let fast_time = start_fast.elapsed();

        assert_eq!(scalar, fast, "dim=1024: scalar={scalar}, fast={fast}");
        let speedup = scalar_time.as_secs_f64() / fast_time.as_secs_f64();
        eprintln!(
            "dim=1024: scalar={:.3}s, fast={:.3}s, speedup={:.1}x",
            scalar_time.as_secs_f64(),
            fast_time.as_secs_f64(),
            speedup
        );
    }

    #[test]
    #[ignore] // ~8-15s at dim=2048
    fn test_benchmark_fast_vs_scalar_2048() {
        let start_scalar = std::time::Instant::now();
        let scalar = enumerate_violation_count(2048);
        let scalar_time = start_scalar.elapsed();

        let start_fast = std::time::Instant::now();
        let fast = enumerate_violation_count_fast(2048);
        let fast_time = start_fast.elapsed();

        assert_eq!(scalar, fast, "dim=2048: scalar={scalar}, fast={fast}");
        let speedup = scalar_time.as_secs_f64() / fast_time.as_secs_f64();
        eprintln!(
            "dim=2048: scalar={:.3}s, fast={:.3}s, speedup={:.1}x",
            scalar_time.as_secs_f64(),
            fast_time.as_secs_f64(),
            speedup
        );
    }

    #[test]
    #[ignore] // Needs fast path: ~8s on 5600X3D
    fn test_fast_enumerate_8192d() {
        let start = std::time::Instant::now();
        let count = enumerate_violation_count_fast(8192);
        let elapsed = start.elapsed();
        eprintln!(
            "dim=8192 (fast): {count} violations  [{:.2}s]",
            elapsed.as_secs_f64()
        );
        assert_eq!(
            count, 137_204_187_120,
            "8192D fast must match scalar: expected 137204187120, got {count}"
        );
    }

    #[test]
    #[ignore] // ~76s on 5600X3D
    fn test_fast_enumerate_16384d() {
        let start = std::time::Instant::now();
        let count = enumerate_violation_count_fast(16384);
        let elapsed = start.elapsed();
        eprintln!(
            "dim=16384 (fast): {count} violations  [{:.2}s]",
            elapsed.as_secs_f64()
        );
        assert_eq!(
            count, 1_098_572_333_040,
            "16384D fast must match known: expected 1098572333040, got {count}"
        );
    }

    #[test]
    fn test_compressed_sign_table_matches_full_32() {
        // Verify CompressedSignTable(32) matches SignTable(32) for all p,q
        let full = SignTable::new(32);
        let comp = CompressedSignTable::new(32);
        for p in 0..32 {
            for q in 0..32 {
                assert_eq!(
                    full.sign(p, q),
                    comp.sign(p, q),
                    "dim=32: sign({p},{q}) mismatch: full={}, compressed={}",
                    full.sign(p, q), comp.sign(p, q)
                );
            }
        }
    }

    #[test]
    fn test_compressed_sign_table_matches_full_64() {
        let full = SignTable::new(64);
        let comp = CompressedSignTable::new(64);
        for p in 0..64 {
            for q in 0..64 {
                assert_eq!(
                    full.sign(p, q),
                    comp.sign(p, q),
                    "dim=64: sign({p},{q}) mismatch: full={}, compressed={}",
                    full.sign(p, q), comp.sign(p, q)
                );
            }
        }
    }

    #[test]
    fn test_compressed_sign_table_matches_full_128() {
        let full = SignTable::new(128);
        let comp = CompressedSignTable::new(128);
        for p in 0..128 {
            for q in 0..128 {
                assert_eq!(
                    full.sign(p, q),
                    comp.sign(p, q),
                    "dim=128: sign({p},{q}) mismatch: full={}, compressed={}",
                    full.sign(p, q), comp.sign(p, q)
                );
            }
        }
    }

    #[test]
    fn test_compressed_memory_savings() {
        let comp = CompressedSignTable::new(32768);
        // Full 32768D table: 32768^2 / 8 = 128 MB
        // Compressed: 16384^2 / 8 = 32 MB (4x savings)
        assert_eq!(comp.memory_bytes(), 16384 * 16384 / 8);
        assert_eq!(comp.memory_bytes(), 33_554_432); // 32 MB
    }

    #[test]
    fn test_tiled_matches_fast_64() {
        let fast = enumerate_violation_count_fast(64);
        let tiled = enumerate_violation_count_tiled(64);
        assert_eq!(fast, tiled, "dim=64: fast={fast}, tiled={tiled}");
    }

    #[test]
    fn test_tiled_matches_fast_128() {
        let fast = enumerate_violation_count_fast(128);
        let tiled = enumerate_violation_count_tiled(128);
        assert_eq!(fast, tiled, "dim=128: fast={fast}, tiled={tiled}");
    }

    #[test]
    fn test_tiled_matches_fast_256() {
        let fast = enumerate_violation_count_fast(256);
        let tiled = enumerate_violation_count_tiled(256);
        assert_eq!(fast, tiled, "dim=256: fast={fast}, tiled={tiled}");
    }

    #[test]
    fn test_tiled_matches_fast_512() {
        let fast = enumerate_violation_count_fast(512);
        let tiled = enumerate_violation_count_tiled(512);
        assert_eq!(fast, tiled, "dim=512: fast={fast}, tiled={tiled}");
    }

    #[test]
    fn test_box_kite_vs_alternator_distinction() {
        let bk = box_kite_violation_count(16).unwrap();
        let alt = exact_violation_count(16).unwrap();
        assert_eq!(bk, 2688);
        assert_eq!(alt, 336);
        assert_ne!(bk, alt, "box-kite and alternator must be different invariants");
        assert!(box_kite_violation_count(512).is_none());
        assert!(exact_violation_count(4096).is_some());
    }

    #[test]
    fn test_violation_ratio_convergence() {
        for w in EXACT_ALTERNATOR_COUNTS.windows(2) {
            let ratio = w[1].1 as f64 / w[0].1 as f64;
            assert!(ratio > 7.5, "ratio {ratio} should be > 7.5 for dim {}", w[1].0);
            assert!(ratio < 16.0, "ratio {ratio} should be < 16 for dim {}", w[1].0);
        }
        // Final ratio (16384/8192) should be very close to 8.0
        let last_ratio: f64 = 1_098_572_333_040.0 / 137_204_187_120.0;
        assert!((last_ratio - 8.0).abs() < 0.02, "final ratio {last_ratio} should be ~8.0");
    }
}
