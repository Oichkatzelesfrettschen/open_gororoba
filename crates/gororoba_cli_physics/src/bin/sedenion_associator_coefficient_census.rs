//! Exact coefficient census of the 16D Cayley-Dickson basis associator.
//!
//! The staple-associator detector is, after unrolling, the norm of a
//! fixed sparse cubic filter over six consecutive magnetometer samples.
//! Its coefficient tensor comes entirely from the basis associator
//!
//!   [e_i, e_j, e_k] = (e_i e_j) e_k - e_i (e_j e_k)
//!                   = ( s(i,j) s(i^j, k) - s(j,k) s(i, j^k) ) e_{i^j^k}
//!
//! where s(i,j) is the Cayley-Dickson twist sign from CdMultTable and ^
//! is XOR on basis indices. Each coefficient therefore lies in
//! {-2, 0, +2}, and the whole detector is characterized by which of the
//! 16^3 = 4096 ordered triples are nonzero and where they land.
//!
//! This binary enumerates all 4096 triples against the SHA-256-verified
//! multiplication table and prints the exact census: value histogram,
//! per-output-component nonzero counts, and the sign balance. The
//! numbers are a derived property of the committed twist recurrence --
//! any change to the table that alters them fails the regression tests
//! below, which pin the census as a machine-checked fact rather than a
//! narrative claim.
//!
//! Usage:
//!   sedenion-associator-coefficient-census [--dim 16]

use cd_kernel::mult_table::CdMultTable;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Exact basis-associator coefficient census for a CD algebra")]
struct Args {
    /// Algebra dimension (power of two).
    #[arg(long, default_value_t = 16)]
    dim: usize,
}

/// Census of the rank-3 associator coefficient tensor of a CD algebra.
#[derive(Debug, PartialEq, Eq)]
pub struct AssociatorCensus {
    pub dim: usize,
    pub zeros: usize,
    pub plus_two: usize,
    pub minus_two: usize,
    /// Values outside {-2, 0, +2}; the CD twist structure forces this
    /// to stay empty, and the census records rather than assumes it.
    pub other: usize,
    /// Nonzero ordered-triple count landing on each output component.
    pub per_output_nonzero: Vec<usize>,
}

/// Coefficient of [e_i, e_j, e_k] on e_{i^j^k}: the signed difference of
/// the two association orders read off the multiplication table.
fn associator_coefficient(table: &CdMultTable, i: usize, j: usize, k: usize) -> i32 {
    let (s_ij, ij) = table.multiply_basis(i, j);
    let (s_ij_k, ijk_left) = table.multiply_basis(ij, k);
    let (s_jk, jk) = table.multiply_basis(j, k);
    let (s_i_jk, ijk_right) = table.multiply_basis(i, jk);
    assert_eq!(
        ijk_left, ijk_right,
        "CD basis products land on e_(i XOR j XOR k) in both association orders"
    );
    i32::from(s_ij) * i32::from(s_ij_k) - i32::from(s_jk) * i32::from(s_i_jk)
}

pub fn census(dim: usize) -> AssociatorCensus {
    let table = CdMultTable::generate(dim);
    let mut zeros = 0usize;
    let mut plus_two = 0usize;
    let mut minus_two = 0usize;
    let mut other = 0usize;
    let mut per_output_nonzero = vec![0usize; dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                let c = associator_coefficient(&table, i, j, k);
                match c {
                    0 => zeros += 1,
                    2 => {
                        plus_two += 1;
                        per_output_nonzero[i ^ j ^ k] += 1;
                    }
                    -2 => {
                        minus_two += 1;
                        per_output_nonzero[i ^ j ^ k] += 1;
                    }
                    _ => other += 1,
                }
            }
        }
    }
    AssociatorCensus {
        dim,
        zeros,
        plus_two,
        minus_two,
        other,
        per_output_nonzero,
    }
}

fn main() {
    let args = Args::parse();
    assert!(args.dim.is_power_of_two(), "dim is a power of two");
    let c = census(args.dim);
    let total = args.dim * args.dim * args.dim;
    let nonzero = c.plus_two + c.minus_two;
    println!("dim = {}", c.dim);
    println!("ordered triples = {}", total);
    println!("zeros = {}", c.zeros);
    println!("+2 = {}", c.plus_two);
    println!("-2 = {}", c.minus_two);
    println!("outside {{-2,0,+2}} = {}", c.other);
    println!(
        "nonzero support = {} ({:.7}%)",
        nonzero,
        100.0 * nonzero as f64 / total as f64
    );
    println!("per-output-component nonzero counts:");
    for (comp, n) in c.per_output_nonzero.iter().enumerate() {
        println!("  e{:<2} {}", comp, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 16D census is an exact machine-checked fact of the committed
    /// twist recurrence: 2248 zeros, 924 of each sign (1848 nonzero,
    /// 45.1171875% support), signs balanced, nothing outside {-2,0,+2}.
    #[test]
    fn sedenion_census_totals() {
        let c = census(16);
        assert_eq!(c.zeros, 2248);
        assert_eq!(c.plus_two, 924);
        assert_eq!(c.minus_two, 924);
        assert_eq!(c.other, 0);
    }

    /// Output-component structure: the scalar e0 receives nothing (the
    /// associator is purely imaginary), e8 receives 168 ordered
    /// contributions, and every other imaginary component receives 120.
    #[test]
    fn sedenion_census_output_components() {
        let c = census(16);
        assert_eq!(c.per_output_nonzero[0], 0);
        assert_eq!(c.per_output_nonzero[8], 168);
        for comp in 1..16 {
            if comp != 8 {
                assert_eq!(c.per_output_nonzero[comp], 120, "component e{}", comp);
            }
        }
    }

    /// Octonions (dim 8) are alternative but not associative: nonzero
    /// associator support exists, and the quaternion subalgebra (dim 4)
    /// is associative with an identically zero tensor.
    #[test]
    fn lower_dim_census_sanity() {
        let quat = census(4);
        assert_eq!(quat.zeros, 64);
        assert_eq!(quat.plus_two + quat.minus_two, 0);
        let oct = census(8);
        assert!(oct.plus_two > 0);
        assert_eq!(oct.plus_two, oct.minus_two);
        assert_eq!(oct.other, 0);
    }
}
