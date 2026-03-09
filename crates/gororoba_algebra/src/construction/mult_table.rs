//! Cayley-Dickson multiplication table generator.
//!
//! Re-exports from `cd_kernel::mult_table`.

pub use cd_kernel::mult_table::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::m3::OctonionTable;

    #[test]
    fn test_dim8_matches_octonion_table() {
        let table = CdMultTable::generate(8);
        let oct = OctonionTable::new();

        for i in 0..8 {
            for j in 0..8 {
                let (oct_sign, oct_idx) = oct.mul_basis(i, j);
                let (cd_sign, cd_idx) = table.multiply_basis(i, j);
                assert_eq!(
                    (cd_sign as i32, cd_idx),
                    (oct_sign, oct_idx),
                    "Mismatch at e_{} * e_{}: CD=({}, {}), Oct=({}, {})",
                    i,
                    j,
                    cd_sign,
                    cd_idx,
                    oct_sign,
                    oct_idx
                );
            }
        }
    }
}
