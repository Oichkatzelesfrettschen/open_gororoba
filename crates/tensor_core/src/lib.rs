pub mod tt_cross;
pub mod tt_train;

pub use tt_cross::{build_rank1_symmetry_adapted, build_rank2_symmetry_adapted};
pub use tt_train::{TTCore, TTTrain};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank1_tt_recovers_separable_function() {
        // f(i,j) = (i+1) * (j+1), a rank-1 separable function
        let d = 2;
        let n = 4;
        let pivot = vec![0, 0]; // pivot at (0,0), f=1
        let tt = build_rank1_symmetry_adapted(d, n, &pivot, |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 1.0)
        });
        // Rank-1 should recover exactly for a separable function
        let val = tt.get(&[2, 3]); // (3)*(4) = 12
        assert!((val - 12.0).abs() < 1e-10, "expected 12.0, got {val}");
    }
}
