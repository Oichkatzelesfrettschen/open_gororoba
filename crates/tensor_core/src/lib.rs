pub mod tt_cross;
pub mod tt_train;

pub use tt_cross::{build_rank1_symmetry_adapted, build_rank2_symmetry_adapted};
pub use tt_train::{TTCore, TTTrain};

#[cfg(test)]
pub(crate) fn assert_uniform_integration_matches_dense_sum_for_rank1() {
    let d = 3;
    let n = 5;
    let pivot = vec![2, 2, 2];
    let tt = build_rank1_symmetry_adapted(d, n, &pivot, |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 2.0) * (idx[2] as f64 + 3.0)
    });
    let weight = 0.25_f64;

    let mut dense = 0.0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                dense += tt.get(&[i, j, k]) * weight.powi(3);
            }
        }
    }

    let integrated = tt.integrate_uniform(weight);
    assert!(
        (integrated - dense).abs() < 1.0e-10,
        "integrated={integrated} dense={dense}"
    );
}

#[cfg(test)]
#[test]
fn uniform_integration_matches_dense_sum_for_rank1() {
    assert_uniform_integration_matches_dense_sum_for_rank1();
}

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

    #[test]
    fn uniform_integration_matches_dense_sum_for_rank1() {
        super::assert_uniform_integration_matches_dense_sum_for_rank1();
    }
}
