#[cfg(test)]
use crate::diagonalizer::PathionDiagonalizer;
#[cfg(test)]
use crate::carlson::PathionCarlson;
#[cfg(test)]
use approx::assert_relative_eq;

#[cfg(test)]
#[test]
fn test_diagonalizer_roundtrip() {
    let diag = PathionDiagonalizer::new();
    let mut original = [0.0; 32];
    for (i, o) in original.iter_mut().enumerate() {
        *o = i as f64 * 0.1;
    }
        
    let projected = diag.project(&original);
    let recomposed = diag.recompose(&projected);

    for (o, r) in original.iter().zip(recomposed.iter()) {
        assert_relative_eq!(o, r, epsilon = 1e-10);
    }
}

#[cfg(test)]
#[test]
fn test_rf_constant_field() {
    let pc = PathionCarlson::new();

    let expected_real = 1.3110287771460599;

    let diag = PathionDiagonalizer::new();
    let mut x_p = [0.0; 32];
    let mut y_p = [0.0; 32];
    let mut z_p = [0.0; 32];

    for &(r, _i) in &diag.planes {
        x_p[r] = 1.0;
        y_p[r] = 2.0;
        z_p[r] = 0.0;
    }

    let res_p = pc.rf_32d(&x_p, &y_p, &z_p).unwrap();

    for &(r, im) in &diag.planes {
        assert_relative_eq!(res_p[r], expected_real, epsilon = 1e-6);
        assert_relative_eq!(res_p[im], 0.0, epsilon = 1e-6);
    }
}
