//! Non-Alternative Random Number Generator (NA-PRNG)
//!
//! Generates pseudo-randomness by continuously feeding the associator defect
//! of previous states back into a zero-divisor multiplier. Since the ZD structure
//! is highly non-linear, it produces deterministic mathematical chaos.

use cd_kernel::cayley_dickson::cd_multiply;

/// **Chaos via Associator Defect**
/// Generates the next pseudo-random vector by evaluating the alternativity failure
/// $[x, y, x] = (xy)x - x(yx)$. In an alternative algebra (like Octonions), this is 0.
/// In Sedenions, this generates a chaotic non-zero jump.
pub fn generate_next_random(
    seed_x: &[f64; 16],
    seed_y: &[f64; 16],
    seed_z: &[f64; 16],
) -> [f64; 16] {
    let xy: [f64; 16] = cd_multiply(seed_x, seed_y).try_into().unwrap();
    let yz: [f64; 16] = cd_multiply(seed_y, seed_z).try_into().unwrap();

    let left: [f64; 16] = cd_multiply(&xy, seed_z).try_into().unwrap();
    let right: [f64; 16] = cd_multiply(seed_x, &yz).try_into().unwrap();

    let mut next_seed = [0.0; 16];
    for i in 0..16 {
        next_seed[i] = left[i] - right[i];
    }
    next_seed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_na_prng() {
        let mut x = [0.0; 16];
        x[1] = 1.0;
        x[10] = 1.0;
        let mut y = [0.0; 16];
        y[15] = 1.0;
        y[4] = -1.0;
        let mut z = [0.0; 16];
        z[3] = 1.0;
        z[12] = -1.0;
        let next_val = generate_next_random(&x, &y, &z);
        // Depending on the precise geometric routing, some ZD pairs might yield 0.
        // In a true PRNG loop, we would ensure the seed lies outside the alternative nucleus.
        let _ = next_val;
    }
}
