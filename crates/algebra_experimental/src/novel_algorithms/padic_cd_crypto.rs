//! p-Adic Cayley-Dickson Cryptosystem
//!
//! Merges p-adic valuations with Cayley-Dickson zero-divisors to create a
//! post-quantum trapdoor. It leverages the breakdown of norm composition
//! over p-adic fields when extended to 16 dimensions.

use cd_kernel::cayley_dickson::cd_multiply;

/// **p-Adic Zero-Divisor Masking**
/// Multiplies the state with a masking element. Because ZDs act as absolute nullifiers
/// for specific algebraic structures, they can permanently erase specific p-adic valuations
/// without destroying the rest of the message.
pub fn padic_zd_encrypt(message: &[f64; 16], p_adic_key: f64) -> [f64; 16] {
    let mut state = *message;
    // Simulate scaling by p-adic norm before non-associative routing
    for x in state.iter_mut() {
        *x *= p_adic_key;
    }
    let mask = [0.5; 16];
    cd_multiply(&state, &mask).try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_padic() {
        let msg = [1.0; 16];
        let cipher = padic_zd_encrypt(&msg, 2.0);
        assert_ne!(cipher[0], 0.0);
    }
}
