#[cfg(test)]
mod tests {
    use crate::{albert_bridge::embed_to_albert, state::TwoQubitState};
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn test_albert_embedding_and_singh_attractor() {
        // Initial state: Triplet mixture
        let t = Matrix3::<f64>::identity().scale(1.0 / 3.0);
        let a = Vector3::zeros();
        let b = Vector3::zeros();
        let state = TwoQubitState::from_ab_t(&a, &b, &t);

        let el = embed_to_albert(&state, &a, &b, &t);
        let d2 = el.delta_squared();

        println!("Triplet state delta^2: {}", d2);

        // Let's search for a state that hits 3/8
        // Singh's delta^2 = 3/8 result is for trace-free elements with unit octonions.
        // Our embedding uses 1/3 diagonal (Tr=1).
        // Let's check the trace-free version.
        let mut el_tf = el.clone();
        el_tf.diag = [1.0, 0.0, -1.0]; // Tr=0

        let d2_tf = el_tf.delta_squared();
        println!("Trace-free variant delta^2: {}", d2_tf);

        // 3/8 = 0.375
        // If we scale the off-diagonals (T components), can we hit 0.375?
    }
}
