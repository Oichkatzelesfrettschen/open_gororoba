//! Value-level bridge from structurable operators into the flavor-lift seam.
//!
//! The key contract is architectural: the middle tier feeds `FlavorLift`
//! through a value-level vector rather than a materialized dense operator.

use crate::lift::FlavorLift;
use gororoba_structurable::{StructurableElement, structurable_v_operator};

pub fn apply_structurable_bridge(
    mass_matrix: &mut faer::Mat<f64>,
    x: &StructurableElement,
    y: &StructurableElement,
    z: &StructurableElement,
    flavor_lift: &dyn FlavorLift,
) {
    let lifted = structurable_v_operator(x, y, z);
    flavor_lift.lift(lifted.coords(), mass_matrix);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumLift;

    impl FlavorLift for SumLift {
        fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
            m.write(0, 0, m.read(0, 0) + v.iter().sum::<f64>());
        }
    }

    #[test]
    fn test_structurable_bridge_passes_value_level_output_to_lift() {
        let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
        let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
        let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

        let expected_sum: f64 = structurable_v_operator(&x, &y, &z).coords().iter().sum();

        let mut mass_matrix = faer::Mat::<f64>::zeros(3, 3);
        apply_structurable_bridge(&mut mass_matrix, &x, &y, &z, &SumLift);

        assert!((mass_matrix.read(0, 0) - expected_sum).abs() < 1e-12);
    }
}
