//! Value-level bridge from structurable operators into the flavor-lift seam.
//!
//! The key contract is architectural: the middle tier feeds `FlavorLift`
//! through a value-level vector rather than a materialized dense operator.

use crate::lift::FlavorLift;
use gororoba_structurable::{
    StructurableElement, StructurableVOperatorReport, structurable_v_operator,
    structurable_v_operator_report,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A machine-stable snapshot of one bridge evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructurableBridgeSnapshot {
    pub operator_report: StructurableVOperatorReport,
    pub mass_matrix: [[f64; 3]; 3],
}

impl StructurableBridgeSnapshot {
    pub fn flattened_mass_matrix(&self) -> [f64; 9] {
        [
            self.mass_matrix[0][0],
            self.mass_matrix[0][1],
            self.mass_matrix[0][2],
            self.mass_matrix[1][0],
            self.mass_matrix[1][1],
            self.mass_matrix[1][2],
            self.mass_matrix[2][0],
            self.mass_matrix[2][1],
            self.mass_matrix[2][2],
        ]
    }

    pub fn summary_lines(&self) -> [String; 2] {
        [
            self.operator_report.summary_line(),
            format!(
                "mass-diagonal=[{:.6}, {:.6}, {:.6}]",
                self.mass_matrix[0][0], self.mass_matrix[1][1], self.mass_matrix[2][2]
            ),
        ]
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl fmt::Display for StructurableBridgeSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lines = self.summary_lines();
        write!(f, "{}\n{}", lines[0], lines[1])
    }
}

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

/// Run the bridge against a zero 3x3 seed and retain both the operator report
/// and the resulting mass matrix in a simple export-friendly form.
pub fn sample_structurable_bridge(
    x: &StructurableElement,
    y: &StructurableElement,
    z: &StructurableElement,
    flavor_lift: &dyn FlavorLift,
) -> StructurableBridgeSnapshot {
    let operator_report = structurable_v_operator_report(x, y, z);
    let mut mass_matrix = faer::Mat::<f64>::zeros(3, 3);
    flavor_lift.lift(&operator_report.output_coords, &mut mass_matrix);

    let mut matrix = [[0.0; 3]; 3];
    for (row_idx, row) in matrix.iter_mut().enumerate() {
        for (col_idx, value) in row.iter_mut().enumerate() {
            *value = mass_matrix[(row_idx, col_idx)];
        }
    }

    StructurableBridgeSnapshot {
        operator_report,
        mass_matrix: matrix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumLift;

    impl FlavorLift for SumLift {
        fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
            m[(0, 0)] += v.iter().sum::<f64>();
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

        assert!((mass_matrix[(0, 0)] - expected_sum).abs() < 1e-12);
    }

    #[test]
    fn test_structurable_bridge_snapshot_is_machine_readable() {
        let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
        let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
        let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

        let snapshot = sample_structurable_bridge(&x, &y, &z, &SumLift);
        let flattened = snapshot.flattened_mass_matrix();

        assert_eq!(snapshot.operator_report.dimension, 4);
        assert_eq!(flattened.len(), 9);
        assert!(
            (flattened[0] - snapshot.operator_report.output_coords.iter().sum::<f64>()).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_structurable_bridge_snapshot_json_roundtrip() {
        let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
        let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
        let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

        let snapshot = sample_structurable_bridge(&x, &y, &z, &SumLift);
        let json = snapshot
            .to_json_pretty()
            .expect("bridge snapshot should serialize");
        let decoded: StructurableBridgeSnapshot =
            serde_json::from_str(&json).expect("bridge snapshot should deserialize");

        assert_eq!(decoded, snapshot);
        assert!(json.contains("\"mass_matrix\""));
        assert!(snapshot.to_string().contains("mass-diagonal"));
    }
}
