//! Structurable and Jordan-pair bridge scaffolding.
//!
//! This crate is the first implementation step for the missing middle tier:
//! Jordan pairs, involutive structurable carriers, and value-level ternary
//! operators that can later feed the `flavor_lifts` seam without materializing
//! dense operators in hot loops.
//!
//! For a quick probe, run `cargo run -p gororoba_structurable --example v_operator_report`.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct JordanPairElement {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

impl JordanPairElement {
    pub fn new(plus: Vec<f64>, minus: Vec<f64>) -> Self {
        assert_eq!(plus.len(), minus.len(), "Jordan pair halves must match");
        Self { plus, minus }
    }

    pub fn dim(&self) -> usize {
        self.plus.len()
    }

    pub fn swap(&self) -> Self {
        Self {
            plus: self.minus.clone(),
            minus: self.plus.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructurableElement {
    coords: Vec<f64>,
}

impl StructurableElement {
    pub fn new(coords: Vec<f64>) -> Self {
        assert!(
            coords.len().is_power_of_two(),
            "Structurable carrier must use a CD-compatible dimension"
        );
        Self { coords }
    }

    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    pub fn norm_squared(&self) -> f64 {
        self.coords.iter().map(|value| value * value).sum()
    }

    pub fn involute(&self) -> Self {
        Self {
            coords: cd_conjugate(&self.coords),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueLevelOperatorCost {
    pub binary_products: usize,
    pub conjugations: usize,
    pub vector_additions: usize,
}

pub const STRUCTURABLE_V_OPERATOR_COST: ValueLevelOperatorCost = ValueLevelOperatorCost {
    binary_products: 5,
    conjugations: 2,
    vector_additions: 2,
};

/// Compact exploration record for the value-level structurable V operator.
///
/// This keeps the output inspectable in tests and experiments without forcing
/// callers to reconstruct norms or cost metadata by hand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructurableVOperatorReport {
    pub dimension: usize,
    pub input_norms: [f64; 3],
    pub output_norm: f64,
    pub output_coords: Vec<f64>,
    pub cost: ValueLevelOperatorCost,
}

impl StructurableVOperatorReport {
    pub fn summary_row(&self) -> [f64; 5] {
        [
            self.dimension as f64,
            self.input_norms[0],
            self.input_norms[1],
            self.input_norms[2],
            self.output_norm,
        ]
    }

    pub fn summary_line(&self) -> String {
        self.to_string()
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl fmt::Display for StructurableVOperatorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dim={} ||x||^2={:.6} ||y||^2={:.6} ||z||^2={:.6} ||V||^2={:.6} cost=({} mul, {} conj, {} add)",
            self.dimension,
            self.input_norms[0],
            self.input_norms[1],
            self.input_norms[2],
            self.output_norm,
            self.cost.binary_products,
            self.cost.conjugations,
            self.cost.vector_additions
        )
    }
}

pub fn structurable_v_operator(
    x: &StructurableElement,
    y: &StructurableElement,
    z: &StructurableElement,
) -> StructurableElement {
    assert_eq!(
        x.dim(),
        y.dim(),
        "structurable V operator requires matching dimensions"
    );
    assert_eq!(
        y.dim(),
        z.dim(),
        "structurable V operator requires matching dimensions"
    );

    let y_bar = cd_conjugate(y.coords());
    let x_bar = cd_conjugate(x.coords());

    let xy_bar = cd_multiply(x.coords(), &y_bar);
    let zy_bar = cd_multiply(z.coords(), &y_bar);
    let zx_bar = cd_multiply(z.coords(), &x_bar);

    let left = cd_multiply(&xy_bar, z.coords());
    let middle = cd_multiply(&zy_bar, x.coords());
    let right = cd_multiply(&zx_bar, y.coords());

    let coords = left
        .into_iter()
        .zip(middle)
        .zip(right)
        .map(|((l, m), r)| l + m - r)
        .collect();

    StructurableElement::new(coords)
}

/// Evaluate the V operator and retain a lightweight exploration report.
pub fn structurable_v_operator_report(
    x: &StructurableElement,
    y: &StructurableElement,
    z: &StructurableElement,
) -> StructurableVOperatorReport {
    let output = structurable_v_operator(x, y, z);
    StructurableVOperatorReport {
        dimension: output.dim(),
        input_norms: [x.norm_squared(), y.norm_squared(), z.norm_squared()],
        output_norm: output.norm_squared(),
        output_coords: output.coords().to_vec(),
        cost: STRUCTURABLE_V_OPERATOR_COST,
    }
}

pub fn jordan_pair_from_structurable(
    plus: StructurableElement,
    minus: StructurableElement,
) -> JordanPairElement {
    assert_eq!(plus.dim(), minus.dim(), "Jordan pair halves must match");
    JordanPairElement::new(plus.coords, minus.coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structurable_v_operator_preserves_dimension() {
        let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
        let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
        let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

        let out = structurable_v_operator(&x, &y, &z);
        assert_eq!(out.dim(), 4);
    }

    #[test]
    fn test_cost_contract_is_value_level() {
        assert_eq!(STRUCTURABLE_V_OPERATOR_COST.binary_products, 5);
    }

    #[test]
    fn test_structurable_v_operator_report_matches_raw_output() {
        let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
        let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
        let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

        let raw = structurable_v_operator(&x, &y, &z);
        let report = structurable_v_operator_report(&x, &y, &z);

        assert_eq!(report.dimension, 4);
        assert_eq!(report.output_coords, raw.coords().to_vec());
        assert_eq!(report.cost, STRUCTURABLE_V_OPERATOR_COST);
        assert_eq!(report.summary_row()[0], 4.0);
    }

    #[test]
    fn test_structurable_v_operator_report_json_roundtrip() {
        let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
        let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
        let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

        let report = structurable_v_operator_report(&x, &y, &z);
        let json = report
            .to_json_pretty()
            .expect("structurable report should serialize");
        let decoded: StructurableVOperatorReport =
            serde_json::from_str(&json).expect("structurable report should deserialize");

        assert_eq!(decoded, report);
        assert!(json.contains("\"output_coords\""));
        assert!(report.summary_line().contains("dim=4"));
    }
}
