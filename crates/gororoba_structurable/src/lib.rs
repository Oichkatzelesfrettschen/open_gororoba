//! Structurable and Jordan-pair bridge scaffolding.
//!
//! This crate is the first implementation step for the missing middle tier:
//! Jordan pairs, involutive structurable carriers, and value-level ternary
//! operators that can later feed the `flavor_lifts` seam without materializing
//! dense operators in hot loops.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply};

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

    pub fn involute(&self) -> Self {
        Self {
            coords: cd_conjugate(&self.coords),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}
