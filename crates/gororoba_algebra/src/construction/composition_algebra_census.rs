//! Composition Algebra Census and Taxonomy Framework
//!
//! Comprehensive classification framework for composition algebras across:
//! - Construction methods (tensor product, recursive doubling, exceptional)
//! - Metric signatures (gamma patterns in recursive families)
//! - Algebraic properties (commutativity, associativity, division status, norm multiplicativity)
//!
//! This module establishes the two-axis taxonomy proven in Phase 10.1-10.2 and
//! extends it to ALL composition algebra families.
//!
//! Date: 2026-02-10, Phase 10.3 Track 2

use std::collections::HashMap;

/// Two-axis taxonomy for composition algebras
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstructionMethod {
    /// Component-wise operations: (a, b) * (c, d) = (ac, bd)
    TensorProduct,
    /// Cayley-Dickson recursive doubling with conjugation
    RecursiveDoubling,
    /// Exceptional algebras (Albert, Sedenion-variants)
    Exceptional,
}

/// Metric signature for Cayley-Dickson families
/// Each element corresponds to gamma_n in the recursive formula
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetricSignature {
    /// Gamma values [gamma_1, gamma_2, ...] where gamma_n in {-1, +1}
    /// gamma = -1: Hurwitz (division algebra, zero-divisor free)
    /// gamma = +1: split (zero-divisors present)
    pub gammas: Vec<i8>,
}

impl MetricSignature {
    pub fn new(gammas: Vec<i8>) -> Self {
        Self { gammas }
    }

    /// Hurwitz signature: all gamma = -1 (e.g., R, C, H, O)
    pub fn hurwitz(dimension_log2: usize) -> Self {
        Self {
            gammas: vec![-1; dimension_log2],
        }
    }

    /// Mixed signature: alternating or partial +1
    pub fn mixed(gammas: Vec<i8>) -> Self {
        Self { gammas }
    }

    pub fn is_hurwitz(&self) -> bool {
        self.gammas.iter().all(|&g| g == -1)
    }

    pub fn has_zero_divisors(&self) -> bool {
        self.gammas.contains(&1)
    }
}

/// Algebraic property census result
#[derive(Debug, Clone)]
pub struct AlgebraProperties {
    pub name: String,
    pub dimension: usize,
    pub construction: ConstructionMethod,
    pub metric_signature: Option<MetricSignature>,

    // Boolean properties
    pub is_commutative: bool,
    pub is_associative: bool,
    pub is_division_algebra: bool,
    pub has_zero_divisors: bool,
    pub preserves_norm_multiplicativity: bool,

    // Statistics
    pub commutativity_percentage: f64,
    pub associativity_percentage: f64,
    pub invertibility_percentage: f64,

    // Cross-validation fields
    pub sri_delta_squared_mean: Option<f64>, // For Albert algebra
    pub num_samples: usize,
}

impl AlgebraProperties {
    pub fn new(
        name: &str,
        dimension: usize,
        construction: ConstructionMethod,
        metric_signature: Option<MetricSignature>,
    ) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            construction,
            metric_signature,
            is_commutative: false,
            is_associative: false,
            is_division_algebra: false,
            has_zero_divisors: false,
            preserves_norm_multiplicativity: false,
            commutativity_percentage: 0.0,
            associativity_percentage: 0.0,
            invertibility_percentage: 0.0,
            sri_delta_squared_mean: None,
            num_samples: 0,
        }
    }

    /// Check consistency between division status and zero-divisor structure
    pub fn validate_division_consistency(&self) -> Result<(), String> {
        if self.is_division_algebra && self.has_zero_divisors {
            return Err(format!(
                "{} cannot be both division algebra and have zero-divisors",
                self.name
            ));
        }
        Ok(())
    }

    /// Verify commutativity is 0% or 100% (not intermediate)
    pub fn validate_commutativity_law(&self) -> Result<(), String> {
        if self.commutativity_percentage > 1e-10 && self.commutativity_percentage < 99.99 {
            return Err(format!(
                "{} has intermediate commutativity ({}%), violates all-or-nothing law",
                self.name, self.commutativity_percentage
            ));
        }
        Ok(())
    }

    /// Verify associativity law (if applicable)
    pub fn validate_associativity_law(&self) -> Result<(), String> {
        if self.associativity_percentage > 1e-10 && self.associativity_percentage < 99.99 {
            return Err(format!(
                "{} has intermediate associativity ({}%)",
                self.name, self.associativity_percentage
            ));
        }
        Ok(())
    }
}

/// Taxonomy node representing an algebra family
#[derive(Debug, Clone)]
pub struct TaxonomyNode {
    pub algebra: AlgebraProperties,
    pub axis1_construction: ConstructionMethod,
    pub axis2_signature: Option<MetricSignature>,

    // Relationships
    pub parent_family: Option<String>,
    pub is_exceptional: bool,
}

impl TaxonomyNode {
    pub fn new(algebra: AlgebraProperties) -> Self {
        let axis1 = algebra.construction;
        let axis2 = algebra.metric_signature.clone();
        let is_exceptional = axis1 == ConstructionMethod::Exceptional;

        Self {
            algebra,
            axis1_construction: axis1,
            axis2_signature: axis2,
            parent_family: None,
            is_exceptional,
        }
    }

    /// Check if this node and another represent the same family
    pub fn same_family(&self, other: &TaxonomyNode) -> bool {
        self.axis1_construction == other.axis1_construction
            && self.is_exceptional == other.is_exceptional
    }
}

/// Complete composition algebra census
pub struct CompositionAlgebraCensus {
    algebras: HashMap<String, TaxonomyNode>,
    families: HashMap<String, Vec<String>>, // Family name -> algebra names
}

impl CompositionAlgebraCensus {
    pub fn new() -> Self {
        Self {
            algebras: HashMap::new(),
            families: HashMap::new(),
        }
    }

    /// Register an algebra in the census
    pub fn register_algebra(&mut self, node: TaxonomyNode) -> Result<(), String> {
        let name = node.algebra.name.clone();

        // Validate before registration
        node.algebra.validate_division_consistency()?;
        node.algebra.validate_commutativity_law()?;
        node.algebra.validate_associativity_law()?;

        // Determine family key
        let family_key = format!("{:?}", node.axis1_construction);

        // Register
        self.algebras.insert(name.clone(), node);
        self.families.entry(family_key).or_default().push(name);

        Ok(())
    }

    /// Get all algebras in a family
    pub fn get_family(&self, family_name: &str) -> Option<Vec<&TaxonomyNode>> {
        self.families.get(family_name).map(|names| {
            names
                .iter()
                .filter_map(|name| self.algebras.get(name))
                .collect()
        })
    }

    /// Verify axis 1 hypothesis: construction method determines commutativity universally
    pub fn verify_axis1_commutativity_law(&self) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        for (family_name, algebra_names) in &self.families {
            let mut comm_values = Vec::new();
            for name in algebra_names {
                if let Some(node) = self.algebras.get(name) {
                    comm_values.push((name.clone(), node.algebra.is_commutative));
                }
            }

            // All algebras in same family must have same commutativity
            if !comm_values.is_empty() {
                let first_comm = comm_values[0].1;
                for (name, comm) in &comm_values {
                    if *comm != first_comm {
                        violations.push(format!(
                            "Family {} has mixed commutativity: {} vs {}",
                            family_name, name, comm_values[0].0
                        ));
                    }
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Verify axis 2 hypothesis: metric signature controls zero-divisor presence (CD only)
    pub fn verify_axis2_signature_law(&self) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        // Only check Cayley-Dickson families
        if let Some(cd_algebras) = self.get_family("RecursiveDoubling") {
            for node in cd_algebras {
                if let Some(sig) = &node.axis2_signature {
                    let predicted_has_zd = sig.has_zero_divisors();
                    let actual_has_zd = node.algebra.has_zero_divisors;

                    if predicted_has_zd != actual_has_zd {
                        violations.push(format!(
                            "{}: signature predicts ZD={}, actual ZD={}",
                            node.algebra.name, predicted_has_zd, actual_has_zd
                        ));
                    }
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Count families and algebras per family
    pub fn statistics(&self) -> (usize, HashMap<String, usize>) {
        let num_families = self.families.len();
        let mut algebras_per_family = HashMap::new();
        for (family, names) in &self.families {
            algebras_per_family.insert(family.clone(), names.len());
        }
        (num_families, algebras_per_family)
    }

    /// Export census as structured data
    pub fn export_summary(&self) -> Vec<AlgebraProperties> {
        self.algebras
            .values()
            .map(|node| node.algebra.clone())
            .collect()
    }
}

impl Default for CompositionAlgebraCensus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_signature_creation() {
        let hurwitz = MetricSignature::hurwitz(3);
        assert!(hurwitz.is_hurwitz());
        assert!(!hurwitz.has_zero_divisors());

        let mixed = MetricSignature::mixed(vec![-1, -1, 1]);
        assert!(!mixed.is_hurwitz());
        assert!(mixed.has_zero_divisors());
    }

    #[test]
    fn test_algebra_properties_validation() {
        let mut props = AlgebraProperties::new(
            "TestAlgebra",
            4,
            ConstructionMethod::RecursiveDoubling,
            Some(MetricSignature::hurwitz(2)),
        );
        props.is_division_algebra = true;
        props.has_zero_divisors = false;

        assert!(props.validate_division_consistency().is_ok());
    }

    #[test]
    fn test_algebra_division_zero_divisor_conflict() {
        let mut props = AlgebraProperties::new(
            "BadAlgebra",
            4,
            ConstructionMethod::RecursiveDoubling,
            Some(MetricSignature::hurwitz(2)),
        );
        props.is_division_algebra = true;
        props.has_zero_divisors = true;

        assert!(props.validate_division_consistency().is_err());
    }

    #[test]
    fn test_taxonomy_node_family_matching() {
        let props1 = AlgebraProperties::new("Algebra1", 4, ConstructionMethod::TensorProduct, None);
        let props2 = AlgebraProperties::new("Algebra2", 8, ConstructionMethod::TensorProduct, None);
        let props3 = AlgebraProperties::new(
            "Algebra3",
            4,
            ConstructionMethod::RecursiveDoubling,
            Some(MetricSignature::hurwitz(2)),
        );

        let node1 = TaxonomyNode::new(props1);
        let node2 = TaxonomyNode::new(props2);
        let node3 = TaxonomyNode::new(props3);

        assert!(node1.same_family(&node2));
        assert!(!node1.same_family(&node3));
    }

    #[test]
    fn test_census_registration() {
        let mut census = CompositionAlgebraCensus::new();

        let mut props =
            AlgebraProperties::new("TestAlgebra", 4, ConstructionMethod::TensorProduct, None);
        props.is_commutative = true;
        props.commutativity_percentage = 100.0;

        let node = TaxonomyNode::new(props);
        assert!(census.register_algebra(node).is_ok());

        let (families, _) = census.statistics();
        assert!(families > 0);
    }

    #[test]
    fn test_commutativity_percentage_validation() {
        let mut props = AlgebraProperties::new(
            "IntermedComm",
            4,
            ConstructionMethod::RecursiveDoubling,
            Some(MetricSignature::hurwitz(2)),
        );
        props.commutativity_percentage = 50.0; // Violates all-or-nothing

        assert!(props.validate_commutativity_law().is_err());
    }

    #[test]
    fn test_axis1_family_hypothesis() {
        let mut census = CompositionAlgebraCensus::new();

        let mut props1 =
            AlgebraProperties::new("TensorProduct1", 4, ConstructionMethod::TensorProduct, None);
        props1.is_commutative = true;
        props1.commutativity_percentage = 100.0;

        let mut props2 =
            AlgebraProperties::new("TensorProduct2", 8, ConstructionMethod::TensorProduct, None);
        props2.is_commutative = true;
        props2.commutativity_percentage = 100.0;

        let node1 = TaxonomyNode::new(props1);
        let node2 = TaxonomyNode::new(props2);

        census.register_algebra(node1).ok();
        census.register_algebra(node2).ok();

        assert!(census.verify_axis1_commutativity_law().is_ok());
    }
}
