use crate::AlgebraResult;
use serde::{Deserialize, Serialize};

/// Represents a 16-D block within the universal algebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: String,
    /// The global indices in the 4096-D vector space that this block occupies.
    pub basis_indices: Vec<usize>,
}

/// Represents a major component (U_A or U_B) of the decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub label: String,
    pub is_associative: bool,
    pub is_commutative: bool,
    pub homotopy_closure: bool,
    pub blocks: Vec<Block>,
}

/// The Universal Algebra Object (4096-D).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalAlgebra {
    pub dim: usize,
    pub components: Vec<Component>,
}

impl UniversalAlgebra {
    /// Create a new empty Universal Algebra of specified dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            components: Vec::new(),
        }
    }

    /// Load the decomposition from the "CSV schema" description.
    /// This is a placeholder for the logic described in the prompt.
    pub fn from_schema_description() -> Self {
        let mut ua = Self::new(4096);

        // Component A: Non-commutative, Associative
        let mut comp_a = Component {
            label: "U_A".to_string(),
            is_associative: true,
            is_commutative: false,
            homotopy_closure: true,
            blocks: Vec::new(),
        };

        // Component B: Non-commutative, Non-associative
        let mut comp_b = Component {
            label: "U_B".to_string(),
            is_associative: false,
            is_commutative: false,
            homotopy_closure: true,
            blocks: Vec::new(),
        };

        // Populate blocks (placeholder logic based on 16-D chunks)
        // 256 blocks total. Assign first half to A, second to B for now.
        for k in 0..128 {
            let start = k * 16;
            let indices = (start..start + 16).collect();
            comp_a.blocks.push(Block {
                id: format!("U_A_{}", k),
                basis_indices: indices,
            });
        }

        for k in 128..256 {
            let start = k * 16;
            let indices = (start..start + 16).collect();
            comp_b.blocks.push(Block {
                id: format!("U_B_{}", k),
                basis_indices: indices,
            });
        }

        ua.components.push(comp_a);
        ua.components.push(comp_b);
        ua
    }

    /// Load from a CSV reader.
    /// Expects columns: "component", "block_id", "basis_indices" (comma-separated).
    pub fn from_csv<R: std::io::Read>(reader: R) -> AlgebraResult<Self> {
        let mut ua = Self::new(4096);
        let mut comp_map: std::collections::HashMap<String, Component> =
            std::collections::HashMap::new();

        let mut rdr = csv::Reader::from_reader(reader);
        for result in rdr.records() {
            let record = result.map_err(|e| crate::AlgebraError::SimdError(e.to_string()))?;
            let comp_label = &record[0];
            let block_id = &record[1];
            let indices_str = &record[2];

            let indices: Vec<usize> = indices_str
                .split(',')
                .map(|s| s.trim().parse::<usize>().unwrap_or(0))
                .collect();

            let comp = comp_map
                .entry(comp_label.to_string())
                .or_insert_with(|| Component {
                    label: comp_label.to_string(),
                    // Defaults, ideally read from another metadata CSV
                    is_associative: comp_label.contains("A"),
                    is_commutative: false,
                    homotopy_closure: true,
                    blocks: Vec::new(),
                });

            comp.blocks.push(Block {
                id: block_id.to_string(),
                basis_indices: indices,
            });
        }

        for (_, comp) in comp_map {
            ua.components.push(comp);
        }
        // Sort components by label for consistency
        ua.components.sort_by(|a, b| a.label.cmp(&b.label));

        Ok(ua)
    }
}
