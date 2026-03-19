//! Summary analysis of Cayley-Dickson Motifs across dimensions.
//!
//! Aggregates component statistics (node counts, edge counts, symmetry types)
//! from individual dimension CSVs into a global summary.
//!
//! Migrated from src/vis_cd_motif_summary.py.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Statistics for a single algebra dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifDimSummary {
    pub dim: usize,
    pub component_count: usize,
    pub active_nodes_total: usize,
    pub max_component_nodes: usize,
    pub max_component_edges: usize,
    pub octahedron_k222_count: usize,
    pub cuboctahedron_count: usize,
    pub k2_multipartite_max_parts: usize,
}

/// Aggregates motif data from CSV files in data/csv.
pub fn collect_motif_summary(dims: &[usize]) -> Vec<MotifDimSummary> {
    let mut summary = Vec::new();
    for &dim in dims {
        let path_str = format!("data/csv/cd_motif_components_{}d.csv", dim);
        let path = Path::new(&path_str);
        if !path.exists() {
            continue;
        }

        // Toy implementation for logic migration: 
        // In a real run, we'd use csv::Reader to parse the files.
        // For now, we provide the aggregation structure.
        let mut reader = csv::Reader::from_path(path).unwrap();
        let mut count = 0;
        let mut total_nodes = 0;
        let mut max_nodes = 0;
        let mut max_edges = 0;
        let mut k222 = 0;
        let mut cuboct = 0;
        let mut max_parts = 0;

        for result in reader.records() {
            let record = result.unwrap();
            count += 1;
            let nodes: usize = record.get(1).unwrap_or("0").parse().unwrap_or(0);
            let edges: usize = record.get(2).unwrap_or("0").parse().unwrap_or(0);
            total_nodes += nodes;
            max_nodes = max_nodes.max(nodes);
            max_edges = max_edges.max(edges);
            
            // Check flags (assume columns 3, 4, 5 for symmetry)
            if record.get(3).unwrap_or("false") == "true" { k222 += 1; }
            if record.get(4).unwrap_or("false") == "true" { cuboct += 1; }
            let parts: usize = record.get(5).unwrap_or("0").parse().unwrap_or(0);
            max_parts = max_parts.max(parts);
        }

        summary.push(MotifDimSummary {
            dim,
            component_count: count,
            active_nodes_total: total_nodes,
            max_component_nodes: max_nodes,
            max_component_edges: max_edges,
            octahedron_k222_count: k222,
            cuboctahedron_count: cuboct,
            k2_multipartite_max_parts: max_parts,
        });
    }
    summary
}
