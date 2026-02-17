//! TDA Bridge for stats_core.
//!
//! Exposes topological data analysis primitives including Persistent Homology.

use crate::hypergraph::TriadHypergraph;

/// Persistent Homology summary for a point cloud.
#[derive(Debug, Clone)]
pub struct PersistentHomology {
    pub barcodes: Vec<Barcode>,
}

/// A single persistent homology interval (birth, death).
#[derive(Debug, Clone, Copy)]
pub struct Barcode {
    pub dimension: usize,
    pub birth: f64,
    pub death: f64,
}

impl Barcode {
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }
}

/// Compute persistent homology for a 3D point cloud via Vietoris-Rips filtration.
pub fn compute_vietoris_rips(points: &[[f64; 3]], max_radius: f64) -> PersistentHomology {
    let mut barcodes = Vec::new();
    let n = points.len();

    // H0: Connected components
    // We can use a simple Union-Find or MST-based approach
    // For this prototype, we'll just report the number of components
    // at various scales.

    // H1: Cycles
    // Simple 1-cycle detection: for every triangle (i, j, k),
    // if edges (i,j), (j,k), (k,i) all exist at radius R, a cycle is born.

    // Toy implementation:
    // We return Betti-0 bars for each point starting at birth=0.
    for _ in 0..n {
        barcodes.push(Barcode {
            dimension: 0,
            birth: 0.0,
            death: max_radius,
        });
    }

    PersistentHomology { barcodes }
}

/// Compute persistent homology for a 3-uniform hypergraph.
pub fn compute_persistence(hg: &TriadHypergraph) -> PersistentHomology {
    let mut barcodes = Vec::new();

    // Betti-0 components
    for _ in 0..hg.betti_0() {
        barcodes.push(Barcode {
            dimension: 0,
            birth: 0.0,
            death: f64::INFINITY,
        });
    }

    // Betti-1 cycles
    for _ in 0..hg.betti_1() {
        barcodes.push(Barcode {
            dimension: 1,
            birth: 1.0, // Fixed birth for cycles in this toy model
            death: f64::INFINITY,
        });
    }

    PersistentHomology { barcodes }
}
