//! Thesis 2: Knotted Filtration of Particle Mass
//!
//! Elementary particle masses emerge from survival depth in the Cayley-Dickson
//! filtration cascade (Lambda_2048 -> Lambda_256).

pub mod basis_index;
pub mod filtration;
pub mod patricia_trie;
pub mod survival_spectrum;

pub use basis_index::{project_to_lattice, BasisIndexCodec};
pub use filtration::{
    simulate_fibonacci_collision_storm, CollisionObservation, CollisionStormStats,
};
pub use patricia_trie::PatriciaIndex;
pub use survival_spectrum::{
    classify_latency_law, inverse_square_r2, radial_bins, LatencyLaw, SpectrumBin,
};
