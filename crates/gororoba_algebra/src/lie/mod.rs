pub mod e7_geometry;
pub mod e7_structure;
pub mod e8_lattice;
pub mod group_theory;
pub mod kac_moody;
pub mod lyndon_basis;
pub mod nilpotent_orbits;
// three_fermion_generations depends only on cd_kernel + rand -- no heavy deps.
// It must be always available because lepton_mass_hierarchy (algebra_experimental)
// imports get_sedenion_subalgebras from it regardless of feature state.
pub mod three_fermion_generations;
#[cfg(feature = "physics-sm")]
pub mod su5_gut;
