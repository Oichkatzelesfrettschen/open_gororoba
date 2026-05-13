//! One-shot dumper: prints all 109 `CrystalStructureInfo` entries as TOML
//! [[structure]] blocks suitable for inclusion in
//! `crates/materials_data/data/crystal/crystal_structures.toml`.
//!
//! Usage:
//!   cargo run --example dump_crystal_structures_toml -p materials_core --release \
//!     > /tmp/crystal_structures_dump.toml
//!
//! Then concatenate the dump into the canonical TOML and remove the
//! inline `make()` calls from
//! `crates/materials_core/src/crystal_symmetry/crystal_structures.rs`.
//!
//! Purpose: #127 Phase 6 full migration of the remaining 101 inline
//! structure entries to TOML codegen. Format matches the existing schema
//! at the top of crystal_structures.toml byte-for-byte.

use materials_core::crystal_symmetry::known_crystal_structures;

fn main() {
    let registry = known_crystal_structures();
    for s in registry.iter() {
        let pg_name = format!("{:?}", s.point_group);
        let ls_name = format!("{:?}", s.lattice_system);
        // Drop the "PointGroup::" / "LatticeSystem::" prefix from the Debug.
        let pg_name = pg_name.rsplit("::").next().unwrap_or(&pg_name);
        let ls_name = ls_name.rsplit("::").next().unwrap_or(&ls_name);
        println!();
        println!("[[structure]]");
        println!("name = {:?}", s.name);
        println!("sg_num = {}", s.space_group_number);
        println!("sg_sym = {:?}", s.space_group_symbol);
        println!("point_group = {:?}", pg_name);
        println!("lattice_system = {:?}", ls_name);
        println!("centering = {:?}", s.bravais_centering.to_string());
        println!("a = {:?}", s.lattice_a_angstrom);
        println!("b = {:?}", s.lattice_b_angstrom);
        println!("c = {:?}", s.lattice_c_angstrom);
        println!("alpha = {:?}", s.alpha_deg);
        println!("beta = {:?}", s.beta_deg);
        println!("gamma = {:?}", s.gamma_deg);
        println!("z = {}", s.atoms_per_unit_cell);
        println!("density = {:?}", s.density_g_cm3);
        println!("reference = {:?}", s.primary_reference);
    }
}
