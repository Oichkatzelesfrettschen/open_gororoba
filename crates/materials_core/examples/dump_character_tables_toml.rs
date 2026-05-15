//! One-shot dumper: prints all 32 (or as many as are inline) point-group
//! character tables from `CharacterTable::for_point_group()` as TOML
//! [[table]] blocks suitable for inclusion in
//! `crates/materials_data/data/crystal/character_tables.toml`.
//!
//! #127 Phase 7 full-migration helper.

use materials_core::crystal_symmetry::{CharacterTable, PointGroup};

const ALL_POINT_GROUPS: &[PointGroup] = &[
    PointGroup::C1,
    PointGroup::Ci,
    PointGroup::C2,
    PointGroup::Cs,
    PointGroup::C2h,
    PointGroup::D2,
    PointGroup::C2v,
    PointGroup::D2h,
    PointGroup::C4,
    PointGroup::S4,
    PointGroup::C4h,
    PointGroup::D4,
    PointGroup::C4v,
    PointGroup::D2d,
    PointGroup::D4h,
    PointGroup::C3,
    PointGroup::C3i,
    PointGroup::C3v,
    PointGroup::D3,
    PointGroup::D3d,
    PointGroup::C6,
    PointGroup::C3h,
    PointGroup::C6h,
    PointGroup::D6,
    PointGroup::C6v,
    PointGroup::D3h,
    PointGroup::D6h,
    PointGroup::T,
    PointGroup::Td,
    PointGroup::Th,
    PointGroup::O,
    PointGroup::Oh,
];

fn pg_name(pg: PointGroup) -> &'static str {
    match pg {
        PointGroup::C1 => "C1",
        PointGroup::Ci => "Ci",
        PointGroup::C2 => "C2",
        PointGroup::Cs => "Cs",
        PointGroup::C2h => "C2h",
        PointGroup::D2 => "D2",
        PointGroup::C2v => "C2v",
        PointGroup::D2h => "D2h",
        PointGroup::C4 => "C4",
        PointGroup::S4 => "S4",
        PointGroup::C4h => "C4h",
        PointGroup::D4 => "D4",
        PointGroup::C4v => "C4v",
        PointGroup::D2d => "D2d",
        PointGroup::D4h => "D4h",
        PointGroup::C3 => "C3",
        PointGroup::C3i => "C3i",
        PointGroup::C3v => "C3v",
        PointGroup::D3 => "D3",
        PointGroup::D3d => "D3d",
        PointGroup::C6 => "C6",
        PointGroup::C3h => "C3h",
        PointGroup::C6h => "C6h",
        PointGroup::D6 => "D6",
        PointGroup::C6v => "C6v",
        PointGroup::D3h => "D3h",
        PointGroup::D6h => "D6h",
        PointGroup::T => "T",
        PointGroup::Td => "Td",
        PointGroup::Th => "Th",
        PointGroup::O => "O",
        PointGroup::Oh => "Oh",
    }
}

fn main() {
    for &pg in ALL_POINT_GROUPS {
        let Some(table) = CharacterTable::for_point_group(pg) else {
            // Skip unimplemented point groups (those return None inline).
            continue;
        };
        println!();
        println!("[[table]]");
        println!("point_group = {:?}", pg_name(pg));
        // Inline `classes` array of tables.
        print!("classes = [");
        for (idx, c) in table.classes.iter().enumerate() {
            if idx > 0 {
                print!(",");
            }
            print!("\n    {{ name = {:?}, count = {} }}", c.name, c.count);
        }
        println!("\n]");
        // Inline `irreps`.
        print!("irreps = [");
        for (idx, i) in table.irreps.iter().enumerate() {
            if idx > 0 {
                print!(",");
            }
            print!(
                "\n    {{ label = {:?}, dimension = {} }}",
                i.label, i.dimension
            );
        }
        println!("\n]");
        // 2D characters: outer list of rows, inner list of [re, im] pairs.
        print!("characters = [");
        for (row_idx, row) in table.characters.iter().enumerate() {
            if row_idx > 0 {
                print!(",");
            }
            print!("\n    [");
            for (col_idx, (re, im)) in row.iter().enumerate() {
                if col_idx > 0 {
                    print!(", ");
                }
                print!("[{:?}, {:?}]", re, im);
            }
            print!("]");
        }
        println!("\n]");
    }
}
