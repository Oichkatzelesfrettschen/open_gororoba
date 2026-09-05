//! Decompose E8 to its simple roots, Cartan matrix, and 240-root Weyl orbit.
//!
//! E8 is the rank-8 exceptional Lie algebra (dim 248 = 8 + 240). This binary
//! prints the Dynkin diagram (branch at node 4), the eight simple roots, the
//! Cartan matrix A_ij = 2(alpha_i, alpha_j)/(alpha_i, alpha_i), the highest
//! root (height 29), the 112 D8 + 128 spinor split, and confirms Weyl
//! generation matches the combinatorial 240.
//!
//! The staple packing index e_8 (lag-2 Bx in 16D sedenions) is not this
//! algebra. The roots span an eight-dimensional Euclidean space; equality
//! of dimension alone establishes no identification with the octonions.
//!
//! ```bash
//! cargo run --profile validation -p gororoba_cli_physics --bin e8-root-decompose
//! ```

use anyhow::{Context, Result, ensure};
use gororoba_algebra::lie::e8::root_system::{
    e8_cartan_matrix, e8_root_type_counts, e8_simple_roots, e8_weyl_group_order, generate_e8_roots,
    generate_e8_roots_by_weyl, height, simple_coordinates, weyl_highest_root,
};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Serialize)]
struct Report {
    rank: u8,
    dimension: u16,
    n_roots: usize,
    n_positive: usize,
    n_d8_integer: usize,
    n_spinor_half: usize,
    weyl_order: u64,
    weyl_orbit_len: usize,
    root_sets_match: bool,
    highest_root_coords: [f64; 8],
    highest_root_simple_coords: [i32; 8],
    highest_root_height: i32,
    simple_roots: Vec<[f64; 8]>,
    cartan: [[i32; 8]; 8],
    dynkin: &'static str,
    height_histogram: Vec<(i32, usize)>,
    reading: String,
}

fn main() -> Result<()> {
    let weyl = generate_e8_roots_by_weyl();
    let combinatorial = generate_e8_roots();
    let root_keys = |roots: &[gororoba_algebra::lie::e8::root_system::E8Root]| {
        roots
            .iter()
            .map(|root| {
                root.coords
                    .map(|coordinate| (2.0 * coordinate).round() as i32)
            })
            .collect::<BTreeSet<_>>()
    };
    let root_sets_match = root_keys(&weyl) == root_keys(&combinatorial);
    ensure!(root_sets_match, "Weyl and combinatorial root sets differ");
    let (d8, spinor) =
        e8_root_type_counts(&weyl).context("Weyl orbit contains invalid E8 roots")?;
    let highest = weyl_highest_root();
    let highest_root_height = height(&highest).context("highest root is outside E8")?;
    let mut hist = [0usize; 61];
    for r in &weyl {
        let h = height(r).context("Weyl orbit contains a vector outside E8")?;
        ensure!(
            (-30..=30).contains(&h),
            "root height {h} exceeds histogram bounds"
        );
        let idx = (h + 30) as usize;
        hist[idx] += 1;
    }
    let height_histogram: Vec<(i32, usize)> = hist
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| (i as i32 - 30, *c))
        .collect();
    let report = Report {
        rank: 8,
        dimension: 248,
        n_roots: combinatorial.len(),
        n_positive: height_histogram
            .iter()
            .filter(|(height, _)| *height > 0)
            .map(|(_, count)| count)
            .sum(),
        n_d8_integer: d8,
        n_spinor_half: spinor,
        weyl_order: e8_weyl_group_order(),
        weyl_orbit_len: weyl.len(),
        root_sets_match,
        highest_root_coords: highest.coords,
        highest_root_simple_coords: simple_coordinates(&highest)
            .context("highest root is outside E8")?,
        highest_root_height,
        simple_roots: e8_simple_roots().map(|r| r.coords).to_vec(),
        cartan: e8_cartan_matrix(),
        dynkin: "alpha0-alpha1-alpha2-alpha3-alpha4-alpha5 ; alpha4-alpha6-alpha7",
        height_histogram,
        reading: format!(
            "E8 = 8 Cartan + 240 roots. Simple reflections close on 240 (got {}). Highest root height {}. Split 112 D8 + 128 spinor. Staple index e8 is a 16D sedenion basis vector, not this 8D root system.",
            weyl.len(),
            highest_root_height
        ),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
