//! Advanced Manifold Projection: Associator Growth and Baire Ultrametricity.
//!
//! This script unifies:
//! 1. Algebraic scaling of the associator sq-norm across CD dimensions (c074).
//! 2. Information-theoretic scaling of ultrametricity across attribute dimensions (c071c).
//!
//! We test whether the 'algebraic fragility' of high-dimensional CD algebras
//! shares a common Jacobian structure with the 'information fragility' of
//! high-dimensional astrophysical data cubes.

use nalgebra::DVector;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use verified_core::coupler_manifold::{CouplerJacobian, CouplerPoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "--- Domain 4: CD-Algebra Associator Growth (c074_associator_growth_empirical.csv) ---"
    );
    let path_alg = "../../data/csv/c074_associator_growth_empirical.csv";
    let file_alg = File::open(path_alg)?;
    let reader_alg = BufReader::new(file_alg);

    let mut alg_points = Vec::new();
    for line in reader_alg.lines() {
        let l = line?;
        if l.starts_with('#') || l.is_empty() || l.starts_with("dimension") {
            continue;
        }

        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        if let (Ok(dim), Ok(norm_sq)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
            alg_points.push(CouplerPoint {
                g: DVector::from_vec(vec![dim]),
                o: DVector::from_vec(vec![norm_sq]),
            });
            println!(
                "  Dim (g): {:.0} | Associator sq-norm (O): {:.4}",
                dim, norm_sq
            );
        }
    }

    if alg_points.len() >= 2 {
        println!("\nAlgebraic Growth Jacobians:");
        for i in 0..(alg_points.len() - 1) {
            if let Ok(jac) =
                CouplerJacobian::estimate_from_delta(&alg_points[i], &alg_points[i + 1])
            {
                println!(
                    "    g range [{:.0}, {:.0}]: J = {:.4}",
                    alg_points[i].g[0],
                    alg_points[i + 1].g[0],
                    jac.j_mat[(0, 0)]
                );
            }
        }
    }

    println!(
        "\n--- Domain 5: Baire-Space Information Scaling (c071c_baire_compact_ultrametric.csv) ---"
    );
    let path_inf = "../../data/csv/c071c_baire_compact_ultrametric.csv";
    let file_inf = File::open(path_inf)?;
    let reader_inf = BufReader::new(file_inf);

    let mut inf_points = Vec::new();
    for line in reader_inf.lines().skip(1) {
        let l = line?;
        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 6 {
            continue;
        }

        // We look for 'n_attributes' as g and 'ultrametric_fraction' as O
        if let (Ok(n_attr), Ok(um_frac)) = (parts[2].parse::<f64>(), parts[5].parse::<f64>()) {
            inf_points.push(CouplerPoint {
                g: DVector::from_vec(vec![n_attr]),
                o: DVector::from_vec(vec![um_frac + 1e-6]),
            });
            println!(
                "  Attributes (g): {:.0} | Ultrametric Fraction (O): {:.4} | Label: {}",
                n_attr, um_frac, parts[0]
            );
        }
    }

    // Filter for FRB points specifically to check scaling across attributes
    let frb_points: Vec<_> = inf_points
        .iter()
        .zip(vec![
            "FRB_3attr",
            "Pulsar_3attr",
            "Combined_3attr",
            "FRB_DM_only",
        ])
        .filter(|(_, label)| label.contains("FRB"))
        .map(|(p, _)| p.clone())
        .collect();

    if frb_points.len() >= 2 {
        // Sort by g
        let mut sorted_frb = frb_points.clone();
        sorted_frb.sort_by(|a, b| a.g[0].partial_cmp(&b.g[0]).unwrap());

        println!("\nInformation Scaling Jacobians (FRB):");
        for i in 0..(sorted_frb.len() - 1) {
            if (sorted_frb[i + 1].g[0] - sorted_frb[i].g[0]).abs() > 0.5
                && let Ok(jac) =
                    CouplerJacobian::estimate_from_delta(&sorted_frb[i], &sorted_frb[i + 1])
            {
                println!(
                    "    g range [{:.0}, {:.0}]: J = {:.4}",
                    sorted_frb[i].g[0],
                    sorted_frb[i + 1].g[0],
                    jac.j_mat[(0, 0)]
                );
            }
        }
    }

    println!("\n--- Cross-Domain Synthesis: Algebra vs Information ---");
    // Extract the 'mid-range' Jacobians
    // Algebra (dim 16 -> 32): saturation of non-associativity
    // Information (attr 1 -> 3): expansion of data cube

    let j_alg_mid = if alg_points.len() >= 3 {
        CouplerJacobian::estimate_from_delta(&alg_points[1], &alg_points[2])?.j_mat[(0, 0)]
    } else {
        0.0
    };

    let j_inf_mid = if frb_points.len() >= 2 {
        let mut sorted = frb_points.clone();
        sorted.sort_by(|a, b| a.g[0].partial_cmp(&b.g[0]).unwrap());
        CouplerJacobian::estimate_from_delta(&sorted[0], &sorted[1])?.j_mat[(0, 0)]
    } else {
        0.0
    };

    println!("Algebraic Jacobian (Sedenion Saturation): {:.4}", j_alg_mid);
    println!(
        "Information Jacobian (Data Cube Expansion): {:.4}",
        j_inf_mid
    );

    let delta = (j_alg_mid - j_inf_mid).abs();
    println!("Delta J: {:.4}", delta);

    if delta < 0.2 {
        println!(
            "!! DISCOVERY: Deep alignment between Algebraic and Informational scaling detected !!"
        );
        println!(
            "The way non-associativity saturates in higher algebras mirrors the way hierarchical information "
        );
        println!(
            "saturates in high-dimensional data cubes. They inhabit the same sector of the Coupler Manifold."
        );
    } else {
        println!("The two domains inhabit distinct scaling sectors.");
    }

    Ok(())
}
