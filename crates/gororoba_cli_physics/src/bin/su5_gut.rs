//! su5-gut: SU(5) GUT generator decomposition and verification.
//!
//! Usage: `su5-gut [--format table|json] [--verbose] [--only generators|fermions|verify]`

use std::process::ExitCode;

use algebra_experimental::su5_gut::{
    self, build_generators, build_quark_sector, classify_all, verify_algebra,
};
use clap::Parser;
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "su5-gut",
    about = "SU(5) GUT generator decomposition and quark sector analysis"
)]
struct Cli {
    /// Output format
    #[arg(long, default_value = "table", value_parser = ["table", "json"])]
    format: String,

    /// Show detailed matrix output and commutator closure table
    #[arg(long)]
    verbose: bool,

    /// Show only one section
    #[arg(long, value_parser = ["generators", "fermions", "verify"])]
    only: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let generators = build_generators();
    let infos = classify_all(&generators);
    let fermions = build_quark_sector();
    let results = verify_algebra();

    let all_passed = results.iter().all(|r| r.passed);

    if cli.format == "json" {
        let gen_json: Vec<_> = infos
            .iter()
            .map(|g| {
                json!({
                    "index": g.index,
                    "sector": g.sector,
                    "generator_type": g.generator_type,
                    "indices": [g.indices.0, g.indices.1],
                    "sm_representation": [
                        g.sm_representation.dim_su3,
                        g.sm_representation.dim_su2,
                        g.sm_representation.hypercharge,
                    ],
                    "physical_name": g.physical_name,
                })
            })
            .collect();

        let fermion_json: Vec<_> = fermions
            .iter()
            .map(|f| {
                json!({
                    "name": f.name,
                    "representation": f.representation,
                    "position": f.position,
                    "quantum_numbers": f.quantum_numbers,
                })
            })
            .collect();

        let verify_json: Vec<_> = results
            .iter()
            .map(|r| {
                json!({
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                })
            })
            .collect();

        let output = json!({
            "generators": gen_json,
            "fermions": fermion_json,
            "verification": verify_json,
            "all_passed": all_passed,
            "metadata": {
                "decomposition": "24 = (8,1,0) + (1,3,0) + (1,1,0) + (3,2,5/6) + (3*,2,-5/6)",
                "group": "SU(5) -> SU(3)_C x SU(2)_L x U(1)_Y",
            },
        });

        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return if all_passed {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    // Table format
    let mut sections = Vec::new();

    if cli.only.is_none() || cli.only.as_deref() == Some("generators") {
        sections.push("=".repeat(72));
        sections.push("SU(5) Generator Classification".to_string());
        sections.push("24 = (8,1,0) + (1,3,0) + (1,1,0) + (3,2,5/6) + (3*,2,-5/6)".to_string());
        sections.push("=".repeat(72));
        sections.push(String::new());
        sections.push(su5_gut::format_generator_table(&infos));
        if cli.verbose {
            sections.push(String::new());
            for g in &infos {
                sections.push(su5_gut::format_matrix(
                    &g.matrix,
                    &format!("T[{}] ({})", g.index, g.physical_name),
                ));
            }
            sections.push(String::new());
            sections.push(su5_gut::format_commutator_table(&generators));
        }
    }

    if cli.only.is_none() || cli.only.as_deref() == Some("fermions") {
        sections.push(String::new());
        sections.push("=".repeat(72));
        sections.push("Quark Sector: Fermion Assignments (one generation)".to_string());
        sections.push("=".repeat(72));
        sections.push(su5_gut::format_fermion_table(&fermions));
        sections.push(String::new());
        sections.push(su5_gut::format_ten_tensor());
    }

    if cli.only.is_none() || cli.only.as_deref() == Some("verify") {
        sections.push(String::new());
        sections.push("=".repeat(72));
        sections.push("Verification Suite".to_string());
        sections.push("=".repeat(72));
        sections.push(String::new());
        sections.push(su5_gut::format_verification_table(&results));
    }

    sections.push(String::new());
    let status = if all_passed {
        "ALL 11 CHECKS PASSED"
    } else {
        "SOME CHECKS FAILED"
    };
    sections.push(format!("Result: {status}"));

    println!("{}", sections.join("\n"));

    if all_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
