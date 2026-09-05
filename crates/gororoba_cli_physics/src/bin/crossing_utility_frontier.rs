//! Render unit-consistent detector utility sensitivity from admitted paired measurements.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use gororoba_cli_physics::{
    detection_utility::{UtilityInput, UtilityInterval, UtilityUnit},
    detection_utility_output::publish_bundle,
};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Parser)]
#[command(about = "Render conditional detector utility frontiers from paired measurements")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    /// New directory for report.json, frontier.csv, and frontier.svg.
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    max_cost_ratio: f64,
    /// Lower additive overhead shift, in benefit-normalized units per decision or hour.
    #[arg(long, allow_hyphen_values = true)]
    min_overhead_shift: f64,
    #[arg(long, allow_hyphen_values = true)]
    max_overhead_shift: f64,
    #[arg(long, default_value_t = 61)]
    grid_size: usize,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Empirical,
    MathematicalFixture,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Reference {
    StrictCatalog,
    Adjudicated,
    MathematicalFixture,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceReceipt {
    path: PathBuf,
    sha256: String,
}

fn validate_source_receipt(request_directory: &Path, receipt: &SourceReceipt) -> Result<()> {
    ensure!(
        !receipt.path.as_os_str().is_empty()
            && receipt
                .path
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
            && receipt.path.as_os_str()
                == receipt.path.components().collect::<PathBuf>().as_os_str(),
        "source receipt must use a normalized relative path without parent components"
    );
    let directory = if request_directory.as_os_str().is_empty() {
        Path::new(".")
    } else {
        request_directory
    };
    let root = directory
        .canonicalize()
        .context("resolve request directory")?;
    let path = root
        .join(&receipt.path)
        .canonicalize()
        .context("resolve receipt source")?;
    ensure!(
        path.starts_with(&root),
        "receipt source resolves outside request directory"
    );
    ensure!(
        path.is_file(),
        "source is not a regular file: {}",
        path.display()
    );
    let contents = fs::read(&path)?;
    ensure!(!contents.is_empty(), "empty source evidence");
    ensure!(
        sha256(&contents) == receipt.sha256,
        "source digest mismatch: {}",
        path.display()
    );
    Ok(())
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    evidence_kind: EvidenceKind,
    reference: Reference,
    measurement_boundary: String,
    sources: Vec<SourceReceipt>,
    utility: UtilityInput,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Region {
    DeployConditional,
    MaintainBaseline,
    ManualReview,
}

fn region(interval: Option<&UtilityInterval>) -> Region {
    match interval {
        Some(bounds) if bounds.lower > 0.0 => Region::DeployConditional,
        Some(bounds) if bounds.upper < 0.0 => Region::MaintainBaseline,
        _ => Region::ManualReview,
    }
}

#[derive(Serialize)]
struct Cell {
    cost_ratio: f64,
    normalized_overhead_shift: f64,
    point_utility: f64,
    interval: Option<UtilityInterval>,
    region: Region,
}

fn calculate(request: &Request, args: &Args) -> Result<Vec<Cell>> {
    ensure!(request.schema_version == 1, "unsupported schema version");
    ensure!(
        !request.measurement_boundary.trim().is_empty(),
        "measurement boundary is required"
    );
    ensure!(
        args.max_cost_ratio.is_finite() && args.max_cost_ratio > 0.0,
        "maximum cost ratio must be finite and positive"
    );
    ensure!(
        args.min_overhead_shift.is_finite()
            && args.max_overhead_shift.is_finite()
            && args.max_overhead_shift > args.min_overhead_shift
            && (args.max_overhead_shift - args.min_overhead_shift).is_finite(),
        "overhead axis must have finite increasing endpoints and span"
    );
    ensure!(
        (2..=201).contains(&args.grid_size),
        "grid size must be 2..=201"
    );
    let mut cells = Vec::with_capacity(args.grid_size * args.grid_size);
    for vertical in 0..args.grid_size {
        let shift = args.min_overhead_shift
            + (args.max_overhead_shift - args.min_overhead_shift)
                * (vertical as f64 / (args.grid_size - 1) as f64);
        for horizontal in 0..args.grid_size {
            let ratio = args.max_cost_ratio * (horizontal as f64 / (args.grid_size - 1) as f64);
            let point_utility = request.utility.evaluate(ratio, shift)?;
            let interval = request.utility.interval(ratio, shift)?;
            cells.push(Cell {
                cost_ratio: ratio,
                normalized_overhead_shift: shift,
                point_utility,
                region: if matches!(request.evidence_kind, EvidenceKind::Empirical)
                    && matches!(request.reference, Reference::Adjudicated)
                {
                    region(interval.as_ref().filter(|bounds| bounds.draws >= 1000))
                } else {
                    Region::ManualReview
                },
                interval,
            });
        }
    }
    Ok(cells)
}

fn plot(request: &Request, args: &Args, cells: &[Cell]) -> Result<String> {
    let coefficients = request.utility.coefficients()?;
    let overhead_span = args.max_overhead_shift - args.min_overhead_shift;
    let unit = match request.utility.unit() {
        UtilityUnit::PerDecision => "per decision",
        UtilityUnit::PerExposureHour => "per exposure hour",
    };
    let ink = RGBColor(23, 33, 43);
    let favorable = RGBColor(134, 197, 218);
    let unfavorable = RGBColor(230, 170, 114);
    let unresolved = RGBColor(213, 216, 220);
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (960, 700)).into_drawing_area();
        root.fill(&WHITE)?;
        let (chart_area, footer) = root.split_vertically(580);
        // Physical tick labels accompany normalized coordinates to bound rendering arithmetic.
        let mut chart = ChartBuilder::on(&chart_area)
            .caption("Conditional detector utility frontier", ("sans-serif", 24))
            .margin_top(18)
            .margin_right(45)
            .margin_bottom(12)
            .margin_left(18)
            .x_label_area_size(64)
            .y_label_area_size(125)
            .build_cartesian_2d(0.0_f64..1.0_f64, 0.0_f64..1.0_f64)?;
        let spacing = 1.0 / (args.grid_size - 1) as f64;
        chart.draw_series(cells.iter().enumerate().map(|(index, cell)| {
            let horizontal = (index % args.grid_size) as f64 * spacing;
            let vertical = (index / args.grid_size) as f64 * spacing;
            let color = match cell.region {
                Region::DeployConditional => favorable,
                Region::MaintainBaseline => unfavorable,
                Region::ManualReview => unresolved,
            };
            Rectangle::new(
                [
                    (
                        (horizontal - spacing / 2.0).max(0.0),
                        (vertical - spacing / 2.0).max(0.0),
                    ),
                    (
                        (horizontal + spacing / 2.0).min(1.0),
                        (vertical + spacing / 2.0).min(1.0),
                    ),
                ],
                color.filled(),
            )
        }))?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(5)
            .y_labels(5)
            .x_label_formatter(&|fraction| format!("{:.3e}", args.max_cost_ratio * fraction))
            .y_label_formatter(&|fraction| {
                format!("{:.3e}", args.min_overhead_shift + overhead_span * fraction)
            })
            .x_desc("False-alarm cost / true-detection benefit")
            .y_desc(format!("Additional normalized overhead shift ({unit})"))
            .axis_desc_style(("sans-serif", 15).into_font().color(&ink))
            .label_style(("sans-serif", 12).into_font().color(&ink))
            .axis_style(ink)
            .draw()?;

        // Clip the affine breakeven contour in physical units before normalizing.
        let left_shift = coefficients.a - coefficients.k;
        let right_shift = left_shift - args.max_cost_ratio * coefficients.b;
        ensure!(
            left_shift.is_finite() && right_shift.is_finite(),
            "contour overflow"
        );
        let mut intersections = Vec::new();
        for (ratio, shift) in [(0.0, left_shift), (args.max_cost_ratio, right_shift)] {
            if (args.min_overhead_shift..=args.max_overhead_shift).contains(&shift) {
                intersections.push((ratio, shift));
            }
        }
        if coefficients.b != 0.0 {
            for shift in [args.min_overhead_shift, args.max_overhead_shift] {
                let ratio = (left_shift - shift) / coefficients.b;
                if ratio.is_finite() && (0.0..=args.max_cost_ratio).contains(&ratio) {
                    intersections.push((ratio, shift));
                }
            }
        }
        intersections.sort_by(|left, right| left.0.total_cmp(&right.0));
        if let (Some(first), Some(last)) = (intersections.first(), intersections.last()) {
            let normalized = |&(ratio, shift): &(f64, f64)| {
                (
                    ratio / args.max_cost_ratio,
                    (shift - args.min_overhead_shift) / overhead_span,
                )
            };
            chart.draw_series(std::iter::once(PathElement::new(
                vec![normalized(first), normalized(last)],
                ink.stroke_width(2),
            )))?;
        }
        for (horizontal, color, label) in [
            (90, favorable, "Deploy candidate*"),
            (355, unfavorable, "Maintain Baseline"),
            (610, unresolved, "Manual Review"),
        ] {
            footer.draw(&Rectangle::new(
                [(horizontal, 10), (horizontal + 18, 28)],
                color.filled(),
            ))?;
            footer.draw(&Text::new(
                label,
                (horizontal + 26, 13),
                ("sans-serif", 14).into_font().color(&ink),
            ))?;
        }
        for (vertical, text) in [
            (
                48,
                "Line: point-estimate breakeven. Colors: adjudicated empirical inputs with at least 1000 paired draws.",
            ),
            (
                70,
                "*Candidate requires operational admission and owner valuations; other inputs stay Manual Review.",
            ),
        ] {
            footer.draw(&Text::new(
                text,
                (90, vertical),
                ("sans-serif", 13).into_font().color(&ink),
            ))?;
        }
        if matches!(request.evidence_kind, EvidenceKind::MathematicalFixture) {
            footer.draw(&Text::new(
                "MATHEMATICAL VALIDATION FIXTURE: these are not measured spacecraft results.",
                (90, 94),
                ("sans-serif", 13).into_font().color(&ink),
            ))?;
        }
        root.present()?;
    }
    Ok(svg)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = fs::read(&args.input).context("read utility request")?;
    let request: Request = serde_json::from_slice(&bytes).context("parse typed utility request")?;
    if matches!(request.evidence_kind, EvidenceKind::Empirical) {
        ensure!(
            !request.sources.is_empty(),
            "empirical input requires source receipts"
        );
        ensure!(
            !matches!(request.reference, Reference::MathematicalFixture),
            "empirical input requires an empirical reference"
        );
    }
    for source in &request.sources {
        validate_source_receipt(args.input.parent().unwrap_or(Path::new(".")), source)?;
    }
    let cells = calculate(&request, &args)?;
    let svg = plot(&request, &args, &cells)?;
    let mut csv = csv::Writer::from_writer(Vec::new());
    csv.write_record([
        "cost_ratio",
        "normalized_overhead_shift",
        "point_normalized_utility",
        "lower",
        "upper",
        "region",
    ])?;
    for cell in &cells {
        csv.write_record([
            cell.cost_ratio.to_string(),
            cell.normalized_overhead_shift.to_string(),
            cell.point_utility.to_string(),
            cell.interval
                .as_ref()
                .map(|bounds| bounds.lower.to_string())
                .unwrap_or_default(),
            cell.interval
                .as_ref()
                .map(|bounds| bounds.upper.to_string())
                .unwrap_or_default(),
            serde_json::to_value(cell.region)?
                .as_str()
                .context("region serialization")?
                .to_owned(),
        ])?;
    }
    let csv_bytes = csv.into_inner()?;
    let report = serde_json::json!({
        "schema_version": 1,
        "input_sha256": sha256(&bytes),
        "request": request,
        "unit": request.utility.unit(),
        "coefficients": request.utility.coefficients()?,
        "frontier_at_measured_overhead": request.utility.ratio_frontier(0.0)?,
        "grid": {"size": args.grid_size, "max_cost_ratio": args.max_cost_ratio, "min_overhead_shift": args.min_overhead_shift, "max_overhead_shift": args.max_overhead_shift},
        "inference_boundary": "Pointwise empirical 2.5/97.5 paired-draw percentiles; supplied resampling design determines coverage. Grid colors are not simultaneous confidence regions or permission to optimize on final data.",
        "plot_admission": "Interval-sign colors require empirical adjudicated inputs and at least 1000 paired draws. The draw floor is a computational safeguard, not a confidence-coverage guarantee. Every other input remains Manual Review.",
        "decision_boundary": "Conditional decision model only. Operational admission, independent reference quality and owner valuations require separate evidence. Missing uncertainty yields Manual Review.",
        "cells": cells,
    });
    let report_bytes = serde_json::to_vec_pretty(&report)?;
    publish_bundle(&args.out_dir, &report_bytes, &csv_bytes, svg.as_bytes())?;
    println!("Wrote conditional frontier to {}", args.out_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_receipts_require_normalized_contained_paths_and_matching_bytes() {
        let parent =
            std::env::temp_dir().join(format!("gororoba-utility-receipts-{}", std::process::id()));
        fs::create_dir(&parent).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let _cleanup = Cleanup(parent.clone());
        let root = parent.join("request");
        fs::create_dir_all(root.join("data")).unwrap();
        fs::write(root.join("data/evidence"), b"paired counts").unwrap();
        fs::write(parent.join("outside"), b"paired counts").unwrap();
        let receipt = |path: PathBuf| SourceReceipt {
            path,
            sha256: sha256(b"paired counts"),
        };
        validate_source_receipt(&root, &receipt(PathBuf::from("data/evidence"))).unwrap();
        for path in [
            parent.join("outside"),
            PathBuf::from("../outside"),
            PathBuf::from("data/../data/evidence"),
            PathBuf::from("./data/evidence"),
            PathBuf::from("data//evidence"),
            PathBuf::from("data/./evidence"),
            PathBuf::new(),
        ] {
            let error = validate_source_receipt(&root, &receipt(path)).unwrap_err();
            assert!(error.to_string().contains("normalized relative path"));
        }
        let mut mismatch = receipt(PathBuf::from("data/evidence"));
        mismatch.sha256 = sha256(b"different bytes");
        assert!(validate_source_receipt(&root, &mismatch).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(parent.join("outside"), root.join("escape")).unwrap();
            let error =
                validate_source_receipt(&root, &receipt(PathBuf::from("escape"))).unwrap_err();
            assert!(error.to_string().contains("outside request directory"));
        }
    }

    fn fixture() -> Request {
        serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "evidence_kind": "mathematical_fixture",
            "reference": "mathematical_fixture",
            "measurement_boundary": "Arithmetic validation only",
            "sources": [],
            "utility": {
                "accounting": {"kind": "event", "exposure_hours": 1.0,
                    "true_events": 0,
                    "baseline": {"true_positives": 0, "false_positives": 0},
                    "augmented": {"true_positives": 0, "false_positives": 0}},
                "benefit_per_true_detection": 1.0,
                "additional_overhead": 0.0,
                "paired_draws": null
            }
        }))
        .unwrap()
    }

    #[test]
    fn finite_large_axes_avoid_intermediate_overflow() {
        let request = fixture();
        let args = Args {
            input: PathBuf::new(),
            out_dir: PathBuf::new(),
            max_cost_ratio: 1e308,
            min_overhead_shift: 0.0,
            max_overhead_shift: 1e308,
            grid_size: 3,
        };
        let cells = calculate(&request, &args).unwrap();
        assert_eq!(cells.len(), 9);
        assert!(cells.iter().all(|cell| cell.region == Region::ManualReview));
        let svg = plot(&request, &args, &cells).unwrap();
        assert!(svg.contains("MATHEMATICAL VALIDATION FIXTURE"));
        assert!(!svg.contains("NaN"));
        assert!(!svg.contains("inf"));
    }

    #[test]
    fn invalid_axis_fails_before_output() {
        let args = Args {
            input: PathBuf::new(),
            out_dir: PathBuf::new(),
            max_cost_ratio: 1.0,
            min_overhead_shift: 2.0,
            max_overhead_shift: 1.0,
            grid_size: 3,
        };
        assert!(calculate(&fixture(), &args).is_err());
    }

    #[test]
    fn uncertainty_regions_require_strict_bounds() {
        let interval = |lower, upper| UtilityInterval {
            lower,
            upper,
            draws: 2000,
            lower_quantile: 0.025,
            upper_quantile: 0.975,
        };
        assert_eq!(region(None), Region::ManualReview);
        assert_eq!(region(Some(&interval(0.0, 1.0))), Region::ManualReview);
        assert_eq!(region(Some(&interval(-1.0, 0.0))), Region::ManualReview);
        assert_eq!(region(Some(&interval(0.1, 1.0))), Region::DeployConditional);
        assert_eq!(
            region(Some(&interval(-1.0, -0.1))),
            Region::MaintainBaseline
        );
    }
}
