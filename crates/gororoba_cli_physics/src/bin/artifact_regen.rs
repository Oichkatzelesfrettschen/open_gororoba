use algebra_analysis::{
    annihilator::annihilator_info,
    boxkites::{cached_sedenion_boxkites, diagonal_zero_products_exact, primitive_assessors},
    homotopy_algebra::SedenionAInfinity,
    reggiani::{standard_zero_divisor_partners, standard_zero_divisors},
};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use csv::{ReaderBuilder, WriterBuilder};
use data_core::catalogs::jarvis;
use nalgebra::{DMatrix, SVD};
use plotters::{coord::types::RangedCoordf64, prelude::*};
use statrs::function::gamma::gamma;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const WIDTH: u32 = 3160;
const HEIGHT: u32 = 2820;
const BACKGROUND: RGBColor = RGBColor(13, 15, 20);
const GRID: RGBColor = RGBColor(55, 65, 81);
const CYAN: RGBColor = RGBColor(0, 247, 255);
const AMBER: RGBColor = RGBColor(255, 176, 0);
const MAGENTA: RGBColor = RGBColor(255, 92, 255);
const TEXT: RGBColor = RGBColor(245, 247, 250);

#[derive(Parser, Debug)]
#[command(
    name = "artifact-regen",
    about = "Rust-native regeneration of deterministic artifact CSV/PNG outputs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    DimensionalGeometry,
    MaterialsSubset(MaterialsSubsetArgs),
    MaterialsEmbedding(MaterialsEmbeddingArgs),
    DeMarraisBoxkites,
    ReggianiAnnihilatorStats,
    M3Table,
    MotifSummary,
}

#[derive(Parser, Debug)]
struct MaterialsSubsetArgs {
    #[arg(long, default_value = "data/external/jarvis_dft_3d.json")]
    input: PathBuf,
    #[arg(long, default_value = "data/csv/materials_jarvis_subset.csv")]
    output: PathBuf,
    #[arg(long, default_value_t = 200)]
    n: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Parser, Debug)]
struct MaterialsEmbeddingArgs {
    #[arg(long, default_value = "data/csv/materials_jarvis_subset.csv")]
    input: PathBuf,
    #[arg(long, default_value = "data/csv/materials_embedding_benchmarks.csv")]
    output: PathBuf,
    #[arg(long, default_value = "data/artifacts/images")]
    image_dir: PathBuf,
}

#[derive(Clone)]
struct MaterialRow {
    formula: String,
    formation_energy_peratom: f64,
    optb88vdw_bandgap: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::DimensionalGeometry => generate_dimensional_geometry(),
        Command::MaterialsSubset(args) => generate_materials_subset(&args),
        Command::MaterialsEmbedding(args) => generate_materials_embedding(&args),
        Command::DeMarraisBoxkites => generate_de_marrais_boxkite_artifacts(),
        Command::ReggianiAnnihilatorStats => generate_reggiani_artifacts(),
        Command::M3Table => generate_m3_table(),
        Command::MotifSummary => generate_motif_summary(),
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent {}", parent.display()))?;
    }
    Ok(())
}

fn plot_error<E: std::fmt::Display>(err: E) -> anyhow::Error {
    anyhow::anyhow!(err.to_string())
}

fn configure_mesh<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<()> {
    chart
        .configure_mesh()
        .label_style(("sans-serif", 24).into_font().color(&TEXT))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(2))
        .light_line_style(ShapeStyle::from(&GRID).stroke_width(1))
        .bold_line_style(ShapeStyle::from(&GRID.mix(0.7)).stroke_width(2))
        .draw()
        .map_err(plot_error)
}

fn generate_dimensional_geometry() -> Result<()> {
    render_dimensional_geometry(
        -4.0,
        16.0,
        Path::new("data/csv/dimensional_geometry_-4_to_16.csv"),
        Path::new("data/artifacts/images/dimensional_geometry_-4_to_16.png"),
    )?;
    render_dimensional_geometry(
        0.0,
        32.0,
        Path::new("data/csv/dimensional_geometry_0_to_32.csv"),
        Path::new("data/artifacts/images/dimensional_geometry_0_to_32.png"),
    )?;
    Ok(())
}

fn render_dimensional_geometry(
    d_min: f64,
    d_max: f64,
    out_csv: &Path,
    out_png: &Path,
) -> Result<()> {
    ensure_parent(out_csv)?;
    ensure_parent(out_png)?;
    let samples = 4001usize;
    let step = (d_max - d_min) / (samples as f64 - 1.0);
    let mut rows = Vec::with_capacity(samples);
    for idx in 0..samples {
        let d = d_min + idx as f64 * step;
        let volume = safe_ball_volume(d);
        let area = safe_sphere_area(d);
        rows.push((d, volume, area));
    }

    let mut writer = WriterBuilder::new().from_path(out_csv)?;
    writer.write_record(["d", "ball_volume_r1", "unit_sphere_area"])?;
    for (d, volume, area) in &rows {
        writer.serialize((d, volume, area))?;
    }
    writer.flush()?;

    let root = BitMapBackend::new(out_png, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;
    let areas = root.split_evenly((2, 2));
    let finite_volume: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|(d, v, _)| v.is_finite().then_some((*d, *v)))
        .collect();
    let finite_area: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|(d, _, a)| a.is_finite().then_some((*d, *a)))
        .collect();
    let log_volume: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|(d, v, _)| {
            (v.is_finite() && *v != 0.0).then_some((*d, (v.abs() + 1e-30).log10()))
        })
        .collect();
    let log_area: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|(d, _, a)| {
            (a.is_finite() && *a != 0.0).then_some((*d, (a.abs() + 1e-30).log10()))
        })
        .collect();

    draw_line_panel(
        &areas[0],
        &finite_volume,
        d_min,
        d_max,
        "Ball volume V_d(1)",
        CYAN,
    )?;
    draw_line_panel(
        &areas[1],
        &finite_area,
        d_min,
        d_max,
        "Sphere area S_{d-1}(1)",
        AMBER,
    )?;
    draw_line_panel(&areas[2], &log_volume, d_min, d_max, "log10 |V_d(1)|", CYAN)?;
    draw_line_panel(
        &areas[3],
        &log_area,
        d_min,
        d_max,
        "log10 |S_{d-1}(1)|",
        AMBER,
    )?;
    root.present().map_err(plot_error)?;
    println!("WROTE {}", out_csv.display());
    println!("WROTE {}", out_png.display());
    Ok(())
}

fn draw_line_panel(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    data: &[(f64, f64)],
    x_min: f64,
    x_max: f64,
    title: &str,
    color: RGBColor,
) -> Result<()> {
    let (y_min, y_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, (_, y)| {
            (acc.0.min(*y), acc.1.max(*y))
        });
    let pad = ((y_max - y_min).abs()).max(1e-6) * 0.05;
    let mut chart = ChartBuilder::on(area)
        .margin(30)
        .caption(title, ("sans-serif", 28).into_font().color(&TEXT))
        .x_label_area_size(60)
        .y_label_area_size(90)
        .build_cartesian_2d(x_min..x_max, (y_min - pad)..(y_max + pad))
        .map_err(plot_error)?;
    configure_mesh(&mut chart)?;
    chart
        .draw_series(LineSeries::new(
            data.iter().copied(),
            ShapeStyle::from(&color).stroke_width(4),
        ))
        .map_err(plot_error)?;
    Ok(())
}

fn safe_ball_volume(d: f64) -> f64 {
    safe_gamma_expression(d, true)
}

fn safe_sphere_area(d: f64) -> f64 {
    safe_gamma_expression(d, false)
}

fn safe_gamma_expression(d: f64, volume: bool) -> f64 {
    let half = d / 2.0;
    let denom_arg = if volume { half + 1.0 } else { half };
    if is_gamma_pole(denom_arg) {
        return f64::NAN;
    }
    let numerator = if volume {
        std::f64::consts::PI.powf(half)
    } else {
        2.0 * std::f64::consts::PI.powf(half)
    };
    let denom = gamma(denom_arg);
    let value = numerator / denom;
    if value.is_finite() { value } else { f64::NAN }
}

fn is_gamma_pole(x: f64) -> bool {
    x <= 0.0 && (x.round() - x).abs() < 1e-9
}

fn generate_materials_subset(args: &MaterialsSubsetArgs) -> Result<()> {
    ensure_parent(&args.output)?;
    let materials = jarvis::parse_jarvis_json(&args.input)
        .with_context(|| format!("parse {}", args.input.display()))?;
    let sampled = data_core::catalogs::jarvis::sample_materials(&materials, args.n, args.seed);
    let mut writer = WriterBuilder::new().from_path(&args.output)?;
    writer.write_record([
        "jid",
        "formula",
        "formation_energy_peratom",
        "optb88vdw_bandgap",
        "nelements",
        "density",
        "volume",
    ])?;
    for row in sampled {
        writer.serialize((
            row.jid,
            row.formula,
            row.formation_energy_peratom.unwrap_or(f64::NAN),
            row.optb88vdw_bandgap.unwrap_or(f64::NAN),
            row.nelements,
            row.density.unwrap_or(f64::NAN),
            row.volume.unwrap_or(f64::NAN),
        ))?;
    }
    writer.flush()?;
    println!("WROTE {}", args.output.display());
    Ok(())
}

fn generate_materials_embedding(args: &MaterialsEmbeddingArgs) -> Result<()> {
    ensure_parent(&args.output)?;
    fs::create_dir_all(&args.image_dir)?;
    let rows = read_material_rows(&args.input)?;
    if rows.is_empty() {
        bail!("no usable material rows found in {}", args.input.display());
    }

    let data = build_composition_matrix(&rows);
    let centered = center_columns(&data);
    let svd = SVD::new(centered.clone(), false, true);
    let v_t = svd.v_t.context("materials PCA missing V^T")?;
    let singular = svd.singular_values;
    let total_var: f64 = singular
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .max(1e-12);

    let formation: Vec<f64> = rows
        .iter()
        .map(|row| row.formation_energy_peratom)
        .collect();
    let band_gap: Vec<f64> = rows.iter().map(|row| row.optb88vdw_bandgap).collect();

    let mut writer = WriterBuilder::new().from_path(&args.output)?;
    writer.write_record([
        "k",
        "spearman_distance_preservation",
        "explained_variance_ratio_sum",
    ])?;

    for &k in &[4usize, 8, 16, 32] {
        let use_k = k.min(v_t.nrows()).max(2);
        let basis = v_t.rows(0, use_k).transpose().into_owned();
        let scores = &centered * basis;
        let hi = matrix_rows(&centered);
        let lo = matrix_rows(&scores);
        let rho = spearman_distance_preservation(&hi, &lo, 5000);
        let explained = singular
            .iter()
            .take(use_k)
            .map(|value| value * value / total_var)
            .sum::<f64>();
        writer.serialize((k, rho, explained))?;
        let image_path = args.image_dir.join(format!("materials_pca_{k}d.png"));
        render_material_scatter(
            &image_path,
            &scores,
            &formation,
            &band_gap,
            &format!("Materials PCA projection (118D -> {k}D -> 2D)"),
        )?;
    }
    writer.flush()?;
    println!("WROTE {}", args.output.display());
    Ok(())
}

fn read_material_rows(path: &Path) -> Result<Vec<MaterialRow>> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let mut rows = Vec::new();
    for record in reader.deserialize::<BTreeMap<String, String>>() {
        let record = record?;
        let Some(formula) = record
            .get("formula")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        rows.push(MaterialRow {
            formula: formula.to_string(),
            formation_energy_peratom: parse_f64(record.get("formation_energy_peratom")),
            optb88vdw_bandgap: parse_f64(record.get("optb88vdw_bandgap")),
        });
    }
    Ok(rows)
}

fn parse_f64(value: Option<&String>) -> f64 {
    value
        .and_then(|item| item.parse::<f64>().ok())
        .filter(|item| item.is_finite())
        .unwrap_or(0.0)
}

fn build_composition_matrix(rows: &[MaterialRow]) -> DMatrix<f64> {
    let mut data = Vec::with_capacity(rows.len() * 118);
    for row in rows {
        data.extend(composition_vector(&row.formula));
    }
    DMatrix::from_row_slice(rows.len(), 118, &data)
}

fn composition_vector(formula: &str) -> [f64; 118] {
    let elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
        "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ];
    let mut vector = [0.0; 118];
    let mut total = 0.0;
    let bytes = formula.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if !(bytes[index] as char).is_ascii_uppercase() {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        if index < bytes.len() && (bytes[index] as char).is_ascii_lowercase() {
            index += 1;
        }
        let symbol = &formula[start..index];
        let count_start = index;
        while index < bytes.len() && (bytes[index] as char).is_ascii_digit() {
            index += 1;
        }
        let count = if count_start < index {
            formula[count_start..index].parse::<f64>().unwrap_or(1.0)
        } else {
            1.0
        };
        if let Some(pos) = elements.iter().position(|item| item == &symbol) {
            vector[pos] += count;
            total += count;
        }
    }
    if total > 0.0 {
        for value in &mut vector {
            *value /= total;
        }
    }
    vector
}

fn center_columns(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut centered = matrix.clone();
    for col in 0..centered.ncols() {
        let mean = centered.column(col).iter().sum::<f64>() / centered.nrows() as f64;
        for row in 0..centered.nrows() {
            centered[(row, col)] -= mean;
        }
    }
    centered
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| matrix.row(row).iter().copied().collect::<Vec<_>>())
        .collect()
}

fn spearman_distance_preservation(hi: &[Vec<f64>], lo: &[Vec<f64>], n_pairs: usize) -> f64 {
    let n = hi.len();
    if n < 3 {
        return f64::NAN;
    }
    let mut hi_dist = Vec::new();
    let mut lo_dist = Vec::new();
    for idx in 0..n_pairs {
        let a = idx % n;
        let b = (idx * 37 + 17) % n;
        if a == b {
            continue;
        }
        hi_dist.push(euclidean(&hi[a], &hi[b]));
        lo_dist.push(euclidean(&lo[a], &lo[b]));
    }
    pearson(&rank(&hi_dist), &rank(&lo_dist))
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn rank(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; values.len()];
    for (rank, (index, _)) in indexed.iter().enumerate() {
        ranks[*index] = rank as f64;
    }
    ranks
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }
    num / (den_a.sqrt() * den_b.sqrt()).max(1e-12)
}

fn render_material_scatter(
    path: &Path,
    scores: &DMatrix<f64>,
    formation: &[f64],
    band_gap: &[f64],
    title: &str,
) -> Result<()> {
    ensure_parent(path)?;
    let x_values: Vec<f64> = scores.column(0).iter().copied().collect();
    let y_values: Vec<f64> = scores.column(1).iter().copied().collect();
    let (x_min, x_max) = bounds(&x_values);
    let (y_min, y_max) = bounds(&y_values);
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .caption(title, ("sans-serif", 32).into_font().color(&TEXT))
        .x_label_area_size(70)
        .y_label_area_size(80)
        .build_cartesian_2d((x_min - 0.1)..(x_max + 0.1), (y_min - 0.1)..(y_max + 0.1))
        .map_err(plot_error)?;
    configure_mesh(&mut chart)?;

    let (c_min, c_max) = bounds(formation);
    let (s_min, s_max) = bounds(band_gap);
    chart
        .draw_series((0..scores.nrows()).map(|idx| {
            let x = scores[(idx, 0)];
            let y = scores[(idx, 1)];
            let color = gradient_color(formation[idx], c_min, c_max);
            let radius = scale_radius(band_gap[idx], s_min, s_max);
            Circle::new((x, y), radius, ShapeStyle::from(&color).filled())
        }))
        .map_err(plot_error)?;
    root.present().map_err(plot_error)?;
    println!("WROTE {}", path.display());
    Ok(())
}

fn gradient_color(value: f64, min: f64, max: f64) -> RGBColor {
    let t = if (max - min).abs() < 1e-12 {
        0.5
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    };
    let r = (68.0 + 187.0 * t) as u8;
    let g = (1.0 + 200.0 * (1.0 - (t - 0.5).abs() * 2.0)) as u8;
    let b = (84.0 + 150.0 * (1.0 - t)) as u8;
    RGBColor(r, g, b)
}

fn scale_radius(value: f64, min: f64, max: f64) -> i32 {
    let t = if (max - min).abs() < 1e-12 {
        0.5
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    };
    (4.0 + 10.0 * t) as i32
}

fn bounds(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            min = min.min(*value);
            max = max.max(*value);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        (-1.0, 1.0)
    } else {
        (min, max)
    }
}

fn generate_de_marrais_boxkite_artifacts() -> Result<()> {
    fs::create_dir_all("data/csv")?;
    let assessors = primitive_assessors();
    let boxkites = cached_sedenion_boxkites();

    let mut assessors_writer =
        WriterBuilder::new().from_path("data/csv/de_marrais_assessors.csv")?;
    assessors_writer.write_record(["low", "high"])?;
    for assessor in &assessors {
        assessors_writer.serialize((assessor.low, assessor.high))?;
    }
    assessors_writer.flush()?;

    let mut boxkite_writer = WriterBuilder::new().from_path("data/csv/de_marrais_boxkites.csv")?;
    boxkite_writer.write_record(["box_kite", "strut_signature", "low", "high"])?;
    for boxkite in boxkites {
        for assessor in &boxkite.assessors {
            boxkite_writer.serialize((
                boxkite.id,
                boxkite.strut_signature,
                assessor.low,
                assessor.high,
            ))?;
        }
    }
    boxkite_writer.flush()?;

    let mut edge_writer =
        WriterBuilder::new().from_path("data/csv/de_marrais_boxkite_edges.csv")?;
    edge_writer.write_record([
        "box_kite",
        "strut_signature",
        "a_low",
        "a_high",
        "b_low",
        "b_high",
        "edge_type",
        "sign_solutions",
    ])?;
    for boxkite in boxkites {
        for (a_idx, b_idx) in &boxkite.edges {
            let a = boxkite.assessors[*a_idx];
            let b = boxkite.assessors[*b_idx];
            let solutions = diagonal_zero_products_exact(16, (a.low, a.high), (b.low, b.high))
                .into_iter()
                .map(|(s, t)| format!("{s:+},{t:+}"))
                .collect::<Vec<_>>()
                .join(";");
            edge_writer.serialize((
                boxkite.id,
                boxkite.strut_signature,
                a.low,
                a.high,
                b.low,
                b.high,
                "coassessor",
                solutions,
            ))?;
        }
    }
    edge_writer.flush()?;

    let mut strut_writer = WriterBuilder::new().from_path("data/csv/de_marrais_strut_table.csv")?;
    strut_writer.write_record([
        "box_kite",
        "strut_signature",
        "A_low",
        "A_high",
        "B_low",
        "B_high",
        "C_low",
        "C_high",
        "D_low",
        "D_high",
        "E_low",
        "E_high",
        "F_low",
        "F_high",
    ])?;
    for boxkite in boxkites {
        let mut assessors = boxkite.assessors.clone();
        assessors.sort_by_key(|assessor| (assessor.low, assessor.high));
        let mut row = vec![boxkite.id.to_string(), boxkite.strut_signature.to_string()];
        for assessor in assessors {
            row.push(assessor.low.to_string());
            row.push(assessor.high.to_string());
        }
        strut_writer.write_record(row)?;
    }
    strut_writer.flush()?;
    println!("WROTE data/csv/de_marrais_*");
    Ok(())
}

fn generate_reggiani_artifacts() -> Result<()> {
    fs::create_dir_all("data/csv")?;
    fs::create_dir_all("data/artifacts/images")?;
    let zds = standard_zero_divisors();
    let mut writer =
        WriterBuilder::new().from_path("data/csv/reggiani_standard_zero_divisors.csv")?;
    writer.write_record([
        "assessor_low",
        "assessor_high",
        "diagonal_sign",
        "left_nullity",
        "right_nullity",
    ])?;
    let mut distribution: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for zd in &zds {
        let info = annihilator_info(&zd.vector, 16, 1.0e-12);
        *distribution
            .entry((info.left_nullity, info.right_nullity))
            .or_default() += 1;
        writer.serialize((
            zd.assessor_low,
            zd.assessor_high,
            zd.diagonal_sign,
            info.left_nullity,
            info.right_nullity,
        ))?;
    }
    writer.flush()?;

    let mut pair_writer =
        WriterBuilder::new().from_path("data/csv/reggiani_standard_zero_divisor_pairs.csv")?;
    pair_writer.write_record(["u_low", "u_high", "u_sign", "v_low", "v_high", "v_sign"])?;
    for zd in &zds {
        for partner in standard_zero_divisor_partners(zd) {
            pair_writer.serialize((
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
                partner.assessor_low,
                partner.assessor_high,
                partner.diagonal_sign,
            ))?;
        }
    }
    pair_writer.flush()?;

    let mut dist_writer =
        WriterBuilder::new().from_path("data/csv/reggiani_annihilator_nullity_distribution.csv")?;
    dist_writer.write_record(["left_nullity", "right_nullity", "count"])?;
    for ((left, right), count) in &distribution {
        dist_writer.serialize((left, right, count))?;
    }
    dist_writer.flush()?;

    let image_path = Path::new("data/artifacts/images/reggiani_annihilator_nullity_3160x2820.png");
    ensure_parent(image_path)?;
    let root = BitMapBackend::new(image_path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .caption(
            "Reggiani standard zero-divisor annihilator nullity distribution",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..8.0, 0.0..(zds.len() as f64 + 4.0))
        .map_err(plot_error)?;
    configure_mesh(&mut chart)?;
    for ((left, _right), count) in &distribution {
        let x0 = *left as f64 - 0.3;
        let x1 = *left as f64 + 0.3;
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, *count as f64)],
                ShapeStyle::from(&MAGENTA.mix(0.8)).filled(),
            )))
            .map_err(plot_error)?;
    }
    root.present().map_err(plot_error)?;
    println!("WROTE data/csv/reggiani_*");
    println!("WROTE {}", image_path.display());
    Ok(())
}

fn generate_m3_table() -> Result<()> {
    fs::create_dir_all("data/csv")?;
    let ainf = SedenionAInfinity::new(16);
    let mut writer = WriterBuilder::new().from_path("data/csv/m3_table.csv")?;
    writer.write_record(["i", "j", "k", "kind", "index", "value"])?;
    for i in 0..16 {
        let e_i = basis_vector(i);
        for j in 0..16 {
            let e_j = basis_vector(j);
            for k in 0..16 {
                let e_k = basis_vector(k);
                let assoc = ainf.m3(&e_i, &e_j, &e_k);
                for (index, value) in assoc.iter().enumerate() {
                    if value.abs() > 1.0e-12 {
                        writer.serialize((i, j, k, "component", index, value))?;
                    }
                }
            }
        }
    }
    writer.flush()?;
    println!("WROTE data/csv/m3_table.csv");
    Ok(())
}

fn basis_vector(index: usize) -> Vec<f64> {
    let mut out = vec![0.0; 16];
    out[index] = 1.0;
    out
}

fn generate_motif_summary() -> Result<()> {
    fs::create_dir_all("data/csv")?;
    fs::create_dir_all("data/artifacts/images")?;
    let dims = [16usize, 32, 64, 128, 256];
    let mut rows = Vec::new();
    for dim in dims {
        let path = PathBuf::from(format!("data/csv/cd_motif_components_{dim}d.csv"));
        if !path.exists() {
            continue;
        }
        let mut reader = ReaderBuilder::new().from_path(&path)?;
        let headers = reader.headers()?.clone();
        let mut component_count = 0usize;
        let mut active_nodes_total = 0usize;
        let mut max_component_nodes = 0usize;
        let mut max_component_edges = 0usize;
        let mut octahedron_k222_count = 0usize;
        let mut cuboctahedron_count = 0usize;
        let mut k2_multipartite_max_parts = 0usize;
        let mut sampled = false;
        let mut sample_max_nodes = 0usize;
        let mut seed = 0usize;
        for record in reader.records() {
            let record = record?;
            component_count += 1;
            active_nodes_total += parse_csv_usize(&headers, &record, "node_count");
            max_component_nodes =
                max_component_nodes.max(parse_csv_usize(&headers, &record, "node_count"));
            max_component_edges =
                max_component_edges.max(parse_csv_usize(&headers, &record, "edge_count"));
            octahedron_k222_count +=
                parse_csv_bool(&headers, &record, "is_octahedron_k222") as usize;
            cuboctahedron_count += parse_csv_bool(&headers, &record, "is_cuboctahedron") as usize;
            k2_multipartite_max_parts = k2_multipartite_max_parts.max(parse_csv_usize(
                &headers,
                &record,
                "k2_multipartite_part_count",
            ));
            sampled |= parse_csv_bool(&headers, &record, "sampled");
            sample_max_nodes =
                sample_max_nodes.max(parse_csv_usize(&headers, &record, "sample_max_nodes"));
            seed = seed.max(parse_csv_usize(&headers, &record, "seed"));
        }
        rows.push((
            dim,
            component_count,
            active_nodes_total,
            max_component_nodes,
            max_component_edges,
            octahedron_k222_count,
            cuboctahedron_count,
            k2_multipartite_max_parts,
            sampled,
            sample_max_nodes,
            seed,
        ));
    }

    let mut writer = WriterBuilder::new().from_path("data/csv/cd_motif_summary_by_dim.csv")?;
    writer.write_record([
        "dim",
        "component_count",
        "active_nodes_total",
        "max_component_nodes",
        "max_component_edges",
        "octahedron_k222_count",
        "cuboctahedron_count",
        "k2_multipartite_max_parts",
        "sampled",
        "sample_max_nodes",
        "seed",
    ])?;
    for row in &rows {
        writer.serialize(row)?;
    }
    writer.flush()?;

    let image_path = Path::new("data/artifacts/images/cd_motif_max_component_nodes_3160x2820.png");
    ensure_parent(image_path)?;
    let root = BitMapBackend::new(image_path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;
    let max_nodes = rows.iter().map(|row| row.3).max().unwrap_or(1) as f64;
    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .caption(
            "Cayley-Dickson motif census: max component nodes by dimension",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .x_label_area_size(80)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0..300.0, 0.0..(max_nodes * 1.2 + 1.0))
        .map_err(plot_error)?;
    configure_mesh(&mut chart)?;
    for row in &rows {
        let x = row.0 as f64;
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x - 8.0, 0.0), (x + 8.0, row.3 as f64)],
                ShapeStyle::from(&CYAN.mix(0.85)).filled(),
            )))
            .map_err(plot_error)?;
    }
    root.present().map_err(plot_error)?;
    println!("WROTE data/csv/cd_motif_summary_by_dim.csv");
    println!("WROTE {}", image_path.display());
    Ok(())
}

fn parse_csv_usize(headers: &csv::StringRecord, record: &csv::StringRecord, column: &str) -> usize {
    headers
        .iter()
        .position(|header| header == column)
        .and_then(|idx| record.get(idx))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

fn parse_csv_bool(headers: &csv::StringRecord, record: &csv::StringRecord, column: &str) -> bool {
    headers
        .iter()
        .position(|header| header == column)
        .and_then(|idx| record.get(idx))
        .map(|value| value.eq_ignore_ascii_case("true") || value == "1")
        .unwrap_or(false)
}
