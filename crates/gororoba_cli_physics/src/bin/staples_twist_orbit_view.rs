//! E-279 viewer: six samples of B in R^3 (hodogram), the 16D lag-channel
//! fiber, and the 96-term twist ladder.
//!
//! THEMIS FGM is (t, Bx, By, Bz). There is no light cone and the LM plane
//! is a derived frame, not the measurement. The hodogram is B-space.
//!
//! ```bash
//! cargo run --profile validation -p gororoba_cli_physics --bin staples-twist-orbit-view
//! cargo run --profile validation -p gororoba_cli_physics --bin staples-twist-orbit-view -- --png data/artifacts/images/science_staples_twist_orbit_plate_3160x2820.png
//! ```

use anyhow::{Context, Result, bail};
use cd_kernel::mult_table::CdMultTable;
use clap::Parser;
use gororoba_cli_physics::staple_associator::STAPLE_DIM;
use gororoba_cli_physics::staple_controls::{
    PG32_LINE_COUNT, SparseCubicTensor, cd_twist, extract_line_orientations,
    line_octonion_incidence, octonion_plane_count, pg32_lines, twist_from_line_orientations,
};
use gororoba_cli_physics::staple_physical::helical_field;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use plotters::prelude::*;
use serde::Deserialize;
use std::{fs, path::PathBuf};

const PLATE_W: u32 = 3160;
const PLATE_H: u32 = 2820;
const VIEW_W: usize = 1680;
const VIEW_H: usize = 1050;
const BACKGROUND: RGBColor = RGBColor(8, 10, 16);
const PANEL: RGBColor = RGBColor(16, 20, 28);
const GRID: RGBColor = RGBColor(48, 58, 74);
const TEXT: RGBColor = RGBColor(236, 241, 247);
const MUTED: RGBColor = RGBColor(140, 154, 172);
const CYAN: RGBColor = RGBColor(94, 210, 255);
const AMBER: RGBColor = RGBColor(255, 186, 73);
const EMERALD: RGBColor = RGBColor(52, 211, 153);
const ROSE: RGBColor = RGBColor(255, 92, 122);
const GOLD: RGBColor = RGBColor(250, 204, 21);

#[derive(Parser, Debug)]
#[command(about = "Interactive E-279 PG(3,2) / twist-orbit viewer")]
struct Args {
    #[arg(long, default_value = "data/output/staples_twist_orbit.json")]
    input: PathBuf,
    #[arg(long)]
    png: Option<PathBuf>,
    #[arg(long, default_value_t = VIEW_W)]
    width: usize,
    #[arg(long, default_value_t = VIEW_H)]
    height: usize,
}

#[derive(Deserialize)]
struct OrbitFile {
    canonical: Canonical,
    invariant_matched_twists: Ensemble,
    isomorphic_orbit: Ensemble,
    invariant_twist_draws: Vec<Draw>,
    isomorphic_orbit_draws: Vec<Draw>,
}

#[derive(Deserialize)]
struct Canonical {
    subsample_auc: f64,
    term_count: usize,
}

#[derive(Deserialize)]
struct Ensemble {
    n: usize,
    mean: f64,
    p2_5: f64,
    p97_5: f64,
    max: f64,
    draws_at_or_above_canonical: usize,
    term_count_min: usize,
    term_count_max: usize,
}

#[derive(Deserialize, Clone)]
struct Draw {
    index: usize,
    auc: f64,
    term_count: usize,
}

struct Scene {
    data: OrbitFile,
    lines: [(usize, usize, usize); PG32_LINE_COUNT],
    cd_signs: [i8; PG32_LINE_COUNT],
    live_signs: [i8; PG32_LINE_COUNT],
    live_terms: usize,
    octonion_planes: usize,
    line_drop: [usize; PG32_LINE_COUNT],
    line_inc: [usize; PG32_LINE_COUNT],
    selected_line: usize,
    hodogram: Vec<[f64; 3]>,
    yaw: f64,
    pitch: f64,
    drag: Option<(f32, f32)>,
    hover: String,
}

fn plot_err(err: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("plotters: {err}")
}

fn rgb_to_argb(buf: &[u8], pixels: &mut [u32]) {
    for (dst, chunk) in pixels.iter_mut().zip(buf.chunks_exact(3)) {
        *dst = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
    }
}

fn term_count_of(
    lines: &[(usize, usize, usize); PG32_LINE_COUNT],
    signs: &[i8; PG32_LINE_COUNT],
) -> usize {
    SparseCubicTensor::from_twist(&twist_from_line_orientations(lines, signs)).term_count()
}

fn k_level(terms: usize) -> i32 {
    ((terms as i32) - 1080) / 96
}

fn lag_of(index: usize) -> usize {
    index / 4
}

fn channel_of(index: usize) -> usize {
    index % 4
}

fn cell_xy(index: usize) -> (f64, f64) {
    (channel_of(index) as f64 + 0.5, 3.5 - lag_of(index) as f64)
}

fn closed_rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
}

fn cell_frame(index: usize) -> Vec<(f64, f64)> {
    let (x, y) = cell_xy(index);
    closed_rect(x - 0.42, y - 0.42, x + 0.42, y + 0.42)
}

fn near_half(value: f64) -> Option<i32> {
    let shifted = value - 0.5;
    let rounded = shifted.round();
    if (shifted - rounded).abs() < 0.08 {
        Some(rounded as i32)
    } else {
        None
    }
}

fn channel_tick(x: f64) -> String {
    match near_half(x) {
        Some(0) => "Bx".to_string(),
        Some(1) => "By".to_string(),
        Some(2) => "Bz".to_string(),
        Some(3) => "|B|".to_string(),
        _ => String::new(),
    }
}

fn lag_tick(y: f64) -> String {
    match near_half(y) {
        Some(cell) => {
            let lag = 3 - cell;
            if (0..4).contains(&lag) {
                format!("t+{lag}")
            } else {
                String::new()
            }
        }
        None => String::new(),
    }
}

fn project_b(b: [f64; 3], yaw: f64, pitch: f64) -> (f64, f64) {
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let x1 = b[0] * cy - b[2] * sy;
    let z1 = b[0] * sy + b[2] * cy;
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let y2 = b[1] * cp - z1 * sp;
    let z2 = b[1] * sp + z1 * cp;
    let depth = 3.4 + z2 * 0.4;
    (x1 / depth, y2 / depth)
}

fn build_scene(data: OrbitFile) -> Scene {
    let lines = pg32_lines();
    let table = CdMultTable::generate(STAPLE_DIM);
    let cd = cd_twist(&table);
    let cd_signs = extract_line_orientations(&cd, &lines);
    let mut line_drop = [0usize; PG32_LINE_COUNT];
    let mut line_inc = [0usize; PG32_LINE_COUNT];
    for (k, &line) in lines.iter().enumerate() {
        line_inc[k] = line_octonion_incidence(&cd, line);
        let mut signs = cd_signs;
        signs[k] = -signs[k];
        line_drop[k] = 1848 - term_count_of(&lines, &signs);
    }
    Scene {
        live_terms: term_count_of(&lines, &cd_signs),
        live_signs: cd_signs,
        octonion_planes: octonion_plane_count(&cd),
        cd_signs,
        lines,
        line_drop,
        line_inc,
        selected_line: 0,
        hodogram: helical_field(6, 0.7, 1.0, 0.45),
        yaw: 0.7,
        pitch: 0.35,
        drag: None,
        hover: "drag rotates B-space  |  arrows cycle a PG(3,2) line  |  Esc quit".to_string(),
        data,
    }
}

fn draw_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    accent: RGBColor,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (w, h) = area.dim_in_pixel();
    area.fill(&PANEL).map_err(plot_err)?;
    area.draw(&Rectangle::new(
        [(0, 0), (w as i32 - 1, h as i32 - 1)],
        ShapeStyle::from(&GRID.mix(0.4)).stroke_width(1),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(0, 0), (w as i32 - 1, 0)],
        ShapeStyle::from(&accent).stroke_width(4),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn draw_b_hodogram<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel(area, CYAN)?;
    let pts: Vec<(f64, f64)> = scene
        .hodogram
        .iter()
        .map(|&b| project_b(b, scene.yaw, scene.pitch))
        .collect();
    let origin = project_b([0.0, 0.0, 0.0], scene.yaw, scene.pitch);
    let axis_x = project_b([1.4, 0.0, 0.0], scene.yaw, scene.pitch);
    let axis_y = project_b([0.0, 1.4, 0.0], scene.yaw, scene.pitch);
    let axis_z = project_b([0.0, 0.0, 1.4], scene.yaw, scene.pitch);
    let mut xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let mut ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
    for p in [origin, axis_x, axis_y, axis_z] {
        xs.push(p.0);
        ys.push(p.1);
    }
    let xmin = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ymin = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = 0.12 * (xmax - xmin).max(ymax - ymin).max(0.4);
    let mut chart = ChartBuilder::on(area)
        .margin(16)
        .caption(
            "B-space hodogram of six samples  |  not a light cone  |  not the LM plane  |  drag to rotate R^3",
            ("sans-serif", 16).into_font().color(&TEXT),
        )
        .build_cartesian_2d((xmin - pad)..(xmax + pad), (ymin - pad)..(ymax + pad))
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .draw()
        .map_err(plot_err)?;
    for (end, label, color) in [
        (axis_x, "Bx", ROSE),
        (axis_y, "By", EMERALD),
        (axis_z, "Bz", AMBER),
    ] {
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![origin, end],
                ShapeStyle::from(&color).stroke_width(2),
            )))
            .map_err(plot_err)?;
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (end.0, end.1),
                ("sans-serif", 14).into_font().color(&color),
            )))
            .map_err(plot_err)?;
    }
    if pts.len() >= 2 {
        chart
            .draw_series(std::iter::once(PathElement::new(
                pts.clone(),
                ShapeStyle::from(&CYAN).stroke_width(3),
            )))
            .map_err(plot_err)?;
    }
    for (i, &p) in pts.iter().enumerate() {
        let color = if i == 0 {
            GOLD
        } else if i + 1 == pts.len() {
            ROSE
        } else {
            CYAN
        };
        chart
            .draw_series(std::iter::once(Circle::new(p, 6, color.filled())))
            .map_err(plot_err)?;
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![origin, p],
                ShapeStyle::from(&MUTED.mix(0.45)).stroke_width(1),
            )))
            .map_err(plot_err)?;
    }
    Ok(())
}

fn draw_basis_grid<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel(area, GOLD)?;
    let (a, b, c) = scene.lines[scene.selected_line];
    let drop = scene.line_drop[scene.selected_line];
    let mut chart = ChartBuilder::on(area)
        .margin(16)
        .caption(
            format!(
                "16D fiber: index = 4*lag + channel  |  e8 = lag2 Bx  |  selected line {{e_{a},e_{b},e_{c}}} drop {drop}  |  rose = pencil through e8"
            ),
            ("sans-serif", 16).into_font().color(&TEXT),
        )
        .x_label_area_size(36)
        .y_label_area_size(52)
        .build_cartesian_2d(-0.2f64..4.2f64, -0.2f64..4.2f64)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .x_desc("channel")
        .x_label_formatter(&|x| channel_tick(*x))
        .y_desc("lag")
        .y_label_formatter(&|y| lag_tick(*y))
        .label_style(("sans-serif", 15).into_font().color(&MUTED))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(1))
        .disable_mesh()
        .draw()
        .map_err(plot_err)?;
    for i in 0..STAPLE_DIM {
        let (x, y) = cell_xy(i);
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x - 0.42, y - 0.42), (x + 0.42, y + 0.42)],
                GRID.mix(0.35).filled(),
            )))
            .map_err(plot_err)?;
    }
    for (k, &(u, v, w)) in scene.lines.iter().enumerate() {
        if scene.line_drop[k] != 288 {
            continue;
        }
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![cell_xy(u), cell_xy(v), cell_xy(w), cell_xy(u)],
                ShapeStyle::from(&ROSE.mix(0.7)).stroke_width(2),
            )))
            .map_err(plot_err)?;
    }
    for index in [a, b, c] {
        chart
            .draw_series(std::iter::once(PathElement::new(
                cell_frame(index),
                ShapeStyle::from(&CYAN).stroke_width(3),
            )))
            .map_err(plot_err)?;
    }
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![cell_xy(a), cell_xy(b), cell_xy(c), cell_xy(a)],
            ShapeStyle::from(&CYAN).stroke_width(3),
        )))
        .map_err(plot_err)?;
    let e8 = cell_xy(8);
    chart
        .draw_series(std::iter::once(Rectangle::new(
            [(e8.0 - 0.42, e8.1 - 0.42), (e8.0 + 0.42, e8.1 + 0.42)],
            GOLD.mix(0.35).filled(),
        )))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(Text::new(
            "e8",
            (e8.0 - 0.18, e8.1 - 0.08),
            ("sans-serif", 16).into_font().color(&GOLD),
        )))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(Text::new(
            "unit e0 = Bx t",
            (cell_xy(0).0 - 0.35, cell_xy(0).1 - 0.08),
            ("sans-serif", 13).into_font().color(&MUTED),
        )))
        .map_err(plot_err)?;
    Ok(())
}

fn draw_ladder<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel(area, AMBER)?;
    let canon = scene.data.canonical.subsample_auc;
    let mut chart = ChartBuilder::on(area)
        .margin(16)
        .caption(
            "AUC on the 1080 + 96 k ladder  |  35-line family lives at k=0..6  |  CD is k=8, below the cloud",
            ("sans-serif", 18).into_font().color(&TEXT),
        )
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(-0.6f64..8.8f64, 0.802f64..0.848f64)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .x_desc("k  where terms = 1080 + 96 k")
        .y_desc("ROC-AUC")
        .label_style(("sans-serif", 15).into_font().color(&MUTED))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(1))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.28)))
        .draw()
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(-0.5, canon), (8.6, canon)],
            ShapeStyle::from(&ROSE).stroke_width(2),
        )))
        .map_err(plot_err)?;
    chart
        .draw_series(scene.data.invariant_twist_draws.iter().map(|d| {
            let k = f64::from(k_level(d.term_count));
            let j = ((d.index % 13) as f64 - 6.0) * 0.018;
            Circle::new((k + j, d.auc), 3, AMBER.mix(0.88).filled())
        }))
        .map_err(plot_err)?;
    chart
        .draw_series(scene.data.isomorphic_orbit_draws.iter().map(|d| {
            let j = ((d.index % 17) as f64 - 8.0) * 0.016;
            Circle::new((8.0 + j, d.auc), 3, EMERALD.mix(0.75).filled())
        }))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(Circle::new(
            (8.0, canon),
            9,
            ROSE.stroke_width(3),
        )))
        .map_err(plot_err)?;
    Ok(())
}

fn draw_densities<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel(area, EMERALD)?;
    let canon = scene.data.canonical.subsample_auc;
    let alt = &scene.data.invariant_matched_twists;
    let iso = &scene.data.isomorphic_orbit;
    let mut chart = ChartBuilder::on(area)
        .margin(16)
        .caption(
            format!(
                "Family densities on 0.80-0.85  |  35-line p2.5={:.4} p97.5={:.4}  |  isomorphic p2.5={:.4} p97.5={:.4}  |  CD={:.4}",
                alt.p2_5, alt.p97_5, iso.p2_5, iso.p97_5, canon
            ),
            ("sans-serif", 16).into_font().color(&TEXT),
        )
        .x_label_area_size(40)
        .y_label_area_size(44)
        .build_cartesian_2d(0.800f64..0.848f64, 0.0f64..0.42f64)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .x_desc("ROC-AUC")
        .y_desc("bin mass")
        .label_style(("sans-serif", 15).into_font().color(&MUTED))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(1))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.28)))
        .draw()
        .map_err(plot_err)?;
    let bins = histogram(
        &scene
            .data
            .invariant_twist_draws
            .iter()
            .map(|d| d.auc)
            .collect::<Vec<_>>(),
        0.800,
        0.848,
        24,
    );
    let iso_bins = histogram(
        &scene
            .data
            .isomorphic_orbit_draws
            .iter()
            .map(|d| d.auc)
            .collect::<Vec<_>>(),
        0.800,
        0.848,
        24,
    );
    chart
        .draw_series(
            bins.iter()
                .map(|&(x0, x1, h)| Rectangle::new([(x0, 0.0), (x1, h)], AMBER.mix(0.45).filled())),
        )
        .map_err(plot_err)?;
    chart
        .draw_series(
            iso_bins.iter().map(|&(x0, x1, h)| {
                Rectangle::new([(x0, 0.0), (x1, h)], EMERALD.mix(0.40).filled())
            }),
        )
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(canon, 0.0), (canon, 0.40)],
            ShapeStyle::from(&ROSE).stroke_width(3),
        )))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(alt.p2_5, 0.015), (alt.p97_5, 0.015)],
            ShapeStyle::from(&AMBER).stroke_width(4),
        )))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(iso.p2_5, 0.028), (iso.p97_5, 0.028)],
            ShapeStyle::from(&EMERALD).stroke_width(4),
        )))
        .map_err(plot_err)?;
    Ok(())
}

fn histogram(values: &[f64], lo: f64, hi: f64, n: usize) -> Vec<(f64, f64, f64)> {
    let mut counts = vec![0.0; n];
    let width = (hi - lo) / n as f64;
    for &v in values {
        if v < lo || v >= hi {
            continue;
        }
        let idx = ((v - lo) / width).floor() as usize;
        counts[idx.min(n - 1)] += 1.0;
    }
    let total = values.len().max(1) as f64;
    counts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let x0 = lo + i as f64 * width;
            (x0, x0 + width, c / total)
        })
        .collect()
}

fn render_layout<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
    footer: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&BACKGROUND).map_err(plot_err)?;
    let alt = &scene.data.invariant_matched_twists;
    let iso = &scene.data.isomorphic_orbit;
    root.draw_text(
        "E-279  Six samples of B in R^3  |  fiber is lag x channel packing, not space  |  e8 is lag-2 Bx",
        &("sans-serif", 30).into_font().color(&TEXT),
        (28, 16),
    )
    .map_err(plot_err)?;
    let summary = format!(
        "canonical AUC {:.4} at {} terms (k=8)  |  {}/15 hyperplanes octonion  |  35-line n={} mean {:.4} sits in k=0..6 (terms {}..{})  |  isomorphic n={} mean {:.4} p2.5 {:.4} p97.5 {:.4} max {:.4}  |  {}/{} 35-line draws >= CD  |  {}/{} isomorphic >= CD  |  {}",
        scene.data.canonical.subsample_auc,
        scene.data.canonical.term_count,
        scene.octonion_planes,
        alt.n,
        alt.mean,
        alt.term_count_min,
        alt.term_count_max,
        iso.n,
        iso.mean,
        iso.p2_5,
        iso.p97_5,
        iso.max,
        alt.draws_at_or_above_canonical,
        alt.n,
        iso.draws_at_or_above_canonical,
        iso.n,
        footer
    );
    root.draw_text(
        &summary,
        &("sans-serif", 16).into_font().color(&MUTED),
        (28, 54),
    )
    .map_err(plot_err)?;
    let body = root.margin(14, 88, 14, 22);
    let rows = body.split_evenly((2, 1));
    let top = rows[0].split_evenly((1, 2));
    let bot = rows[1].split_evenly((1, 2));
    draw_b_hodogram(&top[0], scene)?;
    draw_ladder(&top[1], scene)?;
    draw_basis_grid(&bot[0], scene)?;
    draw_densities(&bot[1], scene)?;
    Ok(())
}

fn write_png(path: &std::path::Path, scene: &Scene) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let root = BitMapBackend::new(path, (PLATE_W, PLATE_H)).into_drawing_area();
    render_layout(
        &root,
        scene,
        "C-1737  |  gold vertices = weight 1  |  rose lines drop 288",
    )?;
    root.present().map_err(plot_err)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

fn render_to_argb(scene: &Scene, width: usize, height: usize, pixels: &mut [u32]) -> Result<()> {
    let mut rgb = vec![0u8; width * height * 3];
    {
        let root =
            BitMapBackend::with_buffer(&mut rgb, (width as u32, height as u32)).into_drawing_area();
        render_layout(&root, scene, &scene.hover)?;
        root.present().map_err(plot_err)?;
    }
    rgb_to_argb(&rgb, pixels);
    Ok(())
}

fn run_window(mut scene: Scene, width: usize, height: usize) -> Result<()> {
    let mut window = Window::new(
        "E-279  B-space hodogram  |  drag rotate R^3  |  arrows cycle line  |  Esc quit",
        width,
        height,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|err| anyhow::anyhow!("minifb window: {err}"))?;
    window.set_target_fps(30);
    let mut pixels = vec![0u32; width * height];
    let mut dirty = true;
    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        if window.is_key_pressed(Key::Right, minifb::KeyRepeat::Yes) {
            scene.selected_line = (scene.selected_line + 1) % PG32_LINE_COUNT;
            dirty = true;
        }
        if window.is_key_pressed(Key::Left, minifb::KeyRepeat::Yes) {
            scene.selected_line = (scene.selected_line + PG32_LINE_COUNT - 1) % PG32_LINE_COUNT;
            dirty = true;
        }
        if window.is_key_pressed(Key::F, minifb::KeyRepeat::No) {
            let k = scene.selected_line;
            scene.live_signs[k] = -scene.live_signs[k];
            scene.live_terms = term_count_of(&scene.lines, &scene.live_signs);
            scene.hover = format!(
                "flipped line {k} (inc {}, drop {}): live terms {}",
                scene.line_inc[k], scene.line_drop[k], scene.live_terms
            );
            dirty = true;
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            scene.live_signs = scene.cd_signs;
            scene.live_terms = term_count_of(&scene.lines, &scene.live_signs);
            scene.hover = "reset to Cayley-Dickson orientation (1848 terms, k=8)".to_string();
            dirty = true;
        }
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            if window.get_mouse_down(MouseButton::Left) {
                if let Some((px, py)) = scene.drag {
                    scene.yaw += f64::from(mx - px) * 0.01;
                    scene.pitch += f64::from(my - py) * 0.01;
                    dirty = true;
                }
                scene.drag = Some((mx, my));
            } else {
                scene.drag = None;
            }
        }
        if dirty {
            render_to_argb(&scene, width, height, &mut pixels)?;
            dirty = false;
        }
        window
            .update_with_buffer(&pixels, width, height)
            .map_err(|err| anyhow::anyhow!("minifb present: {err}"))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let raw = fs::read_to_string(&args.input)
        .with_context(|| format!("read {}", args.input.display()))?;
    let data: OrbitFile =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", args.input.display()))?;
    anyhow::ensure!(
        !data.invariant_twist_draws.is_empty() && !data.isomorphic_orbit_draws.is_empty(),
        "orbit JSON has empty draw lists"
    );
    let scene = build_scene(data);
    if let Some(png) = args.png {
        write_png(&png, &scene)?;
        return Ok(());
    }
    match run_window(scene, args.width, args.height) {
        Ok(()) => Ok(()),
        Err(err) => {
            bail!(
                "{err}\nNo display? Write the plate instead:\n  staples-twist-orbit-view --png data/artifacts/images/science_staples_twist_orbit_plate_3160x2820.png"
            )
        }
    }
}
