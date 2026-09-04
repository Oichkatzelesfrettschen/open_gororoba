//! Interactive E-279 viewer: three AUC families, term-count scatter, and the
//! (Z_2)^4 hypercube with PG(3, 2) lines.
//!
//! Opens a minifb window by default. `--png` writes the 3160x2820 dark plate
//! without a display. Arrow keys cycle the 35 projective lines; dragging the
//! cube rotates the 4-space projection; F flips the selected line sign on the
//! Cayley-Dickson orientation and shows the associator term drop.
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
    PG32_LINE_COUNT, SparseCubicTensor, cd_twist, extract_line_orientations, pg32_lines,
    twist_from_line_orientations,
};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use plotters::prelude::*;
use serde::Deserialize;
use std::{fs, path::PathBuf};

const PLATE_W: u32 = 3160;
const PLATE_H: u32 = 2820;
const VIEW_W: usize = 1600;
const VIEW_H: usize = 1000;
const BACKGROUND: RGBColor = RGBColor(11, 15, 21);
const PANEL: RGBColor = RGBColor(21, 28, 37);
const GRID: RGBColor = RGBColor(55, 65, 81);
const TEXT: RGBColor = RGBColor(241, 245, 249);
const MUTED: RGBColor = RGBColor(148, 163, 184);
const CYAN: RGBColor = RGBColor(56, 189, 248);
const AMBER: RGBColor = RGBColor(251, 191, 36);
const EMERALD: RGBColor = RGBColor(52, 211, 153);
const MAGENTA: RGBColor = RGBColor(244, 114, 182);
const ROSE: RGBColor = RGBColor(251, 113, 133);

#[derive(Parser, Debug)]
#[command(about = "Interactive E-279 twist-orbit viewer and 3160x2820 plate")]
struct Args {
    #[arg(long, default_value = "data/output/staples_twist_orbit.json")]
    input: PathBuf,
    /// Write the grand-visualization plate and exit (no window).
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
    cocycle_draws: Vec<Draw>,
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
    hamming1_terms: [usize; PG32_LINE_COUNT],
    selected_line: usize,
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

fn project_vertex(i: usize, yaw: f64, pitch: f64) -> (f64, f64) {
    let mut x = if i & 1 == 0 { -1.0 } else { 1.0 };
    let mut y = if i & 2 == 0 { -1.0 } else { 1.0 };
    let z = if i & 4 == 0 { -1.0 } else { 1.0 };
    let mut w = if i & 8 == 0 { -1.0 } else { 1.0 };
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let nx = x * cy - w * sy;
    w = x * sy + w * cy;
    x = nx;
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let ny = y * cp - w * sp;
    w = y * sp + w * cp;
    y = ny;
    let depth = 3.2 + z * 0.35 + w * 0.2;
    (x / depth, y / depth)
}

fn term_count_of(
    lines: &[(usize, usize, usize); PG32_LINE_COUNT],
    signs: &[i8; PG32_LINE_COUNT],
) -> usize {
    SparseCubicTensor::from_twist(&twist_from_line_orientations(lines, signs)).term_count()
}

fn build_scene(data: OrbitFile) -> Scene {
    let lines = pg32_lines();
    let table = CdMultTable::generate(STAPLE_DIM);
    let cd_signs = extract_line_orientations(&cd_twist(&table), &lines);
    let mut hamming1_terms = [0usize; PG32_LINE_COUNT];
    for (k, slot) in hamming1_terms.iter_mut().enumerate() {
        let mut signs = cd_signs;
        signs[k] = -signs[k];
        *slot = term_count_of(&lines, &signs);
    }
    Scene {
        live_terms: term_count_of(&lines, &cd_signs),
        live_signs: cd_signs,
        cd_signs,
        lines,
        hamming1_terms,
        selected_line: 0,
        yaw: 0.55,
        pitch: 0.35,
        drag: None,
        hover: "hover a point or bar".to_string(),
        data,
    }
}

fn draw_panel_frame<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    accent: RGBColor,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (w, h) = area.dim_in_pixel();
    area.fill(&PANEL).map_err(plot_err)?;
    area.draw(&Rectangle::new(
        [(1, 1), (w as i32 - 2, h as i32 - 2)],
        ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(0, 0), (w as i32 - 1, 0)],
        ShapeStyle::from(&accent).stroke_width(3),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn draw_beeswarm<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel_frame(area, CYAN)?;
    let canon = scene.data.canonical.subsample_auc;
    let mut chart = ChartBuilder::on(area)
        .margin(18)
        .caption(
            "ROC-AUC by family  |  red = canonical 0.8274  |  cocycles sit at 0.5000, off this scale",
            ("sans-serif", 20).into_font().color(&TEXT),
        )
        .x_label_area_size(36)
        .y_label_area_size(56)
        .build_cartesian_2d(-0.6f64..2.6f64, 0.800f64..0.850f64)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(3)
        .x_label_formatter(&|x| {
            if *x < 0.4 {
                "cocycle".to_string()
            } else if *x < 1.4 {
                "35-line".to_string()
            } else {
                "isomorphic".to_string()
            }
        })
        .y_desc("ROC-AUC")
        .label_style(("sans-serif", 16).into_font().color(&MUTED))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(1))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.25)))
        .draw()
        .map_err(plot_err)?;
    let families: [(&str, &[Draw], RGBColor, f64); 3] = [
        ("cocycle", &scene.data.cocycle_draws, MAGENTA, 0.0),
        ("35-line", &scene.data.invariant_twist_draws, AMBER, 1.0),
        (
            "isomorphic",
            &scene.data.isomorphic_orbit_draws,
            EMERALD,
            2.0,
        ),
    ];
    for (_, draws, color, x0) in families {
        chart
            .draw_series(draws.iter().map(|d| {
                let jitter = ((d.index % 17) as f64 - 8.0) * 0.012;
                Circle::new((x0 + jitter, d.auc), 3, color.mix(0.85).filled())
            }))
            .map_err(plot_err)?;
    }
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(-0.5, canon), (2.5, canon)],
            ShapeStyle::from(&ROSE).stroke_width(2),
        )))
        .map_err(plot_err)?;
    Ok(())
}

fn draw_scatter<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel_frame(area, AMBER)?;
    let canon = scene.data.canonical.subsample_auc;
    let mut chart = ChartBuilder::on(area)
        .margin(18)
        .caption(
            "Associator support vs AUC  |  CD (open red) is 1848 terms, below most 35-line draws",
            ("sans-serif", 20).into_font().color(&TEXT),
        )
        .x_label_area_size(40)
        .y_label_area_size(56)
        .build_cartesian_2d(1000.0f64..1900.0f64, 0.800f64..0.850f64)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .x_desc("nonzero associator terms")
        .y_desc("ROC-AUC")
        .label_style(("sans-serif", 16).into_font().color(&MUTED))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(1))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.25)))
        .draw()
        .map_err(plot_err)?;
    let series = [
        (&scene.data.cocycle_draws, MAGENTA),
        (&scene.data.invariant_twist_draws, AMBER),
        (&scene.data.isomorphic_orbit_draws, EMERALD),
    ];
    for (draws, color) in series {
        chart
            .draw_series(
                draws
                    .iter()
                    .map(|d| Circle::new((d.term_count as f64, d.auc), 3, color.mix(0.8).filled())),
            )
            .map_err(plot_err)?;
    }
    chart
        .draw_series(std::iter::once(Circle::new(
            (scene.data.canonical.term_count as f64, canon),
            8,
            ROSE.stroke_width(2),
        )))
        .map_err(plot_err)?;
    Ok(())
}

fn draw_hypercube<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel_frame(area, EMERALD)?;
    let (a, b, c) = scene.lines[scene.selected_line];
    let mut chart = ChartBuilder::on(area)
        .margin(18)
        .caption(
            format!(
                "(Z_2)^4 cube  |  line {} = {{{a},{b},{c}}}  |  live terms {}",
                scene.selected_line, scene.live_terms
            ),
            ("sans-serif", 20).into_font().color(&TEXT),
        )
        .build_cartesian_2d(-0.72f64..0.72f64, -0.72f64..0.72f64)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .draw()
        .map_err(plot_err)?;
    let pts: Vec<(f64, f64)> = (0..STAPLE_DIM)
        .map(|i| project_vertex(i, scene.yaw, scene.pitch))
        .collect();
    for i in 0..STAPLE_DIM {
        for bit in 0..4 {
            let j = i ^ (1 << bit);
            if j > i {
                chart
                    .draw_series(std::iter::once(PathElement::new(
                        vec![pts[i], pts[j]],
                        ShapeStyle::from(&GRID.mix(0.7)).stroke_width(1),
                    )))
                    .map_err(plot_err)?;
            }
        }
    }
    let tri = [pts[a], pts[b], pts[c], pts[a]];
    chart
        .draw_series(std::iter::once(PathElement::new(
            tri,
            ShapeStyle::from(&CYAN).stroke_width(3),
        )))
        .map_err(plot_err)?;
    chart
        .draw_series((0..STAPLE_DIM).map(|i| {
            let on_line = i == a || i == b || i == c;
            let color = if i == 0 {
                ROSE
            } else if on_line {
                CYAN
            } else {
                MUTED
            };
            Circle::new(pts[i], if on_line { 7 } else { 4 }, color.filled())
        }))
        .map_err(plot_err)?;
    Ok(())
}

fn draw_hamming<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    scene: &Scene,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    draw_panel_frame(area, ROSE)?;
    let max_bar = scene
        .hamming1_terms
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1848) as f64;
    let mut chart = ChartBuilder::on(area)
        .margin(18)
        .caption(
            "Hamming-1 line flips of the CD orientation  |  none keep 1848 terms",
            ("sans-serif", 20).into_font().color(&TEXT),
        )
        .x_label_area_size(36)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..35.0f64, 900.0f64..(max_bar + 40.0))
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .x_desc("flipped PG(3,2) line")
        .y_desc("associator terms")
        .label_style(("sans-serif", 16).into_font().color(&MUTED))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(1))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.25)))
        .draw()
        .map_err(plot_err)?;
    chart
        .draw_series(scene.hamming1_terms.iter().enumerate().map(|(k, &n)| {
            let color = if k == scene.selected_line {
                CYAN
            } else {
                AMBER
            };
            Rectangle::new(
                [(k as f64 + 0.12, 900.0), (k as f64 + 0.88, n as f64)],
                color.mix(0.9).filled(),
            )
        }))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 1848.0), (35.0, 1848.0)],
            ShapeStyle::from(&ROSE).stroke_width(2),
        )))
        .map_err(plot_err)?;
    Ok(())
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
    let (w, h) = root.dim_in_pixel();
    root.draw_text(
        "E-279  |  Cayley-Dickson associator vs matched twist families",
        &("sans-serif", 32).into_font().color(&TEXT),
        (28, 18),
    )
    .map_err(plot_err)?;
    let alt = &scene.data.invariant_matched_twists;
    let iso = &scene.data.isomorphic_orbit;
    let summary = format!(
        "canonical AUC {:.4} (1848 terms)  |  35-line n={} mean {:.4} p2.5 {:.4} p97.5 {:.4} max {:.4} ({}/{} >= CD) terms {}..{}  |  isomorphic n={} mean {:.4} p2.5 {:.4} p97.5 {:.4} max {:.4} ({}/{} >= CD)  |  {}",
        scene.data.canonical.subsample_auc,
        alt.n,
        alt.mean,
        alt.p2_5,
        alt.p97_5,
        alt.max,
        alt.draws_at_or_above_canonical,
        alt.n,
        alt.term_count_min,
        alt.term_count_max,
        iso.n,
        iso.mean,
        iso.p2_5,
        iso.p97_5,
        iso.max,
        iso.draws_at_or_above_canonical,
        iso.n,
        footer
    );
    root.draw_text(
        &summary,
        &("sans-serif", 18).into_font().color(&MUTED),
        (28, 58),
    )
    .map_err(plot_err)?;
    let body = root.margin(16, 96, 16, 28);
    let chunks = body.split_evenly((2, 2));
    draw_beeswarm(&chunks[0], scene)?;
    draw_scatter(&chunks[1], scene)?;
    draw_hypercube(&chunks[2], scene)?;
    draw_hamming(&chunks[3], scene)?;
    let _ = w;
    let _ = h;
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
        "C-1737  |  arrows cycle line, F flips sign, drag rotates cube",
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
        render_layout(
            &root,
            scene,
            &format!(
                "{}  |  arrows=line  F=flip selected  R=reset CD  Esc=quit",
                scene.hover
            ),
        )?;
        root.present().map_err(plot_err)?;
    }
    rgb_to_argb(&rgb, pixels);
    Ok(())
}

fn hover_from_mouse(scene: &Scene, width: usize, height: usize, mx: f32, my: f32) -> String {
    let x = mx / width as f32;
    let y = my / height as f32;
    if (0.50..1.00).contains(&x) && (0.52..1.00).contains(&y) {
        let k = ((x - 0.50) / 0.50 * 35.0).floor() as isize;
        if (0..35).contains(&k) {
            let k = k as usize;
            return format!(
                "flip line {k} -> {} terms (CD 1848)",
                scene.hamming1_terms[k]
            );
        }
    }
    if (0.50..1.00).contains(&x) && y < 0.50 {
        let t = 1000.0 + (x - 0.50) / 0.50 * 900.0;
        let auc = 0.850 - (y / 0.50) * 0.050;
        return format!("scatter ~{t:.0} terms, AUC ~{auc:.4}");
    }
    scene.hover.clone()
}

fn run_window(mut scene: Scene, width: usize, height: usize) -> Result<()> {
    let mut window = Window::new(
        "E-279 twist orbit  |  arrows cycle PG(3,2) line  |  F flip  |  drag cube  |  Esc quit",
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
            scene.hover = format!("flipped line {k}: live terms {}", scene.live_terms);
            dirty = true;
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            scene.live_signs = scene.cd_signs;
            scene.live_terms = term_count_of(&scene.lines, &scene.live_signs);
            scene.hover = "reset to Cayley-Dickson orientation (1848 terms)".to_string();
            dirty = true;
        }
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            let next = hover_from_mouse(&scene, width, height, mx, my);
            if next != scene.hover {
                scene.hover = next;
                dirty = true;
            }
            let down = window.get_mouse_down(MouseButton::Left);
            if down {
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
