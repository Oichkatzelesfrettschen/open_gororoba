//! Static ecliptic-plane chart of the solar system at one epoch, as SVG.
//!
//! Every point on the chart is a kernel evaluation. A body's marker is its
//! position at `--jed`, and its orbit is the path the kernel returns when
//! sampled across one sidereal period centred on that epoch -- not a circle at
//! the current radius and not an ellipse fitted to elements, so a real
//! eccentricity shows as a real eccentricity and Pluto crosses inside Neptune
//! where it does.
//!
//! The projection drops ecliptic latitude: `x = r cos(lat) cos(lon)` and
//! `y = r cos(lat) sin(lon)`, which is the view from ecliptic north with the
//! vernal equinox to the right. Latitude is reported per body in the legend
//! rather than silently discarded, because Pluto reaches 17 degrees and a
//! flat chart would otherwise imply it lies in the plane.
//!
//! SVG is written directly. There is no plotting or GUI dependency here, which
//! keeps the chart out of the dependency-closure question an interactive viewer
//! would raise.
//!
//! Usage:
//!   ephemeris chart --jed 2451545.0 --out data/output/ephemeris/j2000.svg
//!   ephemeris chart --bodies inner --scale linear

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use std::{fmt::Write as _, fs, path::PathBuf};

use gororoba_cli_physics::ephemeris_loader::{HeliocentricEphemeris, SolarSystemBody};

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum RadialScale {
    /// Radius proportional to distance. Honest, and unreadable across 0.31 to
    /// 49 AU, so it suits an inner-planet chart.
    Linear,
    /// Radius proportional to `log10(r)`. Compresses the outer system enough
    /// for all nine bodies to share one frame; radial distances are no longer
    /// comparable by eye, which the axis labels state.
    Log,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum BodySet {
    All,
    /// Mercury through Mars.
    Inner,
    /// Jupiter through Pluto.
    Outer,
}

#[derive(Args)]
pub struct Cli {
    /// JPL DE-series kernel. DE440 covers 1550 through 2650.
    #[arg(long, default_value = "data/external/de440.bsp")]
    kernel: PathBuf,

    /// Julian Ephemeris Date in TDB for the body markers. Defaults to J2000.0.
    #[arg(long, default_value_t = 2_451_545.0)]
    jed: f64,

    /// Where to write the SVG. Omit to write to stdout.
    #[arg(long)]
    out: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = BodySet::All)]
    bodies: BodySet,

    #[arg(long, value_enum, default_value_t = RadialScale::Log)]
    scale: RadialScale,

    /// Samples per orbit. Pluto over 248 years at the default 360 is one
    /// sample per 92 days, which is finer than the curvature at 30 AU.
    #[arg(long, default_value_t = 360)]
    samples: usize,

    /// Side of the square canvas, in SVG user units.
    #[arg(long, default_value_t = 900.0)]
    size: f64,
}

/// One body's traced orbit plus its marker at the requested epoch.
struct Track {
    body: SolarSystemBody,
    /// Ecliptic-plane points across one period, in AU.
    path: Vec<(f64, f64)>,
    /// Position at `--jed`, in AU, with the latitude the projection drops.
    marker: (f64, f64),
    r_au: f64,
    lat_deg: f64,
}

/// Colours run inner to outer and stay distinguishable on either page ground.
const BODY_COLORS: [&str; 9] = [
    "#b07d4a", "#c9a227", "#4f8de0", "#c0503a", "#c9873f", "#a9903f", "#5fa8a0", "#4a6fd4",
    "#8a7f9e",
];

fn selected(set: BodySet) -> Vec<SolarSystemBody> {
    match set {
        BodySet::All => SolarSystemBody::ALL.to_vec(),
        BodySet::Inner => SolarSystemBody::ALL[..4].to_vec(),
        BodySet::Outer => SolarSystemBody::ALL[4..].to_vec(),
    }
}

/// Project heliocentric ecliptic spherical coordinates onto the ecliptic plane.
fn project(r_au: f64, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    (r_au * lat.cos() * lon.cos(), r_au * lat.cos() * lon.sin())
}

fn trace(ephemeris: &HeliocentricEphemeris, body: SolarSystemBody, cli: &Cli) -> Result<Track> {
    let period = body.sidereal_period_days();
    let mut path = Vec::with_capacity(cli.samples + 1);
    for i in 0..=cli.samples {
        let jed = cli.jed - period / 2.0 + period * (i as f64) / (cli.samples as f64);
        let p = ephemeris
            .body_ecliptic(body, jed)
            .with_context(|| format!("{} at JED {jed}", body.name()))?;
        path.push(project(p.r_au, p.lat_deg, p.lon_deg));
    }
    let now = ephemeris.body_ecliptic(body, cli.jed)?;
    Ok(Track {
        body,
        path,
        marker: project(now.r_au, now.lat_deg, now.lon_deg),
        r_au: now.r_au,
        lat_deg: now.lat_deg,
    })
}

/// Map an AU radius onto canvas units under the chosen scale.
///
/// The log branch maps `log10(r)` from the innermost sampled radius to the
/// outermost, so the ring spacing carries decades rather than distance.
fn radial_map(scale: RadialScale, r_au: f64, r_max: f64, r_min: f64, extent: f64) -> f64 {
    match scale {
        RadialScale::Linear => extent * (r_au / r_max),
        RadialScale::Log => {
            let lo = r_min.max(1e-6).log10();
            let hi = r_max.max(1e-6).log10();
            // A single-body chart collapses the range; put it at the edge
            // rather than dividing by zero.
            if (hi - lo).abs() < 1e-12 {
                return extent;
            }
            let t = ((r_au.max(1e-6).log10() - lo) / (hi - lo)).clamp(0.0, 1.0);
            extent * t
        }
    }
}

pub fn run(cli: Cli) -> Result<()> {
    let ephemeris = HeliocentricEphemeris::load(&cli.kernel)?;
    let bodies = selected(cli.bodies);

    let tracks: Vec<Track> = bodies
        .iter()
        .map(|&b| trace(&ephemeris, b, &cli))
        .collect::<Result<_>>()?;

    let radii: Vec<f64> = tracks
        .iter()
        .flat_map(|t| t.path.iter().map(|(x, y)| x.hypot(*y)))
        .collect();
    let r_max = radii.iter().cloned().fold(f64::MIN, f64::max);
    let r_min = radii.iter().cloned().fold(f64::MAX, f64::min);

    let center = cli.size / 2.0;
    let extent = center - 70.0;
    let place = |x: f64, y: f64| {
        let r = x.hypot(y);
        let mapped = radial_map(cli.scale, r, r_max, r_min, extent);
        let (ux, uy) = if r > 0.0 { (x / r, y / r) } else { (0.0, 0.0) };
        // SVG y grows downward; negate so ecliptic longitude runs counter-clockwise.
        (center + ux * mapped, center - uy * mapped)
    };

    let mut svg = String::with_capacity(1 << 16);
    let size = cli.size;
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" font-family="ui-monospace, monospace">"#
    )?;
    writeln!(
        svg,
        r##"<rect width="{size}" height="{size}" fill="#0d1017"/>"##
    )?;

    // Decade or fraction rings, labelled in AU so the radial scale is readable.
    for &ring in &[0.1_f64, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] {
        if ring < r_min * 0.9 || ring > r_max * 1.1 {
            continue;
        }
        let rr = radial_map(cli.scale, ring, r_max, r_min, extent);
        // The innermost ring maps to the origin under the log scale, which
        // stacks its label on the Sun. Drop rings that fall inside the marker.
        if rr < 20.0 {
            continue;
        }
        writeln!(
            svg,
            r##"<circle cx="{center:.1}" cy="{center:.1}" r="{rr:.1}" fill="none" stroke="#232838" stroke-width="1"/>"##
        )?;
        writeln!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" fill="#4a5268" font-size="11">{ring} AU</text>"##,
            center + 4.0,
            center - rr - 4.0
        )?;
    }

    writeln!(
        svg,
        r##"<circle cx="{center:.1}" cy="{center:.1}" r="6" fill="#ffd479"/>"##
    )?;

    for (idx, track) in tracks.iter().enumerate() {
        let color = BODY_COLORS[SolarSystemBody::ALL
            .iter()
            .position(|b| *b == track.body)
            .unwrap_or(idx)];
        let pts: Vec<String> = track
            .path
            .iter()
            .map(|&(x, y)| {
                let (px, py) = place(x, y);
                format!("{px:.2},{py:.2}")
            })
            .collect();
        writeln!(
            svg,
            r#"<polyline points="{}" fill="none" stroke="{color}" stroke-width="1.2" stroke-opacity="0.75"/>"#,
            pts.join(" ")
        )?;
        let (mx, my) = place(track.marker.0, track.marker.1);
        writeln!(
            svg,
            r#"<circle cx="{mx:.2}" cy="{my:.2}" r="4.5" fill="{color}"/>"#
        )?;
        writeln!(
            svg,
            r#"<text x="{:.2}" y="{:.2}" fill="{color}" font-size="12">{}</text>"#,
            mx + 8.0,
            my + 4.0,
            track.body.name()
        )?;
    }

    let scale_note = match cli.scale {
        RadialScale::Linear => "radius linear in AU",
        RadialScale::Log => "radius logarithmic in AU; ring spacing is decades, not distance",
    };
    writeln!(
        svg,
        r##"<text x="16" y="24" fill="#8b93a7" font-size="13">heliocentric ecliptic of J2000, JED(TDB) {:.1}</text>"##,
        cli.jed
    )?;
    writeln!(
        svg,
        r##"<text x="16" y="42" fill="#5c6478" font-size="11">orbits sampled from {} across one sidereal period; {scale_note}</text>"##,
        cli.kernel.display()
    )?;
    let mut y = size - 14.0 - 14.0 * tracks.len() as f64;
    writeln!(
        svg,
        r##"<text x="16" y="{y:.0}" fill="#5c6478" font-size="11">latitude is projected out; it is listed here because Pluto reaches 17 degrees</text>"##
    )?;
    for track in &tracks {
        y += 14.0;
        writeln!(
            svg,
            r##"<text x="16" y="{y:.0}" fill="#8b93a7" font-size="11">{:<22} r = {:>9.4} AU   lat = {:>8.3} deg</text>"##,
            track.body.name(),
            track.r_au,
            track.lat_deg
        )?;
    }
    writeln!(svg, "</svg>")?;

    match &cli.out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
            fs::write(path, &svg).with_context(|| format!("write {}", path.display()))?;
            println!("wrote {} ({} bytes)", path.display(), svg.len());
        }
        None => print!("{svg}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RadialScale, project, radial_map};

    /// Longitude zero is the vernal equinox and lies on the +x axis.
    #[test]
    fn vernal_equinox_projects_to_positive_x() {
        let (x, y) = project(1.0, 0.0, 0.0);
        assert!((x - 1.0).abs() < 1e-12);
        assert!(y.abs() < 1e-12);
    }

    /// Ninety degrees of longitude is a quarter turn counter-clockwise.
    #[test]
    fn ninety_degrees_projects_to_positive_y() {
        let (x, y) = project(1.0, 0.0, 90.0);
        assert!(x.abs() < 1e-12);
        assert!((y - 1.0).abs() < 1e-12);
    }

    /// Latitude foreshortens the in-plane radius by its cosine, which is why
    /// the legend reports it rather than the chart implying it is zero.
    #[test]
    fn latitude_foreshortens_the_projected_radius() {
        let (x, y) = project(1.0, 60.0, 0.0);
        assert!((x - 0.5).abs() < 1e-12, "cos(60 deg) = 0.5, got {x}");
        assert!(y.abs() < 1e-12);
    }

    #[test]
    fn linear_scale_puts_the_outermost_radius_at_the_canvas_edge() {
        assert!((radial_map(RadialScale::Linear, 30.0, 30.0, 0.3, 400.0) - 400.0).abs() < 1e-9);
        assert!(radial_map(RadialScale::Linear, 0.3, 30.0, 0.3, 400.0) < 5.0);
    }

    /// The log branch must map both ends of the sampled range onto the full
    /// extent, so the innermost body is not stacked on the Sun.
    #[test]
    fn log_scale_spans_the_full_extent() {
        assert!(radial_map(RadialScale::Log, 0.3, 30.0, 0.3, 400.0).abs() < 1e-9);
        assert!((radial_map(RadialScale::Log, 30.0, 30.0, 0.3, 400.0) - 400.0).abs() < 1e-9);
        let mid = radial_map(RadialScale::Log, 3.0, 30.0, 0.3, 400.0);
        assert!(
            (mid - 200.0).abs() < 1e-9,
            "a decade is half of two, got {mid}"
        );
    }

    /// `--bodies inner --scale log` spans 0.31 to 1.67 AU, well under a decade.
    /// An earlier form widened the divisor to a full decade to dodge a
    /// degenerate range, which quietly stopped the outermost body reaching the
    /// edge on exactly that chart.
    #[test]
    fn log_scale_spans_a_sub_decade_range() {
        let outer = radial_map(RadialScale::Log, 1.67, 1.67, 0.31, 400.0);
        assert!(
            (outer - 400.0).abs() < 1e-9,
            "outermost body must reach the canvas edge, got {outer}"
        );
        assert!(radial_map(RadialScale::Log, 0.31, 1.67, 0.31, 400.0).abs() < 1e-9);
    }

    /// A one-body chart has no range at all and must not divide by zero.
    #[test]
    fn log_scale_survives_a_collapsed_range() {
        let only = radial_map(RadialScale::Log, 5.2, 5.2, 5.2, 400.0);
        assert!(
            only.is_finite() && (only - 400.0).abs() < 1e-9,
            "got {only}"
        );
    }
}
