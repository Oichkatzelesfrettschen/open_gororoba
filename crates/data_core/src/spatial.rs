//! Lightweight spatial primitives for catalog crossmatching.
//!
//! The current survey lanes mostly need exact point-to-point sky matches with
//! modest query counts against large static catalogs. This module keeps that
//! logic reusable without forcing each binary to rebuild its own angular
//! geometry and sky-cell bookkeeping.

use std::collections::HashMap;
#[cfg(feature = "fits")]
use wide::f64x4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogModality {
    SkyPoint,
    SkyFootprint,
    SkyLocalization,
    NonSpatial,
}

#[derive(Debug, Clone)]
pub struct SkyPoint {
    pub id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiusMatch {
    pub query_index: usize,
    pub candidate_index: usize,
    pub separation_arcsec: f64,
}

#[derive(Debug, Clone)]
pub struct SkyGridIndex {
    cell_size_deg: f64,
    ra_cells: i32,
    cells: HashMap<(i32, i32), Vec<usize>>,
    points: Vec<SkyPoint>,
}

impl SkyGridIndex {
    pub fn from_points(points: Vec<SkyPoint>, cell_size_deg: f64) -> Self {
        let clamped_cell_size = cell_size_deg.max(1.0e-4);
        let ra_cells = (360.0 / clamped_cell_size).ceil() as i32;
        let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (index, point) in points.iter().enumerate() {
            let key = sky_cell_key(point.ra_deg, point.dec_deg, clamped_cell_size, ra_cells);
            cells.entry(key).or_default().push(index);
        }
        Self {
            cell_size_deg: clamped_cell_size,
            ra_cells,
            cells,
            points,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn points(&self) -> &[SkyPoint] {
        &self.points
    }

    pub fn nearest_within(
        &self,
        query_ra_deg: f64,
        query_dec_deg: f64,
        radius_arcsec: f64,
    ) -> Option<RadiusMatch> {
        if self.points.is_empty() || radius_arcsec <= 0.0 {
            return None;
        }

        let radius_deg = radius_arcsec / 3600.0;
        let search_steps = (radius_deg / self.cell_size_deg).ceil() as i32 + 1;
        let (query_ra_cell, query_dec_cell) = sky_cell_key(
            query_ra_deg,
            query_dec_deg,
            self.cell_size_deg,
            self.ra_cells,
        );

        let mut best: Option<RadiusMatch> = None;
        for ra_offset in -search_steps..=search_steps {
            let ra_cell = wrap_ra_cell(query_ra_cell + ra_offset, self.ra_cells);
            for dec_offset in -search_steps..=search_steps {
                let dec_cell = query_dec_cell + dec_offset;
                let Some(indices) = self.cells.get(&(ra_cell, dec_cell)) else {
                    continue;
                };
                for &candidate_index in indices {
                    let candidate = &self.points[candidate_index];
                    let separation_arcsec = angular_separation_arcsec(
                        query_ra_deg,
                        query_dec_deg,
                        candidate.ra_deg,
                        candidate.dec_deg,
                    );
                    if separation_arcsec > radius_arcsec {
                        continue;
                    }
                    match best {
                        Some(current) if current.separation_arcsec <= separation_arcsec => {}
                        _ => {
                            best = Some(RadiusMatch {
                                query_index: 0,
                                candidate_index,
                                separation_arcsec,
                            });
                        }
                    }
                }
            }
        }
        best
    }

    pub fn for_each_within<F>(
        &self,
        query_ra_deg: f64,
        query_dec_deg: f64,
        radius_arcsec: f64,
        mut visitor: F,
    ) where
        F: FnMut(RadiusMatch),
    {
        if self.points.is_empty() || radius_arcsec <= 0.0 {
            return;
        }

        let radius_deg = radius_arcsec / 3600.0;
        let search_steps = (radius_deg / self.cell_size_deg).ceil() as i32 + 1;
        let (query_ra_cell, query_dec_cell) = sky_cell_key(
            query_ra_deg,
            query_dec_deg,
            self.cell_size_deg,
            self.ra_cells,
        );

        for ra_offset in -search_steps..=search_steps {
            let ra_cell = wrap_ra_cell(query_ra_cell + ra_offset, self.ra_cells);
            for dec_offset in -search_steps..=search_steps {
                let dec_cell = query_dec_cell + dec_offset;
                let Some(indices) = self.cells.get(&(ra_cell, dec_cell)) else {
                    continue;
                };
                for &candidate_index in indices {
                    let candidate = &self.points[candidate_index];
                    let separation_arcsec = angular_separation_arcsec(
                        query_ra_deg,
                        query_dec_deg,
                        candidate.ra_deg,
                        candidate.dec_deg,
                    );
                    if separation_arcsec <= radius_arcsec {
                        visitor(RadiusMatch {
                            query_index: 0,
                            candidate_index,
                            separation_arcsec,
                        });
                    }
                }
            }
        }
    }

    pub fn for_each_search_candidate<F>(
        &self,
        query_ra_deg: f64,
        query_dec_deg: f64,
        radius_arcsec: f64,
        mut visitor: F,
    ) where
        F: FnMut(usize),
    {
        if self.points.is_empty() || radius_arcsec <= 0.0 {
            return;
        }

        let radius_deg = radius_arcsec / 3600.0;
        let search_steps = (radius_deg / self.cell_size_deg).ceil() as i32 + 1;
        let (query_ra_cell, query_dec_cell) = sky_cell_key(
            query_ra_deg,
            query_dec_deg,
            self.cell_size_deg,
            self.ra_cells,
        );

        for ra_offset in -search_steps..=search_steps {
            let ra_cell = wrap_ra_cell(query_ra_cell + ra_offset, self.ra_cells);
            for dec_offset in -search_steps..=search_steps {
                let dec_cell = query_dec_cell + dec_offset;
                let Some(indices) = self.cells.get(&(ra_cell, dec_cell)) else {
                    continue;
                };
                for &candidate_index in indices {
                    visitor(candidate_index);
                }
            }
        }
    }
}

pub fn angular_separation_arcsec(ra1_deg: f64, dec1_deg: f64, ra2_deg: f64, dec2_deg: f64) -> f64 {
    let ra1 = ra1_deg.to_radians();
    let dec1 = dec1_deg.to_radians();
    let ra2 = ra2_deg.to_radians();
    let dec2 = dec2_deg.to_radians();
    let delta_ra = ra2 - ra1;
    let delta_dec = dec2 - dec1;
    let sin_ddec = (delta_dec / 2.0).sin();
    let sin_dra = (delta_ra / 2.0).sin();
    let a = sin_ddec * sin_ddec + dec1.cos() * dec2.cos() * sin_dra * sin_dra;
    let c = 2.0 * a.sqrt().min(1.0).asin();
    c.to_degrees() * 3600.0
}

pub fn normalize_ra_deg(ra_deg: f64) -> f64 {
    let wrapped = ra_deg % 360.0;
    if wrapped < 0.0 {
        wrapped + 360.0
    } else {
        wrapped
    }
}

fn sky_cell_key(ra_deg: f64, dec_deg: f64, cell_size_deg: f64, ra_cells: i32) -> (i32, i32) {
    let ra_cell = wrap_ra_cell(
        (normalize_ra_deg(ra_deg) / cell_size_deg).floor() as i32,
        ra_cells,
    );
    let dec_cell = ((dec_deg + 90.0) / cell_size_deg).floor() as i32;
    (ra_cell, dec_cell)
}

fn wrap_ra_cell(cell: i32, ra_cells: i32) -> i32 {
    if ra_cells <= 0 {
        return cell;
    }
    let wrapped = cell % ra_cells;
    if wrapped < 0 {
        wrapped + ra_cells
    } else {
        wrapped
    }
}


pub fn ecliptic_to_equatorial_vector(longitude_rad: f64, latitude_rad: f64) -> [f64; 3] {
    let epsilon = 23.4392911_f64.to_radians();
    let x_ecl = latitude_rad.cos() * longitude_rad.cos();
    let y_ecl = latitude_rad.cos() * longitude_rad.sin();
    let z_ecl = latitude_rad.sin();
    [
        x_ecl,
        y_ecl * epsilon.cos() - z_ecl * epsilon.sin(),
        y_ecl * epsilon.sin() + z_ecl * epsilon.cos(),
    ]
}

pub fn parse_hms_radians(value: &str) -> Option<f64> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() != 3 {
        return None;
    }
    let hours = fields[0].parse::<f64>().ok()?;
    let minutes = fields[1].parse::<f64>().ok()?;
    let seconds = fields[2].parse::<f64>().ok()?;
    let hours_total = hours + minutes / 60.0 + seconds / 3600.0;
    Some((hours_total * 15.0).to_radians())
}

pub fn parse_dms_radians(value: &str) -> Option<f64> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() != 3 {
        return None;
    }
    let deg = fields[0].parse::<f64>().ok()?;
    let min = fields[1].parse::<f64>().ok()?;
    let sec = fields[2].parse::<f64>().ok()?;
    let sign = if deg < 0.0 || value.starts_with('-') {
        -1.0
    } else {
        1.0
    };
    let deg_abs = deg.abs();
    let deg_total = deg_abs + min / 60.0 + sec / 3600.0;
    Some(sign * deg_total.to_radians())
}

// ---- SIMD point-grid cone search -----------------------------------------------
//
// Shared by data_core::catalogs::lotss and gororoba_cli_physics::survey_crossmatch.
// Both had identical PreparedLotss/MatrixPointGrid structs and identical
// for_each_*_match + candidate_dot_chunk functions.  Extracted here so the
// feature-gated CLI binary can use the same kernel via data_core::spatial.

/// A sky-point catalog prepared for SIMD-accelerated cone search.
///
/// Unit vectors (x, y, z) are precomputed for each point; the inner
/// SIMD lane-4 loop does cheap dot-product pre-screening before the
/// more expensive Haversine refinement.  `grid` handles coarse cell
/// filtering so the SIMD loop only sees a small candidate set.
#[cfg(feature = "fits")]
#[derive(Debug)]
pub struct PreparedPointGrid {
    /// Total point count.
    pub point_count: usize,
    /// Cell-level spatial index for candidate pre-filtering.
    pub grid: SkyGridIndex,
    /// Precomputed unit-vector X = cos(dec)*cos(ra) for each point.
    pub unit_x: Vec<f64>,
    /// Precomputed unit-vector Y = cos(dec)*sin(ra) for each point.
    pub unit_y: Vec<f64>,
    /// Precomputed unit-vector Z = sin(dec) for each point.
    pub unit_z: Vec<f64>,
}

/// Build a [`PreparedPointGrid`] from a slice of sky points.
///
/// `match_radius_arcsec` sizes the internal grid cells; set it to the
/// largest search radius you plan to use to avoid cell-boundary misses.
#[cfg(feature = "fits")]
pub fn prepare_point_grid(points: &[SkyPoint], match_radius_arcsec: f64) -> PreparedPointGrid {
    let mut unit_x = Vec::with_capacity(points.len());
    let mut unit_y = Vec::with_capacity(points.len());
    let mut unit_z = Vec::with_capacity(points.len());
    for point in points {
        let [x, y, z] = sky_point_unit_vector(point.ra_deg, point.dec_deg);
        unit_x.push(x);
        unit_y.push(y);
        unit_z.push(z);
    }
    PreparedPointGrid {
        point_count: points.len(),
        grid: SkyGridIndex::from_points(points.to_vec(), (match_radius_arcsec / 3600.0).max(0.01)),
        unit_x,
        unit_y,
        unit_z,
    }
}

/// Convert (RA, Dec) in degrees to a Cartesian unit vector on the unit sphere.
#[cfg(feature = "fits")]
pub fn sky_point_unit_vector(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

/// Angular separation using x87 80-bit extended precision on x86_64, Haversine elsewhere.
///
/// Used inside [`for_each_point_grid_match`] for the final per-candidate
/// distance check after dot-product pre-filtering.
#[cfg(feature = "fits")]
pub fn precise_angular_separation_arcsec(
    ra1_deg: f64,
    dec1_deg: f64,
    ra2_deg: f64,
    dec2_deg: f64,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use cd_kernel::angular_separation_arcsec_ext80_deg;
        let sep = angular_separation_arcsec_ext80_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg);
        if sep.is_finite() {
            sep
        } else {
            angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg)
    }
}

/// SIMD-accelerated cone search visitor over a [`PreparedPointGrid`].
///
/// Calls `visitor(candidate_index, separation_arcsec)` for every point
/// within `match_radius_arcsec` of the query position.  Processes four
/// candidates at a time with `f64x4` dot-product screening; a scalar tail
/// drains any remainder.
#[cfg(feature = "fits")]
pub fn for_each_point_grid_match<F>(
    prepared: &PreparedPointGrid,
    query_ra_deg: f64,
    query_dec_deg: f64,
    match_radius_arcsec: f64,
    mut visitor: F,
) where
    F: FnMut(usize, f64),
{
    let mut candidate_indices = Vec::new();
    prepared.grid.for_each_search_candidate(
        query_ra_deg,
        query_dec_deg,
        match_radius_arcsec,
        |idx| {
            candidate_indices.push(idx);
        },
    );
    if candidate_indices.is_empty() {
        return;
    }

    let [query_x, query_y, query_z] = sky_point_unit_vector(query_ra_deg, query_dec_deg);
    let cos_threshold = (match_radius_arcsec / 3600.0).to_radians().cos();
    let cos_threshold_slack = 1.0e-12;
    let points = prepared.grid.points();

    let mut offset = 0usize;
    while offset + 4 <= candidate_indices.len() {
        let chunk = &candidate_indices[offset..offset + 4];
        let dots = simd_dot_chunk(prepared, query_x, query_y, query_z, chunk);
        let dot_arr = dots.to_array();
        for lane in 0..4 {
            if dot_arr[lane] + cos_threshold_slack < cos_threshold {
                continue;
            }
            let candidate_index = chunk[lane];
            let point = &points[candidate_index];
            let separation_arcsec = precise_angular_separation_arcsec(
                query_ra_deg,
                query_dec_deg,
                point.ra_deg,
                point.dec_deg,
            );
            if separation_arcsec <= match_radius_arcsec {
                visitor(candidate_index, separation_arcsec);
            }
        }
        offset += 4;
    }

    while offset < candidate_indices.len() {
        let candidate_index = candidate_indices[offset];
        let point = &points[candidate_index];
        let separation_arcsec = precise_angular_separation_arcsec(
            query_ra_deg,
            query_dec_deg,
            point.ra_deg,
            point.dec_deg,
        );
        if separation_arcsec <= match_radius_arcsec {
            visitor(candidate_index, separation_arcsec);
        }
        offset += 1;
    }
}

/// Compute f64x4 dot products of four candidate unit vectors against the query vector.
#[cfg(feature = "fits")]
fn simd_dot_chunk(
    prepared: &PreparedPointGrid,
    query_x: f64,
    query_y: f64,
    query_z: f64,
    candidate_indices: &[usize],
) -> f64x4 {
    let idx0 = candidate_indices[0];
    let idx1 = candidate_indices[1];
    let idx2 = candidate_indices[2];
    let idx3 = candidate_indices[3];
    let xs = f64x4::from([
        prepared.unit_x[idx0],
        prepared.unit_x[idx1],
        prepared.unit_x[idx2],
        prepared.unit_x[idx3],
    ]);
    let ys = f64x4::from([
        prepared.unit_y[idx0],
        prepared.unit_y[idx1],
        prepared.unit_y[idx2],
        prepared.unit_y[idx3],
    ]);
    let zs = f64x4::from([
        prepared.unit_z[idx0],
        prepared.unit_z[idx1],
        prepared.unit_z[idx2],
        prepared.unit_z[idx3],
    ]);
    xs * f64x4::splat(query_x) + ys * f64x4::splat(query_y) + zs * f64x4::splat(query_z)
}

/// Converts equatorial (RA, Dec) in degrees to Galactic latitude (b) in degrees.
pub fn equatorial_to_galactic_lat(ra_deg: f64, dec_deg: f64) -> f64 {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let ra_ngp = 192.85948_f64.to_radians();
    let dec_ngp = 27.12825_f64.to_radians();

    let sin_b = dec.sin() * dec_ngp.sin() + dec.cos() * dec_ngp.cos() * (ra - ra_ngp).cos();
    sin_b.clamp(-1.0, 1.0).asin().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_ra_wraps_negative_and_large_values() {
        assert!((normalize_ra_deg(-10.0) - 350.0).abs() < 1.0e-12);
        assert!((normalize_ra_deg(725.0) - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn nearest_match_finds_expected_candidate() {
        let index = SkyGridIndex::from_points(
            vec![
                SkyPoint {
                    id: "a".to_string(),
                    ra_deg: 10.0,
                    dec_deg: 10.0,
                },
                SkyPoint {
                    id: "b".to_string(),
                    ra_deg: 42.0,
                    dec_deg: -2.0,
                },
            ],
            0.1,
        );
        let matched = index.nearest_within(42.0001, -2.0001, 2.0).unwrap();
        assert_eq!(matched.candidate_index, 1);
        assert!(matched.separation_arcsec < 1.0);
    }

    #[test]
    fn for_each_within_visits_all_candidates_in_radius() {
        let index = SkyGridIndex::from_points(
            vec![
                SkyPoint {
                    id: "a".to_string(),
                    ra_deg: 10.0,
                    dec_deg: 10.0,
                },
                SkyPoint {
                    id: "b".to_string(),
                    ra_deg: 10.0001,
                    dec_deg: 10.0,
                },
                SkyPoint {
                    id: "c".to_string(),
                    ra_deg: 11.0,
                    dec_deg: 10.0,
                },
            ],
            0.1,
        );
        let mut seen = Vec::new();
        index.for_each_within(10.00005, 10.0, 1.0, |matched| {
            seen.push(matched.candidate_index);
        });
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1]);
    }

    #[test]
    fn search_candidate_window_visits_neighboring_cells() {
        let index = SkyGridIndex::from_points(
            vec![
                SkyPoint {
                    id: "a".to_string(),
                    ra_deg: 10.0,
                    dec_deg: 10.0,
                },
                SkyPoint {
                    id: "b".to_string(),
                    ra_deg: 10.08,
                    dec_deg: 10.0,
                },
            ],
            0.1,
        );
        let mut seen = Vec::new();
        index.for_each_search_candidate(10.04, 10.0, 30.0, |idx| seen.push(idx));
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1]);
    }
}
