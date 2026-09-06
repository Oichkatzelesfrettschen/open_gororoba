//! Directional moments of fixed, sampled transverse density fields.
//!
//! ArXiv:2212.01324 Eqs. 9 and 12 motivate the density-weighted path ratio.
//! The declared ray is forward, s >= 0, from each cell center. Participant
//! density is piecewise constant inside cells and zero outside the rectangle.
//! Cell-crossing integration introduces no ray-step approximation; binary64
//! arithmetic and cell-center sampling remain numerical approximations.

use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathError {
    InvalidInput(&'static str),
    InvalidGrid,
    InvalidDensity,
    InvalidAngle,
    NonfiniteArithmetic,
    UnresolvedCrossing,
    EmptyNormalization,
    InvalidMoments,
}
impl fmt::Display for PathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "directional path: {self:?}")
    }
}
impl Error for PathError {}

/// Global weighted moments, with length equal to twice their ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalPathMoments {
    /// Joint first-distance moment; dimensionless for densities in fm^-2.
    pub numerator: f64,
    /// Joint line-mass normalization in fm^-1 for densities in fm^-2.
    pub denominator: f64,
    /// Twice the ratio of the retained joint moments, in fm.
    pub length_fm: f64,
}

/// Cartesian cell densities in row-major order, x varying fastest.
#[derive(Debug, Clone)]
pub struct TransverseDensityGrid {
    nx: usize,
    ny: usize,
    x_edges: Vec<f64>,
    y_edges: Vec<f64>,
    x_centers: Vec<f64>,
    y_centers: Vec<f64>,
    cell_area: f64,
    collision_density: Vec<f64>,
    participant_density: Vec<f64>,
}

fn finite(value: f64) -> Result<f64, PathError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PathError::NonfiniteArithmetic)
    }
}

fn axis(count: usize, minimum: f64, spacing: f64) -> Result<(Vec<f64>, Vec<f64>), PathError> {
    if count == 0 || !minimum.is_finite() || !spacing.is_finite() || spacing <= 0.0 {
        return Err(PathError::InvalidGrid);
    }
    let edge_count = count.checked_add(1).ok_or(PathError::InvalidGrid)?;
    let mut edges = Vec::with_capacity(edge_count);
    for index in 0..edge_count {
        edges.push(finite(minimum + index as f64 * spacing)?);
    }
    let mut centers = Vec::with_capacity(count);
    for pair in edges.windows(2) {
        let center = pair[0] * 0.5 + pair[1] * 0.5;
        if !(pair[0] < center && center < pair[1]) {
            return Err(PathError::InvalidGrid);
        }
        centers.push(center);
    }
    Ok((edges, centers))
}

impl TransverseDensityGrid {
    #[allow(
        clippy::too_many_arguments,
        reason = "Grid geometry and its two density fields form one explicit admission boundary"
    )]
    pub fn new(
        nx: usize,
        ny: usize,
        x_min_fm: f64,
        y_min_fm: f64,
        dx_fm: f64,
        dy_fm: f64,
        collision_density: Vec<f64>,
        participant_density: Vec<f64>,
    ) -> Result<Self, PathError> {
        let cells = nx.checked_mul(ny).ok_or(PathError::InvalidGrid)?;
        if cells == 0 || collision_density.len() != cells || participant_density.len() != cells {
            return Err(PathError::InvalidGrid);
        }
        if collision_density
            .iter()
            .chain(&participant_density)
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PathError::InvalidDensity);
        }
        let (x_edges, x_centers) = axis(nx, x_min_fm, dx_fm)?;
        let (y_edges, y_centers) = axis(ny, y_min_fm, dy_fm)?;
        let cell_area = finite(dx_fm * dy_fm)?;
        if cell_area <= 0.0 {
            return Err(PathError::InvalidGrid);
        }
        Ok(Self {
            nx,
            ny,
            x_edges,
            y_edges,
            x_centers,
            y_centers,
            cell_area,
            collision_density,
            participant_density,
        })
    }

    /// Integrate rho_part and s*rho_part on each forward ray, then take the
    /// ratio of collision-density-weighted sums, rather than averaging lengths.
    pub fn directional_moments(&self, angle_rad: f64) -> Result<DirectionalPathMoments, PathError> {
        if !angle_rad.is_finite() {
            return Err(PathError::InvalidAngle);
        }
        let (direction_y, direction_x) = angle_rad.sin_cos();
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for origin_y in 0..self.ny {
            for origin_x in 0..self.nx {
                let origin_index = origin_x + self.nx * origin_y;
                if self.collision_density[origin_index] == 0.0 {
                    continue;
                }
                let origin = [self.x_centers[origin_x], self.y_centers[origin_y]];
                let mut horizontal = origin_x;
                let mut vertical = origin_y;
                let mut distance = 0.0;
                let mut line_mass = 0.0;
                let mut line_moment = 0.0;
                loop {
                    // Infinity denotes a parallel axis or a crossing beyond
                    // binary64 range; only the earliest finite crossing is used.
                    let crossing_x = crossing(&self.x_edges, horizontal, origin[0], direction_x);
                    let crossing_y = crossing(&self.y_edges, vertical, origin[1], direction_y);
                    let next = finite(crossing_x.min(crossing_y))?;
                    if next <= distance {
                        return Err(PathError::UnresolvedCrossing);
                    }
                    let width = finite(next - distance)?;
                    let density = self.participant_density[horizontal + self.nx * vertical];
                    if density > 0.0 {
                        let segment_mass = finite(density * width)?;
                        line_mass = finite(line_mass + segment_mass)?;
                        let midpoint = finite(distance * 0.5 + next * 0.5)?;
                        line_moment = finite(line_moment + finite(segment_mass * midpoint)?)?;
                    }
                    distance = next;
                    let move_x = crossing_x == next;
                    let move_y = crossing_y == next;
                    if (move_x
                        && ((direction_x > 0.0 && horizontal + 1 == self.nx)
                            || (direction_x < 0.0 && horizontal == 0)))
                        || (move_y
                            && ((direction_y > 0.0 && vertical + 1 == self.ny)
                                || (direction_y < 0.0 && vertical == 0)))
                    {
                        break;
                    }
                    if move_x {
                        if direction_x > 0.0 {
                            horizontal += 1;
                        } else {
                            horizontal -= 1;
                        }
                    }
                    if move_y {
                        if direction_y > 0.0 {
                            vertical += 1;
                        } else {
                            vertical -= 1;
                        }
                    }
                }
                let weight = finite(self.cell_area * self.collision_density[origin_index])?;
                numerator = finite(numerator + finite(weight * line_moment)?)?;
                denominator = finite(denominator + finite(weight * line_mass)?)?;
            }
        }
        if denominator <= 0.0 {
            return Err(PathError::EmptyNormalization);
        }
        let length_fm = finite(2.0 * finite(numerator / denominator)?)?;
        if length_fm <= 0.0 {
            return Err(PathError::InvalidMoments);
        }
        Ok(DirectionalPathMoments {
            numerator,
            denominator,
            length_fm,
        })
    }
}

fn crossing(edges: &[f64], index: usize, origin: f64, direction: f64) -> f64 {
    if direction > 0.0 {
        (edges[index + 1] - origin) / direction
    } else if direction < 0.0 {
        (edges[index] - origin) / direction
    } else {
        f64::INFINITY
    }
}

/// Geometric path anisotropy (Ly-Lx)/(Ly+Lx); independent of measured flow.
pub fn path_eccentricity(
    in_plane: &DirectionalPathMoments,
    out_of_plane: &DirectionalPathMoments,
) -> Result<f64, PathError> {
    for moment in [in_plane, out_of_plane] {
        if !moment.numerator.is_finite()
            || moment.numerator <= 0.0
            || !moment.denominator.is_finite()
            || moment.denominator <= 0.0
            || !moment.length_fm.is_finite()
            || moment.length_fm <= 0.0
        {
            return Err(PathError::InvalidMoments);
        }
        let reconstructed = finite(2.0 * finite(moment.numerator / moment.denominator)?)?;
        if reconstructed != moment.length_fm {
            return Err(PathError::InvalidMoments);
        }
    }
    let sum = finite(out_of_plane.length_fm + in_plane.length_fm)?;
    finite((out_of_plane.length_fm - in_plane.length_fm) / sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_inputs_and_empty_normalization() {
        assert!(TransverseDensityGrid::new(0, 1, 0.0, 0.0, 1.0, 1.0, vec![], vec![]).is_err());
        assert!(
            TransverseDensityGrid::new(1, 1, 0.0, 0.0, 0.0, 1.0, vec![1.0], vec![1.0]).is_err()
        );
        assert!(
            TransverseDensityGrid::new(1, 1, 0.0, 0.0, 1.0, 1.0, vec![-1.0], vec![1.0]).is_err()
        );
        let grid =
            TransverseDensityGrid::new(1, 1, 0.0, 0.0, 1.0, 1.0, vec![0.0], vec![1.0]).unwrap();
        assert_eq!(
            grid.directional_moments(0.0),
            Err(PathError::EmptyNormalization)
        );
        assert_eq!(
            grid.directional_moments(f64::NAN),
            Err(PathError::InvalidAngle)
        );
    }
    #[test]
    fn single_cell_axis_and_diagonal_factor_two() {
        let grid =
            TransverseDensityGrid::new(1, 1, -1.0, -1.0, 2.0, 2.0, vec![3.0], vec![5.0]).unwrap();
        let axis = grid.directional_moments(0.0).unwrap();
        assert_eq!(axis.denominator, 60.0);
        assert_eq!(axis.numerator, 30.0);
        assert_eq!(axis.length_fm, 1.0);
        let diagonal = grid
            .directional_moments(std::f64::consts::FRAC_PI_4)
            .unwrap();
        assert!((diagonal.length_fm - 2.0_f64.sqrt()).abs() < 1e-14);
    }
    #[test]
    fn diagonal_internal_corner_and_overflow() {
        let grid = TransverseDensityGrid::new(
            2,
            2,
            0.0,
            0.0,
            1.0,
            1.0,
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0; 4],
        )
        .unwrap();
        let path = grid
            .directional_moments(std::f64::consts::FRAC_PI_4)
            .unwrap();
        assert!((path.length_fm - 1.5 * 2.0_f64.sqrt()).abs() < 1e-14);
        let overflow =
            TransverseDensityGrid::new(1, 1, 0.0, 0.0, 2.0, 2.0, vec![f64::MAX], vec![1.0])
                .unwrap();
        assert_eq!(
            overflow.directional_moments(0.0),
            Err(PathError::NonfiniteArithmetic)
        );
    }
}
