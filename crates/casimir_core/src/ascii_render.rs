//! ASCII cross-section renderer for Casimir energy density and warp geometry.
//!
//! Produces terminal-friendly visualizations using a 10-level character
//! intensity scale. Maps 2D grids of energy density values to an 80x40
//! character output.

/// Character palette from lowest to highest intensity.
const PALETTE: &[u8] = b" .:-=+*#%@";

/// A 2D grid of scalar values with spatial bounds.
pub struct AsciiGrid {
    /// Grid values stored row-major: data[row * cols + col].
    pub data: Vec<f64>,
    /// Number of rows (y-axis).
    pub rows: usize,
    /// Number of columns (x-axis).
    pub cols: usize,
    /// Spatial bounds: (x_min, x_max, y_min, y_max).
    pub bounds: (f64, f64, f64, f64),
}

impl AsciiGrid {
    /// Create a new grid initialized to zero.
    pub fn new(rows: usize, cols: usize, bounds: (f64, f64, f64, f64)) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            bounds,
        }
    }

    /// Set value at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = val;
        }
    }

    /// Get value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col]
        } else {
            0.0
        }
    }

    /// Render the grid to an ASCII string.
    ///
    /// Uses the character palette to map values to intensities.
    /// Negative values (typical for Casimir energy) are mapped with
    /// their absolute value.
    pub fn render(&self) -> String {
        let (min_val, max_val) = self.value_range();
        let abs_max = min_val.abs().max(max_val.abs());

        let mut output = String::with_capacity((self.cols + 1) * self.rows + 100);

        // Header with bounds
        output.push_str(&format!(
            "x: [{:.2}, {:.2}]  y: [{:.2}, {:.2}]  range: [{:.2e}, {:.2e}]\n",
            self.bounds.0, self.bounds.1, self.bounds.2, self.bounds.3, min_val, max_val
        ));

        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get(row, col);
                let intensity = if abs_max > 0.0 {
                    (val.abs() / abs_max * (PALETTE.len() - 1) as f64) as usize
                } else {
                    0
                };
                let idx = intensity.min(PALETTE.len() - 1);
                output.push(PALETTE[idx] as char);
            }
            output.push('\n');
        }

        output
    }

    /// Render with markers for special positions.
    ///
    /// `markers` is a list of (row, col, char) tuples.
    pub fn render_with_markers(&self, markers: &[(usize, usize, char)]) -> String {
        let (min_val, max_val) = self.value_range();
        let abs_max = min_val.abs().max(max_val.abs());

        let mut grid: Vec<Vec<char>> = Vec::with_capacity(self.rows);

        for row in 0..self.rows {
            let mut line = Vec::with_capacity(self.cols);
            for col in 0..self.cols {
                let val = self.get(row, col);
                let intensity = if abs_max > 0.0 {
                    (val.abs() / abs_max * (PALETTE.len() - 1) as f64) as usize
                } else {
                    0
                };
                let idx = intensity.min(PALETTE.len() - 1);
                line.push(PALETTE[idx] as char);
            }
            grid.push(line);
        }

        // Apply markers
        for &(row, col, ch) in markers {
            if row < self.rows && col < self.cols {
                grid[row][col] = ch;
            }
        }

        let mut output = String::with_capacity((self.cols + 1) * self.rows + 100);
        output.push_str(&format!(
            "x: [{:.2}, {:.2}]  y: [{:.2}, {:.2}]  range: [{:.2e}, {:.2e}]\n",
            self.bounds.0, self.bounds.1, self.bounds.2, self.bounds.3, min_val, max_val
        ));
        for row in &grid {
            let line: String = row.iter().collect();
            output.push_str(&line);
            output.push('\n');
        }
        output
    }

    /// Compute (min, max) of the data values.
    fn value_range(&self) -> (f64, f64) {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in &self.data {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        if !min_val.is_finite() {
            min_val = 0.0;
        }
        if !max_val.is_finite() {
            max_val = 0.0;
        }
        (min_val, max_val)
    }
}

/// Create an energy density cross-section grid for a warp bubble.
///
/// Evaluates `energy_fn(y, z)` over a 2D grid in the y-z plane (x=0).
pub fn energy_cross_section<F: Fn(f64, f64) -> f64>(
    energy_fn: F,
    half_size: f64,
    rows: usize,
    cols: usize,
) -> AsciiGrid {
    let mut grid = AsciiGrid::new(rows, cols, (-half_size, half_size, -half_size, half_size));

    for row in 0..rows {
        let z = half_size - (row as f64 + 0.5) * 2.0 * half_size / rows as f64;
        for col in 0..cols {
            let y = -half_size + (col as f64 + 0.5) * 2.0 * half_size / cols as f64;
            grid.set(row, col, energy_fn(y, z));
        }
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = AsciiGrid::new(10, 20, (-1.0, 1.0, -1.0, 1.0));
        assert_eq!(grid.rows, 10);
        assert_eq!(grid.cols, 20);
        assert_eq!(grid.data.len(), 200);
    }

    #[test]
    fn test_render_uniform_zero() {
        let grid = AsciiGrid::new(3, 5, (-1.0, 1.0, -1.0, 1.0));
        let output = grid.render();
        assert!(
            output.contains("     "),
            "zero grid should render as spaces"
        );
    }

    #[test]
    fn test_cross_section() {
        let grid = energy_cross_section(|y, z| -(y * y + z * z), 1.0, 10, 20);
        assert_eq!(grid.rows, 10);
        assert_eq!(grid.cols, 20);
        // Center should have the smallest absolute value (closest to zero)
        let center_val = grid.get(5, 10);
        let edge_val = grid.get(0, 0);
        assert!(
            center_val.abs() < edge_val.abs(),
            "center |{center_val}| should be less than edge |{edge_val}|"
        );
    }
}
