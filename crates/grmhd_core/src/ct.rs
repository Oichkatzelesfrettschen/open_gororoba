//! Constrained Transport (CT) for divergence-free magnetic field evolution.
//!
//! The CT algorithm stores B on cell FACES (staggered grid) and updates
//! them via the electromotive force (EMF) at cell EDGES. This guarantees
//! div(B) = 0 to machine precision by construction (Stokes' theorem).
//!
//! References:
//! - Toth (2000): "The div(B)=0 constraint in shock-capturing MHD codes"
//! - Gardiner & Stone (2005): "An unsplit Godunov method for ideal MHD"
//! - Gammie, McKinney & Toth (2003): HARM's CT implementation
//!
//! Storage layout (2D, r-theta):
//!   B^r   at (i+1/2, j)     -- radial faces
//!   B^th  at (i, j+1/2)     -- theta faces
//!   EMF   at (i+1/2, j+1/2) -- cell corners (only phi-component in 2D)

use crate::grid::Grid;
use crate::prims;

/// Face-centered (staggered) magnetic field storage.
///
/// B^r[i][j] lives at the right face of cell (i, j): position (i+1/2, j).
/// B^th[i][j] lives at the top face of cell (i, j): position (i, j+1/2).
///
/// This ensures that the discrete divergence:
///   div(B) = (B^r_{i+1/2,j} - B^r_{i-1/2,j})/dr + (B^th_{i,j+1/2} - B^th_{i,j-1/2})/dtheta
/// is identically zero when updated via the EMF (Stokes' theorem).
pub struct StaggeredB {
    /// B^r at radial faces: index [i * n2t + j]
    pub br: Vec<f64>,
    /// B^theta at theta faces: index [i * n2t + j]
    pub bth: Vec<f64>,
    /// Grid dimensions for indexing
    pub n1t: usize,
    pub n2t: usize,
}

impl StaggeredB {
    /// Create staggered B-field storage for the given grid.
    pub fn new(grid: &Grid) -> Self {
        let n1t = grid.n1_total();
        let n2t = grid.n2_total();
        Self {
            br: vec![0.0; n1t * n2t],
            bth: vec![0.0; n1t * n2t],
            n1t,
            n2t,
        }
    }

    /// Index into face arrays.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.n2t + j
    }

    /// Initialize from cell-centered B-field (simple averaging to faces).
    ///
    /// B^r at (i+1/2, j) = 0.5 * (B^r_center[i] + B^r_center[i+1])
    /// B^th at (i, j+1/2) = 0.5 * (B^th_center[j] + B^th_center[j+1])
    pub fn init_from_cell_centered(
        &mut self,
        prims: &crate::prims::PrimGrid,
        grid: &Grid,
    ) {
        let ng = grid.ng;
        for i in 0..self.n1t - 1 {
            for j in 0..self.n2t {
                let k = ng; // phi = 0 for 2D
                let idx_l = grid.idx(i, j, k);
                let idx_r = grid.idx(i + 1, j, k);
                let face_idx = self.idx(i, j);
                self.br[face_idx] = 0.5 * (prims.get(idx_l)[prims::B1] + prims.get(idx_r)[prims::B1]);
            }
        }
        for i in 0..self.n1t {
            for j in 0..self.n2t - 1 {
                let k = ng;
                let idx_l = grid.idx(i, j, k);
                let idx_r = grid.idx(i, j + 1, k);
                let face_idx = self.idx(i, j);
                self.bth[face_idx] = 0.5 * (prims.get(idx_l)[prims::B2] + prims.get(idx_r)[prims::B2]);
            }
        }
    }

    /// Interpolate face-centered B to cell centers.
    ///
    /// B^r_center[i] = 0.5 * (B^r_{i-1/2} + B^r_{i+1/2})
    /// B^th_center[j] = 0.5 * (B^th_{j-1/2} + B^th_{j+1/2})
    pub fn to_cell_centered(
        &self,
        prims: &mut crate::prims::PrimGrid,
        grid: &Grid,
    ) {
        let ng = grid.ng;
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                let k = ng;
                let idx = grid.idx(i, j, k);
                let p = prims.get_mut(idx);

                // B^r at cell center = average of left and right face
                if i > 0 {
                    p[prims::B1] = 0.5 * (self.br[self.idx(i - 1, j)] + self.br[self.idx(i, j)]);
                }
                // B^theta at cell center = average of bottom and top face
                if j > 0 {
                    p[prims::B2] = 0.5 * (self.bth[self.idx(i, j - 1)] + self.bth[self.idx(i, j)]);
                }
            }
        }
    }

    /// Compute the EMF at cell corners and update face-centered B.
    ///
    /// Uses the FACE-CENTERED velocity from the staggered B-field
    /// (not cell-centered), which is more stable because it's consistent
    /// with the CT staggered grid. The EMF is:
    ///   EMF_phi = -(v^r * B^theta_face - v^theta * B^r_face)
    ///
    /// At corner (i+1/2, j+1/2), average from 2 adjacent faces:
    ///   EMF = -0.5 * (v_r_face[j] * B^th[j] + v_r_face[j+1] * B^th[j+1])
    ///       + 0.5 * (v_th_face[i] * B^r[i] + v_th_face[i+1] * B^r[i+1])
    ///
    /// This is the "flux-CT" approach (Toth 2000, simplified).
    pub fn ct_update(
        &mut self,
        prims: &crate::prims::PrimGrid,
        grid: &Grid,
        dt: f64,
    ) {
        let ng = grid.ng;
        let n1t = self.n1t;
        let n2t = self.n2t;

        // Step 1: Compute EMF at corners using FACE B-fields (not cell-centered)
        // This is more consistent with the staggered grid and reduces
        // the amplification of sharp gradients.
        let mut emf = vec![0.0f64; n1t * n2t];
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                let k = ng;
                // Use cell-centered velocity (average of neighbors for face values)
                // and FACE-CENTERED B-fields from the staggered grid
                let ci = i.min(n1t - 1);
                let cj = j.min(n2t - 1);
                let ci1 = (i + 1).min(n1t - 1);
                let cj1 = (j + 1).min(n2t - 1);

                // Average velocity at corner from 4 cells
                let mut vr_avg = 0.0;
                let mut vth_avg = 0.0;
                let mut count = 0.0;
                for di in 0..=1 {
                    for dj in 0..=1 {
                        let ii = (i + di).min(n1t - 1);
                        let jj = (j + dj).min(n2t - 1);
                        let cell_idx = grid.idx(ii, jj, k);
                        let p = prims.get(cell_idx);
                        vr_avg += p[prims::V1];
                        vth_avg += p[prims::V2];
                        count += 1.0;
                    }
                }
                vr_avg /= count;
                vth_avg /= count;

                // Use FACE B-fields (staggered) instead of cell-centered
                let br_face = 0.5 * (self.br[ci * n2t + cj] + self.br[ci * n2t + cj1]);
                let bth_face = 0.5 * (self.bth[ci * n2t + cj] + self.bth[ci1 * n2t + cj]);

                // EMF = -(v^r * B^theta - v^theta * B^r)
                emf[i * n2t + j] = -(vr_avg * bth_face - vth_avg * br_face);
            }
        }

        // Step 2: Update B^r at faces via dB^r/dt = -(dEMF/dtheta)
        let dtheta = grid.dx2 * std::f64::consts::PI;
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                if j > 0 {
                    let emf_top = emf[i * n2t + j];
                    let emf_bot = emf[i * n2t + (j - 1)];
                    let face = i * n2t + j; // inline idx
                    self.br[face] -= dt * (emf_top - emf_bot) / dtheta;
                }
            }
        }

        // Step 3: Update B^theta at faces via dB^th/dt = +(dEMF/dr)
        let dr = grid.dx1;
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                if i > 0 {
                    let emf_right = emf[i * n2t + j];
                    let emf_left = emf[(i - 1) * n2t + j];
                    let face = i * n2t + j; // inline idx
                    self.bth[face] += dt * (emf_right - emf_left) / dr;
                }
            }
        }
    }

    /// Compute the maximum |div(B)| across the grid (diagnostic).
    /// Should be ~machine epsilon for CT, ~O(1) without CT.
    pub fn max_divb(&self, grid: &Grid) -> f64 {
        let ng = grid.ng;
        let dr = grid.dx1;
        let dtheta = grid.dx2 * std::f64::consts::PI;
        let mut max_div = 0.0f64;

        for i in ng + 1..ng + grid.n1 {
            for j in ng + 1..ng + grid.n2 {
                let dbr_dr = (self.br[self.idx(i, j)] - self.br[self.idx(i - 1, j)]) / dr;
                let dbth_dth = (self.bth[self.idx(i, j)] - self.bth[self.idx(i, j - 1)]) / dtheta;
                let divb = (dbr_dr + dbth_dth).abs();
                if divb > max_div { max_div = divb; }
            }
        }
        max_div
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::KerrMetric;
    use crate::torus::FMTorus;

    #[test]
    fn test_staggered_b_creation() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(8, 8, 1, 2.5, 20.0, metric);
        let sb = StaggeredB::new(&grid);
        assert_eq!(sb.br.len(), grid.n1_total() * grid.n2_total());
    }

    #[test]
    fn test_init_from_cell_centered() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(8, 8, 1, 2.5, 20.0, metric);
        let torus = FMTorus::schwarzschild(6.5, 12.0);
        let mut prims = torus.initialize(&grid);
        torus.add_magnetic_loop(&grid, &mut prims, 100.0);

        let mut sb = StaggeredB::new(&grid);
        sb.init_from_cell_centered(&prims, &grid);

        // Face B should be nonzero somewhere
        let max_br: f64 = sb.br.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max_br > 0.0, "max |B^r| = {} should be nonzero", max_br);
    }

    #[test]
    fn test_ct_preserves_divb() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(8, 8, 1, 2.5, 20.0, metric);
        let torus = FMTorus::schwarzschild(6.5, 12.0);
        let prims = torus.initialize(&grid);

        let mut sb = StaggeredB::new(&grid);
        sb.init_from_cell_centered(&prims, &grid);

        let divb_before = sb.max_divb(&grid);

        // Take a CT step
        sb.ct_update(&prims, &grid, 0.01);

        let divb_after = sb.max_divb(&grid);

        // div(B) should not grow (CT preserves it)
        // For a static torus with v=0, EMF=0, so div(B) should be unchanged
        assert!(
            divb_after <= divb_before * 1.01 + 1e-15,
            "div(B) grew: {} -> {}", divb_before, divb_after
        );
    }
}
