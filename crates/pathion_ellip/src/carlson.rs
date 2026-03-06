use crate::diagonalizer::PathionDiagonalizer;
use num_complex::Complex64;
use rayon::prelude::*;

/// Computes Carlson elliptic integrals over the 32D Pathion algebra.
pub struct PathionCarlson {
    diagonalizer: PathionDiagonalizer,
}

impl PathionCarlson {
    pub fn new() -> Self {
        Self {
            diagonalizer: PathionDiagonalizer::new(),
        }
    }

    /// Carlson $R_F(x, y, z)$ over 32D Pathion algebra.
    /// 
    /// Evaluated by diagonalizing into 16 complex planes, computing
    /// complex $R_F$ in each plane in parallel, and recomposing.
    pub fn rf_32d(&self, x: &[f64; 32], y: &[f64; 32], z: &[f64; 32]) -> anyhow::Result<[f64; 32]> {
        let x_planes = self.diagonalizer.project(x);
        let y_planes = self.diagonalizer.project(y);
        let z_planes = self.diagonalizer.project(z);

        let mut res_planes = [Complex64::new(0.0, 0.0); 16];

        res_planes.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = carlson_rf_complex(x_planes[i], y_planes[i], z_planes[i]);
        });

        Ok(self.diagonalizer.recompose(&res_planes))
    }

    // Additional integrals (RD, RJ) will be added here.
}

impl Default for PathionCarlson {
    fn default() -> Self {
        Self::new()
    }
}

/// Carlson's symmetric elliptic integral of the first kind $R_F(x,y,z)$ over $\mathbb{C}$.
/// Evaluated using the duplication theorem algorithm.
fn carlson_rf_complex(mut x: Complex64, mut y: Complex64, mut z: Complex64) -> Complex64 {
    let err_tol = 1e-6; // Tolerance
    
    // In practice, a max iteration limit should be used
    for _ in 0..100 {
        let mu = (x + y + z) / 3.0;
        let dx = 1.0 - x / mu;
        let dy = 1.0 - y / mu;
        let dz = 1.0 - z / mu;
        let max_err = dx.norm().max(dy.norm()).max(dz.norm());
        
        if max_err < err_tol {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            let res = (1.0 - (1.0/10.0)*e2 + (1.0/14.0)*e3 + (1.0/24.0)*e2*e2 - (3.0/44.0)*e2*e3) / mu.sqrt();
            return res;
        }
        
        let lx = x.sqrt();
        let ly = y.sqrt();
        let lz = z.sqrt();
        
        let lambda = lx*ly + ly*lz + lz*lx;
        
        x = (x + lambda) / 4.0;
        y = (y + lambda) / 4.0;
        z = (z + lambda) / 4.0;
    }
    
    Complex64::new(f64::NAN, f64::NAN)
}
