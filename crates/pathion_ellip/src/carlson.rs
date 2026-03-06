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
    pub fn rf_32d(
        &self,
        x: &[f64; 32],
        y: &[f64; 32],
        z: &[f64; 32],
    ) -> anyhow::Result<[f64; 32]> {
        let x_planes = self.diagonalizer.project(x);
        let y_planes = self.diagonalizer.project(y);
        let z_planes = self.diagonalizer.project(z);

        let mut res_planes = [Complex64::new(0.0, 0.0); 16];
        res_planes
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, res)| {
                *res = carlson_rf_complex(x_planes[i], y_planes[i], z_planes[i]);
            });
        Ok(self.diagonalizer.recompose(&res_planes))
    }

    /// Carlson $R_D(x, y, z) = R_J(x,y,z,z)$ over 32D Pathion algebra.
    pub fn rd_32d(
        &self,
        x: &[f64; 32],
        y: &[f64; 32],
        z: &[f64; 32],
    ) -> anyhow::Result<[f64; 32]> {
        let x_planes = self.diagonalizer.project(x);
        let y_planes = self.diagonalizer.project(y);
        let z_planes = self.diagonalizer.project(z);

        let mut res_planes = [Complex64::new(0.0, 0.0); 16];
        res_planes
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, res)| {
                *res = carlson_rj_complex(x_planes[i], y_planes[i], z_planes[i], z_planes[i]);
            });
        Ok(self.diagonalizer.recompose(&res_planes))
    }

    /// Carlson $R_J(x, y, z, p)$ over 32D Pathion algebra.
    pub fn rj_32d(
        &self,
        x: &[f64; 32],
        y: &[f64; 32],
        z: &[f64; 32],
        p: &[f64; 32],
    ) -> anyhow::Result<[f64; 32]> {
        let x_planes = self.diagonalizer.project(x);
        let y_planes = self.diagonalizer.project(y);
        let z_planes = self.diagonalizer.project(z);
        let p_planes = self.diagonalizer.project(p);

        let mut res_planes = [Complex64::new(0.0, 0.0); 16];
        res_planes
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, res)| {
                *res = carlson_rj_complex(
                    x_planes[i],
                    y_planes[i],
                    z_planes[i],
                    p_planes[i],
                );
            });
        Ok(self.diagonalizer.recompose(&res_planes))
    }
}

impl Default for PathionCarlson {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Complex Carlson RF (DLMF 19.36.1)
// ---------------------------------------------------------------------------

/// Carlson RF(x,y,z) over C. Duplication algorithm.
pub fn carlson_rf_complex(mut x: Complex64, mut y: Complex64, mut z: Complex64) -> Complex64 {
    let err_tol = 1e-10;

    for _ in 0..100 {
        let mu = (x + y + z) / 3.0;
        let dx = 1.0 - x / mu;
        let dy = 1.0 - y / mu;
        let dz = 1.0 - z / mu;
        let max_err = dx.norm().max(dy.norm()).max(dz.norm());

        if max_err < err_tol {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            return (1.0
                - (1.0 / 10.0) * e2
                + (1.0 / 14.0) * e3
                + (1.0 / 24.0) * e2 * e2
                - (3.0 / 44.0) * e2 * e3)
                / mu.sqrt();
        }

        let lx = x.sqrt();
        let ly = y.sqrt();
        let lz = z.sqrt();
        let lambda = lx * ly + ly * lz + lz * lx;

        x = (x + lambda) / 4.0;
        y = (y + lambda) / 4.0;
        z = (z + lambda) / 4.0;
    }

    Complex64::new(f64::NAN, f64::NAN)
}

// ---------------------------------------------------------------------------
// Complex Carlson RC (DLMF 19.2.17)
// ---------------------------------------------------------------------------

/// Carlson RC(x, y) = RF(x, y, y).
pub fn carlson_rc_complex(x: Complex64, y: Complex64) -> Complex64 {
    carlson_rf_complex(x, y, y)
}

// ---------------------------------------------------------------------------
// Complex Carlson RJ (adapted from Boost/ellip, DLMF 19.36.2)
// ---------------------------------------------------------------------------

/// Carlson RJ(x,y,z,p) over C.
///
/// Adapted from the Boost Math / ellip crate algorithm (Carlson 1995).
/// Uses delta-based RC accumulation with RC(1, 1+en) form.
pub fn carlson_rj_complex(
    x: Complex64,
    y: Complex64,
    z: Complex64,
    p: Complex64,
) -> Complex64 {
    let mut xn = x;
    let mut yn = y;
    let mut zn = z;
    let mut pn = p;

    let a0 = (x + y + z + 2.0 * p) / 5.0;
    let mut an = a0;
    let mut delta = (p - x) * (p - y) * (p - z);
    let mut fmn = Complex64::new(1.0, 0.0);
    let mut rc_sum = Complex64::new(0.0, 0.0);

    let q_factor = {
        let err_scale = Complex64::new(1e-10_f64.powf(-1.0 / 8.0), 0.0);
        let max_dev = (an - x)
            .norm()
            .max((an - y).norm())
            .max((an - z).norm())
            .max((an - p).norm());
        err_scale * Complex64::new(max_dev, 0.0)
    };

    for _ in 0..100 {
        let rx = xn.sqrt();
        let ry = yn.sqrt();
        let rz = zn.sqrt();
        let rp = pn.sqrt();
        let dn = (rp + rx) * (rp + ry) * (rp + rz);
        let en = delta / (dn * dn);

        // RC(1, 1+en) for complex en: use atan/atanh formulation
        let rc1p = complex_rc1p(en);
        rc_sum += fmn / dn * rc1p;

        let lambda = rx * ry + rx * rz + ry * rz;
        an = (an + lambda) / 4.0;
        fmn /= 4.0;

        // Convergence check: fmn * q < |an|
        if (fmn * q_factor).norm() < an.norm() {
            // Compute final series from residuals relative to a0
            let xr = fmn * (a0 - x) / an;
            let yr = fmn * (a0 - y) / an;
            let zr = fmn * (a0 - z) / an;
            let pr = (-xr - yr - zr) / 2.0;
            let xyz = xr * yr * zr;
            let p2 = pr * pr;
            let p3 = p2 * pr;

            let e2 = xr * yr + xr * zr + yr * zr - 3.0 * p2;
            let e3 = xyz + 2.0 * e2 * pr + 4.0 * p3;
            let e4 = (2.0 * xyz + e2 * pr + 3.0 * p3) * pr;
            let e5 = xyz * p2;

            let series = 1.0
                - 3.0 * e2 / 14.0
                + e3 / 6.0
                + 9.0 * e2 * e2 / 88.0
                - 3.0 * e4 / 22.0
                - 9.0 * e2 * e3 / 52.0
                + 3.0 * e5 / 26.0;

            let result = fmn / (an * an.sqrt()) * series;
            return result + 6.0 * rc_sum;
        }

        xn = (xn + lambda) / 4.0;
        yn = (yn + lambda) / 4.0;
        zn = (zn + lambda) / 4.0;
        pn = (pn + lambda) / 4.0;
        delta /= 64.0;
    }

    Complex64::new(f64::NAN, f64::NAN)
}

/// Compute RC(1, 1+y) for complex y.
///
/// For real positive y: atan(sqrt(y)) / sqrt(y)
/// For real negative y with |y| < 1: atanh(sqrt(-y)) / sqrt(-y)
/// For general complex y: use atan formula.
fn complex_rc1p(y: Complex64) -> Complex64 {
    if y.norm() < 1e-14 {
        return Complex64::new(1.0, 0.0);
    }

    // RC(1, 1+y) = RF(1, 1+y, 1+y) -- use the general RF as fallback
    // But for better numerical behavior, use the atan/atanh formula:
    // RC(1, w) = atan(sqrt((w-1)/1)) / sqrt(w-1) when w > 0
    // More generally: RC(1, w) = (1/sqrt(w)) * atanh(sqrt(1 - 1/w)) for |w| > 1
    // Or simply: use RF(1, 1+y, 1+y) which is always correct.
    carlson_rf_complex(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0) + y, Complex64::new(1.0, 0.0) + y)
}

/// Carlson RD(x,y,z) = RJ(x,y,z,z).
pub fn carlson_rd_complex(x: Complex64, y: Complex64, z: Complex64) -> Complex64 {
    carlson_rj_complex(x, y, z, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rf_identity() {
        // RF(1, 1, 1) = 1.0
        let one = Complex64::new(1.0, 0.0);
        let rf = carlson_rf_complex(one, one, one);
        assert_relative_eq!(rf.re, 1.0, epsilon = 1e-8);
        assert!(rf.im.abs() < 1e-10);
    }

    #[test]
    fn rf_known_value() {
        // RF(0, 1, 2) = 1.3110287771
        let rf = carlson_rf_complex(
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        );
        assert_relative_eq!(rf.re, 1.3110287771, epsilon = 1e-6);
        assert!(rf.im.abs() < 1e-8);
    }

    #[test]
    fn rj_known_value_1234() {
        // RJ(1,2,3,4) = 0.23984809974956783 (scipy reference)
        let rj = carlson_rj_complex(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        );
        assert_relative_eq!(rj.re, 0.23984809975, epsilon = 1e-6);
        assert!(rj.im.abs() < 1e-8);
    }

    #[test]
    fn rd_known_value() {
        // RD(0, 2, 1) = 1.7972103521 (scipy reference)
        let rd = carlson_rd_complex(
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        assert_relative_eq!(rd.re, 1.7972103521, epsilon = 1e-5);
        assert!(rd.im.abs() < 1e-6);
    }

    #[test]
    fn rd_equals_rj_z_z() {
        // RD(x,y,z) = RJ(x,y,z,z)
        let x = Complex64::new(1.0, 0.0);
        let y = Complex64::new(2.0, 0.0);
        let z = Complex64::new(3.0, 0.0);
        let rd = carlson_rd_complex(x, y, z);
        let rj = carlson_rj_complex(x, y, z, z);
        assert_relative_eq!(rd.re, rj.re, epsilon = 1e-8);
        assert_relative_eq!(rd.im, rj.im, max_relative = 1e-8);
    }

    #[test]
    fn rj_real_positive() {
        let rj = carlson_rj_complex(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        );
        assert!(rj.re > 0.0);
        assert!(rj.im.abs() < 1e-8);
    }

    #[test]
    fn rc_reduces_to_rf() {
        let x = Complex64::new(2.0, 0.0);
        let y = Complex64::new(3.0, 0.0);
        let rc = carlson_rc_complex(x, y);
        let rf = carlson_rf_complex(x, y, y);
        assert_relative_eq!(rc.re, rf.re, epsilon = 1e-10);
    }

    #[test]
    fn rj_symmetric_in_xyz() {
        let x = Complex64::new(1.0, 0.0);
        let y = Complex64::new(2.0, 0.0);
        let z = Complex64::new(3.0, 0.0);
        let p = Complex64::new(0.5, 0.0);
        let r1 = carlson_rj_complex(x, y, z, p);
        let r2 = carlson_rj_complex(z, x, y, p);
        let r3 = carlson_rj_complex(y, z, x, p);
        assert_relative_eq!(r1.re, r2.re, epsilon = 1e-8);
        assert_relative_eq!(r1.re, r3.re, epsilon = 1e-8);
    }
}
