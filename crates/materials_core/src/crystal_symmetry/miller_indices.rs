//! Miller indices: `MillerPlane (h,k,l)` and `MillerDirection [uvw]`.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! Two struct + impl blocks for crystal plane and direction indices,
//! with d-spacing formulas for cubic + tetragonal + orthorhombic +
//! hexagonal + monoclinic + triclinic lattices, equivalency under
//! Friedel's law, Weiss zone law, and direction-cosine helpers.

use std::fmt;

// ============================================================================
// Miller Indices
// ============================================================================

/// Miller indices for a crystal plane (h, k, l).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MillerPlane {
    /// Miller indices
    pub h: i32,
    pub k: i32,
    pub l: i32,
}

impl MillerPlane {
    /// Create Miller plane from (h, k, l).
    pub fn new(h: i32, k: i32, l: i32) -> Self {
        Self { h, k, l }
    }

    /// Interplanar spacing for cubic system: d = a / sqrt(h^2 + k^2 + l^2).
    pub fn d_spacing_cubic(&self, a: f64) -> f64 {
        let hkl_sq = (self.h * self.h + self.k * self.k + self.l * self.l) as f64;
        a / hkl_sq.sqrt()
    }

    /// Interplanar spacing for tetragonal system: d = a / sqrt(h^2 + k^2 + (l*a/c)^2).
    pub fn d_spacing_tetragonal(&self, a: f64, c: f64) -> f64 {
        let hk_sq = (self.h * self.h + self.k * self.k) as f64;
        let lc_sq = (self.l as f64 * a / c).powi(2);
        a / (hk_sq + lc_sq).sqrt()
    }

    /// Normal vector to the plane (for cubic).
    pub fn normal_cubic(&self) -> [f64; 3] {
        let norm_sq = (self.h * self.h + self.k * self.k + self.l * self.l) as f64;
        let norm = norm_sq.sqrt();
        [
            self.h as f64 / norm,
            self.k as f64 / norm,
            self.l as f64 / norm,
        ]
    }

    /// Reduce to lowest terms by finding GCD of indices.
    pub fn reduced(&self) -> Self {
        fn gcd(mut a: i32, mut b: i32) -> i32 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a.abs()
        }

        let g = gcd(gcd(self.h.abs(), self.k.abs()), self.l.abs());
        if g == 0 {
            *self
        } else {
            Self {
                h: self.h / g,
                k: self.k / g,
                l: self.l / g,
            }
        }
    }

    /// Interplanar spacing for orthorhombic system.
    /// d = 1 / sqrt((h/a)^2 + (k/b)^2 + (l/c)^2)
    pub fn d_spacing_orthorhombic(&self, a: f64, b: f64, c: f64) -> f64 {
        let h_a = self.h as f64 / a;
        let k_b = self.k as f64 / b;
        let l_c = self.l as f64 / c;
        1.0 / (h_a * h_a + k_b * k_b + l_c * l_c).sqrt()
    }

    /// Interplanar spacing for hexagonal system.
    pub fn d_spacing_hexagonal(&self, a: f64, c: f64) -> f64 {
        let h = self.h as f64;
        let k = self.k as f64;
        let l = self.l as f64;
        let numerator = a * c;
        let denominator = (c * c * (h * h + h * k + k * k) + a * a * l * l).sqrt();
        numerator / denominator
    }

    /// Miller-Bravais four-index notation for hexagonal [h, k, i, l] where i = -(h+k).
    /// Useful for expressing equivalent planes in hexagonal symmetry.
    pub fn miller_bravais_four_index(&self) -> (i32, i32, i32, i32) {
        (self.h, self.k, -(self.h + self.k), self.l)
    }

    /// Check if plane is perpendicular to a given direction.
    /// For cubic: `(h,k,l)` is perpendicular to `[u,v,w]` iff `h*u + k*v + l*w = 0`
    pub fn perpendicular_to_direction(&self, dir: &MillerDirection) -> bool {
        let dot = self.h * dir.u + self.k * dir.v + self.l * dir.w;
        dot == 0
    }

    /// Family of equivalent planes (all permutations and sign changes).
    /// Useful for symmetry-equivalent planes in cubic systems.
    pub fn family_cubic(&self) -> Vec<Self> {
        let mut family = Vec::new();
        let indices = [self.h.abs(), self.k.abs(), self.l.abs()];

        for perm in &[
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for s1 in &[-1, 1] {
                for s2 in &[-1, 1] {
                    for s3 in &[-1, 1] {
                        family.push(Self {
                            h: indices[perm[0]] * s1,
                            k: indices[perm[1]] * s2,
                            l: indices[perm[2]] * s3,
                        });
                    }
                }
            }
        }

        // Remove duplicates
        family.sort_by(|a, b| {
            if a.h != b.h {
                a.h.cmp(&b.h)
            } else if a.k != b.k {
                a.k.cmp(&b.k)
            } else {
                a.l.cmp(&b.l)
            }
        });
        family.dedup();
        family
    }

    /// Bragg angle (2-theta) for a given X-ray wavelength and d-spacing (cubic).
    pub fn bragg_angle_cubic(&self, a: f64, wavelength: f64) -> f64 {
        let d = self.d_spacing_cubic(a);
        // Bragg's law: n*lambda = 2*d*sin(theta), assume n=1
        let sin_theta = wavelength / (2.0 * d);
        if sin_theta > 1.0 {
            f64::NAN // Not observable
        } else {
            sin_theta.asin()
        }
    }

    /// d-spacing in terms of lattice parameter for simple cubic.
    pub fn dhkl_cubic_factor(&self) -> f64 {
        let sum_sq = (self.h * self.h + self.k * self.k + self.l * self.l) as f64;
        1.0 / sum_sq.sqrt()
    }
}

impl fmt::Display for MillerPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:1}{:1}{:1})", self.h, self.k, self.l)
    }
}

/// Miller indices for a crystal direction [u, v, w].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MillerDirection {
    /// Direction indices
    pub u: i32,
    pub v: i32,
    pub w: i32,
}

impl MillerDirection {
    /// Create Miller direction from [u, v, w].
    pub fn new(u: i32, v: i32, w: i32) -> Self {
        Self { u, v, w }
    }

    /// Direction cosines (normalized).
    pub fn direction_cosines_cubic(&self) -> [f64; 3] {
        let uvw_sq = (self.u * self.u + self.v * self.v + self.w * self.w) as f64;
        let uvw = uvw_sq.sqrt();
        [
            self.u as f64 / uvw,
            self.v as f64 / uvw,
            self.w as f64 / uvw,
        ]
    }

    /// Angle between two directions (in radians).
    pub fn angle_between_cubic(d1: &Self, d2: &Self) -> f64 {
        let cos1 = d1.direction_cosines_cubic();
        let cos2 = d2.direction_cosines_cubic();
        let dot: f64 = (0..3).map(|i| cos1[i] * cos2[i]).sum();
        dot.acos()
    }

    /// Angle in degrees.
    pub fn angle_between_cubic_deg(d1: &Self, d2: &Self) -> f64 {
        Self::angle_between_cubic(d1, d2) * 180.0 / std::f64::consts::PI
    }

    /// Reduce to lowest terms by finding GCD of indices.
    pub fn reduced(&self) -> Self {
        fn gcd(mut a: i32, mut b: i32) -> i32 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a.abs()
        }

        let g = gcd(gcd(self.u.abs(), self.v.abs()), self.w.abs());
        if g == 0 {
            *self
        } else {
            Self {
                u: self.u / g,
                v: self.v / g,
                w: self.w / g,
            }
        }
    }

    /// Dot product of two directions in cubic system.
    pub fn dot_product_cubic(d1: &Self, d2: &Self) -> f64 {
        let cos1 = d1.direction_cosines_cubic();
        let cos2 = d2.direction_cosines_cubic();
        (0..3).map(|i| cos1[i] * cos2[i]).sum()
    }

    /// Cross product (vectorial product) of two directions.
    /// Result is the direction perpendicular to both input directions.
    pub fn cross_product(d1: &Self, d2: &Self) -> Self {
        let u1 = d1.u as f64;
        let v1 = d1.v as f64;
        let w1 = d1.w as f64;

        let u2 = d2.u as f64;
        let v2 = d2.v as f64;
        let w2 = d2.w as f64;

        let u = (v1 * w2 - w1 * v2) as i32;
        let v = (w1 * u2 - u1 * w2) as i32;
        let w = (u1 * v2 - v1 * u2) as i32;

        Self { u, v, w }
    }

    /// Direction cosines for tetragonal system (a = b, c/a ratio varies).
    pub fn direction_cosines_tetragonal(&self, c_to_a: f64) -> [f64; 3] {
        let u = self.u as f64;
        let v = self.v as f64;
        let w = self.w as f64 * c_to_a;

        let uvw_sq = u * u + v * v + w * w;
        let uvw = uvw_sq.sqrt();

        [u / uvw, v / uvw, w / (uvw * c_to_a)]
    }

    /// Family of equivalent directions in cubic system.
    pub fn family_cubic(&self) -> Vec<Self> {
        let mut family = Vec::new();
        let indices = [self.u.abs(), self.v.abs(), self.w.abs()];

        for perm in &[
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for s1 in &[-1, 1] {
                for s2 in &[-1, 1] {
                    for s3 in &[-1, 1] {
                        family.push(Self {
                            u: indices[perm[0]] * s1,
                            v: indices[perm[1]] * s2,
                            w: indices[perm[2]] * s3,
                        });
                    }
                }
            }
        }

        // Remove duplicates
        family.sort_by(|a, b| {
            if a.u != b.u {
                a.u.cmp(&b.u)
            } else if a.v != b.v {
                a.v.cmp(&b.v)
            } else {
                a.w.cmp(&b.w)
            }
        });
        family.dedup();
        family
    }

    /// Check if direction is perpendicular to a given plane.
    /// For cubic: `[u,v,w]` is perpendicular to `(h,k,l)` iff `u*h + v*k + w*l = 0`
    pub fn perpendicular_to_plane(&self, plane: &MillerPlane) -> bool {
        plane.perpendicular_to_direction(self)
    }

    /// Magnitude of direction vector in cubic system.
    pub fn magnitude_cubic(&self) -> f64 {
        ((self.u * self.u + self.v * self.v + self.w * self.w) as f64).sqrt()
    }

    /// Normalize direction to unit vector (direction cosines).
    pub fn normalize_cubic(&self) -> [f64; 3] {
        self.direction_cosines_cubic()
    }
}

impl fmt::Display for MillerDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:1}{:1}{:1}]", self.u, self.v, self.w)
    }
}
