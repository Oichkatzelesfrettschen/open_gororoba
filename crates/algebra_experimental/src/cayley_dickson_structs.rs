use std::ops::{Add, Mul, Neg, Sub};
use num_complex::Complex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub c0: Complex<f64>,
    pub c1: Complex<f64>,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            c0: Complex::new(0.0, 0.0),
            c1: Complex::new(0.0, 0.0),
        }
    }
}

impl Quaternion {
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            c0: Complex::new(a, b),
            c1: Complex::new(c, d),
        }
    }

    pub fn conj(&self) -> Self {
        Self {
            c0: self.c0.conj(),
            c1: -self.c1,
        }
    }

    pub fn norm_sqr(&self) -> f64 {
        self.c0.norm_sqr() + self.c1.norm_sqr()
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[3])
    }

    pub fn to_slice(&self) -> [f64; 4] {
        [self.c0.re, self.c0.im, self.c1.re, self.c1.im]
    }
}

impl Mul for Quaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a = self.c0;
        let b = self.c1;
        let c = rhs.c0;
        let d = rhs.c1;
        Self {
            c0: a * c - d.conj() * b,
            c1: d * a + b * c.conj(),
        }
    }
}

impl Mul<f64> for Quaternion {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}

impl Add for Quaternion {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { c0: self.c0 + rhs.c0, c1: self.c1 + rhs.c1 }
    }
}

impl Sub for Quaternion {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { c0: self.c0 - rhs.c0, c1: self.c1 - rhs.c1 }
    }
}

impl Neg for Quaternion {
    type Output = Self;
    fn neg(self) -> Self {
        Self { c0: -self.c0, c1: -self.c1 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Octonion {
    pub q0: Quaternion,
    pub q1: Quaternion,
}

impl Octonion {
    pub fn new(q0: Quaternion, q1: Quaternion) -> Self {
        Self { q0, q1 }
    }

    pub fn conj(&self) -> Self {
        Self {
            q0: self.q0.conj(),
            q1: -self.q1,
        }
    }

    pub fn norm_sqr(&self) -> f64 {
        self.q0.norm_sqr() + self.q1.norm_sqr()
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        Self::new(
            Quaternion::from_slice(&slice[0..4]),
            Quaternion::from_slice(&slice[4..8]),
        )
    }

    pub fn to_slice(&self) -> [f64; 8] {
        let mut arr = [0.0; 8];
        arr[0..4].copy_from_slice(&self.q0.to_slice());
        arr[4..8].copy_from_slice(&self.q1.to_slice());
        arr
    }
}

impl Mul for Octonion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a = self.q0;
        let b = self.q1;
        let c = rhs.q0;
        let d = rhs.q1;
        Self {
            q0: a * c - d.conj() * b,
            q1: d * a + b * c.conj(),
        }
    }
}

impl Mul<f64> for Octonion {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            q0: self.q0 * rhs,
            q1: self.q1 * rhs,
        }
    }
}

impl Add for Octonion {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { q0: self.q0 + rhs.q0, q1: self.q1 + rhs.q1 }
    }
}

impl Sub for Octonion {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { q0: self.q0 - rhs.q0, q1: self.q1 - rhs.q1 }
    }
}

impl Neg for Octonion {
    type Output = Self;
    fn neg(self) -> Self {
        Self { q0: -self.q0, q1: -self.q1 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Sedenion {
    pub o0: Octonion,
    pub o1: Octonion,
}

impl Sedenion {
    pub fn new(o0: Octonion, o1: Octonion) -> Self {
        Self { o0, o1 }
    }

    pub fn conj(&self) -> Self {
        Self {
            o0: self.o0.conj(),
            o1: -self.o1,
        }
    }

    pub fn norm_sqr(&self) -> f64 {
        self.o0.norm_sqr() + self.o1.norm_sqr()
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        Self::new(
            Octonion::from_slice(&slice[0..8]),
            Octonion::from_slice(&slice[8..16]),
        )
    }

    pub fn to_slice(&self) -> [f64; 16] {
        let mut arr = [0.0; 16];
        arr[0..8].copy_from_slice(&self.o0.to_slice());
        arr[8..16].copy_from_slice(&self.o1.to_slice());
        arr
    }

    pub fn project_to_subalgebra(&self, subalgebra_indices: &[usize]) -> Self {
        let mut components = self.to_slice();
        for i in 0..16 {
            if !subalgebra_indices.contains(&i) {
                components[i] = 0.0;
            }
        }
        Sedenion::from_slice(&components)
    }
}

impl Mul for Sedenion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a = self.o0;
        let b = self.o1;
        let c = rhs.o0;
        let d = rhs.o1;
        Self {
            o0: a * c - d.conj() * b,
            o1: d * a + b * c.conj(),
        }
    }
}

impl Mul<f64> for Sedenion {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            o0: self.o0 * rhs,
            o1: self.o1 * rhs,
        }
    }
}

impl Add for Sedenion {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { o0: self.o0 + rhs.o0, o1: self.o1 + rhs.o1 }
    }
}

impl Sub for Sedenion {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { o0: self.o0 - rhs.o0, o1: self.o1 - rhs.o1 }
    }
}

impl Neg for Sedenion {
    type Output = Self;
    fn neg(self) -> Self {
        Self { o0: -self.o0, o1: -self.o1 }
    }
}
