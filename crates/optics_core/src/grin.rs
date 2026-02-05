//! Gradient-Index (GRIN) Ray Solver.
//!
//! Solves dT/ds = (grad(n) - (T . grad(n)) T) / n via RK4.
//!
//! Supports both real and complex refractive indices:
//! - Real n: standard GRIN ray bending
//! - Complex n = n_r + i*k: Beer-Lambert amplitude attenuation
//!
//! Ref: CSC KTH (2011) "Ray Tracing in Gradient-Index Media"

use num_complex::Complex64;
use std::f64::consts::PI;

/// 3D vector type.
pub type Vec3 = [f64; 3];

/// Ray state: position and unit tangent direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub pos: Vec3,
    pub dir: Vec3,
}

/// Extended ray state with amplitude and phase for absorbing media.
#[derive(Debug, Clone, Copy)]
pub struct RayState {
    pub ray: Ray,
    pub amplitude: f64,
    pub phase: f64,
}

/// Result of ray tracing.
#[derive(Debug, Clone)]
pub struct RayTraceResult {
    pub positions: Vec<Vec3>,
    pub directions: Vec<Vec3>,
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub arc_lengths: Vec<f64>,
}

/// Trait for media with real refractive index.
pub trait GrinMedium {
    /// Returns (gradient_n, n) at position p.
    fn gradient_and_n(&self, p: Vec3) -> (Vec3, f64);
}

/// Trait for media with complex refractive index.
pub trait AbsorbingGrinMedium {
    /// Returns (gradient_re_n, n_complex) at position p.
    fn gradient_and_n_complex(&self, p: Vec3) -> (Vec3, Complex64);
}

/// Vector operations.
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: Vec3) -> f64 {
    dot(v, v).sqrt()
}

fn normalize(v: Vec3) -> Vec3 {
    let n = norm(v);
    if n < 1e-30 {
        v
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: Vec3, s: f64) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Central difference gradient estimation for real n.
///
/// # Arguments
/// * `p` - Position
/// * `n_func` - Function returning n at a position
/// * `eps` - Finite difference step size
pub fn central_difference_gradient<F>(p: Vec3, n_func: F, eps: f64) -> (Vec3, f64)
where
    F: Fn(Vec3) -> f64,
{
    let nx = (n_func([p[0] + eps, p[1], p[2]]) - n_func([p[0] - eps, p[1], p[2]])) / (2.0 * eps);
    let ny = (n_func([p[0], p[1] + eps, p[2]]) - n_func([p[0], p[1] - eps, p[2]])) / (2.0 * eps);
    let nz = (n_func([p[0], p[1], p[2] + eps]) - n_func([p[0], p[1], p[2] - eps])) / (2.0 * eps);
    let n0 = n_func(p);
    ([nx, ny, nz], n0)
}

/// Central difference gradient for complex n (gradient of real part).
pub fn central_difference_gradient_complex<F>(p: Vec3, n_func: F, eps: f64) -> (Vec3, Complex64)
where
    F: Fn(Vec3) -> Complex64,
{
    let nx = (n_func([p[0] + eps, p[1], p[2]]).re - n_func([p[0] - eps, p[1], p[2]]).re) / (2.0 * eps);
    let ny = (n_func([p[0], p[1] + eps, p[2]]).re - n_func([p[0], p[1] - eps, p[2]]).re) / (2.0 * eps);
    let nz = (n_func([p[0], p[1], p[2] + eps]).re - n_func([p[0], p[1], p[2] - eps]).re) / (2.0 * eps);
    let n0 = n_func(p);
    ([nx, ny, nz], n0)
}

/// One RK4 step for the ray equation (real n).
///
/// y = [pos, dir]
/// dy/ds = [dir, (grad_n - (dir.grad_n)*dir) / n]
pub fn rk4_step<M: GrinMedium>(ray: Ray, dt: f64, medium: &M) -> Ray {
    // Derivative function
    let get_derivatives = |p: Vec3, d: Vec3| -> (Vec3, Vec3) {
        let (grad_n, n) = medium.gradient_and_n(p);
        let dot_vd = dot(d, grad_n);
        let k = scale(sub(grad_n, scale(d, dot_vd)), 1.0 / n);
        (d, k)
    };

    // k1
    let (v1, a1) = get_derivatives(ray.pos, ray.dir);

    // k2
    let p2 = add(ray.pos, scale(v1, dt * 0.5));
    let d2 = normalize(add(ray.dir, scale(a1, dt * 0.5)));
    let (v2, a2) = get_derivatives(p2, d2);

    // k3
    let p3 = add(ray.pos, scale(v2, dt * 0.5));
    let d3 = normalize(add(ray.dir, scale(a2, dt * 0.5)));
    let (v3, a3) = get_derivatives(p3, d3);

    // k4
    let p4 = add(ray.pos, scale(v3, dt));
    let d4 = normalize(add(ray.dir, scale(a3, dt)));
    let (v4, a4) = get_derivatives(p4, d4);

    // Combine
    let new_pos = add(
        ray.pos,
        scale(
            add(add(v1, scale(v2, 2.0)), add(scale(v3, 2.0), v4)),
            dt / 6.0,
        ),
    );
    let new_dir = normalize(add(
        ray.dir,
        scale(
            add(add(a1, scale(a2, 2.0)), add(scale(a3, 2.0), a4)),
            dt / 6.0,
        ),
    ));

    Ray { pos: new_pos, dir: new_dir }
}

/// One RK4 step in absorptive GRIN media (complex n).
///
/// The ray path follows grad(Re(n)); the amplitude decays via
/// Beer-Lambert: dA/ds = -(2*pi/lambda)*Im(n)*A.
///
/// Returns (new_ray, attenuation_factor, phase_advance).
pub fn rk4_step_absorbing<M: AbsorbingGrinMedium>(
    ray: Ray,
    dt: f64,
    medium: &M,
    wavelength_nm: f64,
) -> (Ray, f64, f64) {
    let get_derivatives = |p: Vec3, d: Vec3| -> (Vec3, Vec3, Complex64) {
        let (grad_n_real, n_complex) = medium.gradient_and_n_complex(p);
        let n_real = n_complex.re.max(1e-10);
        let dot_vd = dot(d, grad_n_real);
        let curvature = scale(sub(grad_n_real, scale(d, dot_vd)), 1.0 / n_real);
        (d, curvature, n_complex)
    };

    // k1
    let (v1, a1, _n1) = get_derivatives(ray.pos, ray.dir);

    // k2
    let d2 = normalize(add(ray.dir, scale(a1, dt * 0.5)));
    let (v2, a2, n2) = get_derivatives(add(ray.pos, scale(v1, dt * 0.5)), d2);

    // k3
    let d3 = normalize(add(ray.dir, scale(a2, dt * 0.5)));
    let (v3, a3, _n3) = get_derivatives(add(ray.pos, scale(v2, dt * 0.5)), d3);

    // k4
    let d4 = normalize(add(ray.dir, scale(a3, dt)));
    let (v4, a4, _n4) = get_derivatives(add(ray.pos, scale(v3, dt)), d4);

    let new_pos = add(
        ray.pos,
        scale(
            add(add(v1, scale(v2, 2.0)), add(scale(v3, 2.0), v4)),
            dt / 6.0,
        ),
    );
    let new_dir = normalize(add(
        ray.dir,
        scale(
            add(add(a1, scale(a2, 2.0)), add(scale(a3, 2.0), a4)),
            dt / 6.0,
        ),
    ));

    // Attenuation: Beer-Lambert using midpoint k (Simpson-like estimate)
    let k_mid = n2.im;
    let alpha = 2.0 * PI * k_mid / wavelength_nm;
    let exponent = alpha * dt;
    let atten = if exponent < 500.0 {
        (-exponent).exp()
    } else {
        0.0
    };

    // Phase advance using midpoint Re(n)
    let n_real_mid = n2.re.max(1e-10);
    let phase_advance = 2.0 * PI * n_real_mid * dt / wavelength_nm;

    (Ray { pos: new_pos, dir: new_dir }, atten, phase_advance)
}

/// Trace a ray through a GRIN medium (real n).
pub fn trace_ray<M: GrinMedium>(
    initial_ray: Ray,
    medium: &M,
    step_size: f64,
    max_steps: usize,
) -> RayTraceResult {
    let mut positions = vec![initial_ray.pos];
    let mut directions = vec![initial_ray.dir];
    let mut arc_lengths = vec![0.0];

    let mut ray = initial_ray;
    let mut s = 0.0;

    for _ in 0..max_steps {
        ray = rk4_step(ray, step_size, medium);
        s += step_size;

        positions.push(ray.pos);
        directions.push(ray.dir);
        arc_lengths.push(s);
    }

    RayTraceResult {
        positions,
        directions,
        amplitudes: vec![1.0; arc_lengths.len()],
        phases: vec![0.0; arc_lengths.len()],
        arc_lengths,
    }
}

/// Trace a ray through an absorbing GRIN medium (complex n).
pub fn trace_ray_absorbing<M: AbsorbingGrinMedium>(
    initial_ray: Ray,
    medium: &M,
    step_size: f64,
    wavelength_nm: f64,
    max_steps: usize,
    min_amplitude: f64,
) -> RayTraceResult {
    let mut positions = vec![initial_ray.pos];
    let mut directions = vec![initial_ray.dir];
    let mut amplitudes = vec![1.0];
    let mut phases = vec![0.0];
    let mut arc_lengths = vec![0.0];

    let mut ray = initial_ray;
    let mut amplitude = 1.0;
    let mut phase = 0.0;
    let mut s = 0.0;

    for _ in 0..max_steps {
        let (new_ray, atten, phase_adv) = rk4_step_absorbing(ray, step_size, medium, wavelength_nm);
        ray = new_ray;
        amplitude *= atten;
        phase += phase_adv;
        s += step_size;

        positions.push(ray.pos);
        directions.push(ray.dir);
        amplitudes.push(amplitude);
        phases.push(phase);
        arc_lengths.push(s);

        if amplitude < min_amplitude {
            break;
        }
    }

    RayTraceResult {
        positions,
        directions,
        amplitudes,
        phases,
        arc_lengths,
    }
}

/// Simple homogeneous medium for testing.
pub struct HomogeneousMedium {
    pub n: f64,
}

impl GrinMedium for HomogeneousMedium {
    fn gradient_and_n(&self, _p: Vec3) -> (Vec3, f64) {
        ([0.0, 0.0, 0.0], self.n)
    }
}

/// Gradient-index fiber: n(r) = n0 * sqrt(1 - g^2 * r^2).
pub struct GrinFiber {
    pub n0: f64,
    pub g: f64,
    pub axis: Vec3,  // Fiber axis direction
}

impl GrinMedium for GrinFiber {
    fn gradient_and_n(&self, p: Vec3) -> (Vec3, f64) {
        // Distance from axis
        let proj = scale(self.axis, dot(p, self.axis));
        let r_vec = sub(p, proj);
        let r2 = dot(r_vec, r_vec);

        let arg = 1.0 - self.g * self.g * r2;
        if arg <= 0.0 {
            // Outside fiber core
            return ([0.0, 0.0, 0.0], 1.0);
        }

        let n = self.n0 * arg.sqrt();

        // Gradient: dn/dr * r_hat = -n0 * g^2 * r / sqrt(1 - g^2*r^2)
        let factor = -self.n0 * self.g * self.g / arg.sqrt();
        let grad = scale(r_vec, factor);

        (grad, n)
    }
}

/// Layered absorbing medium for metamaterial simulation.
pub struct LayeredAbsorbingMedium {
    pub layers: Vec<(f64, Complex64)>,  // (z_boundary, n_complex)
}

impl AbsorbingGrinMedium for LayeredAbsorbingMedium {
    fn gradient_and_n_complex(&self, p: Vec3) -> (Vec3, Complex64) {
        let z = p[2];

        // Find which layer we're in
        let mut n = Complex64::new(1.0, 0.0);
        for (z_bound, n_layer) in &self.layers {
            if z < *z_bound {
                break;
            }
            n = *n_layer;
        }

        // Gradient is zero within layers (piecewise constant)
        ([0.0, 0.0, 0.0], n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homogeneous_straight_line() {
        let medium = HomogeneousMedium { n: 1.5 };
        let ray = Ray {
            pos: [0.0, 0.0, 0.0],
            dir: [1.0, 0.0, 0.0],
        };

        let result = trace_ray(ray, &medium, 0.1, 100);

        // In homogeneous medium, ray should travel in straight line
        assert_eq!(result.positions.len(), 101);

        // Final position should be close to (10, 0, 0)
        let final_pos = result.positions.last().unwrap();
        assert_relative_eq!(final_pos[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(final_pos[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(final_pos[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_grin_fiber_bends_ray() {
        let medium = GrinFiber {
            n0: 1.5,
            g: 0.5,  // Stronger gradient for faster focusing
            axis: [0.0, 0.0, 1.0],
        };

        // Ray starting off-axis propagating mostly along z
        let ray = Ray {
            pos: [0.5, 0.0, 0.0],
            dir: normalize([0.0, 0.0, 1.0]),
        };

        let result = trace_ray(ray, &medium, 0.05, 500);

        // Ray should bend toward axis (x should decrease initially)
        // Check that x-position decreases from starting position
        let x_min = result.positions.iter()
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min);

        assert!(x_min < 0.5, "GRIN fiber should focus ray toward axis, min x = {}", x_min);
    }

    #[test]
    fn test_direction_stays_normalized() {
        let medium = GrinFiber {
            n0: 1.5,
            g: 0.2,
            axis: [0.0, 0.0, 1.0],
        };

        let ray = Ray {
            pos: [0.5, 0.5, 0.0],
            dir: normalize([0.1, 0.1, 1.0]),
        };

        let result = trace_ray(ray, &medium, 0.05, 100);

        for dir in &result.directions {
            let n = norm(*dir);
            assert_relative_eq!(n, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_absorbing_medium_attenuates() {
        // Simple absorbing medium: n = 1.5 + 0.1i everywhere
        struct UniformAbsorbing;
        impl AbsorbingGrinMedium for UniformAbsorbing {
            fn gradient_and_n_complex(&self, _p: Vec3) -> (Vec3, Complex64) {
                ([0.0, 0.0, 0.0], Complex64::new(1.5, 0.1))
            }
        }

        let medium = UniformAbsorbing;
        let ray = Ray {
            pos: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
        };

        let result = trace_ray_absorbing(ray, &medium, 1.0, 1550.0, 50, 1e-6);

        // Amplitude should decrease
        let final_amp = *result.amplitudes.last().unwrap();
        assert!(final_amp < 1.0, "Absorbing medium should attenuate ray, got {}", final_amp);
        assert!(final_amp > 0.0, "Amplitude should remain positive");
    }

    #[test]
    fn test_phase_accumulates() {
        // Simple non-absorbing medium: n = 1.5 everywhere
        struct UniformMedium;
        impl AbsorbingGrinMedium for UniformMedium {
            fn gradient_and_n_complex(&self, _p: Vec3) -> (Vec3, Complex64) {
                ([0.0, 0.0, 0.0], Complex64::new(1.5, 0.0))
            }
        }

        let medium = UniformMedium;
        let ray = Ray {
            pos: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
        };

        let result = trace_ray_absorbing(ray, &medium, 1.0, 1550.0, 10, 1e-6);

        // Phase should increase monotonically
        for i in 1..result.phases.len() {
            assert!(result.phases[i] > result.phases[i - 1]);
        }
    }

    #[test]
    fn test_central_difference_gradient() {
        // Linear n(x,y,z) = 1.0 + 0.1*x
        let n_func = |p: Vec3| 1.0 + 0.1 * p[0];

        let (grad, n) = central_difference_gradient([1.0, 0.0, 0.0], n_func, 0.001);

        assert_relative_eq!(n, 1.1, epsilon = 1e-6);
        assert_relative_eq!(grad[0], 0.1, epsilon = 1e-4);
        assert_relative_eq!(grad[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grad[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_early_termination_on_absorption() {
        // Highly absorbing medium: n = 1.5 + 10i
        struct HighlyAbsorbing;
        impl AbsorbingGrinMedium for HighlyAbsorbing {
            fn gradient_and_n_complex(&self, _p: Vec3) -> (Vec3, Complex64) {
                ([0.0, 0.0, 0.0], Complex64::new(1.5, 10.0))
            }
        }

        let medium = HighlyAbsorbing;
        let ray = Ray {
            pos: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 1.0],
        };

        // Should terminate early due to high absorption
        let result = trace_ray_absorbing(ray, &medium, 1.0, 1550.0, 1000, 0.01);

        assert!(result.positions.len() < 1000, "Should terminate early, got {} steps", result.positions.len());
    }

    #[test]
    fn test_arc_length_accumulation() {
        let medium = HomogeneousMedium { n: 1.0 };
        let ray = Ray {
            pos: [0.0, 0.0, 0.0],
            dir: [1.0, 0.0, 0.0],
        };

        let step = 0.25;
        let result = trace_ray(ray, &medium, step, 40);

        // Arc length should match step accumulation
        for (i, &s) in result.arc_lengths.iter().enumerate() {
            assert_relative_eq!(s, i as f64 * step, epsilon = 1e-10);
        }
    }
}
