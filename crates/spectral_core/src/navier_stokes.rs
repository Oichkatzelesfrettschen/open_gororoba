// SPDX-License-Identifier: MIT

//! Numerical Fourier reference for unforced incompressible Navier-Stokes.
//!
//! The domain is `(R / 2 pi Z)^3`, with `u(x) = sum_k u_hat[k] exp(i k.x)`.
//! Integrals use normalized volume `(2 pi)^-3 dx`; kinetic energy is
//! `sum_k |u_hat[k]|^2 / 2`. Missing coefficients describe exact zeros in the
//! finite polynomial supplied to this module, not an estimated continuum tail.
//! All computations use ordinary `f64`. Residuals, discarded coefficients,
//! constraint tolerances, and convergence tests are numerical diagnostics;
//! none supplies interval bounds or a continuum existence certificate.

use std::{collections::BTreeMap, error::Error, fmt};

use ndarray::{Array3, Axis};
use num_complex::Complex64;
use rustfft::FftPlanner;

pub type Wavevector3D = [i32; 3];
pub type ComplexVector3D = [Complex64; 3];

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const IMAGINARY: Complex64 = Complex64::new(0.0, 1.0);

/// Invalid inputs and nonfinite arithmetic remain distinct failure classes.
#[derive(Clone, Debug, PartialEq)]
pub enum SpectralError {
    InvalidInput(String),
    NonfiniteArithmetic(&'static str),
}

impl fmt::Display for SpectralError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid spectral input: {message}"),
            Self::NonfiniteArithmetic(operation) => {
                write!(formatter, "nonfinite spectral arithmetic in {operation}")
            }
        }
    }
}

impl Error for SpectralError {}

/// One complex vector coefficient; real fields include its conjugate partner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourierMode3D {
    pub wavevector: Wavevector3D,
    pub velocity: ComplexVector3D,
}

/// A complete finite Fourier polynomial, including both Hermitian partners.
///
/// Construction validates finite coefficients and unique wavevectors. Call
/// [`Self::validate_incompressible_real`] before treating it as baseline data;
/// arbitrary vector polynomials also represent derivatives and residuals.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorField3D {
    coefficients: BTreeMap<Wavevector3D, ComplexVector3D>,
}

impl VectorField3D {
    pub fn new(modes: impl IntoIterator<Item = FourierMode3D>) -> Result<Self, SpectralError> {
        let mut coefficients = BTreeMap::new();
        for mode in modes {
            if mode.velocity.iter().any(|value| !finite(*value)) {
                return Err(invalid("Fourier coefficients must be finite"));
            }
            if mode.wavevector.contains(&i32::MIN) {
                return Err(invalid("wavevector has no representable conjugate partner"));
            }
            if coefficients
                .insert(mode.wavevector, mode.velocity)
                .is_some()
            {
                return Err(invalid("duplicate Fourier wavevector"));
            }
        }
        Self::from_coefficients(coefficients, "field construction")
    }

    fn from_coefficients(
        mut coefficients: BTreeMap<Wavevector3D, ComplexVector3D>,
        operation: &'static str,
    ) -> Result<Self, SpectralError> {
        if coefficients.values().flatten().any(|value| !finite(*value)) {
            return Err(SpectralError::NonfiniteArithmetic(operation));
        }
        // Exact zero removal changes neither support nor any represented value.
        coefficients.retain(|_, value| value.iter().any(|component| *component != ZERO));
        Ok(Self { coefficients })
    }

    pub fn modes(&self) -> impl Iterator<Item = FourierMode3D> + '_ {
        self.coefficients
            .iter()
            .map(|(&wavevector, &velocity)| FourierMode3D {
                wavevector,
                velocity,
            })
    }

    pub fn coefficient(&self, wavevector: Wavevector3D) -> ComplexVector3D {
        self.coefficients
            .get(&wavevector)
            .copied()
            .unwrap_or([ZERO; 3])
    }

    /// Largest absolute coordinate of a nonzero coefficient.
    pub fn cutoff(&self) -> i32 {
        self.coefficients
            .keys()
            .flatten()
            .map(|value| value.abs())
            .max()
            .unwrap_or(0)
    }

    /// Check zero mean, reality, and solenoidality with a declared relative tolerance.
    ///
    /// Each mode uses scale `max(1, |u_hat[k]|)`; the divergence check also
    /// scales by `|k|`. These checks detect floating-point constraint defects;
    /// passing them is not an exact proof of either constraint.
    pub fn validate_incompressible_real(&self, tolerance: f64) -> Result<(), SpectralError> {
        validate_tolerance(tolerance)?;
        if vector_norm(self.coefficient([0; 3])) > tolerance {
            return Err(invalid("baseline velocity has nonzero mean"));
        }
        for mode in self.modes() {
            let scale = vector_norm(mode.velocity).max(1.0);
            if !scale.is_finite() {
                return Err(SpectralError::NonfiniteArithmetic(
                    "constraint normalization",
                ));
            }
            let partner = self.coefficient(mode.wavevector.map(|value| -value));
            let defect = std::array::from_fn(|axis| partner[axis] - mode.velocity[axis].conj());
            if vector_norm(defect) / scale > tolerance {
                return Err(invalid("Fourier coefficients violate Hermitian symmetry"));
            }
            let wave_norm = wave_number_squared(mode.wavevector).sqrt();
            if wave_norm > 0.0 {
                let divergence = wave_dot(mode.wavevector, mode.velocity);
                if !finite(divergence) {
                    return Err(SpectralError::NonfiniteArithmetic("divergence validation"));
                }
                if divergence.norm() / scale / wave_norm > tolerance {
                    return Err(invalid("Fourier coefficients are not divergence-free"));
                }
            }
        }
        Ok(())
    }

    pub fn scaled(&self, factor: f64) -> Result<Self, SpectralError> {
        if !factor.is_finite() {
            return Err(invalid("field scale must be finite"));
        }
        self.map_coefficients("field scaling", |_, value| value.map(|v| v * factor))
    }

    pub fn add_scaled(&self, other: &Self, factor: f64) -> Result<Self, SpectralError> {
        if !factor.is_finite() {
            return Err(invalid("field scale must be finite"));
        }
        let mut coefficients = self.coefficients.clone();
        for mode in other.modes() {
            let entry = coefficients.entry(mode.wavevector).or_insert([ZERO; 3]);
            for (component, value) in entry.iter_mut().zip(mode.velocity) {
                *component += factor * value;
            }
        }
        Self::from_coefficients(coefficients, "field addition")
    }

    pub fn derivative(&self, axis: usize) -> Result<Self, SpectralError> {
        if axis >= 3 {
            return Err(invalid("derivative axis must be 0, 1, or 2"));
        }
        self.map_coefficients("Fourier differentiation", |wavevector, value| {
            value.map(|v| IMAGINARY * f64::from(wavevector[axis]) * v)
        })
    }

    pub fn laplacian(&self) -> Result<Self, SpectralError> {
        self.map_coefficients("Fourier Laplacian", |wavevector, value| {
            value.map(|v| -wave_number_squared(wavevector) * v)
        })
    }

    pub fn curl(&self) -> Result<Self, SpectralError> {
        self.map_coefficients("Fourier curl", |wavevector, value| {
            let [x, y, z] = wavevector.map(f64::from);
            [
                IMAGINARY * (y * value[2] - z * value[1]),
                IMAGINARY * (z * value[0] - x * value[2]),
                IMAGINARY * (x * value[1] - y * value[0]),
            ]
        })
    }

    /// Evaluate the polynomial without discarding a possible imaginary defect.
    pub fn value_at(&self, point: [f64; 3]) -> Result<ComplexVector3D, SpectralError> {
        if point.iter().any(|value| !value.is_finite()) {
            return Err(invalid("evaluation coordinates must be finite"));
        }
        let mut value = [ZERO; 3];
        for mode in self.modes() {
            let angle: f64 = mode
                .wavevector
                .iter()
                .zip(point)
                .map(|(&k, x)| f64::from(k) * x)
                .sum();
            let phase = Complex64::from_polar(1.0, angle);
            for (component, coefficient) in value.iter_mut().zip(mode.velocity) {
                *component += phase * coefficient;
            }
        }
        if value.iter().any(|value| !finite(*value)) {
            return Err(SpectralError::NonfiniteArithmetic("polynomial evaluation"));
        }
        Ok(value)
    }

    fn map_coefficients(
        &self,
        operation: &'static str,
        map: impl Fn(Wavevector3D, ComplexVector3D) -> ComplexVector3D,
    ) -> Result<Self, SpectralError> {
        Self::from_coefficients(
            self.modes()
                .map(|mode| (mode.wavevector, map(mode.wavevector, mode.velocity)))
                .collect(),
            operation,
        )
    }
}

/// Orthogonal projection with `P_0 = I`; mean removal is a separate operation.
pub struct LerayProjector;

impl LerayProjector {
    pub fn project_mode(mode: FourierMode3D) -> Result<FourierMode3D, SpectralError> {
        if mode.velocity.iter().any(|value| !finite(*value)) {
            return Err(invalid("projection requires finite coefficients"));
        }
        let squared = wave_number_squared(mode.wavevector);
        let velocity = if squared == 0.0 {
            mode.velocity
        } else {
            let parallel = wave_dot(mode.wavevector, mode.velocity) / squared;
            std::array::from_fn(|axis| {
                mode.velocity[axis] - f64::from(mode.wavevector[axis]) * parallel
            })
        };
        if velocity.iter().any(|value| !finite(*value)) {
            return Err(SpectralError::NonfiniteArithmetic("Leray projection"));
        }
        Ok(FourierMode3D {
            wavevector: mode.wavevector,
            velocity,
        })
    }

    pub fn project(field: &VectorField3D) -> Result<VectorField3D, SpectralError> {
        VectorField3D::new(
            field
                .modes()
                .map(Self::project_mode)
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

/// Complete quadratic output partitioned by an explicitly requested cubic cutoff.
#[derive(Clone, Debug)]
pub struct NonlinearityEvaluation {
    pub retained: VectorField3D,
    pub discarded: VectorField3D,
    pub retained_cutoff: i32,
    pub input_cutoff: i32,
    /// `None` identifies direct convolution; a padded FFT records its side length.
    pub fft_grid_size: Option<usize>,
}

impl NonlinearityEvaluation {
    pub fn full(&self) -> Result<VectorField3D, SpectralError> {
        self.retained.add_scaled(&self.discarded, 1.0)
    }

    /// Signed modal energy transfer `Re(conj(u_hat[k]).N_hat[k])`.
    pub fn energy_transfer(
        &self,
        velocity: &VectorField3D,
    ) -> Result<BTreeMap<Wavevector3D, f64>, SpectralError> {
        let mut transfer = BTreeMap::new();
        for mode in self.full()?.modes() {
            let u = velocity.coefficient(mode.wavevector);
            let value: f64 = u
                .iter()
                .zip(mode.velocity)
                .map(|(u, n)| (u.conj() * n).re)
                .sum();
            if !value.is_finite() {
                return Err(SpectralError::NonfiniteArithmetic("modal energy transfer"));
            }
            transfer.insert(mode.wavevector, value);
        }
        Ok(transfer)
    }
}

/// Evaluate `N_k = -i P_k sum_{p+q=k} (q.u_hat[p]) u_hat[q]`.
pub trait NavierStokesNonlinearity {
    fn evaluate(
        &self,
        field: &VectorField3D,
        retained_cutoff: i32,
        constraint_tolerance: f64,
    ) -> Result<NonlinearityEvaluation, SpectralError>;
}

/// Ordered-pair convolution provides an FFT-independent small-support oracle.
pub struct DirectTriadConvolution;

impl NavierStokesNonlinearity for DirectTriadConvolution {
    fn evaluate(
        &self,
        field: &VectorField3D,
        retained_cutoff: i32,
        constraint_tolerance: f64,
    ) -> Result<NonlinearityEvaluation, SpectralError> {
        validate_cutoff(retained_cutoff)?;
        field.validate_incompressible_real(constraint_tolerance)?;
        let mut product = BTreeMap::new();
        for p in field.modes() {
            for q in field.modes() {
                let mut k = [0; 3];
                for (axis, value) in k.iter_mut().enumerate() {
                    *value = p.wavevector[axis]
                        .checked_add(q.wavevector[axis])
                        .filter(|value| *value != i32::MIN)
                        .ok_or_else(|| invalid("quadratic wavevector exceeds the i32 range"))?;
                }
                let contraction = wave_dot(q.wavevector, p.velocity);
                let entry = product.entry(k).or_insert([ZERO; 3]);
                for (component, value) in entry.iter_mut().zip(q.velocity) {
                    *component += -IMAGINARY * contraction * value;
                }
            }
        }
        let projected = LerayProjector::project(&VectorField3D::from_coefficients(
            product,
            "direct convolution",
        )?)?;
        split_output(projected, retained_cutoff, field.cutoff(), None)
    }
}

/// FFT convolution resolves the entire product before any Galerkin truncation.
///
/// For input support `|k_j| <= K`, products have support `|k_j| <= 2K`.
/// Requiring `N > 4K` makes every product frequency unique modulo `N`.
/// This padding is stronger than retained-only 2/3 dealiasing (`N > 3K`)
/// because the discarded coefficients are also reported without wraparound.
/// `max_grid_points` is an explicit resource limit on the `N^3` grid.
/// Allocation-size validation checks representability; callers must still
/// choose a budget that fits available memory for multiple working arrays.
pub struct DealiasedConvolution {
    grid_size: usize,
    max_grid_points: usize,
}

impl DealiasedConvolution {
    pub fn new(grid_size: usize, max_grid_points: usize) -> Result<Self, SpectralError> {
        if grid_size == 0 || grid_size > i32::MAX as usize {
            return Err(invalid("FFT side length must lie in 1..=i32::MAX"));
        }
        let grid_points = grid_size
            .checked_pow(3)
            .ok_or_else(|| invalid("FFT grid size overflows usize"))?;
        let grid_bytes = grid_points
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or_else(|| invalid("FFT array byte size overflows usize"))?;
        if grid_bytes > isize::MAX as usize {
            return Err(invalid("FFT array byte size exceeds isize::MAX"));
        }
        if grid_points > max_grid_points {
            return Err(invalid("FFT grid exceeds the declared resource limit"));
        }
        Ok(Self {
            grid_size,
            max_grid_points,
        })
    }
}

impl NavierStokesNonlinearity for DealiasedConvolution {
    fn evaluate(
        &self,
        field: &VectorField3D,
        retained_cutoff: i32,
        constraint_tolerance: f64,
    ) -> Result<NonlinearityEvaluation, SpectralError> {
        validate_cutoff(retained_cutoff)?;
        field.validate_incompressible_real(constraint_tolerance)?;
        let n = self.grid_size;
        let k = field.cutoff() as usize;
        if k.checked_mul(4).is_none_or(|required| n <= required) {
            return Err(invalid(
                "full-product dealiasing requires FFT side length N > 4 K",
            ));
        }
        let points = n
            .checked_pow(3)
            .ok_or_else(|| invalid("FFT grid size overflows usize"))?;
        if points > self.max_grid_points {
            return Err(invalid("FFT grid exceeds the declared resource limit"));
        }
        let mut velocity: [Array3<Complex64>; 3] =
            std::array::from_fn(|_| Array3::zeros((n, n, n)));
        for mode in field.modes() {
            let index = mode
                .wavevector
                .map(|value| value.rem_euclid(n as i32) as usize);
            for (component, value) in velocity.iter_mut().zip(mode.velocity) {
                component[index] = value;
            }
        }
        let mut planner = FftPlanner::<f64>::new();
        let inverse = planner.plan_fft_inverse(n);
        let forward = planner.plan_fft_forward(n);
        for component in &mut velocity {
            transform_3d(component, inverse.as_ref());
        }
        let mut transformed_product: [Array3<Complex64>; 3] =
            std::array::from_fn(|_| Array3::zeros((n, n, n)));
        for (axis, velocity_component) in velocity.iter().enumerate() {
            for (component_index, product) in transformed_product.iter_mut().enumerate() {
                let mut derivative = Array3::zeros((n, n, n));
                for mode in field.modes() {
                    let index = mode
                        .wavevector
                        .map(|value| value.rem_euclid(n as i32) as usize);
                    derivative[index] = IMAGINARY
                        * f64::from(mode.wavevector[axis])
                        * mode.velocity[component_index];
                }
                transform_3d(&mut derivative, inverse.as_ref());
                for ((result, u), du) in product
                    .iter_mut()
                    .zip(velocity_component.iter())
                    .zip(derivative.iter())
                {
                    *result -= u * du;
                }
            }
        }
        for product in &mut transformed_product {
            transform_3d(product, forward.as_ref());
        }
        // rustfft transforms are unnormalized: the inverse synthesizes the
        // specified Fourier series, and the forward requires division by N^3.
        let normalization = points as f64;
        let mut coefficients = BTreeMap::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let index = [x, y, z];
                    let wavevector = index.map(|index| {
                        if index <= n / 2 {
                            index as i32
                        } else {
                            index as i32 - n as i32
                        }
                    });
                    let velocity = std::array::from_fn(|component| {
                        transformed_product[component][index] / normalization
                    });
                    coefficients.insert(wavevector, velocity);
                }
            }
        }
        // Keep even roundoff-only coefficients outside the exact product
        // support. No amplitude threshold silently removes a Fourier tail.
        let projected = LerayProjector::project(&VectorField3D::from_coefficients(
            coefficients,
            "FFT convolution",
        )?)?;
        split_output(projected, retained_cutoff, field.cutoff(), Some(n))
    }
}

fn transform_3d(field: &mut Array3<Complex64>, fft: &dyn rustfft::Fft<f64>) {
    for axis in 0..3 {
        for mut lane in field.lanes_mut(Axis(axis)) {
            let mut values = lane.to_vec();
            fft.process(&mut values);
            for (target, value) in lane.iter_mut().zip(values) {
                *target = value;
            }
        }
    }
}

fn split_output(
    field: VectorField3D,
    retained_cutoff: i32,
    input_cutoff: i32,
    fft_grid_size: Option<usize>,
) -> Result<NonlinearityEvaluation, SpectralError> {
    let (retained, discarded): (Vec<_>, Vec<_>) = field.modes().partition(|mode| {
        mode.wavevector
            .iter()
            .all(|value| value.abs() <= retained_cutoff)
    });
    Ok(NonlinearityEvaluation {
        retained: VectorField3D::new(retained)?,
        discarded: VectorField3D::new(discarded)?,
        retained_cutoff,
        input_cutoff,
        fft_grid_size,
    })
}

/// Fourier norm conventions use normalized torus volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SobolevConvention {
    /// Weight `|k|^(2s)`; a seminorm for fields with nonzero mean when `s > 0`.
    Homogeneous,
    /// Weight `(1 + |k|^2)^s`, including the mean.
    Inhomogeneous,
}

#[derive(Clone, Copy, Debug)]
pub struct SobolevNorm {
    order: f64,
    convention: SobolevConvention,
}

impl SobolevNorm {
    pub fn new(order: f64, convention: SobolevConvention) -> Result<Self, SpectralError> {
        if !order.is_finite() || order < 0.0 {
            return Err(invalid("Sobolev order must be finite and nonnegative"));
        }
        Ok(Self { order, convention })
    }

    /// Evaluate the complete supplied polynomial without a continuum tail estimate.
    pub fn evaluate(&self, field: &VectorField3D) -> Result<f64, SpectralError> {
        let mut norm = 0.0_f64;
        for mode in field.modes() {
            let base = wave_number_squared(mode.wavevector)
                + match self.convention {
                    SobolevConvention::Homogeneous => 0.0,
                    SobolevConvention::Inhomogeneous => 1.0,
                };
            let weight = base.powf(self.order / 2.0);
            for component in mode.velocity {
                norm = norm.hypot(weight * component.norm());
            }
        }
        if !norm.is_finite() {
            return Err(SpectralError::NonfiniteArithmetic("Sobolev norm"));
        }
        Ok(norm)
    }
}

/// Residual of an explicitly supplied value and time derivative at one time.
///
/// `r = dt(v) - N(v) - nu Delta(v)` includes all quadratic output modes.
/// A pointwise residual alone does not control a temporal reconstruction,
/// unknown continuum tails, or arithmetic errors between sample times.
pub struct ResidualEvaluator;

impl ResidualEvaluator {
    pub fn evaluate(
        value: &VectorField3D,
        time_derivative: &VectorField3D,
        viscosity: f64,
        constraint_tolerance: f64,
    ) -> Result<VectorField3D, SpectralError> {
        validate_viscosity(viscosity)?;
        value.validate_incompressible_real(constraint_tolerance)?;
        time_derivative.validate_incompressible_real(constraint_tolerance)?;
        let nonlinear = DirectTriadConvolution
            .evaluate(value, value.cutoff(), constraint_tolerance)?
            .full()?;
        time_derivative
            .add_scaled(&nonlinear, -1.0)?
            .add_scaled(&value.laplacian()?, -viscosity)
    }
}

/// Fixed-cutoff Galerkin RK4 integrator with explicit positive physical viscosity.
///
/// The torus side is fixed at `2 pi`; changing cutoff and timestep therefore
/// keeps the same continuum equation when viscosity and initial data agree.
/// No automatic stability claim or filter accompanies an accepted step.
pub struct SpectralReferenceSolver<C> {
    nonlinearity: C,
    viscosity: f64,
    retained_cutoff: i32,
    constraint_tolerance: f64,
}

#[derive(Clone, Debug)]
pub struct StepResult {
    pub field: VectorField3D,
    pub timestep: f64,
    /// Nonlinear Galerkin omissions at the four RK stages; ordinary f64 values.
    pub discarded_at_stages: [VectorField3D; 4],
}

impl<C: NavierStokesNonlinearity> SpectralReferenceSolver<C> {
    pub fn new(
        nonlinearity: C,
        viscosity: f64,
        retained_cutoff: i32,
        constraint_tolerance: f64,
    ) -> Result<Self, SpectralError> {
        validate_viscosity(viscosity)?;
        validate_cutoff(retained_cutoff)?;
        validate_tolerance(constraint_tolerance)?;
        Ok(Self {
            nonlinearity,
            viscosity,
            retained_cutoff,
            constraint_tolerance,
        })
    }

    pub fn step(&self, field: &VectorField3D, timestep: f64) -> Result<StepResult, SpectralError> {
        if !timestep.is_finite() || timestep <= 0.0 {
            return Err(invalid("RK4 timestep must be positive and finite"));
        }
        if field.cutoff() > self.retained_cutoff {
            return Err(invalid(
                "initial field exceeds solver cutoff; truncate explicitly",
            ));
        }
        let (a, discarded_a) = self.rhs(field)?;
        let (b, discarded_b) = self.rhs(&field.add_scaled(&a, timestep / 2.0)?)?;
        let (c, discarded_c) = self.rhs(&field.add_scaled(&b, timestep / 2.0)?)?;
        let (d, discarded_d) = self.rhs(&field.add_scaled(&c, timestep)?)?;
        let result = field
            .add_scaled(&a, timestep / 6.0)?
            .add_scaled(&b, timestep / 3.0)?
            .add_scaled(&c, timestep / 3.0)?
            .add_scaled(&d, timestep / 6.0)?;
        result.validate_incompressible_real(self.constraint_tolerance)?;
        Ok(StepResult {
            field: result,
            timestep,
            discarded_at_stages: [discarded_a, discarded_b, discarded_c, discarded_d],
        })
    }

    pub fn rhs(
        &self,
        field: &VectorField3D,
    ) -> Result<(VectorField3D, VectorField3D), SpectralError> {
        if field.cutoff() > self.retained_cutoff {
            return Err(invalid("field exceeds solver cutoff"));
        }
        let evaluated =
            self.nonlinearity
                .evaluate(field, self.retained_cutoff, self.constraint_tolerance)?;
        Ok((
            evaluated
                .retained
                .add_scaled(&field.laplacian()?, self.viscosity)?,
            evaluated.discarded,
        ))
    }
}

fn wave_number_squared(wavevector: Wavevector3D) -> f64 {
    wavevector
        .into_iter()
        .map(|value| f64::from(value).powi(2))
        .sum()
}

fn wave_dot(wavevector: Wavevector3D, vector: ComplexVector3D) -> Complex64 {
    wavevector
        .into_iter()
        .zip(vector)
        .map(|(k, value)| f64::from(k) * value)
        .sum()
}

fn vector_norm(vector: ComplexVector3D) -> f64 {
    vector
        .into_iter()
        .fold(0.0_f64, |norm, value| norm.hypot(value.norm()))
}

fn finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn invalid(message: impl Into<String>) -> SpectralError {
    SpectralError::InvalidInput(message.into())
}

fn validate_tolerance(tolerance: f64) -> Result<(), SpectralError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid(
            "constraint tolerance must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn validate_cutoff(cutoff: i32) -> Result<(), SpectralError> {
    if cutoff < 0 {
        return Err(invalid("retained cutoff must be nonnegative"));
    }
    Ok(())
}

fn validate_viscosity(viscosity: f64) -> Result<(), SpectralError> {
    if !viscosity.is_finite() || viscosity <= 0.0 {
        return Err(invalid("physical viscosity must be positive and finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-11;

    fn real_field(positive_modes: &[(Wavevector3D, ComplexVector3D)]) -> VectorField3D {
        VectorField3D::new(positive_modes.iter().flat_map(|&(wavevector, velocity)| {
            [
                FourierMode3D {
                    wavevector,
                    velocity,
                },
                FourierMode3D {
                    wavevector: wavevector.map(|value| -value),
                    velocity: velocity.map(|value| value.conj()),
                },
            ]
        }))
        .unwrap()
    }

    fn l2(field: &VectorField3D) -> f64 {
        SobolevNorm::new(0.0, SobolevConvention::Inhomogeneous)
            .unwrap()
            .evaluate(field)
            .unwrap()
    }

    fn distance(left: &VectorField3D, right: &VectorField3D) -> f64 {
        l2(&left.add_scaled(right, -1.0).unwrap())
    }

    // p=(1,0,0), q=(0,1,0), k=p+q. The only k contribution is
    // (q.u_p)u_q = e_z, so u_k=i e_z gives T_k=-1.
    fn phase_triad(sign: f64) -> VectorField3D {
        real_field(&[
            ([1, 0, 0], [ZERO, Complex64::new(1.0, 0.0), ZERO]),
            ([0, 1, 0], [ZERO, ZERO, Complex64::new(1.0, 0.0)]),
            ([1, 1, 0], [ZERO, ZERO, IMAGINARY * sign]),
        ])
    }

    fn abc() -> VectorField3D {
        // (sin z + cos y, sin x + cos z, sin y + cos x), curl v = v.
        real_field(&[
            (
                [1, 0, 0],
                [ZERO, -IMAGINARY * 0.5, Complex64::new(0.5, 0.0)],
            ),
            (
                [0, 1, 0],
                [Complex64::new(0.5, 0.0), ZERO, -IMAGINARY * 0.5],
            ),
            (
                [0, 0, 1],
                [-IMAGINARY * 0.5, Complex64::new(0.5, 0.0), ZERO],
            ),
        ])
    }

    fn taylor_green() -> VectorField3D {
        // Three-dimensional Taylor-Green initial data, not an exact viscous
        // solution: (sin x cos y cos z, -cos x sin y cos z, 0).
        let mut modes = Vec::new();
        for x in [-1, 1] {
            for y in [-1, 1] {
                for z in [-1, 1] {
                    modes.push(FourierMode3D {
                        wavevector: [x, y, z],
                        velocity: [
                            -IMAGINARY * f64::from(x) / 8.0,
                            IMAGINARY * f64::from(y) / 8.0,
                            ZERO,
                        ],
                    });
                }
            }
        }
        VectorField3D::new(modes).unwrap()
    }

    fn evolve<C: NavierStokesNonlinearity>(
        solver: &SpectralReferenceSolver<C>,
        initial: &VectorField3D,
        timestep: f64,
        steps: usize,
    ) -> VectorField3D {
        let mut field = initial.clone();
        for _ in 0..steps {
            field = solver.step(&field, timestep).unwrap().field;
        }
        field
    }

    #[test]
    fn phase_reversal_has_exact_signed_transfer_and_cubic_scaling() {
        let first = phase_triad(1.0);
        let reversed = phase_triad(-1.0);
        let a = DirectTriadConvolution.evaluate(&first, 2, 0.0).unwrap();
        let b = DirectTriadConvolution.evaluate(&reversed, 2, 0.0).unwrap();
        assert_eq!(a.energy_transfer(&first).unwrap()[&[1, 1, 0]], -1.0);
        assert_eq!(b.energy_transfer(&reversed).unwrap()[&[1, 1, 0]], 1.0);
        for scale in [-2.0_f64, 0.25, 3.0] {
            let scaled = first.scaled(scale).unwrap();
            let n = DirectTriadConvolution.evaluate(&scaled, 2, 0.0).unwrap();
            assert_eq!(
                n.energy_transfer(&scaled).unwrap()[&[1, 1, 0]],
                -scale.powi(3)
            );
        }
    }

    #[test]
    fn phase_quadrature_and_each_of_three_modes_change_transfer() {
        let original = phase_triad(1.0);
        let quadrature = VectorField3D::new(original.modes().map(|mut mode| {
            if mode.wavevector == [1, 1, 0] || mode.wavevector == [-1, -1, 0] {
                mode.velocity[2] = Complex64::new(1.0, 0.0);
            }
            mode
        }))
        .unwrap();
        let transfer = DirectTriadConvolution
            .evaluate(&quadrature, 2, 0.0)
            .unwrap()
            .energy_transfer(&quadrature)
            .unwrap();
        assert_eq!(transfer[&[1, 1, 0]], 0.0);
        for removed in [[1, 0, 0], [0, 1, 0], [1, 1, 0]] {
            let reduced = VectorField3D::new(original.modes().filter(|mode| {
                mode.wavevector != removed && mode.wavevector != removed.map(|k| -k)
            }))
            .unwrap();
            let transfer = DirectTriadConvolution
                .evaluate(&reduced, 2, 0.0)
                .unwrap()
                .energy_transfer(&reduced)
                .unwrap();
            assert_eq!(transfer.get(&[1, 1, 0]).copied().unwrap_or(0.0), 0.0);
        }
    }

    #[test]
    fn direct_nonlinearity_conserves_energy_and_projects_all_output_modes() {
        let field = LerayProjector::project(&real_field(&[
            (
                [1, 2, -1],
                [
                    Complex64::new(0.3, 0.2),
                    Complex64::new(-0.8, 0.5),
                    Complex64::new(0.2, -0.7),
                ],
            ),
            (
                [0, 1, 1],
                [
                    Complex64::new(0.9, -0.4),
                    Complex64::new(0.1, 0.2),
                    Complex64::new(-0.1, 0.5),
                ],
            ),
            (
                [1, 3, 0],
                [
                    Complex64::new(-0.2, 0.4),
                    Complex64::new(0.5, 0.8),
                    Complex64::new(0.7, 0.6),
                ],
            ),
        ]))
        .unwrap();
        let nonlinearity = DirectTriadConvolution
            .evaluate(&field, 3, TOLERANCE)
            .unwrap();
        let balance: f64 = nonlinearity.energy_transfer(&field).unwrap().values().sum();
        assert!(balance.abs() < 1e-12, "energy balance {balance}");
        nonlinearity
            .full()
            .unwrap()
            .validate_incompressible_real(TOLERANCE)
            .unwrap();
        let without_cutoff = DirectTriadConvolution
            .evaluate(&field, 6, TOLERANCE)
            .unwrap();
        assert!(distance(&nonlinearity.full().unwrap(), &without_cutoff.retained) < 1e-13);
        assert!(l2(&nonlinearity.discarded) > 0.0);
    }

    #[test]
    fn integer_orthogonal_coordinate_changes_preserve_nonlinearity() {
        let field = phase_triad(1.0).add_scaled(&abc(), 0.3).unwrap();
        let reference = DirectTriadConvolution
            .evaluate(&field, 2, TOLERANCE)
            .unwrap()
            .full()
            .unwrap();
        // The cubic integer lattice admits signed permutations, not arbitrary
        // rotations. Test all permutations with every axis-reflection choice.
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for bits in 0..8 {
                let signs: [i32; 3] =
                    std::array::from_fn(|axis| if bits & (1 << axis) == 0 { 1 } else { -1 });
                let transform = |input: &VectorField3D| {
                    VectorField3D::new(input.modes().map(|mode| FourierMode3D {
                        wavevector: std::array::from_fn(|axis| {
                            signs[axis] * mode.wavevector[permutation[axis]]
                        }),
                        velocity: std::array::from_fn(|axis| {
                            f64::from(signs[axis]) * mode.velocity[permutation[axis]]
                        }),
                    }))
                    .unwrap()
                };
                let changed = DirectTriadConvolution
                    .evaluate(&transform(&field), 2, TOLERANCE)
                    .unwrap()
                    .full()
                    .unwrap();
                assert!(distance(&changed, &transform(&reference)) < 1e-12);
            }
        }
    }

    #[test]
    fn fft_matches_direct_full_product_and_reports_discarded_modes() {
        let field = taylor_green()
            .add_scaled(&abc(), 0.37)
            .unwrap()
            .add_scaled(&phase_triad(1.0), 0.19)
            .unwrap();
        let direct = DirectTriadConvolution
            .evaluate(&field, 1, TOLERANCE)
            .unwrap();
        for grid in [5_usize, 6, 9] {
            let fft = DealiasedConvolution::new(grid, grid.pow(3))
                .unwrap()
                .evaluate(&field, 1, TOLERANCE)
                .unwrap();
            assert!(distance(&fft.retained, &direct.retained) < 3e-14);
            assert!(distance(&fft.full().unwrap(), &direct.full().unwrap()) < 5e-14);
            assert!(distance(&fft.discarded, &direct.discarded) < 5e-14);
            assert!(l2(&fft.discarded) > 0.1);
            assert_eq!(fft.fft_grid_size, Some(grid));
        }
    }

    #[test]
    fn fft_rejects_aliasing_and_respects_memory_limit() {
        let field = taylor_green();
        assert!(
            DealiasedConvolution::new(4, 64)
                .unwrap()
                .evaluate(&field, 1, TOLERANCE)
                .is_err()
        );
        assert!(DealiasedConvolution::new(5, 124).is_err());
        assert!(DealiasedConvolution::new(0, 100).is_err());
        assert!(DealiasedConvolution::new(usize::MAX, usize::MAX).is_err());
        // On 64-bit targets N^3 fits usize for both cases; array bytes exceed
        // isize::MAX for the first and overflow usize for the second.
        assert!(DealiasedConvolution::new(1_000_000, usize::MAX).is_err());
        assert!(DealiasedConvolution::new(2_000_000, usize::MAX).is_err());
    }

    #[test]
    fn shear_decay_reproduces_physical_viscosity_and_fourth_order_time_accuracy() {
        let initial = real_field(&[([0, 2, 0], [-IMAGINARY * 0.5, ZERO, ZERO])]);
        let viscosity = 0.3;
        let solver =
            SpectralReferenceSolver::new(DirectTriadConvolution, viscosity, 2, TOLERANCE).unwrap();
        assert_eq!(
            l2(&DirectTriadConvolution
                .evaluate(&initial, 2, TOLERANCE)
                .unwrap()
                .full()
                .unwrap()),
            0.0
        );
        let exact = initial.scaled((-4.0_f64 * viscosity).exp()).unwrap();
        let coarse = evolve(&solver, &initial, 0.1, 10);
        let fine = evolve(&solver, &initial, 0.05, 20);
        let coarse_error = distance(&coarse, &exact);
        let fine_error = distance(&fine, &exact);
        assert!(fine_error < 4e-8);
        assert!(coarse_error / fine_error > 15.0);
        assert!(coarse_error / fine_error < 18.0);
        let derivative = exact.scaled(-4.0 * viscosity).unwrap();
        assert!(
            l2(&ResidualEvaluator::evaluate(&exact, &derivative, viscosity, TOLERANCE).unwrap())
                < 1e-15
        );
    }

    #[test]
    fn abc_has_unit_curl_zero_projected_convection_and_exact_viscous_decay() {
        let initial = abc();
        assert_eq!(distance(&initial.curl().unwrap(), &initial), 0.0);
        assert_eq!(
            l2(&DirectTriadConvolution
                .evaluate(&initial, 2, 0.0)
                .unwrap()
                .full()
                .unwrap()),
            0.0
        );
        assert_eq!(
            distance(
                &initial.laplacian().unwrap(),
                &initial.scaled(-1.0).unwrap()
            ),
            0.0
        );
        let hs = SobolevNorm::new(3.0, SobolevConvention::Homogeneous)
            .unwrap()
            .evaluate(&initial)
            .unwrap();
        let hs1 = SobolevNorm::new(4.0, SobolevConvention::Homogeneous)
            .unwrap()
            .evaluate(&initial)
            .unwrap();
        assert_eq!(hs, hs1);
        assert!((hs - 3.0_f64.sqrt()).abs() < 1e-15);
        let inhomogeneous = SobolevNorm::new(3.0, SobolevConvention::Inhomogeneous)
            .unwrap()
            .evaluate(&initial)
            .unwrap();
        assert!((inhomogeneous - 24.0_f64.sqrt()).abs() < 1e-14);
        let viscosity = 0.2;
        let exact = initial.scaled((-viscosity * 0.5_f64).exp()).unwrap();
        let solver = SpectralReferenceSolver::new(
            DealiasedConvolution::new(5, 125).unwrap(),
            viscosity,
            1,
            TOLERANCE,
        )
        .unwrap();
        let numerical = evolve(&solver, &initial, 0.025, 20);
        assert!(distance(&numerical, &exact) < 1e-11);
        let residual = ResidualEvaluator::evaluate(
            &exact,
            &exact.scaled(-viscosity).unwrap(),
            viscosity,
            TOLERANCE,
        )
        .unwrap();
        assert!(l2(&residual) < 1e-14);
        let value = initial.value_at([0.0, 0.0, 0.0]).unwrap();
        assert_eq!(value, [Complex64::new(1.0, 0.0); 3]);
    }

    #[test]
    fn taylor_green_is_three_dimensional_and_not_an_exact_decay_solution() {
        let initial = taylor_green();
        initial.validate_incompressible_real(0.0).unwrap();
        let origin = initial.value_at([0.2, 0.3, 0.4]).unwrap();
        for axis in 0..3 {
            let mut shifted = [0.2, 0.3, 0.4];
            shifted[axis] += 0.2;
            assert!(
                vector_norm(std::array::from_fn(|component| initial
                    .value_at(shifted)
                    .unwrap()[component]
                    - origin[component]))
                    > 0.005
            );
        }
        let full = DirectTriadConvolution.evaluate(&initial, 1, 0.0).unwrap();
        assert!(l2(&full.discarded) > 0.1);
        assert!(
            full.discarded
                .modes()
                .any(|mode| mode.velocity[2].norm() > 0.0)
        );
        let viscosity = 0.1;
        let assumed_decay_derivative = initial.scaled(-3.0 * viscosity).unwrap();
        let residual =
            ResidualEvaluator::evaluate(&initial, &assumed_decay_derivative, viscosity, TOLERANCE)
                .unwrap();
        assert!(distance(&residual, &full.full().unwrap().scaled(-1.0).unwrap()) < 1e-15);
        assert!(l2(&residual) > 0.1);
    }

    #[test]
    fn nonlinear_trajectory_converges_in_timestep_with_direct_fft_agreement() {
        let initial = taylor_green().add_scaled(&abc(), 0.2).unwrap();
        let solver = SpectralReferenceSolver::new(
            DealiasedConvolution::new(9, 729).unwrap(),
            0.1,
            2,
            TOLERANCE,
        )
        .unwrap();
        let coarse = evolve(&solver, &initial, 0.1, 2);
        let fine = evolve(&solver, &initial, 0.05, 4);
        let finer = evolve(&solver, &initial, 0.025, 8);
        let ratio = distance(&coarse, &fine) / distance(&fine, &finer);
        assert!(
            ratio > 12.0 && ratio < 20.0,
            "observed refinement ratio {ratio}"
        );
        let direct =
            SpectralReferenceSolver::new(DirectTriadConvolution, 0.1, 2, TOLERANCE).unwrap();
        let direct_field = evolve(&direct, &initial, 0.1, 2);
        assert!(distance(&coarse, &direct_field) < 1e-12);
        let heat_only = initial
            .map_coefficients("test heat evolution", |k, u| {
                u.map(|value| (-0.1 * 0.2 * wave_number_squared(k)).exp() * value)
            })
            .unwrap();
        assert!(distance(&finer, &heat_only) > 0.01);
        let step = solver.step(&finer, 0.01).unwrap();
        assert!(
            step.discarded_at_stages
                .iter()
                .all(|discarded| l2(discarded) > 0.0)
        );
        let (galerkin_derivative, omitted) = direct.rhs(&finer).unwrap();
        let residual =
            ResidualEvaluator::evaluate(&finer, &galerkin_derivative, 0.1, TOLERANCE).unwrap();
        assert!(distance(&residual, &omitted.scaled(-1.0).unwrap()) < 1e-13);
        assert!(l2(&residual) > 0.0);
    }

    #[test]
    fn field_validation_rejects_invalid_constraints_and_nonfinite_arithmetic() {
        let mode = FourierMode3D {
            wavevector: [1, 0, 0],
            velocity: [ZERO, Complex64::new(1.0, 0.0), ZERO],
        };
        assert!(VectorField3D::new([mode, mode]).is_err());
        assert!(
            VectorField3D::new([mode])
                .unwrap()
                .validate_incompressible_real(0.0)
                .is_err()
        );
        assert!(
            real_field(&[([1, 0, 0], [Complex64::new(1.0, 0.0), ZERO, ZERO])])
                .validate_incompressible_real(TOLERANCE)
                .is_err()
        );
        let mean = VectorField3D::new([FourierMode3D {
            wavevector: [0; 3],
            velocity: [Complex64::new(1.0, 0.0), ZERO, ZERO],
        }])
        .unwrap();
        assert!(mean.validate_incompressible_real(TOLERANCE).is_err());
        assert_eq!(LerayProjector::project(&mean).unwrap(), mean);
        assert!(
            VectorField3D::new([FourierMode3D {
                velocity: [Complex64::new(f64::NAN, 0.0), ZERO, ZERO],
                ..mode
            }])
            .is_err()
        );
        assert!(
            phase_triad(1.0)
                .scaled(f64::MAX)
                .unwrap()
                .scaled(2.0)
                .is_err()
        );
        assert!(
            SobolevNorm::new(1e308, SobolevConvention::Inhomogeneous)
                .unwrap()
                .evaluate(&phase_triad(1.0))
                .is_err()
        );
        assert!(SpectralReferenceSolver::new(DirectTriadConvolution, 0.0, 1, TOLERANCE).is_err());
        let solver =
            SpectralReferenceSolver::new(DirectTriadConvolution, 0.1, 1, TOLERANCE).unwrap();
        assert!(solver.step(&abc(), -0.1).is_err());
        let out_of_range = real_field(&[([2, 0, 0], [ZERO, Complex64::new(1.0, 0.0), ZERO])]);
        assert!(solver.step(&out_of_range, 0.01).is_err());
        assert!(abc().derivative(3).is_err());
        assert!(abc().value_at([f64::INFINITY, 0.0, 0.0]).is_err());
    }
}
