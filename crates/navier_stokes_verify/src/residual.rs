// SPDX-License-Identifier: MIT

//! Exact Fourier polynomials and enclosures over complete time slabs.
//!
//! The domain is `(R / 2 pi Z)^3`, and all Sobolev norms use normalized
//! volume: `|u|_s^2 = sum_{k != 0} |k|^(2s) |u_hat[k]|^2`.
//! Omitted coefficients of a complete polynomial are exact zeros. They do
//! not represent unobserved coefficients of a measured continuum field.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::interval::{
    ArithmeticError, Rational, check_rational, checked_add, checked_div, checked_mul, checked_sub,
    int, sqrt_enclosure,
};

pub type Wave = [i32; 3];
pub type ExactVector = [ExactComplex; 3];

pub const MAX_INPUT_MODES: usize = 64;
pub const MAX_INPUT_COORDINATE: i32 = 16;
const MAX_DERIVED_MODES: usize = 16_384;
const MAX_DERIVED_COORDINATE: i32 = 32;
const MAX_TEMPORAL_COEFFICIENTS: usize = 4;

/// Input errors and exhausted arithmetic budgets have different meanings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyMathError {
    InvalidInput(&'static str),
    Arithmetic(ArithmeticError),
}

impl VerifyMathError {
    pub fn is_invalid_input(&self) -> bool {
        match self {
            Self::InvalidInput(_) => true,
            Self::Arithmetic(error) => error.is_invalid_input(),
        }
    }

    pub fn is_implementation_failure(&self) -> bool {
        matches!(self, Self::Arithmetic(error) if error.is_implementation_failure())
    }
}

impl fmt::Display for VerifyMathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid Fourier input: {message}"),
            Self::Arithmetic(error) => write!(formatter, "Fourier arithmetic: {error}"),
        }
    }
}

impl Error for VerifyMathError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidInput(_) => None,
            Self::Arithmetic(error) => Some(error),
        }
    }
}

impl From<ArithmeticError> for VerifyMathError {
    fn from(error: ArithmeticError) -> Self {
        Self::Arithmetic(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactComplex {
    pub re: Rational,
    pub im: Rational,
}

impl ExactComplex {
    pub fn zero() -> Self {
        Self {
            re: int(0),
            im: int(0),
        }
    }

    pub fn real(value: Rational) -> Self {
        Self {
            re: value,
            im: int(0),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.re == int(0) && self.im == int(0)
    }

    pub fn conjugate(&self) -> Result<Self, VerifyMathError> {
        check_rational(&self.re)?;
        Ok(Self {
            re: self.re.clone(),
            im: checked_sub(&int(0), &self.im)?,
        })
    }
}

fn zero_vector() -> ExactVector {
    std::array::from_fn(|_| ExactComplex::zero())
}

fn complex_add(a: &ExactComplex, b: &ExactComplex) -> Result<ExactComplex, VerifyMathError> {
    Ok(ExactComplex {
        re: checked_add(&a.re, &b.re)?,
        im: checked_add(&a.im, &b.im)?,
    })
}

fn complex_sub(a: &ExactComplex, b: &ExactComplex) -> Result<ExactComplex, VerifyMathError> {
    Ok(ExactComplex {
        re: checked_sub(&a.re, &b.re)?,
        im: checked_sub(&a.im, &b.im)?,
    })
}

fn complex_mul(a: &ExactComplex, b: &ExactComplex) -> Result<ExactComplex, VerifyMathError> {
    Ok(ExactComplex {
        re: checked_sub(&checked_mul(&a.re, &b.re)?, &checked_mul(&a.im, &b.im)?)?,
        im: checked_add(&checked_mul(&a.re, &b.im)?, &checked_mul(&a.im, &b.re)?)?,
    })
}

fn complex_scale(a: &ExactComplex, scale: &Rational) -> Result<ExactComplex, VerifyMathError> {
    Ok(ExactComplex {
        re: checked_mul(&a.re, scale)?,
        im: checked_mul(&a.im, scale)?,
    })
}

fn multiply_i(a: &ExactComplex) -> Result<ExactComplex, VerifyMathError> {
    check_rational(&a.re)?;
    Ok(ExactComplex {
        re: checked_sub(&int(0), &a.im)?,
        im: a.re.clone(),
    })
}

fn squared_wave(wave: Wave) -> i64 {
    wave.into_iter().map(|x| i64::from(x) * i64::from(x)).sum()
}

fn wave_dot(wave: Wave, vector: &ExactVector) -> Result<ExactComplex, VerifyMathError> {
    let mut value = ExactComplex::zero();
    for (coordinate, component) in wave.into_iter().zip(vector) {
        value = complex_add(
            &value,
            &complex_scale(component, &int(i64::from(coordinate)))?,
        )?;
    }
    Ok(value)
}

fn project(wave: Wave, vector: &ExactVector) -> Result<ExactVector, VerifyMathError> {
    if wave == [0; 3] {
        return Ok(zero_vector());
    }
    let dot = wave_dot(wave, vector)?;
    let denominator = int(squared_wave(wave));
    let mut result = zero_vector();
    for axis in 0..3 {
        let ratio = checked_div(&int(i64::from(wave[axis])), &denominator)?;
        result[axis] = complex_sub(&vector[axis], &complex_scale(&dot, &ratio)?)?;
    }
    Ok(result)
}

fn accumulate(
    modes: &mut BTreeMap<Wave, ExactVector>,
    wave: Wave,
    vector: &ExactVector,
) -> Result<(), VerifyMathError> {
    let target = modes.entry(wave).or_insert_with(zero_vector);
    for axis in 0..3 {
        target[axis] = complex_add(&target[axis], &vector[axis])?;
    }
    if modes.len() > MAX_DERIVED_MODES {
        return Err(
            ArithmeticError::ResourceLimit("derived Fourier support exceeds budget").into(),
        );
    }
    Ok(())
}

/// A complete, exactly Hermitian finite Fourier polynomial.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FourierPolynomial {
    modes: BTreeMap<Wave, ExactVector>,
}

impl FourierPolynomial {
    /// Input limits apply before zero coefficients are removed.
    pub fn from_modes(
        modes: impl IntoIterator<Item = (Wave, ExactVector)>,
    ) -> Result<Self, VerifyMathError> {
        let mut map = BTreeMap::new();
        for (index, (wave, vector)) in modes.into_iter().enumerate() {
            if index >= MAX_INPUT_MODES {
                return Err(VerifyMathError::InvalidInput(
                    "input exceeds 64 Fourier modes",
                ));
            }
            if wave
                .into_iter()
                .any(|k| !(-MAX_INPUT_COORDINATE..=MAX_INPUT_COORDINATE).contains(&k))
            {
                return Err(VerifyMathError::InvalidInput("input coordinate exceeds 16"));
            }
            for component in &vector {
                check_rational(&component.re)?;
                check_rational(&component.im)?;
            }
            if map.insert(wave, vector).is_some() {
                return Err(VerifyMathError::InvalidInput(
                    "duplicate Fourier wavevector",
                ));
            }
        }
        let result = Self::derived(map)?;
        result.validate_hermitian()?;
        Ok(result)
    }

    fn derived(mut modes: BTreeMap<Wave, ExactVector>) -> Result<Self, VerifyMathError> {
        if modes.len() > MAX_DERIVED_MODES {
            return Err(
                ArithmeticError::ResourceLimit("derived Fourier support exceeds budget").into(),
            );
        }
        if modes
            .keys()
            .flatten()
            .any(|k| !(-MAX_DERIVED_COORDINATE..=MAX_DERIVED_COORDINATE).contains(k))
        {
            return Err(ArithmeticError::ResourceLimit(
                "derived Fourier coordinate exceeds budget",
            )
            .into());
        }
        modes.retain(|_, vector| vector.iter().any(|value| !value.is_zero()));
        Ok(Self { modes })
    }

    pub fn zero() -> Self {
        Self::default()
    }

    pub fn modes(&self) -> &BTreeMap<Wave, ExactVector> {
        &self.modes
    }

    pub fn cutoff(&self) -> i32 {
        self.modes
            .keys()
            .flatten()
            .map(|coordinate| coordinate.abs())
            .max()
            .unwrap_or(0)
    }

    pub fn is_zero(&self) -> bool {
        self.modes.is_empty()
    }

    fn validate_hermitian(&self) -> Result<(), VerifyMathError> {
        for (wave, vector) in &self.modes {
            let negative = wave.map(|k| -k);
            let partner = self
                .modes
                .get(&negative)
                .ok_or(VerifyMathError::InvalidInput("missing Hermitian partner"))?;
            for axis in 0..3 {
                if partner[axis] != vector[axis].conjugate()? {
                    return Err(VerifyMathError::InvalidInput(
                        "Fourier coefficients are not Hermitian",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn is_zero_mean_divergence_free(&self) -> Result<bool, VerifyMathError> {
        if self.modes.contains_key(&[0; 3]) {
            return Ok(false);
        }
        for (wave, vector) in &self.modes {
            if !wave_dot(*wave, vector)?.is_zero() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn validate_zero_mean_divergence_free(&self) -> Result<(), VerifyMathError> {
        if !self.is_zero_mean_divergence_free()? {
            return Err(VerifyMathError::InvalidInput(
                "field must have zero mean and zero divergence",
            ));
        }
        Ok(())
    }

    pub fn project_zero_mean(&self) -> Result<Self, VerifyMathError> {
        let mut modes = BTreeMap::new();
        for (wave, vector) in &self.modes {
            modes.insert(*wave, project(*wave, vector)?);
        }
        Self::derived(modes)
    }

    pub fn curl(&self) -> Result<Self, VerifyMathError> {
        let mut modes = BTreeMap::new();
        for (wave, vector) in &self.modes {
            let mut curl = zero_vector();
            for (axis, component) in curl.iter_mut().enumerate() {
                let j = (axis + 1) % 3;
                let k = (axis + 2) % 3;
                *component = multiply_i(&complex_sub(
                    &complex_scale(&vector[k], &int(i64::from(wave[j])))?,
                    &complex_scale(&vector[j], &int(i64::from(wave[k])))?,
                )?)?;
            }
            modes.insert(*wave, curl);
        }
        Self::derived(modes)
    }

    /// Equality is exact; the zero polynomial also satisfies curl(v) = v.
    pub fn is_unit_curl_beltrami(&self) -> Result<bool, VerifyMathError> {
        Ok(self.is_zero_mean_divergence_free()? && self.curl()? == *self)
    }

    pub fn scaled(&self, scale: &Rational) -> Result<Self, VerifyMathError> {
        check_rational(scale)?;
        let mut modes = BTreeMap::new();
        for (wave, vector) in &self.modes {
            let mut value = zero_vector();
            for axis in 0..3 {
                value[axis] = complex_scale(&vector[axis], scale)?;
            }
            modes.insert(*wave, value);
        }
        Self::derived(modes)
    }

    pub fn added(&self, other: &Self) -> Result<Self, VerifyMathError> {
        let mut modes = self.modes.clone();
        for (wave, vector) in &other.modes {
            accumulate(&mut modes, *wave, vector)?;
        }
        Self::derived(modes)
    }

    pub fn difference(&self, other: &Self) -> Result<Self, VerifyMathError> {
        self.added(&other.scaled(&int(-1))?)
    }

    pub fn sobolev_norm_upper(
        &self,
        order: u32,
        precision_bits: u32,
    ) -> Result<Rational, VerifyMathError> {
        let squared = self.sobolev_norm_squared(order)?;
        Ok(sqrt_enclosure(&squared, precision_bits)?.upper)
    }

    pub fn sobolev_norm_squared(&self, order: u32) -> Result<Rational, VerifyMathError> {
        if order != 3 && order != 4 {
            return Err(VerifyMathError::InvalidInput(
                "only Sobolev orders 3 and 4 are supported",
            ));
        }
        let mut norm = int(0);
        for (wave, vector) in &self.modes {
            let square = int(squared_wave(*wave));
            let mut weight = int(1);
            for _ in 0..order {
                weight = checked_mul(&weight, &square)?;
            }
            for component in vector {
                let amplitude = checked_add(
                    &checked_mul(&component.re, &component.re)?,
                    &checked_mul(&component.im, &component.im)?,
                )?;
                norm = checked_add(&norm, &checked_mul(&weight, &amplitude)?)?;
            }
        }
        Ok(norm)
    }

    /// Ordered convolution retains every output through twice the input cutoff.
    pub fn nonlinear(&self) -> Result<Self, VerifyMathError> {
        self.validate_zero_mean_divergence_free()?;
        self.validate_input_size()?;
        bilinear_nonlinearity(self, self)
    }

    fn validate_input_size(&self) -> Result<(), VerifyMathError> {
        if self.modes.len() > MAX_INPUT_MODES || self.cutoff() > MAX_INPUT_COORDINATE {
            return Err(VerifyMathError::InvalidInput(
                "Fourier reconstruction exceeds input support budget",
            ));
        }
        Ok(())
    }
}

fn bilinear_nonlinearity(
    left: &FourierPolynomial,
    right: &FourierPolynomial,
) -> Result<FourierPolynomial, VerifyMathError> {
    let mut modes = BTreeMap::new();
    for (p, u_p) in &left.modes {
        for (q, u_q) in &right.modes {
            let wave = std::array::from_fn(|axis| p[axis] + q[axis]);
            let dot = wave_dot(*q, u_p)?;
            let mut contraction = zero_vector();
            for axis in 0..3 {
                contraction[axis] = complex_mul(&dot, &u_q[axis])?;
            }
            let projected = project(wave, &contraction)?;
            let mut value = zero_vector();
            for axis in 0..3 {
                value[axis] = complex_scale(&multiply_i(&projected[axis])?, &int(-1))?;
            }
            accumulate(&mut modes, wave, &value)?;
        }
    }
    FourierPolynomial::derived(modes)
}

/// Coefficients are powers of theta = (t - start) / (end - start).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalSlab {
    pub start: Rational,
    pub end: Rational,
    pub coefficients: Vec<FourierPolynomial>,
}

impl TemporalSlab {
    pub fn validate(&self) -> Result<(), VerifyMathError> {
        check_rational(&self.start)?;
        check_rational(&self.end)?;
        if self.start < int(0) || self.end <= self.start {
            return Err(VerifyMathError::InvalidInput(
                "slab requires 0 <= start < end",
            ));
        }
        if self.coefficients.is_empty() || self.coefficients.len() > MAX_TEMPORAL_COEFFICIENTS {
            return Err(VerifyMathError::InvalidInput(
                "temporal degree must lie between zero and three",
            ));
        }
        for coefficient in &self.coefficients {
            coefficient.validate_input_size()?;
            coefficient.validate_hermitian()?;
            coefficient.validate_zero_mean_divergence_free()?;
        }
        Ok(())
    }

    pub fn endpoints(&self) -> Result<(FourierPolynomial, FourierPolynomial), VerifyMathError> {
        self.validate()?;
        let mut end = FourierPolynomial::zero();
        for coefficient in &self.coefficients {
            end = end.added(coefficient)?;
        }
        Ok((self.coefficients[0].clone(), end))
    }

    pub fn joins_exactly(&self, next: &Self) -> Result<bool, VerifyMathError> {
        let (_, end) = self.endpoints()?;
        let (start, _) = next.endpoints()?;
        Ok(self.end == next.start && end == start)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlabBounds {
    pub d3_upper: Rational,
    pub d4_upper: Rational,
    pub residual3_upper: Rational,
    /// The complete quadratic convolution is evaluated through this cube.
    pub quadratic_output_cutoff: i32,
    /// Residual outside the union of all reconstruction coefficient supports.
    /// These terms are included in residual3_upper and are never dropped.
    pub discarded_residual3_upper: Rational,
}

/// The residual includes exact time differentiation and all quadratic outputs.
pub fn residual_coefficients(
    slab: &TemporalSlab,
    viscosity: &Rational,
) -> Result<Vec<FourierPolynomial>, VerifyMathError> {
    slab.validate()?;
    check_rational(viscosity)?;
    if viscosity <= &int(0) {
        return Err(VerifyMathError::InvalidInput("viscosity must be positive"));
    }
    let duration = checked_sub(&slab.end, &slab.start)?;
    let degree = slab.coefficients.len() - 1;
    let mut residual = vec![FourierPolynomial::zero(); 2 * degree + 1];
    for (index, coefficient) in slab.coefficients.iter().enumerate() {
        if index > 0 {
            let scale = checked_div(&int(index as i64), &duration)?;
            residual[index - 1] = residual[index - 1].added(&coefficient.scaled(&scale)?)?;
        }
        let mut diffusion = BTreeMap::new();
        for (wave, vector) in coefficient.modes() {
            let scale = checked_mul(viscosity, &int(squared_wave(*wave)))?;
            let mut value = zero_vector();
            for axis in 0..3 {
                value[axis] = complex_scale(&vector[axis], &scale)?;
            }
            diffusion.insert(*wave, value);
        }
        residual[index] = residual[index].added(&FourierPolynomial::derived(diffusion)?)?;
        for (other_index, other) in slab.coefficients.iter().enumerate() {
            let nonlinear = bilinear_nonlinearity(coefficient, other)?;
            residual[index + other_index] = residual[index + other_index].difference(&nonlinear)?;
        }
    }
    Ok(residual)
}

/// The triangle inequality encloses every theta in [0, 1], without sampling.
pub fn bound_slab(
    slab: &TemporalSlab,
    viscosity: &Rational,
    precision_bits: u32,
) -> Result<SlabBounds, VerifyMathError> {
    let residual = residual_coefficients(slab, viscosity)?;
    let mut d3_upper = int(0);
    let mut d4_upper = int(0);
    let mut input_support = BTreeMap::new();
    let mut cutoff = 0;
    for coefficient in &slab.coefficients {
        d3_upper = checked_add(
            &d3_upper,
            &coefficient.sobolev_norm_upper(3, precision_bits)?,
        )?;
        d4_upper = checked_add(
            &d4_upper,
            &coefficient.sobolev_norm_upper(4, precision_bits)?,
        )?;
        cutoff = cutoff.max(coefficient.cutoff());
        for wave in coefficient.modes.keys() {
            input_support.insert(*wave, ());
        }
    }
    let mut residual3_upper = int(0);
    let mut discarded_residual3_upper = int(0);
    for coefficient in residual {
        residual3_upper = checked_add(
            &residual3_upper,
            &coefficient.sobolev_norm_upper(3, precision_bits)?,
        )?;
        let discarded = FourierPolynomial::derived(
            coefficient
                .modes
                .into_iter()
                .filter(|(wave, _)| !input_support.contains_key(wave))
                .collect(),
        )?;
        discarded_residual3_upper = checked_add(
            &discarded_residual3_upper,
            &discarded.sobolev_norm_upper(3, precision_bits)?,
        )?;
    }
    Ok(SlabBounds {
        d3_upper,
        d4_upper,
        residual3_upper,
        quadratic_output_cutoff: 2 * cutoff,
        discarded_residual3_upper,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex(re: i64, im: i64) -> ExactComplex {
        ExactComplex {
            re: int(re),
            im: int(im),
        }
    }

    fn paired(modes: Vec<(Wave, ExactVector)>) -> FourierPolynomial {
        let mut complete = Vec::new();
        for (wave, vector) in modes {
            let conjugate = std::array::from_fn(|axis| vector[axis].conjugate().unwrap());
            complete.push((wave, vector));
            complete.push((wave.map(|k| -k), conjugate));
        }
        FourierPolynomial::from_modes(complete).unwrap()
    }

    fn shear() -> FourierPolynomial {
        paired(vec![(
            [0, 1, 0],
            [complex(1, 0), complex(0, 0), complex(0, 0)],
        )])
    }

    fn abc() -> FourierPolynomial {
        paired(vec![
            ([1, 0, 0], [complex(0, 0), complex(0, -1), complex(1, 0)]),
            ([0, 1, 0], [complex(1, 0), complex(0, 0), complex(0, -1)]),
            ([0, 0, 1], [complex(0, -1), complex(1, 0), complex(0, 0)]),
        ])
    }

    fn constant_slab(field: FourierPolynomial) -> TemporalSlab {
        TemporalSlab {
            start: int(0),
            end: int(1),
            coefficients: vec![field],
        }
    }

    #[test]
    fn exact_shear_nonlinearity_vanishes() {
        let field = shear();
        assert!(field.is_zero_mean_divergence_free().unwrap());
        assert!(field.nonlinear().unwrap().is_zero());
        assert_eq!(field.sobolev_norm_squared(3).unwrap(), int(2));
        assert_eq!(field.sobolev_norm_squared(4).unwrap(), int(2));
        assert!(!field.is_unit_curl_beltrami().unwrap());
    }

    #[test]
    fn abc_has_exact_unit_curl_and_zero_projected_nonlinearity() {
        let field = abc();
        assert_eq!(field.curl().unwrap(), field);
        assert!(field.is_unit_curl_beltrami().unwrap());
        assert!(field.nonlinear().unwrap().is_zero());
        assert_eq!(field.sobolev_norm_squared(3).unwrap(), int(12));
        assert_eq!(field.sobolev_norm_squared(4).unwrap(), int(12));
    }

    #[test]
    fn exact_projection_removes_mean_and_longitudinal_part() {
        let field = paired(vec![(
            [1, 1, 0],
            [complex(1, 0), complex(0, 0), complex(0, 0)],
        )]);
        let mut modes = field.modes().clone();
        modes.insert([0; 3], [complex(1, 0), complex(2, 0), complex(3, 0)]);
        let field = FourierPolynomial::from_modes(modes).unwrap();
        assert!(!field.is_zero_mean_divergence_free().unwrap());
        let projected = field.project_zero_mean().unwrap();
        assert!(projected.is_zero_mean_divergence_free().unwrap());
        assert_eq!(projected.project_zero_mean().unwrap(), projected);
        assert!(!projected.modes().contains_key(&[0; 3]));
        let half = checked_div(&int(1), &int(2)).unwrap();
        assert_eq!(projected.modes()[&[1, 1, 0]][0].re, half);
        assert_eq!(
            projected.modes()[&[1, 1, 0]][1].re,
            checked_sub(&int(0), &half).unwrap()
        );
    }

    #[test]
    fn constant_temporal_residual_is_exact_viscous_defect() {
        let field = shear();
        let slab = constant_slab(field.clone());
        assert_eq!(
            residual_coefficients(&slab, &int(2)).unwrap(),
            vec![field.scaled(&int(2)).unwrap()]
        );
        let bounds = bound_slab(&slab, &int(2), 64).unwrap();
        let exact_square = checked_mul(&bounds.residual3_upper, &bounds.residual3_upper).unwrap();
        assert!(exact_square >= int(8));
        assert_eq!(bounds.discarded_residual3_upper, int(0));
    }

    #[test]
    fn linear_shear_reconstruction_retains_full_time_residual() {
        let field = shear();
        let slab = TemporalSlab {
            start: int(2),
            end: int(4),
            coefficients: vec![field.clone(), field.scaled(&int(-2)).unwrap()],
        };
        let residual = residual_coefficients(&slab, &int(1)).unwrap();
        assert!(residual[0].is_zero());
        assert_eq!(residual[1], field.scaled(&int(-2)).unwrap());
        assert!(residual[2].is_zero());
        let (start, end) = slab.endpoints().unwrap();
        assert_eq!(start, field);
        assert_eq!(end, field.scaled(&int(-1)).unwrap());
        let bounds = bound_slab(&slab, &int(1), 64).unwrap();
        assert!(bounds.residual3_upper > int(2));
    }

    #[test]
    fn nonlinear_residual_preserves_modes_outside_reconstruction_cutoff() {
        let field = paired(vec![
            ([1, 1, 0], [complex(1, 0), complex(-1, 0), complex(0, 0)]),
            ([1, 0, 1], [complex(0, 0), complex(1, 0), complex(0, 0)]),
        ]);
        let nonlinear = field.nonlinear().unwrap();
        assert!(nonlinear.is_zero_mean_divergence_free().unwrap());
        let wave = [2, 1, 1];
        assert!(nonlinear.modes().contains_key(&wave));
        assert!(!field.modes().contains_key(&wave));
        let third = checked_div(&int(1), &int(3)).unwrap();
        let expected = [
            ExactComplex {
                re: int(0),
                im: checked_sub(&int(0), &third).unwrap(),
            },
            ExactComplex {
                re: int(0),
                im: third.clone(),
            },
            ExactComplex {
                re: int(0),
                im: third,
            },
        ];
        assert_eq!(nonlinear.modes()[&wave], expected);
        let slab = constant_slab(field);
        let residual = residual_coefficients(&slab, &int(1)).unwrap();
        assert_eq!(
            residual[0].modes()[&wave],
            nonlinear.scaled(&int(-1)).unwrap().modes()[&wave]
        );
        let bounds = bound_slab(&slab, &int(1), 64).unwrap();
        assert_eq!(bounds.quadratic_output_cutoff, 2);
        assert!(bounds.discarded_residual3_upper > int(0));
        assert!(bounds.residual3_upper >= bounds.discarded_residual3_upper);
    }

    #[test]
    fn non_hermitian_and_duplicate_inputs_fail() {
        let vector = [complex(0, 0), complex(1, 0), complex(0, 0)];
        assert!(
            FourierPolynomial::from_modes([([1, 0, 0], vector.clone())])
                .unwrap_err()
                .is_invalid_input()
        );
        assert!(
            FourierPolynomial::from_modes([([1, 0, 0], vector.clone()), ([1, 0, 0], vector),])
                .unwrap_err()
                .is_invalid_input()
        );
        assert!(
            FourierPolynomial::from_modes([(
                [0; 3],
                [complex(0, 1), complex(0, 0), complex(0, 0)]
            )])
            .is_err()
        );
    }

    #[test]
    fn temporal_and_coordinate_limits_are_enforced() {
        assert!(FourierPolynomial::from_modes([([17, 0, 0], zero_vector())]).is_err());
        assert!(FourierPolynomial::from_modes([([i32::MIN, 0, 0], zero_vector())]).is_err());
        let empty = TemporalSlab {
            start: int(0),
            end: int(1),
            coefficients: vec![],
        };
        assert!(empty.validate().is_err());
        let degree_four = TemporalSlab {
            start: int(0),
            end: int(1),
            coefficients: vec![FourierPolynomial::zero(); 5],
        };
        assert!(degree_four.validate().is_err());
        let backwards = TemporalSlab {
            start: int(1),
            end: int(0),
            coefficients: vec![shear()],
        };
        assert!(backwards.validate().is_err());
        assert!(bound_slab(&constant_slab(shear()), &int(0), 64).is_err());
    }

    #[test]
    fn mode_limits_count_zero_inputs_before_canonicalization() {
        let modes = (-16..=16)
            .flat_map(|x| (-16..=16).map(move |y| ([x, y, 0], zero_vector())))
            .take(65);
        assert!(
            FourierPolynomial::from_modes(modes)
                .unwrap_err()
                .is_invalid_input()
        );
    }

    #[test]
    fn slab_joins_require_exact_time_and_field_match() {
        let left = constant_slab(shear());
        let right = TemporalSlab {
            start: int(1),
            end: int(2),
            coefficients: vec![shear()],
        };
        assert!(left.joins_exactly(&right).unwrap());
        let discontinuous = TemporalSlab {
            start: int(1),
            end: int(2),
            coefficients: vec![abc()],
        };
        assert!(!left.joins_exactly(&discontinuous).unwrap());
        let gap = TemporalSlab {
            start: int(2),
            end: int(3),
            coefficients: vec![shear()],
        };
        assert!(!left.joins_exactly(&gap).unwrap());
    }

    #[test]
    fn divergence_in_a_temporal_coefficient_is_rejected() {
        let field = paired(vec![(
            [1, 0, 0],
            [complex(1, 0), complex(0, 0), complex(0, 0)],
        )]);
        assert!(
            constant_slab(field)
                .validate()
                .unwrap_err()
                .is_invalid_input()
        );
    }
}
