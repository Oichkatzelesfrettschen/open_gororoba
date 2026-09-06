//! Outward enclosures for globally isolated bundles of finite Bloch Hamiltonians.
//!
//! Coefficients denote their exact stored binary64 values. Source coefficient
//! uncertainty requires a separate perturbation bound.

use crate::tight_binding::TightBindingModel;
use faer::{Mat, Side, c64};

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn finite(self) -> Self {
        assert!(
            self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi,
            "Interval arithmetic exhausted finite enclosure range"
        );
        self
    }
    fn point(value: f64) -> Self {
        assert!(value.is_finite());
        Self {
            lo: value,
            hi: value,
        }
    }
    fn add(self, other: Self) -> Self {
        Self {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
        .finite()
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }
    fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        Self {
            lo: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
        .finite()
    }
    fn div(self, other: Self) -> Self {
        assert!(other.lo > 0.0);
        self.mul(
            Self {
                lo: (1.0 / other.hi).next_down(),
                hi: (1.0 / other.lo).next_up(),
            }
            .finite(),
        )
    }
    fn abs_upper(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
    fn widen(self, radius: f64) -> Self {
        self.add(Self {
            lo: -radius,
            hi: radius,
        })
    }
    fn midpoint(self) -> f64 {
        self.lo * 0.5 + self.hi * 0.5
    }
}

fn atan_inverse(denominator: f64) -> Interval {
    let argument = Interval::point(1.0).div(Interval::point(denominator));
    let square = argument.mul(argument);
    let mut power = argument;
    let mut sum = Interval::point(0.0);
    for index in 0..32 {
        let term = power.div(Interval::point(f64::from(2 * index + 1)));
        sum = if index % 2 == 0 {
            sum.add(term)
        } else {
            sum.sub(term)
        };
        power = power.mul(square);
    }
    sum.widen(power.div(Interval::point(65.0)).abs_upper())
}

fn pi() -> Interval {
    atan_inverse(5.0)
        .mul(Interval::point(16.0))
        .sub(atan_inverse(239.0).mul(Interval::point(4.0)))
}

fn trig(argument: Interval, sine: bool) -> Interval {
    let square = argument.mul(argument);
    let mut term = if sine { argument } else { Interval::point(1.0) };
    let mut sum = term;
    for index in 1..24 {
        let degree = 2 * index + i32::from(sine);
        term = term
            .mul(square)
            .neg()
            .div(Interval::point(f64::from(degree * (degree - 1))));
        sum = sum.add(term);
    }
    // Lagrange's theorem bounds the next-order remainder by |x|^m/m!.
    let degree = if sine { 48 } else { 47 };
    let mut remainder = Interval::point(1.0);
    for divisor in 1..=degree {
        remainder = remainder
            .mul(Interval::point(argument.abs_upper()))
            .div(Interval::point(f64::from(divisor)));
    }
    sum.widen(remainder.hi)
}

#[derive(Clone, Copy)]
struct ComplexInterval {
    re: Interval,
    im: Interval,
}
impl ComplexInterval {
    fn point(value: c64) -> Self {
        Self {
            re: Interval::point(value.re),
            im: Interval::point(value.im),
        }
    }
    fn zero() -> Self {
        Self::point(c64::new(0.0, 0.0))
    }
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re.add(other.re),
            im: self.im.add(other.im),
        }
    }
    fn sub(self, other: Self) -> Self {
        self.add(Self {
            re: other.re.neg(),
            im: other.im.neg(),
        })
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: self.im.neg(),
        }
    }
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re.mul(other.re).sub(self.im.mul(other.im)),
            im: self.re.mul(other.im).add(self.im.mul(other.re)),
        }
    }
    fn norm_upper(self) -> Interval {
        Interval::point(self.re.abs_upper()).add(Interval::point(self.im.abs_upper()))
    }
}

/// One terminal dyadic square; coordinates denote integer cell indices.
#[derive(Clone, Debug)]
pub struct GapCell {
    pub depth: u32,
    pub coordinates: [u32; 2],
    pub gap_lower: f64,
    pub eigenvalue_radius: f64,
    pub orthogonality_upper: f64,
    pub residual_upper: f64,
    pub variation_upper: f64,
    pub center_eigenvalues: Vec<f64>,
}

/// Full terminal cover and its bounded admission predicate.
#[derive(Debug)]
pub struct GapCertificate {
    pub leaves: Vec<GapCell>,
    pub admitted: bool,
    pub hopping_row_upper: f64,
}

fn hamiltonian(
    model: &TightBindingModel,
    depth: u32,
    coordinates: [u32; 2],
    pi_bound: Interval,
) -> Vec<Vec<ComplexInterval>> {
    let dimension = model.orbitals.len();
    let mut matrix = vec![vec![ComplexInterval::zero(); dimension]; dimension];
    for (index, orbital) in model.orbitals.iter().enumerate() {
        matrix[index][index].re = Interval::point(orbital.on_site_energy);
    }
    let denominator = 1_i64 << (depth + 1);
    for hopping in &model.hoppings {
        let numerator = i64::from(hopping.cell_offset[0]) * (2 * i64::from(coordinates[0]) + 1)
            + i64::from(hopping.cell_offset[1]) * (2 * i64::from(coordinates[1]) + 1);
        let reduced = (numerator + denominator / 2).rem_euclid(denominator) - denominator / 2;
        // Both integer conversions and division by a power of two are exact.
        let phase = pi_bound.mul(Interval::point(2.0 * reduced as f64 / denominator as f64));
        let exponential = ComplexInterval {
            re: trig(phase, false),
            im: trig(phase, true),
        };
        let contribution = ComplexInterval::point(hopping.amplitude).mul(exponential);
        matrix[hopping.from][hopping.to] = matrix[hopping.from][hopping.to].add(contribution);
        matrix[hopping.to][hopping.from] =
            matrix[hopping.to][hopping.from].add(contribution.conj());
    }
    matrix
}

fn candidate_error(
    matrix: &[Vec<ComplexInterval>],
    eigenvalues: &[f64],
    vectors: &Mat<c64>,
) -> Result<(f64, f64, f64), String> {
    let dimension = eigenvalues.len();
    let mut product = vec![vec![ComplexInterval::zero(); dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            for inner in 0..dimension {
                product[row][column] = product[row][column]
                    .add(matrix[row][inner].mul(ComplexInterval::point(vectors[(inner, column)])));
            }
        }
    }
    let mut eta: f64 = 0.0;
    let mut rho: f64 = 0.0;
    for row in 0..dimension {
        let mut gram_row = Interval::point(0.0);
        let mut residual_row = Interval::point(0.0);
        for column in 0..dimension {
            let mut gram = ComplexInterval::zero();
            let mut transformed = ComplexInterval::zero();
            for inner in 0..dimension {
                let left = ComplexInterval::point(vectors[(inner, row)]).conj();
                gram = gram.add(left.mul(ComplexInterval::point(vectors[(inner, column)])));
                transformed = transformed.add(left.mul(product[inner][column]));
            }
            if row == column {
                gram = gram.sub(ComplexInterval::point(c64::new(1.0, 0.0)));
                transformed =
                    transformed.sub(ComplexInterval::point(c64::new(eigenvalues[row], 0.0)));
            }
            gram_row = gram_row.add(gram.norm_upper());
            residual_row = residual_row.add(transformed.norm_upper());
        }
        eta = eta.max(gram_row.hi);
        rho = rho.max(residual_row.hi);
    }
    if !eta.is_finite() || !rho.is_finite() || eta >= 1.0 {
        return Err("Candidate basis lacks a finite invertibility enclosure".into());
    }
    let diagonal_norm = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let radius = Interval::point(rho)
        .add(Interval::point(diagonal_norm).mul(Interval::point(eta)))
        .div(Interval::point(1.0).sub(Interval::point(eta)))
        .hi;
    if !radius.is_finite() {
        return Err("Nonfinite eigenvalue enclosure".into());
    }
    Ok((radius, eta, rho))
}

/// Certify selected adjacent boundaries over the entire unit reciprocal torus.
///
/// A boundary `b` separates ordered eigenvalues `b-1` and `b`. Exhausted
/// cells remain in the cover and force `admitted=false`.
pub fn certify_global_gaps(
    model: &TightBindingModel,
    boundaries: &[usize],
    maximum_depth: u32,
    gate: f64,
) -> Result<GapCertificate, String> {
    let smallest = std::hint::black_box(f64::from_bits(1));
    let normal = std::hint::black_box(f64::MIN_POSITIVE);
    if smallest * std::hint::black_box(2.0) != f64::from_bits(2)
        || normal * std::hint::black_box(0.5) != f64::from_bits(1_u64 << 51)
    {
        return Err("IEEE gradual underflow is required".into());
    }
    let dimension = model.orbitals.len();
    if maximum_depth > 20
        || boundaries.is_empty()
        || boundaries
            .iter()
            .any(|&boundary| boundary == 0 || boundary >= dimension)
        || !gate.is_finite()
        || gate < 0.0
        || model
            .orbitals
            .iter()
            .any(|orbital| !orbital.on_site_energy.is_finite())
        || model.hoppings.iter().any(|hop| {
            hop.from >= dimension
                || hop.to >= dimension
                || !hop.amplitude.re.is_finite()
                || !hop.amplitude.im.is_finite()
        })
    {
        return Err("Invalid finite model, boundary, depth or gate".into());
    }
    let pi_bound = pi();
    let mut rows = vec![Interval::point(0.0); dimension];
    for hop in &model.hoppings {
        let offset_norm = f64::from(hop.cell_offset[0]).abs() + f64::from(hop.cell_offset[1]).abs();
        let weight = ComplexInterval::point(hop.amplitude)
            .norm_upper()
            .mul(Interval::point(offset_norm));
        rows[hop.from] = rows[hop.from].add(weight);
        rows[hop.to] = rows[hop.to].add(weight);
    }
    let row_bound = rows.iter().map(|value| value.hi).fold(0.0, f64::max);
    let mut pending = vec![(0, [0, 0])];
    let mut leaves = Vec::new();
    while let Some((depth, coordinates)) = pending.pop() {
        let matrix = hamiltonian(model, depth, coordinates, pi_bound);
        let midpoint = Mat::from_fn(dimension, dimension, |row, column| {
            c64::new(
                matrix[row][column].re.midpoint(),
                matrix[row][column].im.midpoint(),
            )
        });
        let eigen = midpoint
            .self_adjoint_eigen(Side::Lower)
            .map_err(|error| format!("Candidate eigensolver: {error:?}"))?;
        let mut ordering: Vec<usize> = (0..dimension).collect();
        ordering.sort_by(|&left, &right| {
            eigen.S().column_vector()[left]
                .re
                .total_cmp(&eigen.S().column_vector()[right].re)
        });
        let values: Vec<f64> = ordering
            .iter()
            .map(|&index| eigen.S().column_vector()[index].re)
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("Nonfinite candidate eigenvalue".into());
        }
        let vectors = Mat::from_fn(dimension, dimension, |row, column| {
            eigen.U()[(row, ordering[column])]
        });
        let (radius, eta, rho) = candidate_error(&matrix, &values, &vectors)?;
        let variation = pi_bound
            .mul(Interval::point(2.0))
            .mul(Interval::point(row_bound))
            .div(Interval::point(f64::from(1_u32 << depth)))
            .hi;
        let gap_lower = boundaries
            .iter()
            .map(|&boundary| {
                Interval::point(values[boundary])
                    .sub(Interval::point(values[boundary - 1]))
                    .sub(Interval::point(radius).mul(Interval::point(2.0)))
                    .sub(Interval::point(variation))
                    .lo
            })
            .fold(f64::INFINITY, f64::min);
        if gap_lower > gate || depth == maximum_depth {
            leaves.push(GapCell {
                depth,
                coordinates,
                gap_lower,
                eigenvalue_radius: radius,
                orthogonality_upper: eta,
                residual_upper: rho,
                variation_upper: variation,
                center_eigenvalues: values,
            });
        } else {
            for horizontal in 0..2 {
                for vertical in 0..2 {
                    pending.push((
                        depth + 1,
                        [
                            2 * coordinates[0] + horizontal,
                            2 * coordinates[1] + vertical,
                        ],
                    ));
                }
            }
        }
    }
    let admitted =
        leaves.iter().all(|cell| cell.gap_lower > gate) && validate_cover(&leaves, maximum_depth);
    Ok(GapCertificate {
        leaves,
        admitted,
        hopping_row_upper: row_bound,
    })
}

/// Verify exact dyadic area and absence of duplicated or ancestor-overlapping leaves.
pub fn validate_cover(leaves: &[GapCell], maximum_depth: u32) -> bool {
    if maximum_depth > 20 {
        return false;
    }
    let mut cells = std::collections::BTreeSet::new();
    let mut area = 0_u64;
    for cell in leaves {
        if cell.depth > maximum_depth
            || cell
                .coordinates
                .iter()
                .any(|&coordinate| coordinate >= 1_u32 << cell.depth)
            || !cells.insert((cell.depth, cell.coordinates))
        {
            return false;
        }
        area += 1_u64 << (2 * (maximum_depth - cell.depth));
    }
    for cell in leaves {
        for ancestor_depth in 0..cell.depth {
            let shift = cell.depth - ancestor_depth;
            if cells.contains(&(
                ancestor_depth,
                [cell.coordinates[0] >> shift, cell.coordinates[1] >> shift],
            )) {
                return false;
            }
        }
    }
    area == 1_u64 << (2 * maximum_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arithmetic_and_trigonometric_enclosures() {
        let third = Interval::point(1.0).div(Interval::point(3.0));
        assert!(
            third.mul(Interval::point(3.0)).lo <= 1.0 && third.mul(Interval::point(3.0)).hi >= 1.0
        );
        let pi_bound = pi();
        let sine = trig(pi_bound.div(Interval::point(2.0)), true);
        assert!(sine.lo <= 1.0 && sine.hi >= 1.0 && sine.hi - sine.lo < 1e-12);
        let cosine = trig(pi_bound, false);
        assert!(cosine.lo <= -1.0 && cosine.hi >= -1.0 && cosine.hi - cosine.lo < 1e-12);
        let matrix = vec![vec![ComplexInterval::zero(); 2]; 2];
        assert!(candidate_error(&matrix, &[0.0, 0.0], &Mat::zeros(2, 2)).is_err());
    }

    #[test]
    fn incorrect_spectrum_is_enclosed_and_overflow_fails_closed() {
        let matrix = vec![vec![ComplexInterval::zero(); 2]; 2];
        let identity = Mat::from_fn(2, 2, |row, column| c64::new(f64::from(row == column), 0.0));
        let (radius, _, _) = candidate_error(&matrix, &[0.0, 100.0], &identity).unwrap();
        assert!(radius >= 100.0);
        assert!(
            std::panic::catch_unwind(|| Interval::point(f64::MAX).mul(Interval::point(2.0)))
                .is_err()
        );
        let subnormal = Interval::point(f64::from_bits(1)).mul(Interval::point(0.5));
        assert!(subnormal.lo <= 0.0 && subnormal.hi >= f64::from_bits(1));
    }
}
