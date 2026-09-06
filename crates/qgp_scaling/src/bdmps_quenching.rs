//! Infinite-energy outgoing BDMPS quenching through its Laplace functional.
//!
//! Arleo hep-ph/0210104v3 Eq. 2.7 supplies the outgoing spectrum;
//! BDMS hep-ph/0106347v1 Eq. 18b supplies the independent-emission functional.
//! Analytic omitted-tail bounds and quadrature-order changes remain separate.
//! Order changes are convergence diagnostics, rather than certified error bars.
use gauss_quad::{FiniteAboveNegOneF64, GaussLaguerre, GaussLegendre};
use std::{
    error::Error,
    f64::consts::{LN_2, PI, SQRT_2},
    fmt,
    num::NonZeroUsize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuenchingError {
    InvalidInput(&'static str),
    NonfiniteArithmetic,
    QuadratureConstruction,
}
impl fmt::Display for QuenchingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BDMPS quenching: {self:?}")
    }
}
impl Error for QuenchingError {}
#[derive(Debug, Clone, Copy)]
pub struct LaplaceEstimate {
    pub value: f64,
    pub exponent: f64,
    pub tail_bound: f64,
    pub quadrature_change: f64,
}
#[derive(Debug, Clone, Copy)]
pub struct ConvolutionEstimate {
    pub value: f64,
    /// Maximum of the two normalized Laguerre sums of analytic inner-tail
    /// bounds. Bounds omitted spectrum tails for those discrete rules only;
    /// the exact continuous Gamma mixture requires separate error admission.
    pub tail_bound: f64,
    pub inner_quadrature_change: f64,
    pub outer_quadrature_change: f64,
}
impl ConvolutionEstimate {
    /// Apply absolute engineering gates without interpreting order changes as bounds.
    pub fn passes_numerical_gates(&self, tolerance: f64) -> bool {
        tolerance.is_finite()
            && tolerance > 0.0
            && self.value.is_finite()
            && (0.0..=1.0).contains(&self.value)
            && [
                self.tail_bound,
                self.inner_quadrature_change,
                self.outer_quadrature_change,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0 && *value <= tolerance)
    }
}
fn finite(value: f64) -> Result<f64, QuenchingError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(QuenchingError::NonfiniteArithmetic)
    }
}
fn settings(order: usize, tolerance: f64) -> Result<(NonZeroUsize, NonZeroUsize), QuenchingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(QuenchingError::InvalidInput(
            "positive finite tolerance required",
        ));
    }
    let coarse = NonZeroUsize::new(order).ok_or(QuenchingError::InvalidInput(
        "positive quadrature order required",
    ))?;
    let fine = order
        .checked_mul(2)
        .and_then(NonZeroUsize::new)
        .ok_or(QuenchingError::InvalidInput("quadrature order overflow"))?;
    Ok((coarse, fine))
}

/// log|cos((1+i)/sqrt(2x))|, excluding the alpha/x spectrum prefactor.
pub fn outgoing_spectrum_shape(x: f64) -> Result<f64, QuenchingError> {
    if !x.is_finite() || x <= 0.0 {
        return Err(QuenchingError::InvalidInput(
            "positive finite spectrum coordinate required",
        ));
    }
    let value = if x >= 1.0 {
        let base = (2.0 / x).powi(2);
        let mut term = base / 24.0;
        let mut sum = term;
        for index in 2..=32 {
            let degree = f64::from(4 * index);
            term *= base / (degree * (degree - 1.0) * (degree - 2.0) * (degree - 3.0));
            let next = sum + term;
            if next == sum {
                break;
            }
            sum = next;
        }
        0.5 * sum.ln_1p()
    } else {
        let argument = finite(1.0 / (SQRT_2 * x.sqrt()))?;
        let decay = (-2.0 * argument).exp();
        let correction = if decay == 0.0 {
            0.0
        } else {
            2.0 * (2.0 * argument).cos() * decay + decay * decay
        };
        argument - LN_2 + 0.5 * correction.ln_1p()
    };
    let value = finite(value)?;
    if value < 0.0 {
        return Err(QuenchingError::NonfiniteArithmetic);
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy)]
pub struct OutgoingBdmps {
    alpha: f64,
    mean: f64,
}
impl OutgoingBdmps {
    pub fn new(alpha_s: f64, casimir: f64) -> Result<Self, QuenchingError> {
        if !alpha_s.is_finite() || alpha_s < 0.0 || !casimir.is_finite() || casimir <= 0.0 {
            return Err(QuenchingError::InvalidInput(
                "nonnegative coupling and positive Casimir required",
            ));
        }
        let product = finite(alpha_s * casimir)?;
        if alpha_s > 0.0 && product == 0.0 {
            return Err(QuenchingError::NonfiniteArithmetic);
        }
        let alpha = finite(product * (2.0 / PI))?;
        let mean = product / 2.0;
        if product > 0.0 && (alpha == 0.0 || mean == 0.0) {
            return Err(QuenchingError::NonfiniteArithmetic);
        }
        Ok(Self { alpha, mean })
    }
    pub fn mean_over_omega_c(&self) -> f64 {
        self.mean
    }
    pub fn spectrum_shape(&self, x: f64) -> Result<f64, QuenchingError> {
        outgoing_spectrum_shape(x)
    }
    pub fn laplace(
        &self,
        s: f64,
        inner_order: usize,
        tolerance: f64,
    ) -> Result<LaplaceEstimate, QuenchingError> {
        let orders = settings(inner_order, tolerance)?;
        if !s.is_finite() || s < 0.0 {
            return Err(QuenchingError::InvalidInput(
                "nonnegative finite Laplace coordinate required",
            ));
        }
        if s == 0.0 || self.alpha == 0.0 {
            return Ok(LaplaceEstimate {
                value: 1.0,
                exponent: 0.0,
                tail_bound: 0.0,
                quadrature_change: 0.0,
            });
        }
        let log_alpha = self.alpha.ln();
        let log_low = (2.0 * (tolerance.ln() - 4.0_f64.ln() - log_alpha - SQRT_2.ln() - s.ln()))
            .min(0.1_f64.ln());
        let log_high = (0.5 * (log_alpha - 6.0_f64.ln() - tolerance.ln())).max(10.0_f64.ln());
        if log_low.exp() == 0.0 || !log_high.exp().is_finite() {
            return Err(QuenchingError::NonfiniteArithmetic);
        }
        let tail_bound = finite(
            (log_alpha + SQRT_2.ln() + s.ln() + log_low / 2.0).exp()
                + (log_alpha - 24.0_f64.ln() - 2.0 * log_high).exp(),
        )?;
        let integrate = |order| -> Result<f64, QuenchingError> {
            let rule = GaussLegendre::new(order);
            let half_width = (log_high - log_low) / 2.0;
            let center = log_low / 2.0 + log_high / 2.0;
            let mut sum = 0.0;
            for (node, weight) in rule.nodes().zip(rule.weights()) {
                let x = (center + half_width * node).exp();
                let sx = s * x;
                // Overflow here has the exact limiting response one; the
                // exponential underflows long before sx exceeds binary64.
                let response = if sx.is_infinite() {
                    1.0
                } else {
                    -(-sx).exp_m1()
                };
                sum = finite(sum + weight * outgoing_spectrum_shape(x)? * response)?;
            }
            finite(self.alpha * half_width * sum)
        };
        let coarse = integrate(orders.0)?;
        let exponent = integrate(orders.1)?;
        if exponent < 0.0 {
            return Err(QuenchingError::NonfiniteArithmetic);
        }
        Ok(LaplaceEstimate {
            value: (-exponent).exp(),
            exponent,
            tail_bound,
            quadrature_change: (exponent - coarse).abs(),
        })
    }

    /// Gamma-mixture convolution in u=pT/omega_c, with 0<n<=128.
    /// Laguerre weights are normalized by their positive finite sum. The
    /// normalization enforces the constant moment and is not a gamma-error bound.
    pub fn raa_omega_c(
        &self,
        u: f64,
        n: f64,
        inner_order: usize,
        outer_order: usize,
        tolerance: f64,
    ) -> Result<ConvolutionEstimate, QuenchingError> {
        settings(inner_order, tolerance)?;
        let orders = settings(outer_order, tolerance)?;
        if !u.is_finite() || u <= 0.0 || !n.is_finite() || n <= 0.0 || n > 128.0 {
            return Err(QuenchingError::InvalidInput(
                "positive finite u and 0<n<=128 required",
            ));
        }
        if self.alpha == 0.0 {
            return Ok(ConvolutionEstimate {
                value: 1.0,
                tail_bound: 0.0,
                inner_quadrature_change: 0.0,
                outer_quadrature_change: 0.0,
            });
        }
        let parameter =
            FiniteAboveNegOneF64::new(n - 1.0).ok_or(QuenchingError::QuadratureConstruction)?;
        let integrate = |order| -> Result<(f64, f64, f64), QuenchingError> {
            let rule = GaussLaguerre::new(order, parameter);
            let mut weight_sum = 0.0;
            for weight in rule.weights() {
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(QuenchingError::QuadratureConstruction);
                }
                weight_sum = finite(weight_sum + weight)?;
            }
            if weight_sum <= 0.0 {
                return Err(QuenchingError::QuadratureConstruction);
            }
            let mut value = 0.0;
            let mut tail = 0.0;
            let mut change = 0.0;
            for (node, weight) in rule.nodes().zip(rule.weights()) {
                if !node.is_finite() || *node <= 0.0 {
                    return Err(QuenchingError::QuadratureConstruction);
                }
                let normalized = weight / weight_sum;
                let laplace = self.laplace(finite(node / u)?, inner_order, tolerance)?;
                value = finite(value + normalized * laplace.value)?;
                tail = finite(tail + normalized * laplace.tail_bound)?;
                change = finite(change + normalized * laplace.quadrature_change)?;
            }
            Ok((value, tail, change))
        };
        let coarse = integrate(orders.0)?;
        let fine = integrate(orders.1)?;
        Ok(ConvolutionEstimate {
            value: fine.0,
            tail_bound: coarse.1.max(fine.1),
            inner_quadrature_change: coarse.2.max(fine.2),
            outer_quadrature_change: (fine.0 - coarse.0).abs(),
        })
    }
    /// Mean-loss coordinate v=pT/<epsilon>, mapped to u=v*<epsilon>/omega_c.
    pub fn raa_mean_loss(
        &self,
        v: f64,
        n: f64,
        inner_order: usize,
        outer_order: usize,
        tolerance: f64,
    ) -> Result<ConvolutionEstimate, QuenchingError> {
        if self.mean <= 0.0 {
            return Err(QuenchingError::InvalidInput(
                "mean coordinate undefined at zero coupling",
            ));
        }
        if !v.is_finite() || v <= 0.0 {
            return Err(QuenchingError::InvalidInput(
                "positive finite mean coordinate required",
            ));
        }
        self.raa_omega_c(
            finite(v * self.mean)?,
            n,
            inner_order,
            outer_order,
            tolerance,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_input_and_zero_coupling() {
        assert!(OutgoingBdmps::new(-0.1, 3.0).is_err());
        assert!(outgoing_spectrum_shape(0.0).is_err());
        let model = OutgoingBdmps::new(0.0, 3.0).unwrap();
        assert_eq!(model.laplace(1.0, 8, 1e-8).unwrap().value, 1.0);
        assert_eq!(model.raa_omega_c(2.0, 6.1, 8, 8, 1e-8).unwrap().value, 1.0);
        assert!(model.raa_mean_loss(2.0, 6.1, 8, 8, 1e-8).is_err());
        assert!(model.laplace(f64::NAN, 8, 1e-8).is_err());
    }
    #[test]
    fn stable_spectrum_and_mean() {
        let model = OutgoingBdmps::new(0.3, 3.0).unwrap();
        assert!((model.mean_over_omega_c() - 0.45).abs() < 1e-15);
        for x in [1e-200, 0.1, 1.0, 10.0, 1e100] {
            assert!(outgoing_spectrum_shape(x).unwrap() > 0.0);
        }
        assert!(
            (outgoing_spectrum_shape(1.0).unwrap() - outgoing_spectrum_shape(1.0 - 1e-12).unwrap())
                .abs()
                < 1e-12
        );
        assert!((outgoing_spectrum_shape(1e6).unwrap() * 12e12 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_coupling_accepts_tiny_positive_spectral_index() {
        let model = OutgoingBdmps::new(0.0, 3.0).unwrap();
        let index = f64::MIN_POSITIVE;
        assert_eq!(index - 1.0, -1.0);
        let estimate = model.raa_omega_c(2.0, index, 8, 8, 1e-8).unwrap();
        assert_eq!(estimate.value, 1.0);
        assert_eq!(estimate.tail_bound, 0.0);
        assert_eq!(estimate.inner_quadrature_change, 0.0);
        assert_eq!(estimate.outer_quadrature_change, 0.0);
        assert!(model.raa_omega_c(2.0, 0.0, 8, 8, 1e-8).is_err());
    }
}
