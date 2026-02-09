use thiserror::Error;

#[derive(Error, Debug)]
pub enum AlgebraError {
    #[error("Dimension mismatch: expected {expected}, got {found}")]
    DimensionMismatch {
        expected: usize,
        found: usize,
    },

    #[error("Division by zero")]
    ZeroDivision,

    #[error("Zero divisor found: {0:?}")]
    ZeroDivisorFound(Vec<f64>),

    #[error("Invalid basis index: {index} for dimension {dim}")]
    InvalidBasisIndex {
        index: usize,
        dim: usize,
    },

    #[error("Operation not supported for dimension {0}")]
    UnsupportedDimension(usize),

    #[error("Simd error: {0}")]
    SimdError(String),
}

pub type AlgebraResult<T> = Result<T, AlgebraError>;
