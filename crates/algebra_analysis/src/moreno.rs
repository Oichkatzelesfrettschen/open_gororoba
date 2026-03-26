//! Compatibility shim for the extracted Moreno (1997) paper crate.
//!
//! The executable Moreno lane now lives in `moreno_1997`. This module keeps the
//! existing `algebra_analysis::moreno` path stable for downstream callers.

pub use moreno_1997::*;
