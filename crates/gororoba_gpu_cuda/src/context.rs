//! CUDA context acquisition.
//!
//! Consolidates the 48+ ad-hoc `CudaContext::new(0)` + `is_ok()` sites
//! across the workspace into a single typed entry point with proper
//! error handling.

use std::{panic::AssertUnwindSafe, sync::Arc};

use cudarc::{
    driver::{CudaContext, CudaStream},
    runtime::result::{RuntimeError, device as cudart_device},
};

use crate::error::{CudaError, Result};

fn normalize_runtime_device_count(
    runtime_result: std::result::Result<i32, RuntimeError>,
) -> Result<usize> {
    let count = runtime_result.map_err(|_| CudaError::RuntimeUnavailable)?;
    Ok(count.max(0) as usize)
}

pub(crate) fn runtime_device_count() -> Result<usize> {
    let runtime_result = std::panic::catch_unwind(cudart_device::get_count)
        .map_err(|_| CudaError::RuntimeUnavailable)?;
    normalize_runtime_device_count(runtime_result)
}

/// Acquired CUDA context, clone-cheap via internal Arc.
#[derive(Clone)]
pub struct Context {
    inner: Arc<CudaContext>,
}

impl Context {
    /// Acquire a CUDA context for the default device (ordinal 0).
    ///
    /// Returns `Err(CudaError::NoDevice)` when no CUDA-capable device
    /// is present rather than panicking.
    pub fn with_default_device() -> Result<Self> {
        Self::with_device(0)
    }

    /// Acquire a CUDA context for the given device ordinal.
    pub fn with_device(ordinal: usize) -> Result<Self> {
        let count = runtime_device_count()?;
        if count == 0 {
            return Err(CudaError::NoDevice);
        }
        if ordinal >= count {
            return Err(CudaError::OrdinalOutOfRange { ordinal, count });
        }
        let ctx = std::panic::catch_unwind(AssertUnwindSafe(|| CudaContext::new(ordinal)))
            .map_err(|_| CudaError::RuntimeUnavailable)??;
        Ok(Self { inner: ctx })
    }

    /// True iff at least one CUDA device is present (does not actually
    /// create a context). Cheap; safe to call from any thread.
    pub fn is_available() -> bool {
        runtime_device_count().is_ok_and(|count| count > 0)
    }

    /// Borrow the underlying `Arc<CudaContext>` for ash-equivalent FFI
    /// calls or for handing to other cudarc APIs.
    pub fn raw(&self) -> &Arc<CudaContext> {
        &self.inner
    }

    /// The default stream for the context. All four call sites in
    /// lbm_3d_cuda use this pattern.
    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.inner.default_stream()
    }
}

#[cfg(test)]
mod tests {
    use cudarc::runtime::{result::RuntimeError, sys::cudaError_t::cudaErrorInsufficientDriver};

    use super::normalize_runtime_device_count;
    use crate::error::CudaError;

    #[test]
    fn runtime_device_count_preserves_runtime_failure_class() {
        let result = normalize_runtime_device_count(Err(RuntimeError(cudaErrorInsufficientDriver)));
        assert!(matches!(result, Err(CudaError::RuntimeUnavailable)));
    }
}
