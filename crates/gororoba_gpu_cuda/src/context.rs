//! CUDA context acquisition.
//!
//! Consolidates the 48+ ad-hoc `CudaContext::new(0)` + `is_ok()` sites
//! across the workspace into a single typed entry point with proper
//! error handling.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use cudarc::runtime::result::device as cudart_device;

use crate::error::{CudaError, Result};

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
        let count = cudart_device::get_count().unwrap_or(0).max(0) as usize;
        if count == 0 {
            return Err(CudaError::NoDevice);
        }
        if ordinal >= count {
            return Err(CudaError::OrdinalOutOfRange { ordinal, count });
        }
        let ctx = CudaContext::new(ordinal)?;
        Ok(Self { inner: ctx })
    }

    /// True iff at least one CUDA device is present (does not actually
    /// create a context). Cheap; safe to call from any thread.
    pub fn is_available() -> bool {
        cudart_device::get_count().unwrap_or(0) > 0
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
