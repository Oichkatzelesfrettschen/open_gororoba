//! `Buffer<T>`: stream-attached CUDA device allocations.
//!
//! Consolidates 60+ `stream.alloc_zeros::<T>(n)` + 40+
//! `stream.memcpy_stod` + `memcpy_dtoh` sites across the workspace.
//! Provides a single owned-buffer type so callers can express upload /
//! download / readback without re-reading cudarc's CudaSlice API per
//! site.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::error::Result;

/// Owned CUDA device buffer of `T` elements, attached to a stream.
pub struct Buffer<T: DeviceRepr> {
    inner: CudaSlice<T>,
    stream: Arc<CudaStream>,
}

impl<T: DeviceRepr + ValidAsZeroBits> Buffer<T> {
    /// Allocate `len` zero-initialised elements on the stream's device.
    pub fn alloc_zeros(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        let inner = stream.alloc_zeros::<T>(len)?;
        Ok(Self {
            inner,
            stream: stream.clone(),
        })
    }
}

impl<T: DeviceRepr> Buffer<T> {
    /// Allocate a device buffer initialised by copying from host data.
    /// Uses cudarc 0.19's `clone_htod`.
    pub fn htod(stream: &Arc<CudaStream>, host: &[T]) -> Result<Self>
    where
        T: Unpin,
    {
        let inner = stream.clone_htod(host)?;
        Ok(Self {
            inner,
            stream: stream.clone(),
        })
    }

    /// Copy device contents back to a host buffer.
    pub fn dtoh(&self, host: &mut [T]) -> Result<()>
    where
        T: Unpin,
    {
        self.stream.memcpy_dtoh(&self.inner, host)?;
        Ok(())
    }

    /// Copy device contents to a freshly-allocated host Vec.
    pub fn dtoh_vec(&self) -> Result<Vec<T>>
    where
        T: Clone + Default + Unpin,
    {
        let mut host = vec![T::default(); self.inner.len()];
        self.dtoh(&mut host)?;
        Ok(host)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True iff len == 0.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Borrow the underlying CudaSlice for direct cudarc API use.
    pub fn raw(&self) -> &CudaSlice<T> {
        &self.inner
    }

    /// Mutable borrow of the underlying CudaSlice.
    pub fn raw_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.inner
    }

    /// Borrow the stream this buffer was allocated on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}
