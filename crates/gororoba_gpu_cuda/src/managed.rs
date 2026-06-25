//! Managed (unified) memory + prefetch hints.
//!
//! Consolidates lbm_3d_cuda/src/managed_memory.rs:96-117 -- the canonical
//! reference for 1024^3 out-of-core sparse LBM. The crate's
//! `ManagedBuffer<T>` type is the lifted version; LBM-specific
//! tile-orchestration logic stays in lbm_3d_cuda.

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::error::Result;

/// Allocation policy for managed memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagedResidency {
    /// Initial residency on the device (CU_MEM_ATTACH_GLOBAL).
    DeviceFirst,
    /// Initial residency on the host (CU_MEM_ATTACH_HOST).
    HostFirst,
}

/// Unified-memory buffer over `T` elements.
///
/// The buffer is allocated via `cuMemAllocManaged` (unified addressing)
/// so both host and device can access it without explicit copies. Use
/// `prefetch_to_device` before kernel launches to avoid page-fault
/// latency on the first access.
pub struct ManagedBuffer<T> {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    ptr: *mut T,
    len: usize,
    residency: ManagedResidency,
}

// SAFETY: Managed memory is unified-addressed; the device side is
// serialised by the CUDA runtime when the host accesses overlap kernel
// execution. Callers must respect that contract.
unsafe impl<T: Send> Send for ManagedBuffer<T> {}
unsafe impl<T: Sync> Sync for ManagedBuffer<T> {}

impl<T> ManagedBuffer<T> {
    /// Allocate `len` elements of managed memory. The allocation is
    /// uninitialised; call `as_slice_mut().fill(...)` before first use
    /// if zero-initialised storage is needed.
    pub fn new(ctx: &Arc<CudaContext>, len: usize, residency: ManagedResidency) -> Result<Self> {
        use cudarc::driver::sys;
        let bytes = len.saturating_mul(std::mem::size_of::<T>());
        let flags = match residency {
            ManagedResidency::DeviceFirst => sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL,
            ManagedResidency::HostFirst => sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_HOST,
        };
        let mut device_ptr: sys::CUdeviceptr = 0;
        ctx.bind_to_thread()?;
        // SAFETY: cuMemAllocManaged writes a device pointer to the
        // stack-local variable. The CUDA context has been made current
        // on this thread before the raw driver call.
        let result =
            unsafe { sys::cuMemAllocManaged(&mut device_ptr as *mut _, bytes, flags as u32) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(crate::error::CudaError::Driver(
                cudarc::driver::DriverError(result),
            ));
        }
        Ok(Self {
            ctx: ctx.clone(),
            ptr: device_ptr as *mut T,
            len,
            residency,
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True iff len == 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Initial residency policy this buffer was allocated with.
    pub fn residency(&self) -> ManagedResidency {
        self.residency
    }

    /// Raw device pointer (host-visible too, since unified).
    pub fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    /// Mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Host-side slice view. SAFETY: caller must ensure no kernel is
    /// concurrently writing the same range.
    ///
    /// # Safety
    /// As above; managed memory unifies host + device address space but
    /// the runtime does not serialise concurrent accesses from both
    /// sides.
    pub unsafe fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
    }

    /// Host-side mutable slice view. Same SAFETY contract as `as_slice`.
    ///
    /// # Safety
    /// See `as_slice` -- caller must ensure no concurrent device access.
    pub unsafe fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        if !self.ptr.is_null() {
            self.ctx.record_err(self.ctx.bind_to_thread());
            // SAFETY: pointer was returned by cuMemAllocManaged in
            // Self::new, has not been freed, and the context bind result
            // has been recorded before issuing the raw driver free.
            let result = unsafe { sys::cuMemFree_v2(self.ptr as sys::CUdeviceptr) };
            if result != sys::CUresult::CUDA_SUCCESS {
                self.ctx
                    .record_err::<()>(Err(cudarc::driver::DriverError(result)));
            }
        }
    }
}
