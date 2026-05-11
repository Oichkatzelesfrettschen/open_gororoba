//! AoSoA (Array-of-Structures-of-Arrays) layout helpers for the 3D D3Q19
//! distribution-function buffer.
//!
//! The chunk size of 4 f64 lanes matches one AVX2 256-bit YMM register
//! (`wide::f64x4`). Memory layout per chunk:
//!
//!   [Dir0(c0,c1,c2,c3), Dir1(c0,c1,c2,c3), ..., Dir18(c0,c1,c2,c3)]
//!
//! One chunk footprint is 19 * 4 * 8 = 608 bytes -- ~2% of a 32 KB L1D
//! line, which avoids the 8-way associativity thrashing that pure SoA
//! causes on x86. The pull-streaming and MRT collision passes both read
//! and write through `UnsafeAoSoAPtr`, which carries a SAFETY contract
//! that the parallel index math is disjoint across rayon threads.

use wide::f64x4;

/// AoSoA chunk size in f64 lanes (one 256-bit YMM register on AVX2).
pub const AOSOA_CHUNK: usize = 4;

/// Compute AoSoA index for a given cell and direction.
///
/// Maps `(cell, dir)` to a flat index into the AoSoA f-vector:
///   chunk_idx = cell / CHUNK
///   lane      = cell % CHUNK
///   index     = chunk_idx * 19 * CHUNK + dir * CHUNK + lane
#[inline(always)]
pub fn aosoa_idx(cell: usize, dir: usize) -> usize {
    let chunk = cell / AOSOA_CHUNK;
    let lane = cell % AOSOA_CHUNK;
    chunk * 19 * AOSOA_CHUNK + dir * AOSOA_CHUNK + lane
}

/// Round up to the nearest multiple of AOSOA_CHUNK.
#[inline(always)]
pub(super) fn aosoa_pad(n: usize) -> usize {
    n.div_ceil(AOSOA_CHUNK) * AOSOA_CHUNK
}

/// Zero-cost wrapper to bypass the compiler's inability to prove disjoint
/// index math across parallel threads.
///
/// SAFETY contract: the caller guarantees via `aosoa_idx` algebra that no
/// two rayon threads will ever read-write or write-write the same address
/// simultaneously. This holds because `aosoa_idx(a, d) != aosoa_idx(b, d)`
/// for `a != b`, and the collision step only accesses indices belonging to
/// its own cell. Pull streaming (phase 2) is serial and uses a separate
/// scratch buffer, so no aliasing occurs there either.
#[derive(Copy, Clone)]
pub struct UnsafeAoSoAPtr<T>(pub *mut T);

// SAFETY: UnsafeAoSoAPtr wraps a pointer into a Vec-backed buffer. Send+Sync
// is safe because: (1) rayon's par_iter partitions index ranges so no two
// threads access the same offset, (2) the backing Vec outlives all rayon tasks
// via the enclosing scope, (3) writes target disjoint offsets only.
unsafe impl<T> Send for UnsafeAoSoAPtr<T> {}
unsafe impl<T> Sync for UnsafeAoSoAPtr<T> {}

impl<T> UnsafeAoSoAPtr<T> {
    /// Read a value from the specified offset without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `offset` is in bounds and no concurrent write
    /// to the same address occurs.
    #[inline(always)]
    pub unsafe fn read(&self, offset: usize) -> T {
        // SAFETY: offset is computed from grid indices (i,j,k,q) bounded by
        // grid dimensions. The total buffer length is n1*n2*n3*Q which exceeds
        // any valid offset. The caller guarantees no concurrent write.
        unsafe { core::ptr::read(self.0.add(offset)) }
    }

    /// Write a value to the specified offset without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `offset` is in bounds and no concurrent access
    /// (read or write) to the same address occurs.
    #[inline(always)]
    pub unsafe fn write(&self, offset: usize, val: T) {
        // SAFETY: offset is computed from grid indices (i,j,k,q) bounded by
        // grid dimensions. The total buffer length is n1*n2*n3*Q which exceeds
        // any valid offset. The caller guarantees no concurrent access.
        unsafe { core::ptr::write(self.0.add(offset), val) }
    }
}

impl UnsafeAoSoAPtr<f64> {
    /// Read an aligned CHUNK=4 f64 slice as f64x4 (256-bit VMOVAPD).
    ///
    /// # Safety
    /// Caller must ensure `offset` points to 4 contiguous, valid f64 values
    /// and no concurrent write to the same addresses occurs.
    #[inline(always)]
    pub unsafe fn read_x4(&self, offset: usize) -> f64x4 {
        unsafe {
            let arr = core::ptr::read(self.0.add(offset) as *const [f64; 4]);
            f64x4::new(arr)
        }
    }

    /// Write an f64x4 (256-bit VMOVAPD) to an aligned CHUNK=4 f64 slice.
    ///
    /// # Safety
    /// Caller must ensure `offset` points to 4 writable f64 slots and no
    /// concurrent access to the same addresses occurs.
    #[inline(always)]
    pub unsafe fn write_x4(&self, offset: usize, val: f64x4) {
        unsafe {
            let arr: [f64; 4] = val.to_array();
            core::ptr::write(self.0.add(offset) as *mut [f64; 4], arr);
        }
    }
}
