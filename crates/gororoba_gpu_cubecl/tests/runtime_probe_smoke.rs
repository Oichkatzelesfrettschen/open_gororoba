//! Smoke test: `Runtime::probe()` returns deterministically (true or false)
//! without panicking, regardless of whether a wgpu adapter is reachable.

#![cfg(feature = "cubecl")]

use gororoba_gpu_cubecl::Runtime;

#[test]
fn probe_returns_without_panic() {
    let result = Runtime::probe();
    // We do not assert true OR false -- CI runners may lack a wgpu adapter.
    // The contract is that probe() returns a bool without panicking.
    let _ = result;
}
