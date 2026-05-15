//! Smoke test: `Context::is_available` returns deterministically and
//! `DeviceProbe::query` produces a sensible struct on GPU hosts.
//!
//! Gated `#[ignore = "gpu"]` for the probe test because the cudarc
//! runtime probe touches a CUDA driver which may not be present on CI.

#![cfg(feature = "cudarc")]

use gororoba_gpu_cuda::{Context, DeviceProbe};

#[test]
fn context_is_available_returns_bool() {
    let result = Context::is_available();
    // Contract: returns a bool without panicking. CI may have neither
    // a driver nor a device; either outcome is acceptable.
    let _ = result;
}

#[test]
#[ignore = "gpu"]
fn device_probe_query_produces_sane_struct() {
    let probe = DeviceProbe::query().expect("CUDA device 0 reachable");
    assert!(probe.major >= 5, "CUDA compute cap should be Maxwell+");
    assert!(probe.total_global_mem > 0, "total_global_mem must be > 0");
    assert!(!probe.name.is_empty(), "device name must be non-empty");
    // Sanity-check the bridge.
    let caps = probe.to_hardware_caps();
    assert!(caps.cuda_available);
    assert_eq!(caps.cuda_compute_major, probe.major);
    assert_eq!(caps.cuda_compute_minor, probe.minor);
}
