//! Smoke test for the gororoba_gpu_vulkan InstanceBuilder.
//!
//! Gated `#[ignore = "gpu"]` because the Vulkan loader is not always
//! available on CI runners. Run locally with:
//!
//!     cargo test -p gororoba_gpu_vulkan --features ash -- --ignored

#![cfg(feature = "ash")]

use gororoba_gpu_vulkan::{InstanceBuilder, ValidationPolicy};

#[test]
#[ignore = "gpu"]
fn instance_builder_constructs_and_drops() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = InstanceBuilder::new("gpu_vulkan_smoke_test")
        .validation(ValidationPolicy::Disable)
        .build()
        .expect("Vulkan loader present + driver supports requested api_version");
    assert!(matches!(
        instance.validation(),
        ValidationPolicy::Disable | ValidationPolicy::Enable
    ));
    // Drop runs vk::DestroyInstance; the test passes if it does not panic.
}

#[test]
#[ignore = "gpu"]
fn validation_default_matches_build_profile() {
    let default = ValidationPolicy::default_for_profile();
    if cfg!(debug_assertions) {
        assert_eq!(default, ValidationPolicy::Enable);
    } else {
        assert_eq!(default, ValidationPolicy::Disable);
    }
}
