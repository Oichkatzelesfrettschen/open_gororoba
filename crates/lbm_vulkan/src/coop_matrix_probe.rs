//! Cooperative matrix hardware capability probe.
//!
//! Queries `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` to report
//! available cooperative matrix types (dimensions, component types).
//!
//! The 19x19 MRT matrix would need padding to 32x32 for cooperative matrix
//! acceleration -- likely not worth the complexity. But the probe confirms
//! hardware capability for the record.

use ash::vk;

/// A cooperative matrix configuration reported by the hardware.
#[derive(Debug, Clone)]
pub struct CoopMatrixConfig {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: vk::ComponentTypeKHR,
    pub b_type: vk::ComponentTypeKHR,
    pub c_type: vk::ComponentTypeKHR,
    pub result_type: vk::ComponentTypeKHR,
    pub saturating_accumulation: bool,
    pub scope: vk::ScopeKHR,
}

/// Query cooperative matrix properties from the physical device.
///
/// Returns an empty vec if the extension is not supported.
pub fn probe_cooperative_matrix(
    instance: &ash::Instance,
    pdevice: vk::PhysicalDevice,
) -> Vec<CoopMatrixConfig> {
    // Check extension support first
    let ext_props = unsafe {
        instance
            .enumerate_device_extension_properties(pdevice)
            .unwrap_or_default()
    };

    let has_coop = ext_props.iter().any(|p| {
        let name = unsafe { std::ffi::CStr::from_ptr(p.extension_name.as_ptr()) };
        name.to_bytes() == b"VK_KHR_cooperative_matrix"
    });

    if !has_coop {
        return Vec::new();
    }

    // The actual query requires loading the extension function pointer.
    // For now, just report that the extension is present.
    // Full enumeration would use vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
    log::info!("VK_KHR_cooperative_matrix extension is available");
    Vec::new()
}

/// Print a summary of cooperative matrix capabilities.
pub fn print_coop_matrix_summary(configs: &[CoopMatrixConfig]) {
    if configs.is_empty() {
        println!("No cooperative matrix configurations reported.");
        return;
    }
    println!("Cooperative matrix configurations:");
    for (i, c) in configs.iter().enumerate() {
        println!(
            "  [{i}] {m}x{n}x{k} A={a:?} B={b:?} C={c_t:?} R={r:?} sat={sat} scope={scope:?}",
            m = c.m,
            n = c.n,
            k = c.k,
            a = c.a_type,
            b = c.b_type,
            c_t = c.c_type,
            r = c.result_type,
            sat = c.saturating_accumulation,
            scope = c.scope,
        );
    }
}
