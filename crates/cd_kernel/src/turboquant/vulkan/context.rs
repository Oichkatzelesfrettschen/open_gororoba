//! Vulkan device capability detection and context management.
//!
//! Probes Vulkan physical device for features relevant to TurboQuant:
//! subgroup operations, cooperative matrix, buffer device address.
//! No ash dependency required for the capability struct itself -- the
//! actual Vulkan initialization lives behind the `vulkan` feature gate.

/// Vulkan device capabilities relevant to TurboQuant compute shaders.
#[derive(Clone, Debug)]
pub struct VulkanCapabilities {
    /// Device name.
    pub name: String,
    /// Vulkan API version (major, minor, patch).
    pub api_version: (u32, u32, u32),
    /// Subgroup size (typically 32 for NVIDIA, 64 for AMD).
    pub subgroup_size: u32,
    /// Whether subgroup basic operations are supported (Vulkan 1.1+).
    pub subgroup_basic: bool,
    /// Whether subgroup arithmetic (add/min/max) is supported.
    pub subgroup_arithmetic: bool,
    /// Whether subgroup shuffle operations are supported.
    pub subgroup_shuffle: bool,
    /// Whether subgroup extended types (8/16-bit) are supported.
    pub subgroup_extended_types: bool,
    /// Whether VK_KHR_cooperative_matrix is available.
    pub cooperative_matrix: bool,
    /// Whether bufferDeviceAddress is available (Vulkan 1.2+).
    pub buffer_device_address: bool,
    /// Whether 16-bit storage (Float16/Int16 in SSBO) is supported.
    pub storage_16bit: bool,
    /// Whether 8-bit storage (Int8 in SSBO) is supported.
    pub storage_8bit: bool,
    /// Max compute workgroup size (x dimension).
    pub max_workgroup_size_x: u32,
    /// Max compute shared memory in bytes.
    pub max_shared_memory: u32,
}

impl VulkanCapabilities {
    /// Determine the best TurboQuant shader variant for this device.
    pub fn recommended_shader_tier(&self) -> VulkanShaderTier {
        if self.cooperative_matrix && self.subgroup_shuffle {
            VulkanShaderTier::CooperativeMatrix
        } else if self.subgroup_arithmetic && self.storage_8bit {
            VulkanShaderTier::SubgroupOptimized
        } else if self.subgroup_basic {
            VulkanShaderTier::SubgroupBaseline
        } else {
            VulkanShaderTier::Portable
        }
    }

    /// Whether this device can run TurboQuant at all.
    pub fn is_compute_capable(&self) -> bool {
        self.max_workgroup_size_x >= 64
    }
}

/// Vulkan shader optimization tier.
///
/// Analogous to the CUDA KernelTier -- select the most optimized shader
/// variant that the hardware supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanShaderTier {
    /// No subgroup ops -- pure portable compute shader.
    Portable,
    /// Subgroup basic (broadcast, ballot) for reduction.
    SubgroupBaseline,
    /// Subgroup arithmetic + 8-bit storage for packed indices.
    SubgroupOptimized,
    /// Cooperative matrix for fused dequant+matmul.
    CooperativeMatrix,
}

impl std::fmt::Display for VulkanShaderTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanShaderTier::Portable => write!(f, "portable"),
            VulkanShaderTier::SubgroupBaseline => write!(f, "subgroup-baseline"),
            VulkanShaderTier::SubgroupOptimized => write!(f, "subgroup-optimized"),
            VulkanShaderTier::CooperativeMatrix => write!(f, "cooperative-matrix"),
        }
    }
}

/// Probe Vulkan device capabilities.
///
/// When the `vulkan` feature is enabled, uses ash to enumerate physical
/// devices and query features.  Without it, returns None.
#[cfg(feature = "vulkan")]
pub fn probe_vulkan() -> Option<VulkanCapabilities> {
    // Vulkan initialization via ash is heavyweight -- defer to
    // a dedicated initialization path.  For now, return None
    // to indicate "not yet initialized" rather than "not available".
    // The actual ash::Instance/PhysicalDevice setup will be added
    // when the Vulkan compute shaders are ready.
    None
}

#[cfg(not(feature = "vulkan"))]
pub fn probe_vulkan() -> Option<VulkanCapabilities> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_nvidia() -> VulkanCapabilities {
        VulkanCapabilities {
            name: "NVIDIA RTX 4070 Ti".into(),
            api_version: (1, 3, 0),
            subgroup_size: 32,
            subgroup_basic: true,
            subgroup_arithmetic: true,
            subgroup_shuffle: true,
            subgroup_extended_types: true,
            cooperative_matrix: false, // not all drivers expose this
            buffer_device_address: true,
            storage_16bit: true,
            storage_8bit: true,
            max_workgroup_size_x: 1024,
            max_shared_memory: 49152,
        }
    }

    fn mock_amd() -> VulkanCapabilities {
        VulkanCapabilities {
            name: "AMD RX 7900 XTX".into(),
            api_version: (1, 3, 0),
            subgroup_size: 64,
            subgroup_basic: true,
            subgroup_arithmetic: true,
            subgroup_shuffle: true,
            subgroup_extended_types: true,
            cooperative_matrix: false,
            buffer_device_address: true,
            storage_16bit: true,
            storage_8bit: true,
            max_workgroup_size_x: 1024,
            max_shared_memory: 65536,
        }
    }

    fn mock_integrated() -> VulkanCapabilities {
        VulkanCapabilities {
            name: "Intel UHD 770".into(),
            api_version: (1, 2, 0),
            subgroup_size: 16,
            subgroup_basic: true,
            subgroup_arithmetic: false,
            subgroup_shuffle: false,
            subgroup_extended_types: false,
            cooperative_matrix: false,
            buffer_device_address: true,
            storage_16bit: true,
            storage_8bit: false,
            max_workgroup_size_x: 512,
            max_shared_memory: 32768,
        }
    }

    #[test]
    fn test_nvidia_tier() {
        let caps = mock_nvidia();
        assert_eq!(
            caps.recommended_shader_tier(),
            VulkanShaderTier::SubgroupOptimized
        );
        assert!(caps.is_compute_capable());
    }

    #[test]
    fn test_amd_tier() {
        let caps = mock_amd();
        assert_eq!(
            caps.recommended_shader_tier(),
            VulkanShaderTier::SubgroupOptimized
        );
    }

    #[test]
    fn test_integrated_tier() {
        let caps = mock_integrated();
        // No subgroup arithmetic -> baseline only
        assert_eq!(
            caps.recommended_shader_tier(),
            VulkanShaderTier::SubgroupBaseline
        );
    }

    #[test]
    fn test_cooperative_matrix_tier() {
        let mut caps = mock_nvidia();
        caps.cooperative_matrix = true;
        assert_eq!(
            caps.recommended_shader_tier(),
            VulkanShaderTier::CooperativeMatrix
        );
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(format!("{}", VulkanShaderTier::Portable), "portable");
        assert_eq!(
            format!("{}", VulkanShaderTier::CooperativeMatrix),
            "cooperative-matrix"
        );
    }
}
