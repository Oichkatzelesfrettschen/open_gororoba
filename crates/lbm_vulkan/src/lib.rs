use ash::{Device, Entry, Instance, vk};
use gpu_allocator::vulkan::*;
use std::{ffi::CStr, sync::Arc};

pub mod besag_clifford_vulkan;
pub mod compute;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuTier {
    Constrained = 0, // < 2GB VRAM
    Standard = 1,    // 2-4GB VRAM
    High = 2,        // 4-8GB VRAM
    Ultra = 3,       // > 8GB VRAM
}

#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub tier: GpuTier,
    pub vram_mb: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_name: String,
    pub supports_fp16: bool,
    pub supports_fp64: bool,
    pub max_compute_shared_memory_size: u32,
}

#[derive(Clone)]
pub struct VulkanContext {
    pub entry: Arc<Entry>,
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub allocator: Arc<std::sync::Mutex<Allocator>>,
    pub caps: HardwareCapabilities,
}

impl VulkanContext {
    pub fn new(enable_validation: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load() }?;
        let app_name = c"Gororoba Vulkan LBM";

        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: app_name.as_ptr(),
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

        let layer_names = if enable_validation {
            vec![c"VK_LAYER_KHRONOS_validation"]
        } else {
            vec![]
        };
        let layers_raw: Vec<*const i8> = layer_names.iter().map(|name| name.as_ptr()).collect();
        let _extensions: Vec<*const i8> = vec![];

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_layer_count: layers_raw.len() as u32,
            pp_enabled_layer_names: layers_raw.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        let instance = Arc::new(instance);

        let pdevices = unsafe { instance.enumerate_physical_devices() }?;
        let (pdevice, queue_family_index) = pdevices
            .iter()
            .find_map(|pdevice| {
                unsafe { instance.get_physical_device_queue_family_properties(*pdevice) }
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        if info
                            .queue_flags
                            .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
                        {
                            Some((*pdevice, index as u32))
                        } else {
                            None
                        }
                    })
            })
            .ok_or("No suitable physical device found")?;

        // Extract capabilities
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let mut vram_bytes = 0;
        for i in 0..mem_props.memory_heap_count {
            let heap = mem_props.memory_heaps[i as usize];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                vram_bytes = std::cmp::max(vram_bytes, heap.size);
            }
        }
        let vram_mb = (vram_bytes / (1024 * 1024)) as u32;

        let tier = if vram_mb < 2048 {
            GpuTier::Constrained
        } else if vram_mb < 4096 {
            GpuTier::Standard
        } else if vram_mb < 8192 {
            GpuTier::High
        } else {
            GpuTier::Ultra
        };

        // Check for FP16 support
        let mut features16 = vk::PhysicalDeviceFloat16Int8FeaturesKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
            ..Default::default()
        };
        let mut features2 = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: &mut features16 as *mut _ as *mut _,
            ..Default::default()
        };
        unsafe { instance.get_physical_device_features2(pdevice, &mut features2) };

        let caps = HardwareCapabilities {
            tier,
            vram_mb,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_name: unsafe {
                CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            },
            supports_fp16: features16.shader_float16 == vk::TRUE,
            supports_fp64: features2.features.shader_float64 == vk::TRUE,
            max_compute_shared_memory_size: props.limits.max_compute_shared_memory_size,
        };

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures {
            s_type: vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            dynamic_rendering: vk::TRUE,
            ..Default::default()
        };

        // Chain f16 device features when hardware supports them.
        // Float16Int8 enables f16 arithmetic in shaders;
        // 16BitStorage enables f16 in storage buffers (SPIR-V StorageBuffer16BitAccess).
        let mut storage16_features = vk::PhysicalDevice16BitStorageFeaturesKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
            storage_buffer16_bit_access: if caps.supports_fp16 { vk::TRUE } else { vk::FALSE },
            ..Default::default()
        };

        let mut float16_features = vk::PhysicalDeviceFloat16Int8FeaturesKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
            shader_float16: if caps.supports_fp16 { vk::TRUE } else { vk::FALSE },
            p_next: &mut storage16_features as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };

        // Build the p_next chain: dynamic_rendering -> float16 -> storage16
        dynamic_rendering.p_next = &mut float16_features as *mut _ as *mut std::ffi::c_void;

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            p_next: &mut dynamic_rendering as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(pdevice, &device_create_info, None) }?;
        let device = Arc::new(device);
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: (*instance).clone(),
            device: (*device).clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        let allocator = Arc::new(std::sync::Mutex::new(allocator));

        Ok(Self {
            entry: Arc::new(entry),
            instance,
            device,
            physical_device: pdevice,
            queue,
            queue_family_index,
            allocator,
            caps,
        })
    }

    pub fn get_scaling_parameters(&self) -> ScalingParameters {
        let n = self.optimal_grid_dim();
        ScalingParameters {
            grid_dim: (n, n, n),
            precision: if self.caps.supports_fp16 && self.caps.tier != GpuTier::Ultra {
                Precision::FP16
            } else {
                Precision::FP32
            },
            math_complexity: match self.caps.tier {
                GpuTier::Constrained => MathComplexity::Newtonian,
                GpuTier::Standard => MathComplexity::Schwarzschild,
                GpuTier::High => MathComplexity::Kerr,
                GpuTier::Ultra => MathComplexity::KerrNewman,
            },
            render_resolution_scale: match self.caps.tier {
                GpuTier::Constrained => 0.5,
                GpuTier::Standard => 0.75,
                _ => 1.0,
            },
        }
    }

    fn optimal_grid_dim(&self) -> u32 {
        let vram_bytes = self.caps.vram_mb as u64 * 1024 * 1024;
        let target_bytes = (vram_bytes as f64 * 0.7) as u64;

        let bytes_per_cell = match self.caps.tier {
            GpuTier::Constrained => 128,
            _ => 256,
        };

        let max_cells = target_bytes / bytes_per_cell;
        let n = (max_cells as f64).cbrt() as u32;
        let n = (n / 8) * 8;
        std::cmp::max(n, 32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP16,
    FP32,
    FP64,
}

impl Precision {
    /// Size of one distribution-function element in bytes.
    #[must_use]
    pub fn bytes_per_element(self) -> usize {
        match self {
            Precision::FP16 => 2,
            Precision::FP32 => 4,
            Precision::FP64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathComplexity {
    Newtonian,
    Schwarzschild,
    Kerr,
    KerrNewman,
}

#[derive(Debug, Clone)]
pub struct ScalingParameters {
    pub grid_dim: (u32, u32, u32),
    pub precision: Precision,
    pub math_complexity: MathComplexity,
    pub render_resolution_scale: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiering_logic() {
        let vram_1gb = 1024;
        let tier = if vram_1gb < 2048 {
            GpuTier::Constrained
        } else {
            GpuTier::Standard
        };
        assert_eq!(tier, GpuTier::Constrained);

        let vram_12gb = 12 * 1024;
        let tier = if vram_12gb < 8192 {
            GpuTier::High
        } else {
            GpuTier::Ultra
        };
        assert_eq!(tier, GpuTier::Ultra);
    }

    #[test]
    fn gpu_tier_ordering() {
        assert!(GpuTier::Constrained < GpuTier::Standard);
        assert!(GpuTier::Standard < GpuTier::High);
        assert!(GpuTier::High < GpuTier::Ultra);
    }

    #[test]
    fn tier_boundary_values() {
        // Verify all four tier boundaries
        let cases: &[(u32, GpuTier)] = &[
            (1024, GpuTier::Constrained),
            (2047, GpuTier::Constrained),
            (2048, GpuTier::Standard),
            (4095, GpuTier::Standard),
            (4096, GpuTier::High),
            (8191, GpuTier::High),
            (8192, GpuTier::Ultra),
            (16384, GpuTier::Ultra),
        ];
        for &(vram_mb, expected) in cases {
            let tier = if vram_mb < 2048 {
                GpuTier::Constrained
            } else if vram_mb < 4096 {
                GpuTier::Standard
            } else if vram_mb < 8192 {
                GpuTier::High
            } else {
                GpuTier::Ultra
            };
            assert_eq!(tier, expected, "vram_mb={vram_mb}");
        }
    }

    #[test]
    fn scaling_parameters_precision_selection() {
        // Constrained + FP16 support -> FP16
        let caps = HardwareCapabilities {
            tier: GpuTier::Constrained,
            vram_mb: 1024,
            vendor_id: 0,
            device_id: 0,
            device_name: "test".to_string(),
            supports_fp16: true,
            supports_fp64: false,
            max_compute_shared_memory_size: 32768,
        };
        let ctx_stub = VulkanContextStub { caps: caps.clone() };
        let params = ctx_stub.get_scaling_parameters();
        assert_eq!(params.precision, Precision::FP16);
        assert_eq!(params.math_complexity, MathComplexity::Newtonian);
        assert_eq!(params.render_resolution_scale, 0.5);

        // Standard without FP16 -> FP32
        let caps_std = HardwareCapabilities {
            tier: GpuTier::Standard,
            supports_fp16: false,
            vram_mb: 3072,
            ..caps.clone()
        };
        let ctx_std = VulkanContextStub { caps: caps_std };
        let params_std = ctx_std.get_scaling_parameters();
        assert_eq!(params_std.precision, Precision::FP32);
        assert_eq!(params_std.math_complexity, MathComplexity::Schwarzschild);
        assert_eq!(params_std.render_resolution_scale, 0.75);

        // Standard WITH FP16 -> FP16 (non-Ultra tiers get FP16 when supported)
        let caps_std_fp16 = HardwareCapabilities {
            tier: GpuTier::Standard,
            supports_fp16: true,
            vram_mb: 3072,
            ..caps.clone()
        };
        let ctx_std_fp16 = VulkanContextStub { caps: caps_std_fp16 };
        assert_eq!(ctx_std_fp16.get_scaling_parameters().precision, Precision::FP16);

        // Ultra -> full complexity, stays FP32 even with FP16 support
        let caps_ultra = HardwareCapabilities {
            tier: GpuTier::Ultra,
            vram_mb: 16384,
            ..caps
        };
        let ctx_ultra = VulkanContextStub { caps: caps_ultra };
        let params_ultra = ctx_ultra.get_scaling_parameters();
        assert_eq!(params_ultra.math_complexity, MathComplexity::KerrNewman);
        assert_eq!(params_ultra.render_resolution_scale, 1.0);
        // Ultra with FP16 support still stays FP32 (plenty of VRAM)
        assert_eq!(params_ultra.precision, Precision::FP32);
    }

    #[test]
    fn precision_bytes_per_element() {
        assert_eq!(Precision::FP16.bytes_per_element(), 2);
        assert_eq!(Precision::FP32.bytes_per_element(), 4);
        assert_eq!(Precision::FP64.bytes_per_element(), 8);
    }

    #[test]
    fn optimal_grid_dim_minimum_is_32() {
        // Even with tiny VRAM, grid dim must be at least 32
        let caps = HardwareCapabilities {
            tier: GpuTier::Constrained,
            vram_mb: 1, // 1 MB -- very small
            vendor_id: 0,
            device_id: 0,
            device_name: "tiny".to_string(),
            supports_fp16: false,
            supports_fp64: false,
            max_compute_shared_memory_size: 0,
        };
        let ctx = VulkanContextStub { caps };
        let n = ctx.optimal_grid_dim();
        assert!(n >= 32, "minimum grid dim must be 32, got {n}");
        assert_eq!(n % 8, 0, "grid dim must be 8-aligned, got {n}");
    }

    #[test]
    fn optimal_grid_dim_scales_with_vram() {
        let make_caps = |vram_mb: u32, tier: GpuTier| HardwareCapabilities {
            tier,
            vram_mb,
            vendor_id: 0,
            device_id: 0,
            device_name: "test".to_string(),
            supports_fp16: false,
            supports_fp64: false,
            max_compute_shared_memory_size: 0,
        };
        let small = VulkanContextStub {
            caps: make_caps(2048, GpuTier::Standard),
        };
        let large = VulkanContextStub {
            caps: make_caps(16384, GpuTier::Ultra),
        };
        let n_small = small.optimal_grid_dim();
        let n_large = large.optimal_grid_dim();
        assert!(
            n_large >= n_small,
            "more VRAM should allow larger grid: {n_large} < {n_small}"
        );
    }

    /// Stub that mirrors VulkanContext's pure computation methods
    /// without requiring an actual Vulkan device.
    struct VulkanContextStub {
        caps: HardwareCapabilities,
    }

    impl VulkanContextStub {
        fn get_scaling_parameters(&self) -> ScalingParameters {
            let n = self.optimal_grid_dim();
            ScalingParameters {
                grid_dim: (n, n, n),
                precision: if self.caps.supports_fp16 && self.caps.tier != GpuTier::Ultra {
                    Precision::FP16
                } else {
                    Precision::FP32
                },
                math_complexity: match self.caps.tier {
                    GpuTier::Constrained => MathComplexity::Newtonian,
                    GpuTier::Standard => MathComplexity::Schwarzschild,
                    GpuTier::High => MathComplexity::Kerr,
                    GpuTier::Ultra => MathComplexity::KerrNewman,
                },
                render_resolution_scale: match self.caps.tier {
                    GpuTier::Constrained => 0.5,
                    GpuTier::Standard => 0.75,
                    _ => 1.0,
                },
            }
        }

        fn optimal_grid_dim(&self) -> u32 {
            let vram_bytes = self.caps.vram_mb as u64 * 1024 * 1024;
            let target_bytes = (vram_bytes as f64 * 0.7) as u64;
            let bytes_per_cell = match self.caps.tier {
                GpuTier::Constrained => 128,
                _ => 256,
            };
            let max_cells = target_bytes / bytes_per_cell;
            let n = (max_cells as f64).cbrt() as u32;
            let n = (n / 8) * 8;
            std::cmp::max(n, 32)
        }
    }

    #[test]
    fn compile_wgsl_f16_lbm_shader() {
        let src = include_str!("../shaders/lbm_f16.wgsl");
        crate::compute::compile_wgsl(src).expect("f16 LBM shader must compile via naga");
    }

    #[test]
    fn compile_wgsl_f16_init_shader() {
        let src = include_str!("../shaders/init_f16.wgsl");
        crate::compute::compile_wgsl(src).expect("f16 init shader must compile via naga");
    }

    #[test]
    fn compile_wgsl_f32_lbm_shader() {
        let src = include_str!("../shaders/lbm.wgsl");
        crate::compute::compile_wgsl(src).expect("f32 LBM shader must compile via naga");
    }

    #[test]
    fn compile_wgsl_f32_init_shader() {
        let src = include_str!("../shaders/init.wgsl");
        crate::compute::compile_wgsl(src).expect("f32 init shader must compile via naga");
    }

    #[test]
    fn compile_wgsl_pathion_sink_shader() {
        let src = include_str!("../shaders/pathion_sink.wgsl");
        crate::compute::compile_wgsl(src).expect("pathion sink v1 shader must compile via naga");
    }

    #[test]
    fn compile_wgsl_pathion_sink_v2_shader() {
        let src = include_str!("../shaders/pathion_sink_v2.wgsl");
        crate::compute::compile_wgsl(src).expect("pathion sink v2 shader must compile via naga");
    }

    #[test]
    fn compile_wgsl_non_newtonian_shader() {
        let src = include_str!("../shaders/lbm_non_newtonian.wgsl");
        crate::compute::compile_wgsl(src).expect("non-Newtonian LBM shader must compile via naga");
    }
}
