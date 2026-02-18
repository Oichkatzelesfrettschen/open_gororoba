use ash::{vk, Entry, Instance, Device};
use std::ffi::CStr;
use std::sync::Arc;
use gpu_allocator::vulkan::*;

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
                        if info.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS) {
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
            device_name: unsafe { CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned() },
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
            precision: if self.caps.supports_fp16 && self.caps.tier == GpuTier::Constrained {
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
        let tier = if vram_1gb < 2048 { GpuTier::Constrained } else { GpuTier::Standard };
        assert_eq!(tier, GpuTier::Constrained);

        let vram_12gb = 12 * 1024;
        let tier = if vram_12gb < 8192 { GpuTier::High } else { GpuTier::Ultra };
        assert_eq!(tier, GpuTier::Ultra);
    }
}
