//! Host-visible Vulkan buffer wrapper.
//!
//! This covers the small compute-kernel buffers used by parity and roadmap
//! launcher code. It intentionally chooses HOST_VISIBLE | HOST_COHERENT memory
//! so callers can upload/read back without staging buffers.

use std::sync::Arc;

use ash::vk;

use crate::{
    adapter::Adapter,
    device::Device,
    error::{Result, VulkanError},
};

/// Owned VkBuffer + VkDeviceMemory pair with host-coherent mapping helpers.
pub struct HostVisibleBuffer {
    device: Arc<Device>,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl HostVisibleBuffer {
    pub fn storage(device: &Device, adapter: &Adapter, size: vk::DeviceSize) -> Result<Self> {
        Self::new(device, adapter, size, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    pub fn uniform(device: &Device, adapter: &Adapter, size: vk::DeviceSize) -> Result<Self> {
        Self::new(device, adapter, size, vk::BufferUsageFlags::UNIFORM_BUFFER)
    }

    pub fn new(
        device: &Device,
        adapter: &Adapter,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: device outlives the returned buffer via Arc<Device>.
        let raw = unsafe { device.raw().create_buffer(&buffer_ci, None) }?;
        let requirements = unsafe { device.raw().get_buffer_memory_requirements(raw) };
        let mem_type_index =
            host_visible_coherent_memory_type(adapter, requirements.memory_type_bits)?;
        let memory_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(mem_type_index);
        // SAFETY: allocation size/type come from Vulkan requirements +
        // adapter memory properties.
        let memory = match unsafe { device.raw().allocate_memory(&memory_ci, None) } {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: raw was created on this device and has not been
                // bound or transferred to an owner yet.
                unsafe {
                    device.raw().destroy_buffer(raw, None);
                }
                return Err(err.into());
            }
        };
        // SAFETY: buffer + memory were created on the same logical device.
        if let Err(err) = unsafe { device.raw().bind_buffer_memory(raw, memory, 0) } {
            // SAFETY: both handles were created on this device and have not
            // been transferred to the returned owner.
            unsafe {
                device.raw().destroy_buffer(raw, None);
                device.raw().free_memory(memory, None);
            }
            return Err(err.into());
        }
        Ok(Self {
            device: Arc::new(device.clone()),
            raw,
            memory,
            size,
        })
    }

    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn write_u32_slice(&self, values: &[u32]) -> Result<()> {
        let byte_len = std::mem::size_of_val(values) as vk::DeviceSize;
        if byte_len > self.size {
            return Err(VulkanError::BufferBounds {
                requested: byte_len,
                available: self.size,
            });
        }
        if values.is_empty() {
            return Ok(());
        }
        // SAFETY: memory was allocated HOST_VISIBLE | HOST_COHERENT. The
        // bounds check above proves the copy fits in the allocation.
        unsafe {
            let ptr = self.device.raw().map_memory(
                self.memory,
                0,
                byte_len,
                vk::MemoryMapFlags::empty(),
            )? as *mut u32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            self.device.raw().unmap_memory(self.memory);
        }
        Ok(())
    }

    pub fn write_f32_slice(&self, values: &[f32]) -> Result<()> {
        let byte_len = std::mem::size_of_val(values) as vk::DeviceSize;
        if byte_len > self.size {
            return Err(VulkanError::BufferBounds {
                requested: byte_len,
                available: self.size,
            });
        }
        if values.is_empty() {
            return Ok(());
        }
        // SAFETY: memory was allocated HOST_VISIBLE | HOST_COHERENT. The
        // bounds check above proves the copy fits in the allocation.
        unsafe {
            let ptr = self.device.raw().map_memory(
                self.memory,
                0,
                byte_len,
                vk::MemoryMapFlags::empty(),
            )? as *mut f32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            self.device.raw().unmap_memory(self.memory);
        }
        Ok(())
    }

    pub fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        let byte_len = bytes.len() as vk::DeviceSize;
        if byte_len > self.size {
            return Err(VulkanError::BufferBounds {
                requested: byte_len,
                available: self.size,
            });
        }
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: memory was allocated HOST_VISIBLE | HOST_COHERENT. The
        // bounds check above proves the copy fits in the allocation.
        unsafe {
            let ptr = self.device.raw().map_memory(
                self.memory,
                0,
                byte_len,
                vk::MemoryMapFlags::empty(),
            )? as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            self.device.raw().unmap_memory(self.memory);
        }
        Ok(())
    }

    pub fn read_u32_slice(&self, len: usize) -> Result<Vec<u32>> {
        let byte_len = (len * std::mem::size_of::<u32>()) as vk::DeviceSize;
        if byte_len > self.size {
            return Err(VulkanError::BufferBounds {
                requested: byte_len,
                available: self.size,
            });
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut values = vec![0u32; len];
        // SAFETY: memory is HOST_VISIBLE | HOST_COHERENT and the bounds check
        // proves the copy fits the mapped allocation.
        unsafe {
            let ptr = self.device.raw().map_memory(
                self.memory,
                0,
                byte_len,
                vk::MemoryMapFlags::empty(),
            )? as *const u32;
            std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), len);
            self.device.raw().unmap_memory(self.memory);
        }
        Ok(values)
    }

    pub fn read_f32_slice(&self, len: usize) -> Result<Vec<f32>> {
        let byte_len = (len * std::mem::size_of::<f32>()) as vk::DeviceSize;
        if byte_len > self.size {
            return Err(VulkanError::BufferBounds {
                requested: byte_len,
                available: self.size,
            });
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut values = vec![0.0f32; len];
        // SAFETY: memory is HOST_VISIBLE | HOST_COHERENT and the bounds check
        // proves the copy fits the mapped allocation.
        unsafe {
            let ptr = self.device.raw().map_memory(
                self.memory,
                0,
                byte_len,
                vk::MemoryMapFlags::empty(),
            )? as *const f32;
            std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), len);
            self.device.raw().unmap_memory(self.memory);
        }
        Ok(values)
    }
}

impl Drop for HostVisibleBuffer {
    fn drop(&mut self) {
        // SAFETY: both handles were created by Self::new above; Arc<Device>
        // keeps the device alive past this Drop.
        unsafe {
            self.device.raw().destroy_buffer(self.raw, None);
            self.device.raw().free_memory(self.memory, None);
        }
    }
}

fn host_visible_coherent_memory_type(adapter: &Adapter, memory_type_bits: u32) -> Result<u32> {
    let mem_props = adapter.memory_properties;
    for i in 0..mem_props.memory_type_count {
        let supported = (memory_type_bits & (1 << i)) != 0;
        let flags = mem_props.memory_types[i as usize].property_flags;
        if supported
            && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(i);
        }
    }
    Err(VulkanError::UnsupportedFeature(
        "HOST_VISIBLE | HOST_COHERENT memory type",
    ))
}
