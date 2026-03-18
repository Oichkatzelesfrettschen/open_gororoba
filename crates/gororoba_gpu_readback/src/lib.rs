//! Lightweight host-staging and GPU readback contracts.
//!
//! This crate is intentionally descriptor-first. It standardizes how crates
//! describe host-readable buffers, subrange copies, and transfer reports
//! without taking ownership of CUDA streams, Vulkan command buffers, or any
//! solver-specific frame semantics.

/// Logical shape of a host-readable buffer or image-like readback surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadbackBufferShape {
    /// Logical width in elements or pixels.
    pub width: u32,
    /// Logical height in elements or pixels.
    pub height: u32,
    /// Logical depth for 3D volumes; `1` for 2D readback.
    pub depth: u32,
    /// Number of scalar elements stored at each logical point.
    pub elements_per_point: u32,
}

impl ReadbackBufferShape {
    /// Number of logical points in the shape.
    #[must_use]
    pub fn point_count(self) -> u64 {
        self.width as u64 * self.height as u64 * self.depth as u64
    }

    /// Total number of scalar elements in the shape.
    #[must_use]
    pub fn element_count(self) -> u64 {
        self.point_count() * self.elements_per_point as u64
    }
}

/// Scalar element type used by a readback surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackElementType {
    U8,
    U16,
    U32,
    F16,
    BF16,
    F32,
    F64,
}

impl ReadbackElementType {
    /// Encoded byte width of one scalar element.
    #[must_use]
    pub const fn bytes_per_element(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Logical packing/layout of the host-readable payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackLayout {
    Packed,
    Soa,
    Aos,
    ImageRgba8,
}

/// Residency model used by the readable staging surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackResidency {
    PinnedHost,
    MappedHost,
    UnifiedManaged,
    RendererOwned,
    DirectHost,
}

/// Fully described host-readable surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadbackDescriptor {
    /// Human-readable backend name, such as `CUDA` or `Vulkan`.
    pub backend_name: String,
    /// Human-readable label for the surface or buffer.
    pub label: String,
    /// Logical shape of the readable data.
    pub shape: ReadbackBufferShape,
    /// Scalar element type carried by the readable data.
    pub element_type: ReadbackElementType,
    /// Layout used by the readable data.
    pub layout: ReadbackLayout,
    /// Residency model for the readable surface.
    pub residency: ReadbackResidency,
}

impl ReadbackDescriptor {
    /// Total encoded byte length for the full readable surface.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.shape.element_count() * self.element_type.bytes_per_element() as u64
    }
}

/// Byte-range or logical subregion copy request over a readable surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadbackRange {
    /// Byte offset into the source surface.
    pub byte_offset: u64,
    /// Byte length of the requested range.
    pub byte_len: u64,
    /// Optional logical origin if the range maps to a subvolume/subimage.
    pub logical_origin: Option<[u32; 3]>,
    /// Optional logical extent if the range maps to a subvolume/subimage.
    pub logical_extent: Option<[u32; 3]>,
}

impl ReadbackRange {
    /// Full-range request for the entire descriptor.
    #[must_use]
    pub fn full(descriptor: &ReadbackDescriptor) -> Self {
        Self {
            byte_offset: 0,
            byte_len: descriptor.byte_len(),
            logical_origin: Some([0, 0, 0]),
            logical_extent: Some([
                descriptor.shape.width,
                descriptor.shape.height,
                descriptor.shape.depth,
            ]),
        }
    }
}

/// Measured readback-transfer report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadbackReport {
    /// Number of bytes transferred.
    pub transferred_bytes: u64,
    /// Wall-clock duration in seconds.
    pub elapsed_seconds: f64,
}

impl ReadbackReport {
    /// Effective transfer rate in GiB/s if elapsed time is positive.
    #[must_use]
    pub fn gib_per_second(self) -> Option<f64> {
        if self.elapsed_seconds > 0.0 {
            Some(self.transferred_bytes as f64 / self.elapsed_seconds / 1024.0_f64.powi(3))
        } else {
            None
        }
    }
}

/// Trait for types that can describe a host-readable staging surface.
pub trait HostReadableBuffer {
    /// Descriptor for the full readable surface.
    fn descriptor(&self) -> ReadbackDescriptor;

    /// Full readable range for the surface.
    fn full_range(&self) -> ReadbackRange {
        ReadbackRange::full(&self.descriptor())
    }
}

/// Trait for simple descriptor-first readback planners.
pub trait ReadbackPlanner {
    /// Plan a full readback.
    fn plan_full_readback(&self, descriptor: &ReadbackDescriptor) -> ReadbackRange {
        ReadbackRange::full(descriptor)
    }

    /// Plan a bounded tiled readback using a byte budget.
    fn plan_tiled_readback(
        &self,
        descriptor: &ReadbackDescriptor,
        max_bytes_per_range: u64,
    ) -> Vec<ReadbackRange>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_byte_len_matches_shape_and_element_width() {
        let descriptor = ReadbackDescriptor {
            backend_name: "CUDA".to_string(),
            label: "rho".to_string(),
            shape: ReadbackBufferShape {
                width: 16,
                height: 8,
                depth: 4,
                elements_per_point: 1,
            },
            element_type: ReadbackElementType::F32,
            layout: ReadbackLayout::Packed,
            residency: ReadbackResidency::PinnedHost,
        };
        assert_eq!(descriptor.byte_len(), 16 * 8 * 4 * 4);
    }

    #[test]
    fn full_range_covers_entire_descriptor() {
        let descriptor = ReadbackDescriptor {
            backend_name: "Vulkan".to_string(),
            label: "rgba".to_string(),
            shape: ReadbackBufferShape {
                width: 32,
                height: 32,
                depth: 1,
                elements_per_point: 4,
            },
            element_type: ReadbackElementType::U8,
            layout: ReadbackLayout::ImageRgba8,
            residency: ReadbackResidency::MappedHost,
        };
        let range = ReadbackRange::full(&descriptor);
        assert_eq!(range.byte_len, 32 * 32 * 4);
        assert_eq!(range.logical_extent, Some([32, 32, 1]));
    }

    #[test]
    fn readback_report_derives_transfer_rate() {
        let report = ReadbackReport {
            transferred_bytes: 1024 * 1024 * 1024,
            elapsed_seconds: 0.5,
        };
        assert_eq!(report.gib_per_second(), Some(2.0));
    }
}
