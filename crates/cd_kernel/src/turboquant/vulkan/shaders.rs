//! Embedded Vulkan compute shader sources for TurboQuant.
//!
//! GLSL sources are compiled to SPIR-V at build time (or at runtime via
//! shaderc/glslang).  This module provides the raw GLSL source strings
//! for compilation.

/// Boundary-search quantization shader (quantize.comp).
pub const QUANTIZE_GLSL: &str = include_str!("shaders/quantize.comp");

/// Dequant + dot product shader (dequant_dot.comp).
pub const DEQUANT_DOT_GLSL: &str = include_str!("shaders/dequant_dot.comp");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_sources_embedded() {
        assert!(!QUANTIZE_GLSL.is_empty());
        assert!(QUANTIZE_GLSL.contains("TurboQuant"));
        assert!(QUANTIZE_GLSL.contains("local_size_x"));

        assert!(!DEQUANT_DOT_GLSL.is_empty());
        assert!(DEQUANT_DOT_GLSL.contains("dequant"));
        assert!(DEQUANT_DOT_GLSL.contains("local_size_x"));
    }

    #[test]
    fn test_shader_push_constants() {
        // Verify push constant blocks exist
        assert!(QUANTIZE_GLSL.contains("push_constant"));
        assert!(DEQUANT_DOT_GLSL.contains("push_constant"));
    }

    #[test]
    fn test_shader_buffer_bindings() {
        // Quantize: 3 bindings (values, boundaries, indices)
        assert!(QUANTIZE_GLSL.contains("binding = 0"));
        assert!(QUANTIZE_GLSL.contains("binding = 1"));
        assert!(QUANTIZE_GLSL.contains("binding = 2"));

        // Dequant+dot: 5 bindings (queries, indices, centroids, norms, scores)
        assert!(DEQUANT_DOT_GLSL.contains("binding = 0"));
        assert!(DEQUANT_DOT_GLSL.contains("binding = 4"));
    }
}
