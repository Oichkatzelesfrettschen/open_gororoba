//! Vulkan precision-tier shader dispatch layer.
//!
//! Manages 31 WGSL compute shaders across 10 precision tiers, 2 collision
//! modes, and multiple streaming layouts. Uses lazy compilation with
//! HashMap caching to avoid startup penalty while enabling zero-latency
//! hot-swapping after first use.
//!
//! # Usage
//!
//! ```ignore
//! let registry = ShaderRegistry::new();
//! let spv = registry.get_or_compile(ShaderKey {
//!     collision: CollisionType::Mrt,
//!     precision: PrecisionTier::Fp32,
//!     streaming: StreamingLayout::AlternateAccess,
//! })?;
//! ```

use crate::compute::{VulkanEngineError, compile_wgsl};
use std::collections::HashMap;

/// Collision operator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollisionType {
    /// Single-relaxation-time Bhatnagar-Gross-Krook.
    Bgk,
    /// Multiple-relaxation-time (d'Humieres D3Q19).
    Mrt,
}

/// Precision tier for distribution storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionTier {
    Fp32,
    Fp64,
    Fp16,
    Bf16,
    Int8,
    Int16,
    Int4,
    Fp8E4m3,
    Fp8E5m2,
}

/// Streaming and memory access layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamingLayout {
    /// Standard push-scheme streaming.
    Push,
    /// Pull-scheme (tiled shared-memory).
    Tiled,
    /// Alternate-Access single-buffer streaming.
    AlternateAccess,
    /// Thread-coarsened (2 cells per thread).
    Coarsened,
    /// Non-Newtonian power-law viscosity.
    NonNewtonian,
}

/// Key identifying a specific shader configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderKey {
    pub collision: CollisionType,
    pub precision: PrecisionTier,
    pub streaming: StreamingLayout,
}

impl ShaderKey {
    /// Return the WGSL shader source for this configuration, if it exists.
    pub fn shader_source(&self) -> Option<&'static str> {
        use CollisionType::*;
        use PrecisionTier::*;
        use StreamingLayout::*;

        match (self.collision, self.precision, self.streaming) {
            // FP32 variants (most complete)
            (Bgk, Fp32, Push) => Some(include_str!("../shaders/lbm.wgsl")),
            (Mrt, Fp32, Push) => Some(include_str!("../shaders/lbm_mrt.wgsl")),
            (Bgk, Fp32, Tiled) => Some(include_str!("../shaders/lbm_tiled.wgsl")),
            (Mrt, Fp32, Tiled) => Some(include_str!("../shaders/lbm_mrt_tiled.wgsl")),
            (Bgk, Fp32, AlternateAccess) => Some(include_str!("../shaders/lbm_aa.wgsl")),
            (Mrt, Fp32, AlternateAccess) => Some(include_str!("../shaders/lbm_mrt_aa.wgsl")),
            (Bgk, Fp32, Coarsened) => Some(include_str!("../shaders/lbm_coarsened.wgsl")),
            (Bgk, Fp32, NonNewtonian) => Some(include_str!("../shaders/lbm_non_newtonian.wgsl")),

            // FP64 variants
            (Bgk, Fp64, Push) => Some(include_str!("../shaders/lbm_fp64.wgsl")),
            (Mrt, Fp64, Push) => Some(include_str!("../shaders/lbm_mrt_fp64.wgsl")),

            // FP16 variants
            (Bgk, Fp16, Push) => Some(include_str!("../shaders/lbm_f16.wgsl")),
            (Mrt, Fp16, Push) => Some(include_str!("../shaders/lbm_mrt_f16.wgsl")),

            // BF16 variants
            (Bgk, Bf16, Push) => Some(include_str!("../shaders/lbm_bf16.wgsl")),
            (Mrt, Bf16, Push) => Some(include_str!("../shaders/lbm_mrt_bf16.wgsl")),

            // INT8
            (Bgk, Int8, Push) => Some(include_str!("../shaders/lbm_int8.wgsl")),

            // INT16
            (Bgk, Int16, Push) => Some(include_str!("../shaders/lbm_int16.wgsl")),

            // INT4 (bandwidth ceiling only)
            (Bgk, Int4, Push) => Some(include_str!("../shaders/lbm_int4.wgsl")),

            // FP8 e4m3
            (Bgk, Fp8E4m3, Push) => Some(include_str!("../shaders/lbm_fp8.wgsl")),

            // FP8 e5m2
            (Bgk, Fp8E5m2, Push) => Some(include_str!("../shaders/lbm_fp8_e5m2.wgsl")),

            _ => None, // unsupported combination
        }
    }

    /// Human-readable name for this shader configuration.
    pub fn display_name(&self) -> String {
        format!(
            "{:?}_{:?}_{:?}",
            self.collision, self.precision, self.streaming
        )
    }
}

/// Cached SPIR-V compilation result.
pub struct CompiledShader {
    /// SPIR-V words (u32 array ready for vkCreateShaderModule).
    pub spirv_words: Vec<u32>,
    /// Number of SPIR-V words (for size tracking).
    pub word_count: usize,
}

/// Registry of compiled WGSL shaders with lazy HashMap caching.
///
/// On first request for a ShaderKey, compiles the WGSL source via Naga
/// to SPIR-V and caches the result. Subsequent requests return instantly.
pub struct ShaderRegistry {
    cache: HashMap<ShaderKey, CompiledShader>,
}

impl ShaderRegistry {
    /// Create an empty registry. No shaders compiled until requested.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or lazily compile the SPIR-V for the given shader configuration.
    ///
    /// Returns None if the (collision, precision, streaming) combination
    /// has no WGSL shader source defined.
    pub fn get_or_compile(&mut self, key: ShaderKey) -> Result<&CompiledShader, VulkanEngineError> {
        use std::collections::hash_map::Entry;
        match self.cache.entry(key) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let src = key.shader_source().ok_or_else(|| {
                    VulkanEngineError::ShaderError(format!(
                        "No WGSL shader for configuration: {}",
                        key.display_name()
                    ))
                })?;
                let words = compile_wgsl(src)?;
                let word_count = words.len();
                Ok(e.insert(CompiledShader {
                    spirv_words: words,
                    word_count,
                }))
            }
        }
    }

    /// Number of shaders currently compiled and cached.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Total SPIR-V memory usage across all cached shaders (bytes).
    pub fn total_spirv_bytes(&self) -> usize {
        self.cache.values().map(|s| s.word_count * 4).sum()
    }

    /// List all supported shader configurations.
    pub fn supported_keys() -> Vec<ShaderKey> {
        let mut keys = Vec::new();
        let collisions = [CollisionType::Bgk, CollisionType::Mrt];
        let precisions = [
            PrecisionTier::Fp32,
            PrecisionTier::Fp64,
            PrecisionTier::Fp16,
            PrecisionTier::Bf16,
            PrecisionTier::Int8,
            PrecisionTier::Int16,
            PrecisionTier::Int4,
            PrecisionTier::Fp8E4m3,
            PrecisionTier::Fp8E5m2,
        ];
        let streamings = [
            StreamingLayout::Push,
            StreamingLayout::Tiled,
            StreamingLayout::AlternateAccess,
            StreamingLayout::Coarsened,
            StreamingLayout::NonNewtonian,
        ];
        for &c in &collisions {
            for &p in &precisions {
                for &s in &streamings {
                    let key = ShaderKey {
                        collision: c,
                        precision: p,
                        streaming: s,
                    };
                    if key.shader_source().is_some() {
                        keys.push(key);
                    }
                }
            }
        }
        keys
    }

    /// Pre-compile all supported shader configurations.
    /// Returns the number of shaders compiled.
    pub fn precompile_all(&mut self) -> Result<usize, VulkanEngineError> {
        let keys = Self::supported_keys();
        for key in &keys {
            self.get_or_compile(*key)?;
        }
        Ok(keys.len())
    }
}

impl Default for ShaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_keys_count() {
        let keys = ShaderRegistry::supported_keys();
        // Should have at least 19 supported configurations
        // (the 31 WGSL shaders minus utility shaders like init/render/smagorinsky)
        assert!(
            keys.len() >= 19,
            "expected >= 19 supported shader configs, got {}",
            keys.len()
        );
    }

    #[test]
    fn test_lazy_compilation() {
        let mut registry = ShaderRegistry::new();
        assert_eq!(registry.cached_count(), 0);

        let key = ShaderKey {
            collision: CollisionType::Bgk,
            precision: PrecisionTier::Fp32,
            streaming: StreamingLayout::Push,
        };
        registry.get_or_compile(key).unwrap();
        assert_eq!(registry.cached_count(), 1);

        // Second call should use cache
        registry.get_or_compile(key).unwrap();
        assert_eq!(registry.cached_count(), 1);
    }

    #[test]
    fn test_precompile_all() {
        let mut registry = ShaderRegistry::new();
        let count = registry.precompile_all().unwrap();
        assert!(count >= 19, "precompiled {count} shaders, expected >= 19");
        assert_eq!(registry.cached_count(), count);
        // Verify non-trivial SPIR-V output
        assert!(
            registry.total_spirv_bytes() > 50_000,
            "total SPIR-V too small: {} bytes",
            registry.total_spirv_bytes()
        );
    }

    #[test]
    fn test_unsupported_configuration() {
        let mut registry = ShaderRegistry::new();
        let key = ShaderKey {
            collision: CollisionType::Mrt,
            precision: PrecisionTier::Int4,
            streaming: StreamingLayout::Tiled,
        };
        assert!(registry.get_or_compile(key).is_err());
    }

    #[test]
    fn test_mrt_aa_compiles() {
        let mut registry = ShaderRegistry::new();
        let key = ShaderKey {
            collision: CollisionType::Mrt,
            precision: PrecisionTier::Fp32,
            streaming: StreamingLayout::AlternateAccess,
        };
        let compiled = registry.get_or_compile(key).unwrap();
        assert!(
            compiled.word_count > 3000,
            "MRT A-A should be > 3000 SPIR-V words, got {}",
            compiled.word_count
        );
    }
}
