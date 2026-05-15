//! Vulkan `Instance` builder.
//!
//! Consolidates the Entry::load + ApplicationInfo + InstanceCreateInfo
//! pattern from:
//!   - lbm_vulkan/src/lib.rs:111-137 (with VK_LAYER_KHRONOS_validation toggle)
//!   - cd_kernel/turboquant/vulkan/quantizer.rs:95-107 (no validation layer)
//!   - 8 binaries under gororoba_cli_physics/src/bin/ and gororoba_cli/src/bin/
//!
//! All sites previously hand-rolled the same ~25 lines with minor differences
//! in API version (1.2 vs 1.3) and validation toggle. This builder centralises
//! that with explicit options.

use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

use ash::{Entry, vk};

use crate::error::Result;

/// Validation-layer policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Enable `VK_LAYER_KHRONOS_validation` when the loader exposes it.
    /// Falls back to no-validation if the layer is absent (e.g. drivers-only
    /// install without the SDK).
    Enable,
    /// Never request validation. Use this for performance-critical
    /// production builds.
    Disable,
}

impl ValidationPolicy {
    /// Default-disabled in release builds. Default-enabled in debug builds.
    pub fn default_for_profile() -> Self {
        if cfg!(debug_assertions) {
            Self::Enable
        } else {
            Self::Disable
        }
    }
}

/// Builder for a Vulkan `Instance`.
///
/// # Example
///
/// ```ignore
/// use gororoba_gpu_vulkan::InstanceBuilder;
///
/// let instance = InstanceBuilder::new("my_app")
///     .api_version(ash::vk::API_VERSION_1_3)
///     .validation(gororoba_gpu_vulkan::instance::ValidationPolicy::Disable)
///     .build()?;
/// # Ok::<(), gororoba_gpu_vulkan::VulkanError>(())
/// ```
pub struct InstanceBuilder {
    app_name: CString,
    api_version: u32,
    validation: ValidationPolicy,
    extensions: Vec<CString>,
}

impl InstanceBuilder {
    /// Start a new builder. `app_name` is forwarded to `vk::ApplicationInfo`
    /// and is visible in validation-layer logs.
    pub fn new(app_name: &str) -> Self {
        Self {
            app_name: CString::new(app_name)
                .unwrap_or_else(|_| CString::new("gororoba").expect("static literal")),
            api_version: vk::API_VERSION_1_3,
            validation: ValidationPolicy::default_for_profile(),
            extensions: Vec::new(),
        }
    }

    /// Override the requested Vulkan API version (default 1.3).
    pub fn api_version(mut self, version: u32) -> Self {
        self.api_version = version;
        self
    }

    /// Override the validation policy.
    pub fn validation(mut self, policy: ValidationPolicy) -> Self {
        self.validation = policy;
        self
    }

    /// Request an additional instance-level extension by name.
    pub fn extension(mut self, name: &CStr) -> Self {
        self.extensions.push(name.to_owned());
        self
    }

    /// Load the Vulkan loader, create the instance, and wrap it.
    pub fn build(self) -> Result<Instance> {
        // SAFETY: Entry::load() loads libvulkan.so/.dll via dlopen. The
        // returned Entry owns the library handle and is kept alive for the
        // lifetime of the Instance (via Arc).
        let entry = unsafe { Entry::load() }?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(self.app_name.as_c_str())
            .application_version(0)
            .engine_name(c"open_gororoba")
            .engine_version(0)
            .api_version(self.api_version);

        let layer_names: Vec<CString> = match self.validation {
            ValidationPolicy::Enable => vec![
                CString::new("VK_LAYER_KHRONOS_validation")
                    .expect("static literal is valid CString"),
            ],
            ValidationPolicy::Disable => Vec::new(),
        };
        let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|n| n.as_ptr().cast()).collect();
        let extension_ptrs: Vec<*const i8> =
            self.extensions.iter().map(|n| n.as_ptr().cast()).collect();

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs)
            .enabled_extension_names(&extension_ptrs);

        // SAFETY: app_info, layer_ptrs, extension_ptrs all outlive this call;
        // ash validates pointer non-null. Returned instance is owned and we
        // wrap it in Arc + Drop-destroyed below.
        let instance = unsafe { entry.create_instance(&instance_ci, None) }?;

        Ok(Instance {
            inner: Arc::new(InstanceInner {
                entry,
                instance,
                validation: self.validation,
            }),
        })
    }
}

/// Vulkan `Instance` handle with deterministic Drop-time destruction.
///
/// Clone-cheap via internal `Arc`; the last clone Drop calls
/// `vk::DestroyInstance`.
#[derive(Clone)]
pub struct Instance {
    inner: Arc<InstanceInner>,
}

struct InstanceInner {
    entry: Entry,
    instance: ash::Instance,
    validation: ValidationPolicy,
}

impl Instance {
    /// Access the underlying ash entry (for ash::extensions::* constructors).
    pub fn entry(&self) -> &Entry {
        &self.inner.entry
    }

    /// Access the underlying ash::Instance (for raw FFI calls).
    pub fn raw(&self) -> &ash::Instance {
        &self.inner.instance
    }

    /// Was validation requested at build time?
    pub fn validation(&self) -> ValidationPolicy {
        self.inner.validation
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        // SAFETY: self.instance was created by Entry::create_instance above
        // and has not been destroyed. No outstanding child handles can
        // exist because Device/Allocator/etc. each hold an Arc<InstanceInner>
        // which keeps this Drop deferred until they release.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
