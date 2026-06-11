//! Module + kernel-function registry.
//!
//! Consolidates 28+ `ctx.load_module(ptx)` + `module.load_function(name)`
//! sites that each track function handles manually. The registry caches
//! the loaded module and exposes typed lookups.

use std::{collections::BTreeMap, sync::Arc};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule},
    nvrtc::Ptx,
};

use crate::{
    error::{CudaError, Result},
    nvrtc::CompileOptions,
};

/// Owned kernel function handle.
pub type KernelHandle = CudaFunction;

/// Module + named-function registry.
pub struct ModuleRegistry {
    module: Arc<CudaModule>,
    functions: BTreeMap<String, KernelHandle>,
}

impl ModuleRegistry {
    /// Compile CUDA C source with the workspace NVRTC wrapper and load all
    /// requested kernels into the given context.
    pub fn compile_and_load(
        ctx: &Arc<CudaContext>,
        source: &str,
        opts: &CompileOptions,
        kernel_names: &[&str],
    ) -> Result<Self> {
        let ptx = CompileOptions::compile_ptx(source, opts)?;
        Self::load(ctx, ptx, kernel_names)
    }

    /// Load a PTX module into the given context and pre-resolve a set
    /// of named kernel functions.
    pub fn load(ctx: &Arc<CudaContext>, ptx: Ptx, kernel_names: &[&str]) -> Result<Self> {
        let module = ctx.load_module(ptx)?;
        let mut functions = BTreeMap::new();
        for name in kernel_names {
            let func = module
                .load_function(name)
                .map_err(|_| CudaError::KernelNotFound {
                    name: (*name).to_string(),
                })?;
            functions.insert((*name).to_string(), func);
        }
        Ok(Self { module, functions })
    }

    /// Look up a pre-resolved kernel function by name.
    pub fn get(&self, name: &str) -> Result<KernelHandle> {
        self.functions
            .get(name)
            .cloned()
            .ok_or_else(|| CudaError::KernelNotFound {
                name: name.to_string(),
            })
    }

    /// Borrow the underlying `Arc<CudaModule>` (for cudarc APIs that
    /// require the raw handle).
    pub fn module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    /// Iterate over registered kernel names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.functions.keys().map(String::as_str)
    }
}
