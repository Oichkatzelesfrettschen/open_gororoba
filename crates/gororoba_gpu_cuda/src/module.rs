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
use sha2::{Digest, Sha256};

use crate::{
    error::{CudaError, Result},
    nvrtc::CompileOptions,
};

/// Owned kernel function handle.
pub type KernelHandle = CudaFunction;

/// Stable identity for a loaded CUDA module.
///
/// NVRTC modules carry the source and compile-option digests. Opaque PTX or
/// CUBIN loads carry a deterministic identity over the supplied label and
/// requested symbols, while `source_sha256` remains `None` until the caller
/// supplies an artifact digest through [`ModuleRegistry::load_with_identity`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleProvenance {
    /// Human-readable source or artifact label supplied by the caller.
    pub source_label: String,
    /// SHA-256 digest of the CUDA source when source text is available.
    pub source_sha256: Option<String>,
    /// SHA-256 digest of the canonical NVRTC option set, when applicable.
    pub compile_options_sha256: Option<String>,
    /// Stable digest tying source, options, label, and registered symbols.
    pub module_id: String,
    /// Symbols pre-resolved from the module, in lexical order.
    pub kernel_names: Vec<String>,
}

/// Provenance attached to one resolved kernel function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelProvenance {
    /// Stable module identity.
    pub module_id: String,
    /// Source or artifact label for the containing module.
    pub source_label: String,
    /// Source digest inherited from the containing module, when available.
    pub source_sha256: Option<String>,
    /// Compile-option digest inherited from the containing module, when available.
    pub compile_options_sha256: Option<String>,
    /// Resolved CUDA entry point.
    pub kernel_name: String,
}

/// Module + named-function registry.
pub struct ModuleRegistry {
    module: Arc<CudaModule>,
    functions: BTreeMap<String, KernelHandle>,
    provenance: ModuleProvenance,
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
        Self::compile_and_load_named(ctx, "<inline>", source, opts, kernel_names)
    }

    /// Compile CUDA C source with an explicit source label and load its kernels.
    ///
    /// The label is part of the module identity, so two identical source
    /// strings embedded by different owners remain distinguishable in runtime
    /// dispatch records.
    pub fn compile_and_load_named(
        ctx: &Arc<CudaContext>,
        source_label: &str,
        source: &str,
        opts: &CompileOptions,
        kernel_names: &[&str],
    ) -> Result<Self> {
        let ptx = CompileOptions::compile_ptx(source, opts)?;
        let source_sha256 = sha256_hex(source.as_bytes());
        let compile_options_sha256 = opts.fingerprint();
        let provenance = ModuleProvenance::new(
            source_label,
            Some(source_sha256),
            Some(compile_options_sha256),
            kernel_names,
        );
        Self::load_with_provenance(ctx, ptx, provenance)
    }

    /// Load a PTX module into the given context and pre-resolve a set
    /// of named kernel functions.
    pub fn load(ctx: &Arc<CudaContext>, ptx: Ptx, kernel_names: &[&str]) -> Result<Self> {
        Self::load_with_identity(ctx, ptx, "<ptx>", None, kernel_names)
    }

    /// Load a PTX or CUBIN module with an explicit artifact identity.
    ///
    /// AOT callers can pass the digest of the checked-in PTX/CUBIN. Runtime
    /// callers that only have an opaque cudarc [`Ptx`] can pass `None`; the
    /// registry still records a stable module identity over the label and
    /// symbol set, but does not claim a content hash it cannot observe.
    pub fn load_with_identity(
        ctx: &Arc<CudaContext>,
        ptx: Ptx,
        source_label: &str,
        source_sha256: Option<&str>,
        kernel_names: &[&str],
    ) -> Result<Self> {
        let provenance = ModuleProvenance::new(
            source_label,
            source_sha256.map(str::to_owned),
            None,
            kernel_names,
        );
        Self::load_with_provenance(ctx, ptx, provenance)
    }

    fn load_with_provenance(
        ctx: &Arc<CudaContext>,
        ptx: Ptx,
        provenance: ModuleProvenance,
    ) -> Result<Self> {
        let module = ctx.load_module(ptx)?;
        let mut functions = BTreeMap::new();
        for name in &provenance.kernel_names {
            let func = module
                .load_function(name)
                .map_err(|_| CudaError::KernelNotFound { name: name.clone() })?;
            functions.insert(name.clone(), func);
        }
        Ok(Self {
            module,
            functions,
            provenance,
        })
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

    /// Look up a kernel and return its handle together with dispatch provenance.
    pub fn get_with_provenance(&self, name: &str) -> Result<(KernelHandle, KernelProvenance)> {
        let handle = self.get(name)?;
        Ok((handle, self.provenance.kernel(name)))
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

    /// Borrow the source, option, and module identity record.
    pub fn provenance(&self) -> &ModuleProvenance {
        &self.provenance
    }
}

impl ModuleProvenance {
    fn new(
        source_label: &str,
        source_sha256: Option<String>,
        compile_options_sha256: Option<String>,
        kernel_names: &[&str],
    ) -> Self {
        let mut names = kernel_names
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();

        let mut identity = String::from("gororoba-cuda-module-v1\0");
        append_identity_field(&mut identity, source_label);
        append_identity_field(
            &mut identity,
            source_sha256.as_deref().unwrap_or("<unknown-source>"),
        );
        append_identity_field(
            &mut identity,
            compile_options_sha256
                .as_deref()
                .unwrap_or("<non-nvrtc-artifact>"),
        );
        for name in &names {
            append_identity_field(&mut identity, name);
        }

        Self {
            source_label: source_label.to_string(),
            source_sha256,
            compile_options_sha256,
            module_id: sha256_hex(identity.as_bytes()),
            kernel_names: names,
        }
    }

    fn kernel(&self, name: &str) -> KernelProvenance {
        KernelProvenance {
            module_id: self.module_id.clone(),
            source_label: self.source_label.clone(),
            source_sha256: self.source_sha256.clone(),
            compile_options_sha256: self.compile_options_sha256.clone(),
            kernel_name: name.to_string(),
        }
    }
}

fn append_identity_field(identity: &mut String, value: &str) {
    identity.push_str(&value.len().to_string());
    identity.push(':');
    identity.push_str(value);
    identity.push('|');
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::ModuleProvenance;

    #[test]
    fn module_identity_binds_source_options_and_symbols() {
        let first = ModuleProvenance::new(
            "KERNEL_INT8_SOA_SRC",
            Some("source-a".to_string()),
            Some("options-a".to_string()),
            &["step", "init", "step"],
        );
        let reordered = ModuleProvenance::new(
            "KERNEL_INT8_SOA_SRC",
            Some("source-a".to_string()),
            Some("options-a".to_string()),
            &["init", "step"],
        );
        let changed_source = ModuleProvenance::new(
            "KERNEL_INT8_SOA_SRC",
            Some("source-b".to_string()),
            Some("options-a".to_string()),
            &["init", "step"],
        );

        assert_eq!(
            first.kernel_names,
            vec!["init".to_string(), "step".to_string()]
        );
        assert_eq!(first.module_id, reordered.module_id);
        assert_ne!(first.module_id, changed_source.module_id);
        assert_eq!(first.source_sha256.as_deref(), Some("source-a"));
    }

    #[test]
    fn kernel_provenance_keeps_module_identity() {
        let module = ModuleProvenance::new(
            "KERNEL_FP64_SOA_SRC",
            Some("source".to_string()),
            Some("options".to_string()),
            &["step"],
        );
        let kernel = module.kernel("step");

        assert_eq!(kernel.kernel_name, "step");
        assert_eq!(kernel.module_id, module.module_id);
        assert_eq!(kernel.source_label, "KERNEL_FP64_SOA_SRC");
        assert_eq!(kernel.compile_options_sha256.as_deref(), Some("options"));
    }
}
