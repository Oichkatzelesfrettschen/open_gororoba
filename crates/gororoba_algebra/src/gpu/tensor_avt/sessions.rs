use gororoba_gpu_bridge::ComputeBackend;

#[cfg(feature = "gpu")]
use super::cuda::{TensorAvtMulGpuWorkspace, TensorAvtNormGpuWorkspace};
#[cfg(feature = "vulkan")]
use super::vulkan::{TensorAvtMulVulkanWorkspace, TensorAvtNormVulkanWorkspace};
use super::{
    TensorAVT,
    cpu::{TensorAvtCpuMulSession, TensorAvtCpuNormSession},
};

pub(crate) enum TensorAvtMulSessionInner {
    Cpu(TensorAvtCpuMulSession),
    #[cfg(feature = "gpu")]
    Cuda(TensorAvtMulGpuWorkspace),
    #[cfg(feature = "vulkan")]
    Vulkan(Box<TensorAvtMulVulkanWorkspace>),
}

pub struct TensorAvtMulSession {
    pub(crate) backend: ComputeBackend,
    pub(crate) dim: usize,
    pub(crate) max_batch_size: usize,
    pub(crate) inner: TensorAvtMulSessionInner,
}

impl TensorAvtMulSession {
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn load_left(&mut self, a: &[f32]) -> Result<(), String> {
        if a.len() != self.dim {
            return Err(format!(
                "left input length {} must equal dim {}",
                a.len(),
                self.dim
            ));
        }
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.left.copy_from_slice(a);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => workspace.upload_a(a),
            #[cfg(feature = "vulkan")]
            TensorAvtMulSessionInner::Vulkan(workspace) => workspace.upload_a(a),
        }
    }

    pub fn load_right(&mut self, x: &[f32], batch_size: usize) -> Result<(), String> {
        if batch_size == 0 || batch_size > self.max_batch_size {
            return Err(format!(
                "batch_size must be in 1..={}, got {}",
                self.max_batch_size, batch_size
            ));
        }
        let expected = batch_size * self.dim;
        if x.len() != expected {
            return Err(format!(
                "right input length {} must equal batch_size * dim {}",
                x.len(),
                expected
            ));
        }
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.right[..expected].copy_from_slice(x);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => {
                workspace.upload_x(x, batch_size, self.dim)
            }
            #[cfg(feature = "vulkan")]
            TensorAvtMulSessionInner::Vulkan(workspace) => {
                workspace.upload_x(x, batch_size, self.dim)
            }
        }
    }

    pub fn run_single(&mut self, avt: &TensorAVT) -> Result<(), String> {
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.output =
                    avt.compute_cd_mul_cpu(&session.left, &session.right[..self.dim])?;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => {
                avt.launch_cd_mul_with_workspace(workspace)
            }
            #[cfg(feature = "vulkan")]
            TensorAvtMulSessionInner::Vulkan(workspace) => {
                avt.launch_cd_mul_vulkan_with_workspace(workspace.as_mut())
            }
        }
    }

    pub fn run_batch(&mut self, avt: &TensorAVT, batch_size: usize) -> Result<(), String> {
        if batch_size == 0 || batch_size > self.max_batch_size {
            return Err(format!(
                "batch_size must be in 1..={}, got {}",
                self.max_batch_size, batch_size
            ));
        }
        let expected = batch_size * self.dim;
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                session.output = avt.compute_cd_mul_batch_cpu(
                    &session.left,
                    &session.right[..expected],
                    batch_size,
                )?;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => {
                avt.launch_cd_mul_batch_with_workspace(batch_size, workspace)
            }
            #[cfg(feature = "vulkan")]
            TensorAvtMulSessionInner::Vulkan(workspace) => {
                avt.launch_cd_mul_batch_vulkan_with_workspace(batch_size, workspace.as_mut())
            }
        }
    }

    pub fn download_output(&mut self, len: usize) -> Result<Vec<f32>, String> {
        if len > self.dim * self.max_batch_size {
            return Err(format!(
                "output length {} exceeds session capacity {}",
                len,
                self.dim * self.max_batch_size
            ));
        }
        match &mut self.inner {
            TensorAvtMulSessionInner::Cpu(session) => {
                if len > session.output.len() {
                    return Err(format!(
                        "requested output length {} exceeds computed output length {}",
                        len,
                        session.output.len()
                    ));
                }
                Ok(session.output[..len].to_vec())
            }
            #[cfg(feature = "gpu")]
            TensorAvtMulSessionInner::Cuda(workspace) => workspace.download_y(len),
            #[cfg(feature = "vulkan")]
            TensorAvtMulSessionInner::Vulkan(workspace) => workspace.download_y(len),
        }
    }
}

pub(crate) enum TensorAvtNormSessionInner {
    Cpu(TensorAvtCpuNormSession),
    #[cfg(feature = "gpu")]
    Cuda(TensorAvtNormGpuWorkspace),
    #[cfg(feature = "vulkan")]
    Vulkan(Box<TensorAvtNormVulkanWorkspace>),
}

pub struct TensorAvtNormSession {
    pub(crate) backend: ComputeBackend,
    pub(crate) dim: usize,
    pub(crate) max_vectors: usize,
    pub(crate) inner: TensorAvtNormSessionInner,
}

impl TensorAvtNormSession {
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub fn max_vectors(&self) -> usize {
        self.max_vectors
    }

    pub fn load_vectors(&mut self, vectors: &[f32], n_vectors: usize) -> Result<(), String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        let expected = n_vectors * self.dim;
        if vectors.len() != expected {
            return Err(format!(
                "vectors length {} must equal n_vectors * dim {}",
                vectors.len(),
                expected
            ));
        }
        match &mut self.inner {
            TensorAvtNormSessionInner::Cpu(session) => {
                session.vectors[..expected].copy_from_slice(vectors);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtNormSessionInner::Cuda(workspace) => {
                workspace.upload_vectors(vectors, n_vectors, self.dim)
            }
            #[cfg(feature = "vulkan")]
            TensorAvtNormSessionInner::Vulkan(workspace) => {
                workspace.upload_vectors(vectors, n_vectors, self.dim)
            }
        }
    }

    pub fn run_norms(&mut self, avt: &TensorAVT, n_vectors: usize) -> Result<(), String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        let expected = n_vectors * self.dim;
        match &mut self.inner {
            TensorAvtNormSessionInner::Cpu(session) => {
                session.norms =
                    avt.compute_norm_sq_batch_cpu(&session.vectors[..expected], n_vectors)?;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorAvtNormSessionInner::Cuda(workspace) => {
                avt.launch_norm_sq_batch_with_workspace(n_vectors, workspace)
            }
            #[cfg(feature = "vulkan")]
            TensorAvtNormSessionInner::Vulkan(workspace) => {
                avt.launch_norm_sq_batch_vulkan_with_workspace(n_vectors, workspace.as_mut())
            }
        }
    }

    pub fn download_norms(&mut self, n_vectors: usize) -> Result<Vec<f32>, String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        match &mut self.inner {
            TensorAvtNormSessionInner::Cpu(session) => {
                if n_vectors > session.norms.len() {
                    return Err(format!(
                        "requested norms length {} exceeds computed length {}",
                        n_vectors,
                        session.norms.len()
                    ));
                }
                Ok(session.norms[..n_vectors].to_vec())
            }
            #[cfg(feature = "gpu")]
            TensorAvtNormSessionInner::Cuda(workspace) => workspace.download_norms(n_vectors),
            #[cfg(feature = "vulkan")]
            TensorAvtNormSessionInner::Vulkan(workspace) => workspace.download_norms(n_vectors),
        }
    }
}
