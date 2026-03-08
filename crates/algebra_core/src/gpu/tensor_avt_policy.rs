use gororoba_gpu_bridge::{ComputeBackend, probe_simd};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use super::{TensorAVT, cuda, vulkan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TensorAvtAutoOp {
    CdMul,
    CdMulBatch,
    NormSqBatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TensorAvtCalibrationKey {
    op: TensorAvtAutoOp,
    dim: usize,
    count: usize,
    cuda_available: bool,
    simd_available: bool,
}

pub(crate) fn simd_available() -> bool {
    probe_simd().any()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorAvtCalibrationMode {
    LazyInProcess,
    Disabled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TensorAvtThresholdOverrides {
    pub cd_mul_min_problem_size: Option<usize>,
    pub cd_mul_batch_min_problem_size: Option<usize>,
    pub norm_sq_batch_min_problem_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorAvtAutoConfig {
    pub backend_order: [ComputeBackend; 4],
    pub calibration: TensorAvtCalibrationMode,
    pub threshold_overrides: TensorAvtThresholdOverrides,
}

impl Default for TensorAvtAutoConfig {
    fn default() -> Self {
        Self {
            backend_order: [
                ComputeBackend::Cuda,
                ComputeBackend::Vulkan,
                ComputeBackend::CpuSimd,
                ComputeBackend::CpuScalar,
            ],
            calibration: TensorAvtCalibrationMode::LazyInProcess,
            threshold_overrides: TensorAvtThresholdOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorAvtAutoResult<T> {
    pub backend: ComputeBackend,
    pub value: T,
    pub calibrated_this_call: bool,
}

fn tensor_avt_calibration_cache() -> &'static Mutex<HashMap<TensorAvtCalibrationKey, ComputeBackend>>
{
    static CACHE: OnceLock<Mutex<HashMap<TensorAvtCalibrationKey, ComputeBackend>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tensor_avt_backend_is_cpu(backend: ComputeBackend) -> bool {
    matches!(backend, ComputeBackend::CpuScalar | ComputeBackend::CpuSimd)
}

fn tensor_avt_backend_is_gpu(backend: ComputeBackend) -> bool {
    matches!(backend, ComputeBackend::Cuda | ComputeBackend::Vulkan)
}

impl TensorAVT {
    fn auto_threshold_for(
        &self,
        op: TensorAvtAutoOp,
        config: &TensorAvtAutoConfig,
    ) -> Option<usize> {
        match op {
            TensorAvtAutoOp::CdMul => config.threshold_overrides.cd_mul_min_problem_size,
            TensorAvtAutoOp::CdMulBatch => config.threshold_overrides.cd_mul_batch_min_problem_size,
            TensorAvtAutoOp::NormSqBatch => {
                config.threshold_overrides.norm_sq_batch_min_problem_size
            }
        }
    }

    fn first_available_backend(&self, order: &[ComputeBackend; 4]) -> Option<ComputeBackend> {
        order
            .iter()
            .copied()
            .find(|backend| self.backend_is_available(*backend))
    }

    fn first_available_cpu_backend(&self, order: &[ComputeBackend; 4]) -> Option<ComputeBackend> {
        order.iter().copied().find(|backend| {
            tensor_avt_backend_is_cpu(*backend) && self.backend_is_available(*backend)
        })
    }

    fn first_available_gpu_backend(&self, order: &[ComputeBackend; 4]) -> Option<ComputeBackend> {
        order.iter().copied().find(|backend| {
            tensor_avt_backend_is_gpu(*backend) && self.backend_is_available(*backend)
        })
    }

    fn backend_is_available(&self, backend: ComputeBackend) -> bool {
        let _ = self;
        match backend {
            ComputeBackend::CpuScalar => true,
            ComputeBackend::CpuSimd => simd_available(),
            ComputeBackend::Vulkan => vulkan::tensor_avt_vulkan_available(),
            ComputeBackend::Cuda => cuda::tensor_avt_cuda_available(),
        }
    }

    fn select_backend_from_override(
        &self,
        op: TensorAvtAutoOp,
        problem_size: usize,
        config: &TensorAvtAutoConfig,
    ) -> Option<ComputeBackend> {
        let threshold = self.auto_threshold_for(op, config)?;
        if problem_size >= threshold {
            self.first_available_gpu_backend(&config.backend_order)
                .or_else(|| self.first_available_cpu_backend(&config.backend_order))
        } else {
            self.first_available_cpu_backend(&config.backend_order)
                .or_else(|| self.first_available_gpu_backend(&config.backend_order))
        }
    }

    fn calibration_key(&self, op: TensorAvtAutoOp, count: usize) -> TensorAvtCalibrationKey {
        TensorAvtCalibrationKey {
            op,
            dim: self.dim,
            count,
            cuda_available: cuda::tensor_avt_cuda_available(),
            simd_available: simd_available(),
        }
    }

    fn calibration_inputs(&self, len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|idx| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407 + idx as u64);
                let unit = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
                (unit * 2.0) - 1.0
            })
            .collect()
    }

    fn calibration_runs(&self, op: TensorAvtAutoOp, count: usize) -> (usize, usize) {
        let problem_size = match op {
            TensorAvtAutoOp::CdMul => self.dim,
            TensorAvtAutoOp::CdMulBatch | TensorAvtAutoOp::NormSqBatch => self.dim * count,
        };
        if matches!(op, TensorAvtAutoOp::CdMulBatch) && problem_size >= 65_536 {
            (1, 1)
        } else {
            (1, 3)
        }
    }

    fn measure_cd_mul_cpu_ns(
        &self,
        a: &[f32],
        x: &[f32],
        warmup: usize,
        runs: usize,
    ) -> Result<u128, String> {
        for _ in 0..warmup {
            let _ = self.compute_cd_mul_cpu(a, x)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = self.compute_cd_mul_cpu(a, x)?;
        }
        Ok(start.elapsed().as_nanos() / runs as u128)
    }

    fn measure_cd_mul_batch_cpu_ns(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
        warmup: usize,
        runs: usize,
    ) -> Result<u128, String> {
        for _ in 0..warmup {
            let _ = self.compute_cd_mul_batch_cpu(a, x_batch, batch_size)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = self.compute_cd_mul_batch_cpu(a, x_batch, batch_size)?;
        }
        Ok(start.elapsed().as_nanos() / runs as u128)
    }

    fn measure_norm_sq_batch_cpu_ns(
        &self,
        vectors: &[f32],
        n_vectors: usize,
        warmup: usize,
        runs: usize,
    ) -> Result<u128, String> {
        for _ in 0..warmup {
            let _ = self.compute_norm_sq_batch_cpu(vectors, n_vectors)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = self.compute_norm_sq_batch_cpu(vectors, n_vectors)?;
        }
        Ok(start.elapsed().as_nanos() / runs as u128)
    }

    #[cfg(feature = "gpu")]
    fn measure_cd_mul_cuda_ns(
        &self,
        a: &[f32],
        x: &[f32],
        warmup: usize,
        runs: usize,
    ) -> Result<u128, String> {
        for _ in 0..warmup {
            let _ = self.compute_cd_mul(a, x)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = self.compute_cd_mul(a, x)?;
        }
        Ok(start.elapsed().as_nanos() / runs as u128)
    }

    #[cfg(feature = "gpu")]
    fn measure_cd_mul_batch_cuda_ns(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
        warmup: usize,
        runs: usize,
    ) -> Result<u128, String> {
        for _ in 0..warmup {
            let _ = self.compute_cd_mul_batch(a, x_batch, batch_size)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = self.compute_cd_mul_batch(a, x_batch, batch_size)?;
        }
        Ok(start.elapsed().as_nanos() / runs as u128)
    }

    #[cfg(feature = "gpu")]
    fn measure_norm_sq_batch_cuda_ns(
        &self,
        vectors: &[f32],
        n_vectors: usize,
        warmup: usize,
        runs: usize,
    ) -> Result<u128, String> {
        for _ in 0..warmup {
            let _ = self.compute_norm_sq_batch(vectors, n_vectors)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = self.compute_norm_sq_batch(vectors, n_vectors)?;
        }
        Ok(start.elapsed().as_nanos() / runs as u128)
    }

    fn calibrate_auto_backend(
        &self,
        op: TensorAvtAutoOp,
        count: usize,
        config: &TensorAvtAutoConfig,
    ) -> Result<Option<(ComputeBackend, bool)>, String> {
        if config.calibration != TensorAvtCalibrationMode::LazyInProcess {
            return Ok(None);
        }
        let cpu_backend = match self.first_available_cpu_backend(&config.backend_order) {
            Some(backend) => backend,
            None => {
                return Ok(self
                    .first_available_backend(&config.backend_order)
                    .map(|b| (b, false)));
            }
        };
        let cuda_available = self
            .first_available_gpu_backend(&config.backend_order)
            .is_some_and(|backend| backend == ComputeBackend::Cuda);
        if !cuda_available {
            return Ok(Some((cpu_backend, false)));
        }

        let key = self.calibration_key(op, count);
        {
            let cache = tensor_avt_calibration_cache()
                .lock()
                .expect("tensor_avt calibration cache poisoned");
            if let Some(backend) = cache.get(&key).copied() {
                return Ok(Some((backend, false)));
            }
        }

        let (warmup, runs) = self.calibration_runs(op, count);
        let chosen = match op {
            TensorAvtAutoOp::CdMul => {
                let a = self.calibration_inputs(self.dim, 0xA11CE);
                let x = self.calibration_inputs(self.dim, 0xBADC0DE);
                let cpu_ns = self.measure_cd_mul_cpu_ns(&a, &x, warmup, runs)?;
                #[cfg(feature = "gpu")]
                let cuda_ns = self.measure_cd_mul_cuda_ns(&a, &x, warmup, runs)?;
                #[cfg(not(feature = "gpu"))]
                let cuda_ns = u128::MAX;
                if cuda_ns < cpu_ns {
                    ComputeBackend::Cuda
                } else {
                    cpu_backend
                }
            }
            TensorAvtAutoOp::CdMulBatch => {
                let a = self.calibration_inputs(self.dim, 0xCAFE);
                let x_batch = self.calibration_inputs(self.dim * count, 0xFACEFEED);
                let cpu_ns = self.measure_cd_mul_batch_cpu_ns(&a, &x_batch, count, warmup, runs)?;
                #[cfg(feature = "gpu")]
                let cuda_ns =
                    self.measure_cd_mul_batch_cuda_ns(&a, &x_batch, count, warmup, runs)?;
                #[cfg(not(feature = "gpu"))]
                let cuda_ns = u128::MAX;
                if cuda_ns < cpu_ns {
                    ComputeBackend::Cuda
                } else {
                    cpu_backend
                }
            }
            TensorAvtAutoOp::NormSqBatch => {
                let vectors = self.calibration_inputs(self.dim * count, 0xDEADBEEF);
                let cpu_ns = self.measure_norm_sq_batch_cpu_ns(&vectors, count, warmup, runs)?;
                #[cfg(feature = "gpu")]
                let cuda_ns = self.measure_norm_sq_batch_cuda_ns(&vectors, count, warmup, runs)?;
                #[cfg(not(feature = "gpu"))]
                let cuda_ns = u128::MAX;
                if cuda_ns < cpu_ns {
                    ComputeBackend::Cuda
                } else {
                    cpu_backend
                }
            }
        };

        let mut cache = tensor_avt_calibration_cache()
            .lock()
            .expect("tensor_avt calibration cache poisoned");
        cache.insert(key, chosen);
        Ok(Some((chosen, true)))
    }

    fn select_auto_backend(
        &self,
        op: TensorAvtAutoOp,
        count: usize,
        problem_size: usize,
        config: &TensorAvtAutoConfig,
    ) -> Result<(ComputeBackend, bool), String> {
        if let Some(backend) = self.select_backend_from_override(op, problem_size, config) {
            return Ok((backend, false));
        }
        if let Some((backend, calibrated)) = self.calibrate_auto_backend(op, count, config)? {
            return Ok((backend, calibrated));
        }
        match self.first_available_backend(&config.backend_order) {
            Some(backend) => Ok((backend, false)),
            None => Err("no TensorAVT backend is available for the requested configuration".into()),
        }
    }

    pub fn compute_cd_mul_auto(
        &self,
        a: &[f32],
        x: &[f32],
    ) -> Result<TensorAvtAutoResult<Vec<f32>>, String> {
        self.compute_cd_mul_auto_with_config(a, x, &TensorAvtAutoConfig::default())
    }

    pub fn compute_cd_mul_auto_with_config(
        &self,
        a: &[f32],
        x: &[f32],
        config: &TensorAvtAutoConfig,
    ) -> Result<TensorAvtAutoResult<Vec<f32>>, String> {
        let (backend, calibrated_this_call) =
            self.select_auto_backend(TensorAvtAutoOp::CdMul, 1, self.dim, config)?;
        let value = match backend {
            ComputeBackend::CpuScalar | ComputeBackend::CpuSimd => self.compute_cd_mul_cpu(a, x)?,
            #[cfg(feature = "gpu")]
            ComputeBackend::Cuda => self.compute_cd_mul(a, x)?,
            #[cfg(not(feature = "gpu"))]
            ComputeBackend::Cuda => {
                return Err(
                    "TensorAVT CUDA backend requires building algebra_core with --features gpu"
                        .into(),
                );
            }
            ComputeBackend::Vulkan => return Err(vulkan::tensor_avt_vulkan_error()),
        };
        Ok(TensorAvtAutoResult {
            backend,
            value,
            calibrated_this_call,
        })
    }

    pub fn compute_cd_mul_batch_auto(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<TensorAvtAutoResult<Vec<f32>>, String> {
        self.compute_cd_mul_batch_auto_with_config(
            a,
            x_batch,
            batch_size,
            &TensorAvtAutoConfig::default(),
        )
    }

    pub fn compute_cd_mul_batch_auto_with_config(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
        config: &TensorAvtAutoConfig,
    ) -> Result<TensorAvtAutoResult<Vec<f32>>, String> {
        let problem_size = self.dim * batch_size;
        let (backend, calibrated_this_call) = self.select_auto_backend(
            TensorAvtAutoOp::CdMulBatch,
            batch_size,
            problem_size,
            config,
        )?;
        let value = match backend {
            ComputeBackend::CpuScalar | ComputeBackend::CpuSimd => {
                self.compute_cd_mul_batch_cpu(a, x_batch, batch_size)?
            }
            #[cfg(feature = "gpu")]
            ComputeBackend::Cuda => self.compute_cd_mul_batch(a, x_batch, batch_size)?,
            #[cfg(not(feature = "gpu"))]
            ComputeBackend::Cuda => {
                return Err(
                    "TensorAVT CUDA backend requires building algebra_core with --features gpu"
                        .into(),
                );
            }
            ComputeBackend::Vulkan => return Err(vulkan::tensor_avt_vulkan_error()),
        };
        Ok(TensorAvtAutoResult {
            backend,
            value,
            calibrated_this_call,
        })
    }

    pub fn compute_norm_sq_batch_auto(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<TensorAvtAutoResult<Vec<f32>>, String> {
        self.compute_norm_sq_batch_auto_with_config(
            vectors,
            n_vectors,
            &TensorAvtAutoConfig::default(),
        )
    }

    pub fn compute_norm_sq_batch_auto_with_config(
        &self,
        vectors: &[f32],
        n_vectors: usize,
        config: &TensorAvtAutoConfig,
    ) -> Result<TensorAvtAutoResult<Vec<f32>>, String> {
        let problem_size = self.dim * n_vectors;
        let (backend, calibrated_this_call) = self.select_auto_backend(
            TensorAvtAutoOp::NormSqBatch,
            n_vectors,
            problem_size,
            config,
        )?;
        let value = match backend {
            ComputeBackend::CpuScalar | ComputeBackend::CpuSimd => {
                self.compute_norm_sq_batch_cpu(vectors, n_vectors)?
            }
            #[cfg(feature = "gpu")]
            ComputeBackend::Cuda => self.compute_norm_sq_batch(vectors, n_vectors)?,
            #[cfg(not(feature = "gpu"))]
            ComputeBackend::Cuda => {
                return Err(
                    "TensorAVT CUDA backend requires building algebra_core with --features gpu"
                        .into(),
                );
            }
            ComputeBackend::Vulkan => return Err(vulkan::tensor_avt_vulkan_error()),
        };
        Ok(TensorAvtAutoResult {
            backend,
            value,
            calibrated_this_call,
        })
    }
}
