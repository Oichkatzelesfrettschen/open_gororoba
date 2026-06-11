//! cubecl-wgpu PEPS row contraction.
//!
//! WebGPU-class shader targets do not provide a portable FP64 contract, so
//! the cubecl path narrows `faer::c64` components to FP32 before dispatch and
//! widens the readback to `c64`. CUDA remains the FP64 backend when the `gpu`
//! feature is enabled.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use faer::c64;

#[cube(launch_unchecked)]
pub fn peps_contract_rows_kernel(
    upper_re: &Array<f32>,
    upper_im: &Array<f32>,
    lower_re: &Array<f32>,
    lower_im: &Array<f32>,
    result_re: &mut Array<f32>,
    result_im: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= result_re.len() {
        terminate!();
    }

    let a_re = upper_re[idx];
    let a_im = upper_im[idx];
    let b_re = lower_re[idx];
    let b_im = lower_im[idx];

    result_re[idx] = a_re * b_re - a_im * b_im;
    result_im[idx] = a_re * b_im + a_im * b_re;
}

pub struct PepsCubeclKernel;

impl PepsCubeclKernel {
    pub fn is_available() -> bool {
        peps_cubecl_available()
    }

    pub fn contract_rows_fp32(upper: &[c64], lower: &[c64]) -> Result<Vec<c64>, String> {
        let prepared = PreparedPepsRows::new(upper, lower)?;
        if prepared.len == 0 {
            return Ok(Vec::new());
        }
        if !Self::is_available() {
            return Err("PEPS cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let upper_re = client.create_from_slice(&encode_f32_slice(&prepared.upper_re)?);
        let upper_im = client.create_from_slice(&encode_f32_slice(&prepared.upper_im)?);
        let lower_re = client.create_from_slice(&encode_f32_slice(&prepared.lower_re)?);
        let lower_im = client.create_from_slice(&encode_f32_slice(&prepared.lower_im)?);
        let result_re = client.empty(prepared.byte_len()?);
        let result_im = client.empty(prepared.byte_len()?);
        let result_re_readback = result_re.clone();
        let result_im_readback = result_im.clone();

        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(prepared.len.div_ceil(256) as u32);

        // SAFETY: every buffer has exactly `prepared.len` f32 slots and the
        // kernel terminates threads with idx >= result_re.len().
        unsafe {
            peps_contract_rows_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(upper_re, prepared.len),
                ArrayArg::from_raw_parts(upper_im, prepared.len),
                ArrayArg::from_raw_parts(lower_re, prepared.len),
                ArrayArg::from_raw_parts(lower_im, prepared.len),
                ArrayArg::from_raw_parts(result_re, prepared.len),
                ArrayArg::from_raw_parts(result_im, prepared.len),
            );
        }

        let result_re = decode_f32_output(
            &client.read_one_unchecked(result_re_readback),
            prepared.len,
            "result_re",
        )?;
        let result_im = decode_f32_output(
            &client.read_one_unchecked(result_im_readback),
            prepared.len,
            "result_im",
        )?;

        Ok(result_re
            .into_iter()
            .zip(result_im)
            .map(|(re, im)| c64::new(re as f64, im as f64))
            .collect())
    }
}

pub fn peps_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

pub fn cubecl_contract_rows_peps_fp32(upper: &[c64], lower: &[c64]) -> Vec<c64> {
    if upper.len() != lower.len() {
        return lower.to_vec();
    }

    if let Ok(result) = PepsCubeclKernel::contract_rows_fp32(upper, lower) {
        return result;
    }

    peps_contract_rows_cpu(upper, lower)
}

pub fn peps_contract_rows_cpu(upper: &[c64], lower: &[c64]) -> Vec<c64> {
    upper
        .iter()
        .zip(lower.iter())
        .map(|(a, b)| c64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re))
        .collect()
}

struct PreparedPepsRows {
    upper_re: Vec<f32>,
    upper_im: Vec<f32>,
    lower_re: Vec<f32>,
    lower_im: Vec<f32>,
    len: usize,
}

impl PreparedPepsRows {
    fn new(upper: &[c64], lower: &[c64]) -> Result<Self, String> {
        if upper.len() != lower.len() {
            return Err(format!(
                "PEPS cubecl row lengths differ: upper {}, lower {}",
                upper.len(),
                lower.len()
            ));
        }
        if upper.len() > u32::MAX as usize {
            return Err(format!(
                "PEPS cubecl row length {} exceeds u32 dispatch",
                upper.len()
            ));
        }

        let mut upper_re = Vec::with_capacity(upper.len());
        let mut upper_im = Vec::with_capacity(upper.len());
        let mut lower_re = Vec::with_capacity(lower.len());
        let mut lower_im = Vec::with_capacity(lower.len());

        for (index, value) in upper.iter().enumerate() {
            upper_re.push(narrow_component(value.re, index, "upper_re")?);
            upper_im.push(narrow_component(value.im, index, "upper_im")?);
        }
        for (index, value) in lower.iter().enumerate() {
            lower_re.push(narrow_component(value.re, index, "lower_re")?);
            lower_im.push(narrow_component(value.im, index, "lower_im")?);
        }

        Ok(Self {
            upper_re,
            upper_im,
            lower_re,
            lower_im,
            len: upper.len(),
        })
    }

    fn byte_len(&self) -> Result<usize, String> {
        self.len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("PEPS cubecl row length {} overflows bytes", self.len))
    }
}

fn narrow_component(value: f64, index: usize, label: &str) -> Result<f32, String> {
    if !value.is_finite() {
        return Err(format!("PEPS cubecl {label}[{index}] is not finite"));
    }
    if value.abs() > f32::MAX as f64 {
        return Err(format!("PEPS cubecl {label}[{index}] exceeds f32 range"));
    }
    Ok(value as f32)
}

fn encode_f32_slice(values: &[f32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "PEPS cubecl buffer size overflows".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_f32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("PEPS cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "PEPS cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peps_cubecl_available_does_not_panic() {
        let _ = PepsCubeclKernel::is_available();
    }

    #[test]
    fn peps_cpu_reference_multiplies_complex_rows() {
        let upper = vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)];
        let lower = vec![c64::new(2.0, 1.0), c64::new(1.0, 1.0)];
        let result = peps_contract_rows_cpu(&upper, &lower);
        assert_eq!(result, vec![c64::new(0.0, 5.0), c64::new(-1.0, 7.0)]);
    }

    #[test]
    fn peps_preparation_rejects_invalid_components() {
        let upper = vec![c64::new(f64::INFINITY, 0.0)];
        let lower = vec![c64::new(1.0, 0.0)];
        assert!(PreparedPepsRows::new(&upper, &lower).is_err());
    }

    #[test]
    fn peps_f32_roundtrip_decodes_outputs() {
        let values = [0.0f32, 5.0, -1.0, 7.0];
        let bytes = encode_f32_slice(&values).unwrap();
        assert_eq!(
            decode_f32_output(&bytes, values.len(), "test").unwrap(),
            values
        );
    }

    #[test]
    fn peps_cubecl_matches_cpu_when_adapter_available() {
        if !PepsCubeclKernel::is_available() {
            return;
        }

        let upper = vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)];
        let lower = vec![c64::new(2.0, 1.0), c64::new(1.0, 1.0)];
        let cubecl = PepsCubeclKernel::contract_rows_fp32(&upper, &lower).unwrap();
        let cpu = peps_contract_rows_cpu(&upper, &lower);
        assert_eq!(cubecl, cpu);
    }
}
