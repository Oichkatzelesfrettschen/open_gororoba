#[cfg(test)]
pub(crate) const DEQUANT_DOT_KERNEL_PARAM_ORDER: [&str; 9] = [
    "queries",
    "key_indices",
    "centroids",
    "key_norms",
    "scores",
    "d",
    "n_queries",
    "n_keys",
    "n_levels",
];

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DequantDotKernelDims {
    pub(crate) d: i32,
    pub(crate) n_queries: i32,
    pub(crate) n_keys: i32,
    pub(crate) n_levels: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValidatedDequantDotContract {
    pub(crate) expected_scores: usize,
    #[cfg(feature = "cuda")]
    d: usize,
    #[cfg(feature = "cuda")]
    n_queries: usize,
    #[cfg(feature = "cuda")]
    n_keys: usize,
    #[cfg(feature = "cuda")]
    n_levels: usize,
}

impl ValidatedDequantDotContract {
    #[cfg(feature = "cuda")]
    pub(crate) fn kernel_dims_i32(self) -> Result<DequantDotKernelDims, String> {
        let d = i32::try_from(self.d).map_err(|_| {
            format!(
                "d exceeds i32 kernel ABI limit for turboquant_dequant_dot: {}",
                self.d
            )
        })?;
        let n_queries = i32::try_from(self.n_queries).map_err(|_| {
            format!(
                "n_queries exceeds i32 kernel ABI limit for turboquant_dequant_dot: {}",
                self.n_queries
            )
        })?;
        let n_keys = i32::try_from(self.n_keys).map_err(|_| {
            format!(
                "n_keys exceeds i32 kernel ABI limit for turboquant_dequant_dot: {}",
                self.n_keys
            )
        })?;
        let n_levels = i32::try_from(self.n_levels).map_err(|_| {
            format!(
                "centroids length exceeds i32 kernel ABI limit for turboquant_dequant_dot: {}",
                self.n_levels
            )
        })?;
        Ok(DequantDotKernelDims {
            d,
            n_queries,
            n_keys,
            n_levels,
        })
    }
}

pub(crate) fn validate_dequant_dot_contract(
    queries_len: usize,
    key_indices_len: usize,
    n_levels: usize,
    key_norms_len: usize,
    n_queries: usize,
    n_keys: usize,
    d: usize,
) -> Result<ValidatedDequantDotContract, String> {
    if d == 0 {
        return Err("d must be > 0".to_string());
    }

    let expected_queries = n_queries
        .checked_mul(d)
        .ok_or_else(|| format!("n_queries*d overflow: {n_queries}*{d}"))?;
    if queries_len != expected_queries {
        return Err(format!(
            "queries length mismatch: got {}, expected {} (n_queries*d)",
            queries_len, expected_queries
        ));
    }

    let expected_indices = d
        .checked_mul(n_keys)
        .ok_or_else(|| format!("d*n_keys overflow: {d}*{n_keys}"))?;
    if key_indices_len != expected_indices {
        return Err(format!(
            "key_indices length mismatch: got {}, expected {} (d*n_keys)",
            key_indices_len, expected_indices
        ));
    }

    if n_levels == 0 {
        return Err("centroids must not be empty".to_string());
    }
    if n_levels > (u8::MAX as usize + 1) {
        return Err(format!(
            "centroids length exceeds u8 index space: {} > {}",
            n_levels,
            u8::MAX as usize + 1
        ));
    }

    if key_norms_len != n_keys {
        return Err(format!(
            "key_norms length mismatch: got {}, expected {} (n_keys)",
            key_norms_len, n_keys
        ));
    }

    let expected_scores = n_queries
        .checked_mul(n_keys)
        .ok_or_else(|| format!("n_queries*n_keys overflow: {n_queries}*{n_keys}"))?;

    Ok(ValidatedDequantDotContract {
        expected_scores,
        #[cfg(feature = "cuda")]
        d,
        #[cfg(feature = "cuda")]
        n_queries,
        #[cfg(feature = "cuda")]
        n_keys,
        #[cfg(feature = "cuda")]
        n_levels,
    })
}

#[cfg(test)]
pub(crate) fn verify_dequant_dot_kernel_signature(kernel_src: &str) -> Result<(), String> {
    let actual = extract_dequant_dot_kernel_param_names(kernel_src)?;
    let expected: Vec<&str> = DEQUANT_DOT_KERNEL_PARAM_ORDER.into_iter().collect();
    if actual != expected {
        return Err(format!(
            "turboquant_dequant_dot ABI drift detected: expected {:?}, got {:?}",
            DEQUANT_DOT_KERNEL_PARAM_ORDER, actual
        ));
    }
    Ok(())
}

#[cfg(test)]
fn extract_dequant_dot_kernel_param_names(kernel_src: &str) -> Result<Vec<&str>, String> {
    let marker = "__global__ void turboquant_dequant_dot(";
    let start = kernel_src
        .find(marker)
        .ok_or_else(|| "turboquant_dequant_dot kernel declaration not found".to_string())?;
    let after_marker = &kernel_src[start + marker.len()..];
    let close = after_marker
        .find(')')
        .ok_or_else(|| "turboquant_dequant_dot declaration missing ')'".to_string())?;
    let params = &after_marker[..close];

    let mut names = Vec::with_capacity(DEQUANT_DOT_KERNEL_PARAM_ORDER.len());
    for param in params.split(',') {
        let candidate = param
            .split_whitespace()
            .last()
            .ok_or_else(|| format!("unable to parse kernel parameter from '{param}'"))?
            .trim_matches(|c: char| c == ',' || c == '*' || c == '&');
        if candidate.is_empty() {
            return Err(format!(
                "unable to parse non-empty kernel parameter from '{param}'"
            ));
        }
        names.push(candidate);
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_dequant_contract_accepts_valid_shapes() {
        let validated = validate_dequant_dot_contract(8, 12, 8, 3, 2, 3, 4).expect("valid shapes");
        assert_eq!(validated.expected_scores, 6);
        #[cfg(feature = "cuda")]
        {
            let dims = validated.kernel_dims_i32().expect("i32 ABI dims");
            assert_eq!(dims.d, 4);
            assert_eq!(dims.n_queries, 2);
            assert_eq!(dims.n_keys, 3);
            assert_eq!(dims.n_levels, 8);
        }
    }

    #[test]
    fn test_validate_dequant_contract_rejects_shape_mismatch() {
        let err = validate_dequant_dot_contract(7, 12, 8, 3, 2, 3, 4)
            .expect_err("query mismatch should fail");
        assert!(err.contains("queries length mismatch"));
    }

    #[test]
    fn test_validate_dequant_contract_rejects_invalid_d() {
        let err = validate_dequant_dot_contract(0, 0, 8, 0, 0, 0, 0).expect_err("d=0 should fail");
        assert!(err.contains("d must be > 0"));
    }

    #[test]
    fn test_verify_dequant_dot_kernel_signature() {
        let src = include_str!("cuda/kernels/turboquant.cu");
        verify_dequant_dot_kernel_signature(src).expect("kernel signature should match contract");
    }
}
