//! CosmoMC MCMC chain format parser.
//!
//! CosmoMC chains consist of:
//! - A `.paramnames` file mapping column indices to parameter names/labels
//! - One or more chain files (whitespace-delimited) with columns:
//!   `weight  -log(like)  param1  param2  ...`
//!
//! Used by both WMAP 9yr and Planck 2018 posterior chains.
//!
//! Reference: Lewis & Bridle (2002), PRD 66, 103511

use crate::fetcher::FetchError;
use std::path::Path;

/// Parsed MCMC chain with parameter metadata and weighted samples.
#[derive(Debug, Clone)]
pub struct McmcChain {
    /// Parameter names (from .paramnames file).
    pub param_names: Vec<String>,
    /// Parameter labels (LaTeX-style, from .paramnames file).
    pub param_labels: Vec<String>,
    /// Sample weights (column 0 of chain files).
    pub weights: Vec<f64>,
    /// Negative log-likelihood values (column 1 of chain files).
    pub neg_log_like: Vec<f64>,
    /// Parameter values: param_values[sample_idx][param_idx].
    pub param_values: Vec<Vec<f64>>,
}

/// Parse a CosmoMC .paramnames file.
///
/// Format: one line per parameter, `name\tlatex_label` or `name  latex_label`.
/// Lines starting with '#' are comments.
pub fn parse_paramnames(path: &Path) -> Result<(Vec<String>, Vec<String>), FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read paramnames: {}", e)))?;

    let mut names = Vec::new();
    let mut labels = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Split on first whitespace: name then label
        let mut parts = trimmed.splitn(2, ['\t', ' ']);
        let name = parts.next().unwrap_or("").trim().to_string();
        let label = parts.next().unwrap_or("").trim().to_string();
        if !name.is_empty() {
            names.push(name);
            labels.push(label);
        }
    }

    Ok((names, labels))
}

/// Parse one or more CosmoMC chain files into a unified chain.
///
/// Chain file format: whitespace-delimited, no header.
/// Column 0 = weight, column 1 = -log(like), columns 2..N = parameters.
pub fn parse_cosmomc_chains(paramnames: &Path, chains: &[&Path]) -> Result<McmcChain, FetchError> {
    let (param_names, param_labels) = parse_paramnames(paramnames)?;
    let n_params = param_names.len();

    let mut weights = Vec::new();
    let mut neg_log_like = Vec::new();
    let mut param_values = Vec::new();

    for chain_path in chains {
        let content = std::fs::read_to_string(chain_path).map_err(|e| {
            FetchError::Validation(format!("Read chain {}: {}", chain_path.display(), e))
        })?;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let fields: Vec<f64> = trimmed
                .split_whitespace()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();

            // Need at least weight + neg_log_like + some params
            if fields.len() < 3 {
                continue;
            }

            weights.push(fields[0]);
            neg_log_like.push(fields[1]);

            // Take up to n_params parameter values
            let params: Vec<f64> = fields[2..].iter().take(n_params).copied().collect();
            param_values.push(params);
        }
    }

    Ok(McmcChain {
        param_names,
        param_labels,
        weights,
        neg_log_like,
        param_values,
    })
}

impl McmcChain {
    /// Number of samples in the chain.
    pub fn n_samples(&self) -> usize {
        self.weights.len()
    }

    /// Total effective sample count (sum of weights).
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Find the column index for a parameter name.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.param_names.iter().position(|n| n == name)
    }

    /// Weighted mean of a parameter by column index.
    pub fn weighted_mean(&self, param_idx: usize) -> Option<f64> {
        if param_idx >= self.param_names.len() {
            return None;
        }

        let total_w: f64 = self.weights.iter().sum();
        if total_w <= 0.0 {
            return None;
        }

        let mean: f64 = self
            .weights
            .iter()
            .zip(self.param_values.iter())
            .filter_map(|(w, params)| params.get(param_idx).map(|v| w * v))
            .sum::<f64>()
            / total_w;

        Some(mean)
    }

    /// Weighted standard deviation of a parameter by column index.
    pub fn weighted_std(&self, param_idx: usize) -> Option<f64> {
        let mean = self.weighted_mean(param_idx)?;
        let total_w: f64 = self.weights.iter().sum();
        if total_w <= 0.0 {
            return None;
        }

        let var: f64 = self
            .weights
            .iter()
            .zip(self.param_values.iter())
            .filter_map(|(w, params)| params.get(param_idx).map(|v| w * (v - mean).powi(2)))
            .sum::<f64>()
            / total_w;

        Some(var.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn make_test_files(dir: &Path) -> (PathBuf, PathBuf) {
        let paramnames = dir.join("test.paramnames");
        let chain = dir.join("test_1.txt");

        let mut pf = std::fs::File::create(&paramnames).unwrap();
        writeln!(pf, "omegabh2\t\\Omega_b h^2").unwrap();
        writeln!(pf, "omegach2\t\\Omega_c h^2").unwrap();
        writeln!(pf, "theta\t100\\theta_{{MC}}").unwrap();

        let mut cf = std::fs::File::create(&chain).unwrap();
        // weight  -loglike  omegabh2  omegach2  theta
        writeln!(cf, "1.0 2.5 0.02237 0.1200 1.04092").unwrap();
        writeln!(cf, "2.0 2.3 0.02240 0.1195 1.04095").unwrap();
        writeln!(cf, "1.0 2.4 0.02235 0.1205 1.04090").unwrap();

        (paramnames, chain)
    }

    #[test]
    fn test_parse_paramnames() {
        let dir = std::env::temp_dir().join("mcmc_test_paramnames");
        std::fs::create_dir_all(&dir).unwrap();
        let (pn, _) = make_test_files(&dir);

        let (names, labels) = parse_paramnames(&pn).unwrap();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "omegabh2");
        assert_eq!(labels[0], "\\Omega_b h^2");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_weighted_mean() {
        let dir = std::env::temp_dir().join("mcmc_test_wmean");
        std::fs::create_dir_all(&dir).unwrap();
        let (pn, chain) = make_test_files(&dir);

        let mc = parse_cosmomc_chains(&pn, &[chain.as_path()]).unwrap();
        assert_eq!(mc.n_samples(), 3);

        // Weighted mean of omegabh2: (1*0.02237 + 2*0.02240 + 1*0.02235) / 4
        let mean = mc.weighted_mean(0).unwrap();
        let expected = (1.0 * 0.02237 + 2.0 * 0.02240 + 1.0 * 0.02235) / 4.0;
        assert!(
            (mean - expected).abs() < 1e-10,
            "mean={} expected={}",
            mean,
            expected
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_param_index() {
        let dir = std::env::temp_dir().join("mcmc_test_pidx");
        std::fs::create_dir_all(&dir).unwrap();
        let (pn, chain) = make_test_files(&dir);

        let mc = parse_cosmomc_chains(&pn, &[chain.as_path()]).unwrap();
        assert_eq!(mc.param_index("omegach2"), Some(1));
        assert_eq!(mc.param_index("theta"), Some(2));
        assert_eq!(mc.param_index("nonexistent"), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
