//! Spatial correlation analysis for frustration-topology coupling.
//!
//! Provides tools to partition 3D grids into subregions and compute
//! correlations between local frustration density and other spatially-varying
//! observables (Betti numbers, velocity magnitudes, viscosity).
//!
//! Key functions:
//! - `grid_partition_3d`: Partition grid into cubic subregions
//! - `regional_means`: Compute mean of a field within each subregion
//! - `spearman_correlation`: Rank-based correlation (robust to nonlinearity)
//! - `pearson_correlation`: Standard linear correlation

/// Partition a 3D grid into roughly equal cubic subregions.
///
/// Divides each axis into `n_sub` segments, producing `n_sub^3` subregions.
/// Each subregion is a list of `(x, y, z)` grid coordinates belonging to it.
///
/// # Arguments
/// * `nx`, `ny`, `nz` - Grid dimensions
/// * `n_sub` - Number of subdivisions per axis (total regions = n_sub^3)
///
/// # Returns
/// Vector of subregion cell lists, each containing `(x, y, z)` tuples.
pub fn grid_partition_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    n_sub: usize,
) -> Vec<Vec<(usize, usize, usize)>> {
    assert!(n_sub > 0, "n_sub must be > 0");
    assert!(nx > 0 && ny > 0 && nz > 0, "grid dimensions must be > 0");

    let n_regions = n_sub * n_sub * n_sub;
    let mut regions: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_regions];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let rx = (x * n_sub / nx).min(n_sub - 1);
                let ry = (y * n_sub / ny).min(n_sub - 1);
                let rz = (z * n_sub / nz).min(n_sub - 1);
                let region_idx = rz * n_sub * n_sub + ry * n_sub + rx;
                regions[region_idx].push((x, y, z));
            }
        }
    }

    regions
}

/// Compute mean value of a field within each subregion.
///
/// # Arguments
/// * `field` - Flat field of length `nx * ny * nz` (z-major ordering: z*(nx*ny) + y*nx + x)
/// * `regions` - Subregion partitions from `grid_partition_3d`
/// * `nx`, `ny` - Grid x and y dimensions (needed for linearization)
///
/// # Returns
/// Vector of per-region mean values.
pub fn regional_means(
    field: &[f64],
    regions: &[Vec<(usize, usize, usize)>],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    regions
        .iter()
        .map(|cells| {
            if cells.is_empty() {
                return 0.0;
            }
            let sum: f64 = cells
                .iter()
                .map(|&(x, y, z)| field[z * (nx * ny) + y * nx + x])
                .sum();
            sum / cells.len() as f64
        })
        .collect()
}

/// Compute Pearson product-moment correlation coefficient.
///
/// Returns r in [-1, 1]. Returns 0.0 if either vector has zero variance.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;

    let x_mean: f64 = x[..n].iter().sum::<f64>() / n_f;
    let y_mean: f64 = y[..n].iter().sum::<f64>() / n_f;

    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;

    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }

    let denom = (sxx * syy).sqrt();
    if denom < 1e-30 {
        0.0
    } else {
        (sxy / denom).clamp(-1.0, 1.0)
    }
}

/// Compute Spearman rank correlation coefficient.
///
/// Converts values to ranks (average ranks for ties), then computes
/// Pearson correlation on ranks. More robust than Pearson for nonlinear
/// monotonic relationships.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let rank_x = to_ranks(&x[..n]);
    let rank_y = to_ranks(&y[..n]);
    pearson_correlation(&rank_x, &rank_y)
}

/// Convert values to ranks (1-based), with average rank for ties.
fn to_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for tied group [i..=j]
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

/// Compute velocity magnitude field from velocity vectors.
///
/// Takes a slice of `[f64; 3]` velocity vectors and returns `|u|` at each point.
pub fn velocity_magnitude_field(velocities: &[[f64; 3]]) -> Vec<f64> {
    velocities
        .iter()
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .collect()
}

/// Summary of spatial correlation analysis for one parameter set.
#[derive(Debug, Clone)]
pub struct SpatialCorrelationResult {
    /// Number of subregions used
    pub n_regions: usize,
    /// Spearman correlation between regional frustration and observable
    pub spearman_r: f64,
    /// Pearson correlation between regional frustration and observable
    pub pearson_r: f64,
    /// Per-region mean frustration values
    pub regional_frustration: Vec<f64>,
    /// Per-region mean observable values
    pub regional_observable: Vec<f64>,
}

/// Compute spatial correlation between frustration density and an observable field.
///
/// Partitions the grid, computes regional means of both fields, and returns
/// both Pearson and Spearman correlations.
///
/// # Arguments
/// * `frustration_field` - Frustration density, length `nx*ny*nz`
/// * `observable_field` - Any scalar observable, length `nx*ny*nz`
/// * `nx`, `ny`, `nz` - Grid dimensions
/// * `n_sub` - Subdivisions per axis
pub fn spatial_correlation(
    frustration_field: &[f64],
    observable_field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    n_sub: usize,
) -> SpatialCorrelationResult {
    let regions = grid_partition_3d(nx, ny, nz, n_sub);
    let regional_frustration = regional_means(frustration_field, &regions, nx, ny);
    let regional_observable = regional_means(observable_field, &regions, nx, ny);

    let spearman_r = spearman_correlation(&regional_frustration, &regional_observable);
    let pearson_r = pearson_correlation(&regional_frustration, &regional_observable);

    SpatialCorrelationResult {
        n_regions: regions.len(),
        spearman_r,
        pearson_r,
        regional_frustration,
        regional_observable,
    }
}

/// Compute Jaccard overlap between two 3D point clouds.
///
/// Two points are considered "shared" if they are within `tolerance` Euclidean
/// distance. Returns the fraction of points in the union that appear in both
/// clouds (Jaccard index in [0, 1]).
///
/// This diagnoses whether two top-k extractions selected the same spatial
/// locations. W2=0 with high overlap means the transform preserves extrema
/// ordering; W2=0 with low overlap would indicate accidental topological
/// equivalence from different spatial regions (unlikely but worth checking).
pub fn point_cloud_overlap(a: &[[f64; 3]], b: &[[f64; 3]], tolerance: f64) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let tol2 = tolerance * tolerance;
    let mut matched_b = vec![false; b.len()];
    let mut n_matched = 0_usize;

    for pa in a {
        for (j, pb) in b.iter().enumerate() {
            if matched_b[j] {
                continue;
            }
            let d2 = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);
            if d2 <= tol2 {
                matched_b[j] = true;
                n_matched += 1;
                break;
            }
        }
    }

    // Jaccard = intersection / union = matched / (|a| + |b| - matched)
    let union = a.len() + b.len() - n_matched;
    if union == 0 {
        1.0
    } else {
        n_matched as f64 / union as f64
    }
}

/// Coefficient of variation: std / |mean|.
///
/// Measures relative dispersion. A field with small CV has been compressed
/// into a narrow band (e.g., sigmoid saturation), explaining why spatial
/// correlation degrades despite identical point cloud topology.
/// Returns 0.0 for constant data or zero mean.
pub fn coefficient_of_variation(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean = x.iter().sum::<f64>() / n_f;
    if mean.abs() < 1e-30 {
        return 0.0;
    }
    let variance = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
    variance.sqrt() / mean.abs()
}

/// Nonlinearity index: measures how much a transform deviates from linearity.
///
/// Computed as `Spearman_r - Pearson_r`. For a linear monotonic transform,
/// both correlations are equal (index = 0). For a nonlinear monotonic transform
/// (e.g., sigmoid saturation), Spearman captures the rank ordering perfectly
/// while Pearson degrades, producing a positive index.
///
/// Interpretation:
/// - 0.0: perfectly linear relationship
/// - 0.0-0.1: approximately linear
/// - 0.1-0.3: moderately nonlinear
/// - >0.3: strongly nonlinear (e.g., saturating sigmoid)
pub fn nonlinearity_index(x: &[f64], y: &[f64]) -> f64 {
    let sp = spearman_correlation(x, y);
    let pe = pearson_correlation(x, y);
    sp.abs() - pe.abs()
}

/// Dynamic range ratio: ratio of standard deviations between two fields.
///
/// Measures how much a transform compresses or expands the field's variability.
/// A sigmoid with saturation produces ratio << 1 (compression). A power-law
/// with exponent > 1 produces ratio > 1 (expansion in the tail).
///
/// Returns std(observable) / std(frustration). Zero if frustration has no variance.
pub fn dynamic_range_ratio(frustration: &[f64], observable: &[f64]) -> f64 {
    let std_f = std_dev(frustration);
    let std_o = std_dev(observable);
    if std_f < 1e-30 {
        return 0.0;
    }
    std_o / std_f
}

fn std_dev(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean = x.iter().sum::<f64>() / n_f;
    let variance = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_partition_covers_all_cells() {
        let (nx, ny, nz) = (16, 16, 16);
        let n_sub = 2;
        let regions = grid_partition_3d(nx, ny, nz, n_sub);
        assert_eq!(regions.len(), 8); // 2^3

        let total_cells: usize = regions.iter().map(|r| r.len()).sum();
        assert_eq!(total_cells, nx * ny * nz);
    }

    #[test]
    fn test_grid_partition_non_cubic() {
        let (nx, ny, nz) = (10, 8, 6);
        let n_sub = 3;
        let regions = grid_partition_3d(nx, ny, nz, n_sub);
        assert_eq!(regions.len(), 27); // 3^3

        let total_cells: usize = regions.iter().map(|r| r.len()).sum();
        assert_eq!(total_cells, nx * ny * nz);

        // Every region should have at least 1 cell (for these dimensions)
        for r in &regions {
            assert!(!r.is_empty());
        }
    }

    #[test]
    fn test_regional_means_uniform() {
        let (nx, ny, nz) = (8, 8, 8);
        let n_sub = 2;
        let field = vec![0.375; nx * ny * nz];
        let regions = grid_partition_3d(nx, ny, nz, n_sub);
        let means = regional_means(&field, &regions, nx, ny);

        for &m in &means {
            assert!((m - 0.375).abs() < 1e-14);
        }
    }

    #[test]
    fn test_pearson_perfect_positive() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect();
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-12, "Expected r=1.0, got {}", r);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..10).map(|i| -3.0 * i as f64 + 50.0).collect();
        let r = pearson_correlation(&x, &y);
        assert!((r + 1.0).abs() < 1e-12, "Expected r=-1.0, got {}", r);
    }

    #[test]
    fn test_spearman_monotonic_nonlinear() {
        // x = 1..20, y = x^3 (nonlinear but perfectly monotonic)
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3)).collect();
        let rs = spearman_correlation(&x, &y);
        // Spearman should be 1.0 for any perfectly monotonic relationship
        assert!((rs - 1.0).abs() < 1e-12, "Expected rs=1.0, got {}", rs);
    }

    #[test]
    fn test_spatial_correlation_gradient() {
        // Construct a field where frustration increases linearly in x,
        // and observable also increases linearly in x => positive correlation
        let (nx, ny, nz) = (16, 16, 16);
        let n_cells = nx * ny * nz;
        let mut frustration = vec![0.0; n_cells];
        let mut observable = vec![0.0; n_cells];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    frustration[idx] = x as f64 / nx as f64;
                    observable[idx] = 0.5 + 0.3 * (x as f64 / nx as f64);
                }
            }
        }

        let result = spatial_correlation(&frustration, &observable, nx, ny, nz, 2);
        assert!(
            result.spearman_r > 0.9,
            "Expected strong positive correlation, got {}",
            result.spearman_r
        );
        assert!(
            result.pearson_r > 0.9,
            "Expected strong Pearson r, got {}",
            result.pearson_r
        );
    }

    #[test]
    fn test_velocity_magnitude_field() {
        let vels = vec![[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let mag = velocity_magnitude_field(&vels);
        assert!((mag[0] - 5.0).abs() < 1e-14);
        assert!((mag[1] - 0.0).abs() < 1e-14);
        assert!((mag[2] - 3.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_point_cloud_overlap_identical() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = a.clone();
        let j = point_cloud_overlap(&a, &b, 1e-10);
        assert!((j - 1.0).abs() < 1e-14, "Identical clouds should have Jaccard=1, got {}", j);
    }

    #[test]
    fn test_point_cloud_overlap_disjoint() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = vec![[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]];
        let j = point_cloud_overlap(&a, &b, 0.1);
        assert!(j.abs() < 1e-14, "Disjoint clouds should have Jaccard=0, got {}", j);
    }

    #[test]
    fn test_point_cloud_overlap_partial() {
        // 3 points in a, 3 in b, 2 shared
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let b = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]];
        let j = point_cloud_overlap(&a, &b, 1e-10);
        // intersection=2, union=3+3-2=4, Jaccard=2/4=0.5
        assert!((j - 0.5).abs() < 1e-14, "Expected Jaccard=0.5, got {}", j);
    }

    #[test]
    fn test_coefficient_of_variation_constant() {
        let x = vec![5.0; 100];
        let cv = coefficient_of_variation(&x);
        assert!(cv.abs() < 1e-14, "Constant data should have CV=0, got {}", cv);
    }

    #[test]
    fn test_coefficient_of_variation_spread() {
        // Known: x = [1, 2, 3, 4, 5], mean=3, std=sqrt(2.5)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cv = coefficient_of_variation(&x);
        let expected_cv = (2.5_f64).sqrt() / 3.0;
        assert!(
            (cv - expected_cv).abs() < 1e-12,
            "Expected CV={:.6}, got {:.6}",
            expected_cv,
            cv
        );
    }

    #[test]
    fn test_nonlinearity_index_linear() {
        // Linear relationship: nonlinearity should be ~0
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 7.0).collect();
        let nli = nonlinearity_index(&x, &y);
        assert!(nli.abs() < 1e-10, "Linear transform should have NLI=0, got {}", nli);
    }

    #[test]
    fn test_nonlinearity_index_saturating() {
        // Sigmoid-like saturation: x in [0,10], y = 1/(1+exp(-2*(x-5)))
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 1.0 / (1.0 + (-2.0 * (xi - 5.0)).exp()))
            .collect();
        let nli = nonlinearity_index(&x, &y);
        // Spearman should be ~1.0 (monotonic), Pearson < 1.0 (S-curve)
        assert!(nli > 0.0, "Sigmoid should have positive NLI, got {}", nli);
    }

    #[test]
    fn test_dynamic_range_ratio_identity() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ratio = dynamic_range_ratio(&x, &x);
        assert!((ratio - 1.0).abs() < 1e-12, "Same field should have ratio=1, got {}", ratio);
    }

    #[test]
    fn test_dynamic_range_ratio_compressed() {
        // observable = 0.1 * frustration -> ratio should be 0.1
        let f: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let o: Vec<f64> = f.iter().map(|&v| 0.1 * v).collect();
        let ratio = dynamic_range_ratio(&f, &o);
        assert!((ratio - 0.1).abs() < 1e-12, "Expected ratio=0.1, got {}", ratio);
    }
}
