use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct PathionToyLayer {
    pub layer_id: usize,
    pub pathion_indices: String,
    pub diag_abs_mean: String,
    pub n_real: String,
    pub n_imag: String,
    pub thickness_nm: String,
    pub tmm_absorptance: String,
    pub degeneracy_note: String,
}

pub fn build_pathion_diagonal(dim: usize) -> Result<Vec<f64>, String> {
    if dim != 32 {
        return Err("C-053 toy mapping is fixed to dim=32".to_string());
    }
    Ok((0..dim)
        .map(|idx| if idx % 2 == 0 { 1.0 } else { -1.0 })
        .collect())
}

pub fn diagonal_to_layers(
    diagonal: &[f64],
    group_size: usize,
) -> Result<Vec<PathionToyLayer>, String> {
    if group_size == 0 {
        return Err("group_size must be positive".to_string());
    }
    if !diagonal.len().is_multiple_of(group_size) {
        return Err("diagonal length must be divisible by group_size".to_string());
    }

    let mut layers = Vec::new();
    for layer_id in 0..(diagonal.len() / group_size) {
        let start = layer_id * group_size;
        let stop = start + group_size;
        let block = &diagonal[start..stop];
        let magnitude = block.iter().map(|value| value.abs()).sum::<f64>() / block.len() as f64;
        let n_real = 1.75 + 0.0 * magnitude;
        let n_imag = 0.05;
        let thickness_nm = 40.0;
        let absorptance = 1.0 - ((n_real - 1.0) / (n_real + 1.0)).powi(2);

        layers.push(PathionToyLayer {
            layer_id,
            pathion_indices: format!("{}-{}", start, stop - 1),
            diag_abs_mean: format_fixed6(magnitude),
            n_real: format_fixed6(n_real),
            n_imag: format_fixed6(n_imag),
            thickness_nm: format_fixed6(thickness_nm),
            tmm_absorptance: format_fixed6(absorptance),
            degeneracy_note: "diagonal-only toy map: uniform layer response".to_string(),
        });
    }
    Ok(layers)
}

pub fn default_c053_layers() -> Result<Vec<PathionToyLayer>, String> {
    let diagonal = build_pathion_diagonal(32)?;
    diagonal_to_layers(&diagonal, 4)
}

pub fn write_c053_summary(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("path {} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("creating parent directory {}: {}", parent.display(), err))?;

    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("creating CSV {}: {}", path.display(), err))?;
    for row in default_c053_layers()? {
        writer
            .serialize(row)
            .map_err(|err| format!("serializing row into {}: {}", path.display(), err))?;
    }
    writer
        .flush()
        .map_err(|err| format!("flushing CSV {}: {}", path.display(), err))?;
    Ok(())
}

fn format_fixed6(value: f64) -> String {
    format!("{value:.6}")
}

#[cfg(test)]
mod tests {
    use super::{build_pathion_diagonal, default_c053_layers, diagonal_to_layers};

    #[test]
    fn fixed_diagonal_is_sign_alternating() {
        let diagonal = build_pathion_diagonal(32).expect("default diagonal");
        assert_eq!(diagonal.len(), 32);
        assert_eq!(diagonal[0], 1.0);
        assert_eq!(diagonal[1], -1.0);
        assert_eq!(diagonal[30], 1.0);
        assert_eq!(diagonal[31], -1.0);
    }

    #[test]
    fn layers_are_uniform_for_diagonal_only_mapping() {
        let diagonal = build_pathion_diagonal(32).expect("default diagonal");
        let rows = diagonal_to_layers(&diagonal, 4).expect("default rows");
        assert_eq!(rows.len(), 8);
        assert!(rows.iter().all(|row| row.diag_abs_mean == "1.000000"));
        assert!(rows.iter().all(|row| row.n_real == "1.750000"));
        assert!(rows.iter().all(|row| row.n_imag == "0.050000"));
        assert!(rows.iter().all(|row| row.thickness_nm == "40.000000"));
        assert!(rows.iter().all(|row| row.tmm_absorptance == "0.925620"));
    }

    #[test]
    fn default_rows_wrap_generator() {
        let rows = default_c053_layers().expect("default rows");
        assert_eq!(
            rows.first().expect("at least one row").pathion_indices,
            "0-3"
        );
        assert_eq!(
            rows.last().expect("at least one row").pathion_indices,
            "28-31"
        );
    }
}
