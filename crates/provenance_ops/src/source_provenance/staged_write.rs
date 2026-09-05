// SPDX-License-Identifier: GPL-2.0-or-later
//
// Two-phase writer for the generated artifact provenance registries.

use anyhow::{Context, Result, bail};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Fraction of rows an output may lose before the write is refused.
pub const DEFAULT_SHRINK_THRESHOLD: f64 = 0.05;

#[derive(Clone, Debug)]
pub struct ShrinkPolicy {
    pub max_shrink_fraction: f64,
    pub allow_shrink: bool,
}

impl Default for ShrinkPolicy {
    fn default() -> Self {
        Self {
            max_shrink_fraction: DEFAULT_SHRINK_THRESHOLD,
            allow_shrink: false,
        }
    }
}

impl ShrinkPolicy {
    /// Accepts any row count. Used by callers that own no previous file.
    pub fn permissive() -> Self {
        Self {
            max_shrink_fraction: 1.0,
            allow_shrink: true,
        }
    }
}

/// Row counts observed for one output across the two-phase write.
#[derive(Clone, Debug)]
pub struct RowCountReport {
    pub path: PathBuf,
    pub before: Option<usize>,
    pub after: usize,
}

struct StagedOutput {
    path: PathBuf,
    contents: String,
    row_marker: &'static str,
}

/// Renders every output to a sibling temp path, parses it back, compares its
/// row count against the file it replaces, and only then renames the whole set
/// in one pass. A failure at any point removes the temp files, so the originals
/// stay byte-identical.
#[derive(Default)]
pub struct StagedWriteSet {
    outputs: Vec<StagedOutput>,
}

impl StagedWriteSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage(&mut self, path: &Path, contents: String, row_marker: &'static str) {
        self.outputs.push(StagedOutput {
            path: path.to_path_buf(),
            contents,
            row_marker,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    pub fn commit(self, policy: &ShrinkPolicy) -> Result<Vec<RowCountReport>> {
        let mut reports = Vec::new();
        let mut temp_paths: Vec<(PathBuf, PathBuf)> = Vec::new();
        let result = self.stage_all(policy, &mut reports, &mut temp_paths);
        if let Err(error) = result {
            for (temp, _) in &temp_paths {
                let _ = fs::remove_file(temp);
            }
            return Err(error);
        }
        for (temp, final_path) in &temp_paths {
            fs::rename(temp, final_path)
                .with_context(|| format!("rename into {}", final_path.display()))?;
        }
        Ok(reports)
    }

    fn stage_all(
        &self,
        policy: &ShrinkPolicy,
        reports: &mut Vec<RowCountReport>,
        temp_paths: &mut Vec<(PathBuf, PathBuf)>,
    ) -> Result<()> {
        for output in &self.outputs {
            toml::from_str::<toml::Value>(&output.contents)
                .with_context(|| format!("staged output does not parse: {}", output.path.display()))?;
            let after = count_rows(&output.contents, output.row_marker);
            let before = if output.path.exists() {
                let previous = fs::read_to_string(&output.path)
                    .with_context(|| format!("read {}", output.path.display()))?;
                Some(count_rows(&previous, output.row_marker))
            } else {
                None
            };
            if let Some(before) = before {
                check_shrink(&output.path, before, after, policy)?;
            }
            reports.push(RowCountReport {
                path: output.path.clone(),
                before,
                after,
            });
            let parent = output
                .path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            fs::create_dir_all(&parent).with_context(|| format!("create {}", parent.display()))?;
            let file_name = output
                .path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".to_string());
            let temp = parent.join(format!(".{file_name}.staged"));
            fs::write(&temp, &output.contents)
                .with_context(|| format!("write {}", temp.display()))?;
            temp_paths.push((temp, output.path.clone()));
        }
        Ok(())
    }
}

fn check_shrink(path: &Path, before: usize, after: usize, policy: &ShrinkPolicy) -> Result<()> {
    if policy.allow_shrink || before == 0 || after >= before {
        return Ok(());
    }
    let lost = before - after;
    let fraction = lost as f64 / before as f64;
    if fraction > policy.max_shrink_fraction {
        bail!(
            "{} would lose {lost} of {before} rows ({:.2}% > {:.2}% threshold); \
             pass --allow-shrink to accept the loss",
            path.display(),
            fraction * 100.0,
            policy.max_shrink_fraction * 100.0
        );
    }
    Ok(())
}

fn count_rows(text: &str, row_marker: &str) -> usize {
    text.lines()
        .filter(|line| line.trim() == row_marker)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_write_refuses_a_shrink_beyond_the_threshold() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("lane.toml");
        let before = (0..100)
            .map(|index| format!("[[artifact]]\nid = \"A-{index}\"\n"))
            .collect::<String>();
        fs::write(&path, &before).expect("seed");
        let after = (0..50)
            .map(|index| format!("[[artifact]]\nid = \"A-{index}\"\n"))
            .collect::<String>();
        let mut set = StagedWriteSet::new();
        set.stage(&path, after, "[[artifact]]");
        let error = set
            .commit(&ShrinkPolicy::default())
            .expect_err("50 percent loss must be refused");
        assert!(error.to_string().contains("50 of 100 rows"), "{error}");
        assert_eq!(fs::read_to_string(&path).expect("read"), before);
    }

    #[test]
    fn staged_write_accepts_a_shrink_under_the_threshold() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("lane.toml");
        let before = (0..100)
            .map(|index| format!("[[artifact]]\nid = \"A-{index}\"\n"))
            .collect::<String>();
        fs::write(&path, &before).expect("seed");
        let after = (0..97)
            .map(|index| format!("[[artifact]]\nid = \"A-{index}\"\n"))
            .collect::<String>();
        let mut set = StagedWriteSet::new();
        set.stage(&path, after, "[[artifact]]");
        let reports = set.commit(&ShrinkPolicy::default()).expect("3 percent loss");
        assert_eq!(reports[0].before, Some(100));
        assert_eq!(reports[0].after, 97);
    }

    #[test]
    fn staged_write_leaves_every_original_untouched_when_one_output_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = dir.path().join("good.toml");
        let bad = dir.path().join("bad.toml");
        fs::write(&good, "[[artifact]]\nid = \"keep\"\n").expect("seed good");
        fs::write(&bad, "[[artifact]]\nid = \"keep\"\n").expect("seed bad");
        let mut set = StagedWriteSet::new();
        set.stage(&good, "[[artifact]]\nid = \"new\"\n".to_string(), "[[artifact]]");
        set.stage(&bad, "id = = broken".to_string(), "[[artifact]]");
        let error = set
            .commit(&ShrinkPolicy::default())
            .expect_err("unparsable output must abort the set");
        assert!(error.to_string().contains("does not parse"), "{error}");
        assert_eq!(
            fs::read_to_string(&good).expect("read good"),
            "[[artifact]]\nid = \"keep\"\n"
        );
        assert_eq!(
            fs::read_to_string(&bad).expect("read bad"),
            "[[artifact]]\nid = \"keep\"\n"
        );
        assert!(!dir.path().join(".good.toml.staged").exists());
    }
}
