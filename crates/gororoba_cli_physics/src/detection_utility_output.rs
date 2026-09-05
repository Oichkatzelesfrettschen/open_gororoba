//! Publish an owned utility bundle with the report as its completion marker.

use anyhow::{Context, Result};
use std::{fs, path::Path};

/// Publish three artifacts into a new directory and remove the owned directory on error.
///
/// Existing output paths remain untouched. The report appears after both data artifacts;
/// interrupted processes may leave an incomplete directory without a completion report.
pub fn publish_bundle(
    out_dir: &Path,
    report_bytes: &[u8],
    csv_bytes: &[u8],
    svg_bytes: &[u8],
) -> Result<()> {
    publish_with_writer(
        out_dir,
        report_bytes,
        csv_bytes,
        svg_bytes,
        |path, bytes| fs::write(path, bytes),
    )
}

fn publish_with_writer(
    out_dir: &Path,
    report_bytes: &[u8],
    csv_bytes: &[u8],
    svg_bytes: &[u8],
    mut write: impl FnMut(&Path, &[u8]) -> std::io::Result<()>,
) -> Result<()> {
    fs::create_dir(out_dir).context("output directory must be new and its parent must exist")?;
    let result = (|| -> Result<()> {
        write(&out_dir.join("frontier.csv"), csv_bytes).context("write utility CSV")?;
        write(&out_dir.join("frontier.svg"), svg_bytes).context("write utility SVG")?;
        let pending_report = out_dir.join(".report.json.pending");
        write(&pending_report, report_bytes).context("stage utility completion report")?;
        fs::rename(&pending_report, out_dir.join("report.json"))
            .context("publish utility completion report")?;
        Ok(())
    })();
    if let Err(error) = result {
        return match fs::remove_dir_all(out_dir) {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(error.context(format!(
                "failed to clean newly owned output directory {}: {cleanup_error}",
                out_dir.display()
            ))),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
            let parent = std::env::temp_dir().join(format!(
                "gororoba-utility-output-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&parent).unwrap();
            Self(parent)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn assert_complete(out_dir: &Path) {
        assert_eq!(fs::read(out_dir.join("report.json")).unwrap(), b"report");
        assert_eq!(fs::read(out_dir.join("frontier.csv")).unwrap(), b"csv");
        assert_eq!(fs::read(out_dir.join("frontier.svg")).unwrap(), b"svg");
        assert_eq!(fs::read_dir(out_dir).unwrap().count(), 3);
    }

    #[test]
    fn completion_report_appears_only_after_both_artifacts() {
        let parent = TestDirectory::new();
        let out_dir = parent.0.join("bundle");
        let mut writes = Vec::new();
        publish_with_writer(&out_dir, b"report", b"csv", b"svg", |path, bytes| {
            assert!(!out_dir.join("report.json").exists());
            writes.push(path.file_name().unwrap().to_string_lossy().into_owned());
            fs::write(path, bytes)
        })
        .unwrap();
        assert_eq!(
            writes,
            ["frontier.csv", "frontier.svg", ".report.json.pending"]
        );
        assert_complete(&out_dir);
    }

    #[test]
    fn partial_second_or_final_write_failure_removes_owned_bundle_and_allows_retry() {
        for failed_write in [2, 3] {
            let parent = TestDirectory::new();
            let out_dir = parent.0.join("bundle");
            let sibling = parent.0.join("preserved");
            fs::write(&sibling, b"unrelated").unwrap();
            let mut write_count = 0;
            let result = publish_with_writer(&out_dir, b"report", b"csv", b"svg", |path, bytes| {
                write_count += 1;
                if write_count == failed_write {
                    fs::write(path, b"partial")?;
                    return Err(std::io::Error::other("injected write failure"));
                }
                fs::write(path, bytes)
            });
            assert!(result.is_err());
            assert!(!out_dir.exists());
            assert_eq!(fs::read(&sibling).unwrap(), b"unrelated");
            publish_bundle(&out_dir, b"report", b"csv", b"svg").unwrap();
            assert_complete(&out_dir);
        }
    }

    #[test]
    fn rename_failure_removes_owned_bundle_and_allows_retry() {
        let parent = TestDirectory::new();
        let out_dir = parent.0.join("bundle");
        let result = publish_with_writer(&out_dir, b"report", b"csv", b"svg", |path, bytes| {
            fs::write(path, bytes)?;
            if path.file_name().unwrap() == ".report.json.pending" {
                fs::create_dir(out_dir.join("report.json"))?;
            }
            Ok(())
        });
        assert!(result.is_err());
        assert!(!out_dir.exists());
        publish_bundle(&out_dir, b"report", b"csv", b"svg").unwrap();
        assert_complete(&out_dir);
    }

    #[test]
    fn existing_empty_directory_nonempty_directory_and_file_remain_unchanged() {
        let parent = TestDirectory::new();
        let empty = parent.0.join("empty");
        fs::create_dir(&empty).unwrap();
        let nonempty = parent.0.join("nonempty");
        fs::create_dir(&nonempty).unwrap();
        fs::write(nonempty.join("report.json"), b"historical").unwrap();
        let file = parent.0.join("file");
        fs::write(&file, b"retained").unwrap();
        for existing in [&empty, &nonempty, &file] {
            assert!(
                publish_with_writer(existing, b"report", b"csv", b"svg", |_, _| {
                    panic!("existing output must fail before writing")
                })
                .is_err()
            );
        }
        assert_eq!(fs::read_dir(empty).unwrap().count(), 0);
        assert_eq!(
            fs::read(nonempty.join("report.json")).unwrap(),
            b"historical"
        );
        assert_eq!(fs::read_dir(nonempty).unwrap().count(), 1);
        assert_eq!(fs::read(file).unwrap(), b"retained");
    }
}
