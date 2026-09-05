//! Host-observed mirror status and retained-path classification.

use super::{
    HashMap, LinkObservation, Path, RetentionSet, UnifiedArtifact, artifact_retention, dedupe,
};

pub(super) fn first_local_identity(
    artifact: &UnifiedArtifact,
    repo_root: &Path,
) -> Option<artifact_retention::FileIdentity> {
    artifact
        .downloaded_paths
        .iter()
        .chain(artifact.host_only_paths.iter())
        .find_map(|path| artifact_retention::file_identity(&repo_root.join(path)))
}

pub(super) fn classify_paths_and_mirrors(
    artifact: &mut UnifiedArtifact,
    observations: &HashMap<String, Vec<LinkObservation>>,
    download_map: &HashMap<String, Vec<String>>,
    retention: &RetentionSet,
) {
    let mut working = Vec::new();
    let mut working_pdf = Vec::new();
    let mut nonworking = Vec::new();
    let mut unverified = Vec::new();
    let mut downloaded = artifact.local_paths.clone();

    for url in &artifact.links {
        let obs_list = observations.get(url).cloned().unwrap_or_default();
        let statuses = obs_list
            .iter()
            .map(|obs| obs.status.as_str())
            .collect::<Vec<_>>();
        let has_pdf_ok = obs_list.iter().any(|obs| obs.status == "pdf_ok");
        let has_ok = obs_list.iter().any(|obs| obs.status == "ok_nonpdf");
        let has_nonworking = statuses.iter().any(|status| {
            (status.starts_with("http_")
                && !matches!(
                    *status,
                    "http_200" | "http_201" | "http_202" | "http_203" | "http_204"
                ))
                || *status == "failed"
        });
        if has_pdf_ok {
            working.push(url.clone());
            working_pdf.push(url.clone());
        } else if has_ok {
            working.push(url.clone());
        } else if has_nonworking {
            nonworking.push(url.clone());
        } else {
            unverified.push(url.clone());
        }
        if let Some(paths) = download_map.get(url) {
            downloaded.extend(paths.clone());
        }
    }

    artifact.working_mirrors = dedupe(working);
    artifact.working_pdf_mirrors = dedupe(working_pdf);
    artifact.nonworking_mirrors = dedupe(nonworking);
    artifact.unverified_mirrors = dedupe(unverified);

    // Split the observed paths by the retention predicate. A path git
    // tracks is repository truth and keeps the `downloaded` status; a path
    // that exists only in this checkout is host state, so it leaves the
    // registry row and moves to the materialization manifest.
    let observed_paths = dedupe(downloaded);
    let (retained, host_only): (Vec<String>, Vec<String>) = observed_paths
        .into_iter()
        .partition(|path| retention.contains(path));
    artifact.downloaded_paths = retained;
    artifact.host_only_paths = host_only;
}
