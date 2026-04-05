#[path = "../../gororoba_cli_data/src/source_provenance.rs"]
pub mod source_provenance;

pub mod data_ingest;

#[cfg(test)]
mod tests {
    use super::source_provenance;

    #[test]
    fn default_repo_root_exists() {
        let root = source_provenance::default_repo_root();
        assert!(root.exists(), "repo root {root:?} should exist");
        assert!(
            root.join("Cargo.toml").exists(),
            "repo root should contain Cargo.toml"
        );
    }
}
