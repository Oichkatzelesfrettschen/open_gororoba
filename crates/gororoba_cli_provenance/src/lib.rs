//! gororoba_cli_provenance -- slim provenance and download recovery operator binaries.
//!
//! The library owns bounded repository-verification mechanisms shared by the
//! provenance operator binaries.

pub mod finite_frontier;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        // Smoke test: the crate and its dependency graph compile.
        let _ = ();
    }
}
