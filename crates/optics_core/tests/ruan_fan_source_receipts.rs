//! Source byte identities and the curve-fitting method stated in the retained paper.

use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf};

fn root() -> Result<PathBuf, Box<dyn Error>> {
    let current = std::env::current_dir()?;
    current
        .ancestors()
        .find(|path| path.join("crates/optics_core/Cargo.toml").is_file())
        .map(PathBuf::from)
        .ok_or_else(|| "launch source-receipt tests from the checkout".into())
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn admitted_archive_tex_and_pdf_have_distinct_verified_identities() -> Result<(), Box<dyn Error>> {
    let directory =
        root()?.join("data/output/audit/claim-family-evidence-adjudication/optics-replay");
    for (file, expected) in [
        (
            "arxiv-source.download",
            "086721ff3a9a96a32d73bfe453be68026461472b4f751d1293ac6b2bdaaeba75",
        ),
        (
            "Fano_Scattering.tex",
            "625f2dc473e5d50abba49daa3d1382ee9bd70b17746cd9452bfaf9b81c70bc28",
        ),
        (
            "ruan-fan-0909.3323v2.pdf",
            "a355dc5a9358d05e6eeae3475c4722a37fb3d521fa457e6aac474d71a06d5c9a",
        ),
    ] {
        assert_eq!(expected.len(), 64);
        assert_eq!(hash(&fs::read(directory.join(file))?), expected);
    }
    let runner = include_str!("../src/bin/p2b_ruan_fan_reproduction.rs");
    let legacy = runner
        .lines()
        .find(|line| line.starts_with("const SOURCE_TEX_SHA256:"))
        .and_then(|line| line.split('"').nth(1))
        .ok_or("historical source-hash declaration missing")?;
    assert_eq!(legacy.len(), 61);
    assert_ne!(
        legacy,
        hash(&fs::read(directory.join("Fano_Scattering.tex"))?)
    );
    Ok(())
}

#[test]
fn source_method_and_rounded_landmarks_are_explicit() -> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(root()?.join(
        "data/output/audit/claim-family-evidence-adjudication/optics-replay/Fano_Scattering.tex",
    ))?;
    let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(normalized.contains("We use the above theory to fit these curves."));
    assert!(normalized.contains(
        "The circles are from the Lorentz-Mie method, and the solid lines are the theoretical fit"
    ));
    assert!(normalized.contains(r"$\omega=0.2282\omega_p$"));
    assert!(normalized.contains(r"$0.03(2\lambda/\pi)$"));
    assert!(normalized.contains(r"$0.32(2\lambda/\pi)$"));
    Ok(())
}
