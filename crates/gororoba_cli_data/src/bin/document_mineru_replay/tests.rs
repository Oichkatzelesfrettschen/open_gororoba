//! Unit and integration coverage for the document inventory.
//!
//! Fixtures are generated in-process; no test depends on the private external
//! corpus. PDFs are built with `lopdf` so page counts and parseability are real
//! rather than mocked.

use super::*;
use lopdf::{Document as PdfDocument, Object, dictionary};
use std::path::Path;

// ---- Fixture helpers -----------------------------------------------------

/// Write a genuinely parseable PDF with `pages` empty pages.
fn write_pdf(path: &Path, pages: usize) {
    let mut doc = PdfDocument::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut kids: Vec<Object> = Vec::new();
    for _ in 0..pages {
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
        });
        kids.push(page_id.into());
    }
    let count = kids.len() as i64;
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).expect("save fixture pdf");
}

/// Write a docpipe-style paper.toml with the given full_text and array counts.
fn write_extraction(dir: &Path, full_text: &str, equations: usize, tables: usize, figures: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let mut s = String::new();
    s.push_str("sections = []\n");
    s.push_str(&format!("equations = {}\n", empty_tables(equations)));
    s.push_str(&format!("tables = {}\n", empty_tables(tables)));
    s.push_str(&format!("figures = {}\n", empty_tables(figures)));
    s.push_str("full_text = ");
    s.push_str(&toml::Value::String(full_text.to_string()).to_string());
    s.push('\n');
    std::fs::write(dir.join("paper.toml"), s).unwrap();
}

fn empty_tables(n: usize) -> String {
    if n == 0 {
        return "[]".to_string();
    }
    let items: Vec<String> = (0..n).map(|_| "{}".to_string()).collect();
    format!("[{}]", items.join(", "))
}

fn base_args(root: &Path) -> InventoryArgs {
    InventoryArgs {
        ingest_manifest: root.join("MANIFEST.toml"),
        papers_manifest: root.join("corpus/MANIFEST.toml"),
        source_roots: vec![root.join("ingest").to_string_lossy().to_string()],
        extraction_roots: vec![root.join("extracted").to_string_lossy().to_string()],
        sources: root.join("SOURCES.toml"),
        out: root.join("out/inventory.toml"),
        only: None,
        include_corpus: false,
    }
}

fn write_manifest(root: &Path, entries: &[(&str, &Path)]) {
    let mut s = String::new();
    for (id, path) in entries {
        s.push_str("[[paper]]\n");
        s.push_str(&format!("id = {}\n", toml::Value::String((*id).to_string())));
        s.push_str(&format!(
            "local_pdf = {}\n\n",
            toml::Value::String(path.to_string_lossy().to_string())
        ));
    }
    std::fs::write(root.join("MANIFEST.toml"), s).unwrap();
}

fn read_out(args: &InventoryArgs) -> toml::Value {
    let text = std::fs::read_to_string(&args.out).unwrap();
    toml::from_str(&text).unwrap()
}

fn docs(v: &toml::Value) -> Vec<toml::Value> {
    v.get("document")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default()
}

fn routing_of(doc: &toml::Value) -> String {
    doc.get("routing").unwrap().as_str().unwrap().to_string()
}

// ---- Pure function tests -------------------------------------------------

#[test]
fn media_type_detected_from_bytes_not_suffix() {
    assert_eq!(detect_media_type(b"%PDF-1.7\n..."), MediaType::Pdf);
    assert_eq!(
        detect_media_type(b"<!DOCTYPE html><html><body>404</body></html>"),
        MediaType::Html
    );
    assert_eq!(detect_media_type(b"   <html>hi"), MediaType::Html);
    assert_eq!(detect_media_type(b""), MediaType::Empty);
    assert_eq!(detect_media_type(b"just some text"), MediaType::Other);
}

#[test]
fn sha256_matches_known_vector() {
    // SHA-256 of the empty input.
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn corruption_ratios_measure_expected_signals() {
    assert!(replacement_char_ratio("a\u{FFFD}b\u{FFFD}") > 0.0);
    assert_eq!(replacement_char_ratio("clean"), 0.0);
    assert!(control_char_ratio("a\u{0002}b").as_positive());
    assert_eq!(control_char_ratio("normal\ttext\nline\r"), 0.0);
    // Three of four non-empty lines are duplicates: ratio 0.75.
    let repeated = "x\nx\nx\nx";
    assert!(repeated_line_ratio(repeated) > 0.7);
    assert_eq!(repeated_line_ratio("a\nb\nc"), 0.0);
}

trait Positive {
    fn as_positive(&self) -> bool;
}
impl Positive for f64 {
    fn as_positive(&self) -> bool {
        *self > 0.0
    }
}

#[test]
fn math_ratio_and_gutters() {
    let math = "\u{2211}x \u{222B}y = \u{03B1} + \u{03B2}^2";
    assert!(math_char_ratio(math) > 0.020);
    assert!(math_char_ratio("plain english prose without symbols") < 0.020);
    // Two multi-space gutters split three columns.
    assert_eq!(count_gutters("col1    col2    col3"), 2);
    assert_eq!(count_gutters("single spaced words here"), 0);
}

#[test]
fn median_density_odd_and_even() {
    assert_eq!(median_nonws_per_page(&["aa", "bbbb", "c"]), 2.0);
    assert_eq!(median_nonws_per_page(&["aa", "bbbb"]), 3.0);
    assert_eq!(median_nonws_per_page(&[]), 0.0);
}

#[test]
fn multicolumn_detection() {
    // Left band x=10, right band x=300, five row-aligned lines per column so
    // the row-matched overlaps clear the minimum-pairs threshold.
    let mut spans = Vec::new();
    for row in 0..5 {
        let y = 100.0 + f64::from(row) * 20.0;
        spans.push(PositionedSpan { x: 10.0, y, height: 10.0 });
        spans.push(PositionedSpan { x: 300.0, y, height: 10.0 });
    }
    assert!(detect_multicolumn(&spans));
    // Single column: all spans share one x band.
    let single = vec![
        PositionedSpan { x: 10.0, y: 100.0, height: 10.0 },
        PositionedSpan { x: 11.0, y: 120.0, height: 10.0 },
        PositionedSpan { x: 10.0, y: 140.0, height: 10.0 },
        PositionedSpan { x: 11.0, y: 160.0, height: 10.0 },
    ];
    assert!(!detect_multicolumn(&single));
}

#[test]
fn license_policy_mapping() {
    assert_eq!(license_policy("public").0, "known_permissive");
    assert_eq!(license_policy("open").1, "track_compact");
    assert_eq!(license_policy("manual_or_licensed").0, "known_restricted");
    assert_eq!(license_policy("manual_or_licensed").1, "no_new_tracked_content");
    assert_eq!(license_policy("mystery").0, "unknown");
    assert_eq!(license_policy("mystery").1, "no_new_tracked_content");
}

#[test]
fn contract_glob_matches_alias() {
    let contracts = vec![SourceContract {
        id: "SRC-X".to_string(),
        path_glob: "data/papers/documents_ingest/**".to_string(),
        access_class: "manual_or_licensed".to_string(),
        manual_manifest_refs: vec!["ref.txt".to_string()],
    }];
    let mut aliases = BTreeSet::new();
    aliases.insert("data/papers/documents_ingest/a/b.pdf".to_string());
    let (id, refp, class) = match_contract(&contracts, &aliases);
    assert_eq!(id, "SRC-X");
    assert_eq!(refp, "ref.txt");
    assert_eq!(class, "manual_or_licensed");

    let mut uncovered = BTreeSet::new();
    uncovered.insert("data/other/z.pdf".to_string());
    assert_eq!(match_contract(&contracts, &uncovered).0, "");
}

#[test]
fn match_contract_most_restrictive_wins_across_aliases() {
    // A permissive contract listed first must not govern an identity whose
    // aliases also match a restricted contract: the restricted class wins
    // regardless of TOML order, so track_compact stays gated.
    let permissive = || SourceContract {
        id: "SRC-PUBLIC".to_string(),
        path_glob: "data/pub/**".to_string(),
        access_class: "public".to_string(),
        manual_manifest_refs: vec!["pub.txt".to_string()],
    };
    let restricted = || SourceContract {
        id: "SRC-LICENSED".to_string(),
        path_glob: "data/lic/**".to_string(),
        access_class: "manual_or_licensed".to_string(),
        manual_manifest_refs: vec!["lic.txt".to_string()],
    };
    let mut aliases = BTreeSet::new();
    aliases.insert("data/pub/paper.pdf".to_string());
    aliases.insert("data/lic/paper.pdf".to_string());

    // Permissive-first ordering.
    let forward = vec![permissive(), restricted()];
    let (id, refp, class) = match_contract(&forward, &aliases);
    assert_eq!(id, "SRC-LICENSED");
    assert_eq!(refp, "lic.txt");
    assert_eq!(class, "manual_or_licensed");
    assert_eq!(license_policy(&class).1, "no_new_tracked_content");

    // Reversed ordering yields the same governing contract.
    let reversed = vec![restricted(), permissive()];
    assert_eq!(match_contract(&reversed, &aliases).0, "SRC-LICENSED");
}

#[test]
fn classify_formula_risk_on_math_dense_zero_equations() {
    let identity = pdf_identity(20);
    let mut metrics = ExtractionMetrics {
        non_whitespace_chars: 4000,
        math_char_ratio: 0.05,
        mean_nonws_chars_per_page: 2000.0,
        ..ExtractionMetrics::default()
    };
    metrics.equation_count = 0;
    let signals = evaluate_signals(&identity, "reconciled");
    let (routing, reasons, _) = classify(&identity, &metrics, "reconciled", true, &signals);
    assert_eq!(routing, "mineru_candidate");
    assert!(reasons.iter().any(|r| r == "formula_risk"));
}

#[test]
fn classify_retains_clean_extraction() {
    let identity = pdf_identity(20);
    let metrics = ExtractionMetrics {
        non_whitespace_chars: 40000,
        mean_nonws_chars_per_page: 2000.0,
        math_char_ratio: 0.001,
        ..ExtractionMetrics::default()
    };
    let signals = evaluate_signals(&identity, "reconciled");
    let (routing, reasons, _) = classify(&identity, &metrics, "reconciled", true, &signals);
    assert_eq!(routing, "retain_docpipe");
    assert!(reasons.is_empty());
}

#[test]
fn classify_scan_risk_on_sparse_text() {
    let identity = pdf_identity(30);
    let metrics = ExtractionMetrics {
        non_whitespace_chars: 300,
        mean_nonws_chars_per_page: 10.0,
        ..ExtractionMetrics::default()
    };
    let signals = evaluate_signals(&identity, "reconciled");
    let (routing, reasons, _) = classify(&identity, &metrics, "reconciled", true, &signals);
    assert_eq!(routing, "mineru_candidate");
    assert!(reasons.iter().any(|r| r == "scan_risk"));
}

fn pdf_identity(pages: u64) -> SourceIdentity {
    let mut aliases = BTreeSet::new();
    aliases.insert("data/papers/documents_ingest/x.pdf".to_string());
    let mut ids = BTreeSet::new();
    ids.insert("x".to_string());
    SourceIdentity {
        sha256: "a".repeat(64),
        size_bytes: 1000,
        media_type: MediaType::Pdf,
        page_count: Some(pages),
        parse_status: ParseStatus::Parsed,
        aliases,
        manifest_ids: ids,
        corpus_lanes: BTreeSet::new(),
    }
}

// ---- Integration tests over run_inventory --------------------------------

#[test]
fn valid_pdf_with_reconciled_extraction_retains_docpipe() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let pdf = root.join("ingest/paper.pdf");
    write_pdf(&pdf, 10);
    write_manifest(root, &[("goodpaper", &pdf)]);
    // Dense native text, matched equations.
    let body = "Introduction. ".repeat(2000);
    write_extraction(&root.join("extracted/goodpaper"), &body, 3, 1, 2);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let out = read_out(&args);
    let d = docs(&out);
    assert_eq!(d.len(), 1);
    assert_eq!(routing_of(&d[0]), "retain_docpipe");
    assert_eq!(
        d[0].get("signal_state_reading_order").unwrap().as_str().unwrap(),
        "positioned_text_unavailable"
    );
}

#[test]
fn valid_pdf_without_extraction_requires_mineru() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let pdf = root.join("ingest/lonely.pdf");
    write_pdf(&pdf, 4);
    write_manifest(root, &[("lonely", &pdf)]);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(routing_of(&d[0]), "mineru_required");
}

#[test]
fn html_under_pdf_suffix_is_blocked_by_content() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let fake = root.join("ingest/malitson.pdf");
    std::fs::write(&fake, b"<!DOCTYPE html><html><body>404 Not Found</body></html>").unwrap();
    write_manifest(root, &[("malitson", &fake)]);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(routing_of(&d[0]), "blocked_source");
    assert_eq!(d[0].get("media_type").unwrap().as_str().unwrap(), "html");
    let reasons: Vec<String> = d[0]
        .get("reason_codes")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap().to_string())
        .collect();
    assert!(reasons.contains(&"extension_signature_mismatch".to_string()));
}

#[test]
fn identical_bytes_collapse_into_one_record_with_two_aliases() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let a = root.join("ingest/copy_a.pdf");
    let b = root.join("ingest/copy_b.pdf");
    write_pdf(&a, 3);
    std::fs::copy(&a, &b).unwrap();
    write_manifest(root, &[("id_a", &a), ("id_b", &b)]);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(d.len(), 1, "identical bytes must collapse");
    let aliases = d[0].get("alias_paths").unwrap().as_array().unwrap();
    assert_eq!(aliases.len(), 2);
    let ids = d[0].get("manifest_ids").unwrap().as_array().unwrap();
    assert_eq!(ids.len(), 2);
}

#[test]
fn one_manifest_id_two_byte_streams_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let a = root.join("ingest/one.pdf");
    let b = root.join("ingest/two.pdf");
    write_pdf(&a, 1);
    write_pdf(&b, 7); // distinct bytes
    // Same id resolves to two files.
    write_manifest(root, &[("dupid", &a), ("dupid", &b)]);

    let args = base_args(root);
    let err = run_inventory(&args).unwrap_err();
    assert!(err.to_string().contains("two byte streams"));
}

#[test]
fn missing_source_routes_blocked() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let ghost = root.join("ingest/ghost.pdf"); // never created
    write_manifest(root, &[("ghost", &ghost)]);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(routing_of(&d[0]), "blocked_source");
    assert_eq!(d[0].get("parse_status").unwrap().as_str().unwrap(), "missing");
}

#[test]
fn missing_structured_extraction_requires_mineru() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let pdf = root.join("ingest/noresult.pdf");
    write_pdf(&pdf, 5);
    write_manifest(root, &[("noresult", &pdf)]);
    // Directory exists but paper.toml is absent -> orphaned/unusable.
    std::fs::create_dir_all(root.join("extracted/noresult")).unwrap();

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(routing_of(&d[0]), "mineru_required");
}

#[test]
fn image_only_extraction_with_zero_text_requires_mineru() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let pdf = root.join("ingest/scan.pdf");
    write_pdf(&pdf, 12);
    write_manifest(root, &[("scan", &pdf)]);
    // Valid PDF, extraction present, but no native text layer; figures only.
    write_extraction(&root.join("extracted/scan"), "", 0, 0, 5);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(routing_of(&d[0]), "mineru_required");
}

#[test]
fn repeated_runs_are_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let p1 = root.join("ingest/one.pdf");
    let p2 = root.join("ingest/two.pdf");
    write_pdf(&p1, 6);
    write_pdf(&p2, 2);
    write_manifest(root, &[("one", &p1), ("two", &p2)]);
    write_extraction(&root.join("extracted/one"), &"text ".repeat(3000), 2, 0, 1);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let first = std::fs::read_to_string(&args.out).unwrap();
    run_inventory(&args).unwrap();
    let second = std::fs::read_to_string(&args.out).unwrap();
    assert_eq!(first, second);
}

#[test]
fn paths_with_spaces_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let spaced = root.join("ingest/my paper draft.pdf");
    write_pdf(&spaced, 2);
    write_manifest(root, &[("spaced", &spaced)]);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    let alias = d[0].get("alias_paths").unwrap().as_array().unwrap()[0]
        .as_str()
        .unwrap();
    assert!(alias.contains("my paper draft.pdf"));
}

#[test]
fn emit_path_encodes_reversibly_but_normalize_preserves_bytes() {
    // emit_path is ASCII-only for the tracked artifact.
    // Bytes: en-dash U+2013 -> UTF-8 E2 80 93 -> %E2%80%93.
    assert_eq!(emit_path("a/b\u{2013}c.pdf"), "a/b%E2%80%93c.pdf");
    // A literal percent is escaped so decoding is unambiguous.
    assert_eq!(emit_path("a/50%.pdf"), "a/50%25.pdf");
    // Pure ASCII including spaces is unchanged.
    assert_eq!(emit_path("my paper.pdf"), "my paper.pdf");
    // normalize_rel keeps the exact bytes so the path still resolves on disk;
    // it only swaps separators.
    assert_eq!(normalize_rel("a\\b\u{2013}c.pdf"), "a/b\u{2013}c.pdf");
}

#[test]
fn non_ascii_named_source_resolves_and_is_not_blocked() {
    // Guards against encoding the filesystem path: a real en-dash filename must
    // be read, hashed, and routed by content, with only the emitted alias
    // percent-encoded.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let named = root.join("ingest/E10 Billiards Zero\u{2011}Divisor.pdf");
    write_pdf(&named, 4);
    write_manifest(root, &[("e10", &named)]);
    write_extraction(&root.join("extracted/e10"), &"body ".repeat(3000), 1, 0, 0);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(d.len(), 1);
    assert_ne!(routing_of(&d[0]), "blocked_source");
    assert!(!d[0].get("source_sha256").unwrap().as_str().unwrap().is_empty());
    let alias = d[0].get("alias_paths").unwrap().as_array().unwrap()[0]
        .as_str()
        .unwrap();
    assert!(alias.contains("Zero%E2%80%91Divisor.pdf"), "alias must be encoded: {alias}");
}

#[test]
fn duplicate_twin_extractions_are_not_orphaned() {
    // Two byte-identical sources, each with its own docpipe slug. The identity
    // collapses; both extraction dirs must be consumed, not surfaced as
    // sourceless orphans.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let a = root.join("ingest/twin_a.pdf");
    let b = root.join("ingest/twin_b.pdf");
    write_pdf(&a, 5);
    std::fs::copy(&a, &b).unwrap();
    write_manifest(root, &[("twin_a", &a), ("twin_b", &b)]);
    let body = "text ".repeat(3000);
    write_extraction(&root.join("extracted/twin_a"), &body, 1, 0, 0);
    write_extraction(&root.join("extracted/twin_b"), &body, 1, 0, 0);

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(d.len(), 1, "twins collapse to one record");
    assert_ne!(routing_of(&d[0]), "manual_review");
    // No orphan record emitted.
    assert!(!d.iter().any(|r| r
        .get("canonical_id")
        .unwrap()
        .as_str()
        .unwrap()
        .starts_with("orphan:")));
}

#[test]
fn only_filter_narrows_without_changing_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let p1 = root.join("ingest/keep.pdf");
    let p2 = root.join("ingest/drop.pdf");
    write_pdf(&p1, 3);
    write_pdf(&p2, 9);
    write_manifest(root, &[("keep", &p1), ("drop", &p2)]);

    let mut args = base_args(root);
    args.only = Some("keep".to_string());
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].get("canonical_id").unwrap().as_str().unwrap(), "keep");
    // Schema fields still present.
    assert!(d[0].get("source_sha256").is_some());
    assert!(d[0].get("routing").is_some());
}

#[test]
fn legitimate_json_derivative_remains_hashable_under_raw_semantics() {
    // The inventory hashes the compact paper.toml derivative it reads; a JSON
    // sidecar alongside it is not rejected as a class. Content, never the
    // suffix, decides what is hashable.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("ingest")).unwrap();
    let pdf = root.join("ingest/withjson.pdf");
    write_pdf(&pdf, 8);
    write_manifest(root, &[("withjson", &pdf)]);
    let ext = root.join("extracted/withjson");
    write_extraction(&ext, &"body text ".repeat(3000), 1, 1, 0);
    std::fs::write(ext.join("meta.json"), b"{\"k\":1}").unwrap();

    let args = base_args(root);
    run_inventory(&args).unwrap();
    let d = docs(&read_out(&args));
    assert!(d[0].get("paper_toml_sha256").is_some());
}
