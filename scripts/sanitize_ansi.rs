use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = [
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_theorem_provers_idris2_linear_types.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_lambda_core_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_constraint_probabilistic_probabilistic_lambda_intro.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_formal_methods_linear_logic_girard.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_formal_methods_linear_logic_programming.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_hott_formal_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_theorem_provers_hott_agda.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_game_geometry_differential_linear_logic.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_topos_rewriting_graphical_linear_algebra.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_constraint_probabilistic_linear_constraints.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_semantics_linear_haskell.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_supporting_misc_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_topos_rewriting_synthetic_topology.rs",
        "crates/data_core/src/registry_mirrors/data_papers_intake_2026_02_15_cayley_dickson_sources_supplemental_docs_linear_transport_documents_primary_1_analysis_data_pdf_corpus_markdown_agl_library_game_geometry_goi_linear_logic.rs",
    ];

    for path in files {
        let p = Path::new(path);
        if !p.exists() {
            println!("SKIP: {} not found", path);
            continue;
        }
        let content = fs::read_to_string(p)?;
        let mut clean = String::new();
        for ch in content.chars() {
            let val = ch as u32;
            if (val >= 32 && val <= 126) || ch == '\n' || ch == '\r' || ch == '\t' {
                clean.push(ch);
            } else {
                // Replace everything else with space to be safe
                clean.push(' ');
            }
        }
        fs::write(p, clean)?;
        println!("CLEANED: {}", path);
    }
    Ok(())
}
