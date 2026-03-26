use anyhow::{Context, Result};
use clap::Parser;
use lopdf::{
    Document, Object, ObjectId, Stream,
    content::{Content, Operation},
    dictionary,
};
use regex::Regex;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "cd-gap-binders",
    about = "Pure-Rust builder for Cayley-Dickson exact-gap reconstruction binders"
)]
struct Cli {
    #[arg(long, default_value = "/home/eirikr/Documents/Projects/CayleyDickson")]
    cache_root: PathBuf,
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct Recipe {
    key: &'static str,
    title: &'static str,
    subject: &'static str,
    keywords: &'static str,
    output_rel: &'static str,
    sidecar_rel: &'static str,
    appendix_rels: &'static [&'static str],
    source_pdf_rels: &'static [&'static str],
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let recipes = recipes();
    let mut report_rows = Vec::new();

    for recipe in recipes {
        let output_path = cli.cache_root.join(recipe.output_rel);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }

        let mut docs = Vec::new();
        docs.push(build_text_document(
            &cli.cache_root.join(recipe.sidecar_rel),
            recipe.title,
            recipe.subject,
            recipe.keywords,
        )?);

        for appendix_rel in recipe.appendix_rels {
            docs.push(build_text_document(
                &cli.cache_root.join(appendix_rel),
                recipe.title,
                recipe.subject,
                recipe.keywords,
            )?);
        }

        for source_rel in recipe.source_pdf_rels {
            let path = cli.cache_root.join(source_rel);
            let doc = Document::load(&path)
                .with_context(|| format!("load source pdf {}", path.display()))?;
            docs.push(doc);
        }

        let mut merged = merge_documents(docs)?;
        set_info(
            &mut merged,
            recipe.title,
            recipe.subject,
            recipe.keywords,
            "open_gororoba cd-gap-binders",
        );
        merged.compress();
        merged
            .save(&output_path)
            .with_context(|| format!("save {}", output_path.display()))?;

        let page_count = merged.get_pages().len();
        report_rows.push(format!(
            "{}\t{}\t{}",
            recipe.key,
            render_repo_path(&cli.cache_root, &output_path),
            page_count
        ));
    }

    if let Some(report_path) = cli.report {
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
        let mut body = String::from("key\toutput_rel\tpages\n");
        for row in report_rows {
            body.push_str(&row);
            body.push('\n');
        }
        fs::write(&report_path, body)
            .with_context(|| format!("write report {}", report_path.display()))?;
    }

    Ok(())
}

fn recipes() -> Vec<Recipe> {
    vec![
        Recipe {
            key: "jacobson_p1",
            title: "Jacobson 1958 Spliced Reconstruction Priority 1",
            subject: "Reconstructed reference binder made from official TOC witnesses and official preview fragments; exact original still missing.",
            keywords: "Jacobson,1958,composition algebras,reconstruction,official fragment,official TOC",
            output_rel: "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_spliced_reconstruction_priority1.pdf",
            sidecar_rel: "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_spliced_reconstruction_priority1.md",
            appendix_rels: &[],
            source_pdf_rels: &[
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_collected_mathematical_papers_vol2_dnb_toc.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_collected_mathematical_papers_vol2_hbz_toc.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_preview.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_composition_algebras_and_their_automorphisms_chapter_preview_pageone.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_composition_algebras_and_their_automorphisms_collected_papers_preview.pdf",
            ],
        },
        Recipe {
            key: "jacobson_p2",
            title: "Jacobson 1958 Spliced Reconstruction Priority 2",
            subject: "Extended reconstructed dossier with embedded provenance appendix plus official TOC witnesses and official preview fragments; exact original still missing.",
            keywords: "Jacobson,1958,composition algebras,reconstruction,dossier,official fragment,official TOC",
            output_rel: "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_spliced_reconstruction_priority2.pdf",
            sidecar_rel: "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_spliced_reconstruction_priority2.md",
            appendix_rels: &[
                "metadata/browser_traces/2026-03-26_jacobson_no_cost_placeholder_packet.md",
            ],
            source_pdf_rels: &[
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_collected_mathematical_papers_vol2_dnb_toc.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_collected_mathematical_papers_vol2_hbz_toc.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_preview.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_composition_algebras_and_their_automorphisms_chapter_preview_pageone.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1989_composition_algebras_and_their_automorphisms_collected_papers_preview.pdf",
            ],
        },
        Recipe {
            key: "freudenthal_p1",
            title: "Freudenthal 1951 Spliced Reconstruction Priority 1",
            subject: "Reconstructed reference binder from official fragment, translation or rewriting, and support reconstruction material; exact Utrecht 1951 original still missing.",
            keywords: "Freudenthal,1951,octonions,octave geometry,reconstruction,translation,support",
            output_rel: "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1951_oktaven_ausnahmegruppen_und_oktavengeometrie_spliced_reconstruction_priority1.pdf",
            sidecar_rel: "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1951_oktaven_ausnahmegruppen_und_oktavengeometrie_spliced_reconstruction_priority1.md",
            appendix_rels: &[],
            source_pdf_rels: &[
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1985_oktaven_ausnahmegruppen_und_oktavengeometrie_preview.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1985_translation_oktaven_ausnahmegruppen_oktavengeometrie.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1954_beziehungen_der_e7_und_e8_zur_oktavenebene_part1_uu.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1951_russian_translation_oktavy_osobye_gruppy_i_oktavnaya_geometriya_mat7.pdf",
            ],
        },
        Recipe {
            key: "freudenthal_p2",
            title: "Freudenthal 1951 Spliced Reconstruction Priority 2",
            subject: "Extended reconstructed dossier with embedded provenance appendix plus same-lane descendants and later support reconstructions; exact Utrecht 1951 original still missing.",
            keywords: "Freudenthal,1951,octonions,octave geometry,reconstruction,dossier,translation,support",
            output_rel: "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1951_oktaven_ausnahmegruppen_und_oktavengeometrie_spliced_reconstruction_priority2.pdf",
            sidecar_rel: "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1951_oktaven_ausnahmegruppen_und_oktavengeometrie_spliced_reconstruction_priority2.md",
            appendix_rels: &[
                "metadata/browser_traces/2026-03-26_freudenthal_no_cost_placeholder_packet.md",
            ],
            source_pdf_rels: &[
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1985_oktaven_ausnahmegruppen_und_oktavengeometrie_preview.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1985_translation_oktaven_ausnahmegruppen_oktavengeometrie.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1954_beziehungen_der_e7_und_e8_zur_oktavenebene_part1_uu.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1951_russian_translation_oktavy_osobye_gruppy_i_oktavnaya_geometriya_mat7.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_lie_groups_in_the_foundations_of_geometry_ranicki_scan.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/veldkamp_1968_unitary_groups_projective_octave_planes.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/brada_1986_elements_geometrie_octaves_cayley.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/eschenburg_2010_oktaven_ausnahmegruppen_und_oktavengeometrie_notes.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/eschenburg_2018_geometry_of_octonions.pdf",
                "tier1_core_cd_algebra/composition_alternative_algebras/pinto_2021_octonionic_planes_ufmg.pdf",
            ],
        },
    ]
}

fn render_repo_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|rel| rel.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

fn build_text_document(
    path: &Path,
    title: &str,
    subject: &str,
    keywords: &str,
) -> Result<Document> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let plain = markdown_to_plaintext(&raw);
    let lines = wrap_lines(&plain, 96);
    let mut doc = text_document_from_lines(title, &lines)?;
    set_info(
        &mut doc,
        title,
        subject,
        keywords,
        "open_gororoba cd-gap-binders",
    );
    Ok(doc)
}

fn markdown_to_plaintext(input: &str) -> String {
    let link_re = Regex::new(r"\[([^\]]+)\]\([^)]+\)").expect("valid regex");
    let code_re = Regex::new(r"`([^`]*)`").expect("valid regex");
    let mut out = String::new();
    for line in input.lines() {
        let mut clean = line.replace('\t', "    ");
        clean = link_re.replace_all(&clean, "$1").to_string();
        clean = code_re.replace_all(&clean, "$1").to_string();
        if clean.starts_with("# ") {
            clean = clean.trim_start_matches("# ").to_string();
        } else if clean.starts_with("## ") {
            clean = clean.trim_start_matches("## ").to_string();
        } else if clean.starts_with("### ") {
            clean = clean.trim_start_matches("### ").to_string();
        }
        out.push_str(clean.trim_end());
        out.push('\n');
    }
    out
}

fn wrap_lines(input: &str, width: usize) -> Vec<String> {
    let mut wrapped = Vec::new();
    for raw_line in input.lines() {
        if raw_line.trim().is_empty() {
            wrapped.push(String::new());
            continue;
        }

        let indent_len = raw_line.chars().take_while(|ch| ch.is_whitespace()).count();
        let indent = " ".repeat(indent_len.min(width.saturating_sub(1)));
        let available = width.saturating_sub(indent.len()).max(12);
        let mut current = String::new();

        for word in raw_line.split_whitespace() {
            if current.is_empty() {
                current.push_str(&indent);
                current.push_str(word);
            } else if current.len() + 1 + word.len() <= width {
                current.push(' ');
                current.push_str(word);
            } else if word.len() > available {
                wrapped.push(current);
                current = indent.clone();
                let mut remaining = word;
                while remaining.len() > available {
                    let (head, tail) = remaining.split_at(available);
                    wrapped.push(format!("{indent}{head}"));
                    remaining = tail;
                }
                current.push_str(remaining);
            } else {
                wrapped.push(current);
                current = indent.clone();
                current.push_str(word);
            }
        }

        if !current.is_empty() {
            wrapped.push(current);
        }
    }
    wrapped
}

fn text_document_from_lines(title: &str, lines: &[String]) -> Result<Document> {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Courier",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });

    let lines_per_page = 46usize;
    let mut page_ids = Vec::new();
    for chunk in lines.chunks(lines_per_page) {
        let content_id = doc.add_object(Stream::new(dictionary! {}, build_content(chunk)?));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        page_ids.push(page_id);
    }

    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => page_ids.iter().copied().map(Object::Reference).collect::<Vec<_>>(),
        "Count" => page_ids.len() as i64,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    set_info(
        &mut doc,
        title,
        "Generated provenance cover sheet",
        "reconstruction provenance cover",
        "open_gororoba cd-gap-binders",
    );
    Ok(doc)
}

fn build_content(lines: &[String]) -> Result<Vec<u8>> {
    let mut operations = vec![
        Operation::new("BT", vec![]),
        Operation::new("Tf", vec!["F1".into(), 11.into()]),
        Operation::new("TL", vec![14.into()]),
        Operation::new("Td", vec![50.into(), 740.into()]),
    ];
    for line in lines {
        operations.push(Operation::new(
            "Tj",
            vec![Object::string_literal(line.as_str())],
        ));
        operations.push(Operation::new("T*", vec![]));
    }
    operations.push(Operation::new("ET", vec![]));
    Ok(Content { operations }.encode()?)
}

fn merge_documents(documents: Vec<Document>) -> Result<Document> {
    let mut max_id = 1;
    let mut documents_pages = BTreeMap::new();
    let mut documents_objects = BTreeMap::new();
    let mut document = Document::with_version("1.5");

    for mut doc in documents {
        doc.renumber_objects_with(max_id);
        max_id = doc.max_id + 1;

        let pages = doc.get_pages();
        pages
            .into_values()
            .map(|object_id| {
                (
                    object_id,
                    doc.get_object(object_id)
                        .expect("page object exists")
                        .to_owned(),
                )
            })
            .for_each(|(key, value)| {
                documents_pages.insert(key, value);
            });

        documents_objects.extend(doc.objects);
    }

    let mut catalog_object: Option<(ObjectId, Object)> = None;
    let mut pages_object: Option<(ObjectId, Object)> = None;

    for (object_id, object) in documents_objects {
        match object.type_name().unwrap_or(b"") {
            b"Catalog" => {
                catalog_object = Some((
                    if let Some((id, _)) = catalog_object {
                        id
                    } else {
                        object_id
                    },
                    object,
                ));
            }
            b"Pages" => {
                if let Ok(dictionary) = object.as_dict() {
                    let mut dictionary = dictionary.clone();
                    if let Some((_, ref existing)) = pages_object
                        && let Ok(old_dictionary) = existing.as_dict()
                    {
                        dictionary.extend(old_dictionary);
                    }
                    pages_object = Some((
                        if let Some((id, _)) = pages_object {
                            id
                        } else {
                            object_id
                        },
                        Object::Dictionary(dictionary),
                    ));
                }
            }
            b"Page" | b"Outlines" | b"Outline" => {}
            _ => {
                document.objects.insert(object_id, object);
            }
        }
    }

    let (catalog_id, catalog_obj) = catalog_object.context("catalog root not found")?;
    let (pages_id, pages_obj) = pages_object.context("pages root not found")?;

    for (object_id, object) in &documents_pages {
        if let Ok(dictionary) = object.as_dict() {
            let mut dictionary = dictionary.clone();
            dictionary.set("Parent", pages_id);
            document
                .objects
                .insert(*object_id, Object::Dictionary(dictionary));
        }
    }

    if let Ok(dictionary) = pages_obj.as_dict() {
        let mut dictionary = dictionary.clone();
        dictionary.set("Count", documents_pages.len() as u32);
        dictionary.set(
            "Kids",
            documents_pages
                .keys()
                .copied()
                .map(Object::Reference)
                .collect::<Vec<_>>(),
        );
        document
            .objects
            .insert(pages_id, Object::Dictionary(dictionary));
    }

    if let Ok(dictionary) = catalog_obj.as_dict() {
        let mut dictionary = dictionary.clone();
        dictionary.set("Pages", pages_id);
        dictionary.remove(b"Outlines");
        document
            .objects
            .insert(catalog_id, Object::Dictionary(dictionary));
    }

    document.trailer.set("Root", catalog_id);
    document.max_id = document.objects.len() as u32;
    document.renumber_objects();
    document.adjust_zero_pages();
    Ok(document)
}

fn set_info(doc: &mut Document, title: &str, subject: &str, keywords: &str, creator: &str) {
    let info_id = doc.add_object(dictionary! {
        "Title" => Object::string_literal(title),
        "Subject" => Object::string_literal(subject),
        "Author" => Object::string_literal("OpenAI Codex for Erick Hilgartner"),
        "Creator" => Object::string_literal(creator),
        "Producer" => Object::string_literal("lopdf"),
        "Keywords" => Object::string_literal(keywords),
    });
    doc.trailer.set("Info", info_id);
}
