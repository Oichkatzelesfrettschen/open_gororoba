//! Render mdBook pages from the hand-authored book registry.

use std::{fs, path::Path};
use toml::Value;

pub fn emit_legacy(repo_root: &Path, input: &Path, allow_unicode: bool) -> Result<(), String> {
    let source =
        fs::read_to_string(input).map_err(|error| format!("read {}: {error}", input.display()))?;
    let registry: Value =
        toml::from_str(&source).map_err(|error| format!("parse {}: {error}", input.display()))?;
    let mut documents = registry
        .get("document")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table)
        .collect::<Vec<_>>();
    documents.sort_by_key(|document| field(document, "source_markdown").trim().to_owned());

    for document in documents {
        let relative_path = field(document, "source_markdown").trim();
        if relative_path.is_empty() {
            continue;
        }
        let path = Path::new(relative_path);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
            || !path.starts_with("docs/book/src")
        {
            return Err(format!(
                "book page path escapes docs/book/src: {relative_path}"
            ));
        }
        let body = field(document, "body_markdown");
        let mut lines = if body.starts_with("<!-- AUTO-GENERATED: DO NOT EDIT -->") {
            Vec::new()
        } else {
            vec![
                "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_owned(),
                "<!-- Source of truth: registry/book_docs.toml -->".to_owned(),
                String::new(),
            ]
        };
        if body.is_empty() {
            let preferred = field(document, "title").trim();
            let title = if preferred.is_empty() {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or(relative_path)
            } else {
                preferred
            };
            lines.extend([
                format!("# {title}"),
                String::new(),
                "(No body_markdown captured in registry/book_docs.toml.)".to_owned(),
            ]);
        } else {
            lines.extend(body.lines().map(ToOwned::to_owned));
        }
        lines.push(String::new());
        let output = lines.join("\n");
        if !allow_unicode {
            let invalid = output
                .chars()
                .filter(|character| !character.is_ascii())
                .take(20)
                .collect::<String>();
            if !invalid.is_empty() {
                return Err(format!(
                    "non-ASCII output in {}: {invalid:?}",
                    repo_root.join(path).display()
                ));
            }
        }
        let destination = repo_root.join(path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("mkdir {}: {error}", parent.display()))?;
        }
        fs::write(&destination, output)
            .map_err(|error| format!("write {}: {error}", destination.display()))?;
    }
    Ok(())
}

fn field<'a>(document: &'a toml::map::Map<String, Value>, key: &str) -> &'a str {
    document
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::emit_legacy;
    use std::{fs, path::PathBuf, time::SystemTime};

    fn test_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system time is after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("book-docs-{}-{nonce}", std::process::id()));
        fs::create_dir(&root).expect("create isolated fixture directory");
        root
    }

    #[test]
    fn renders_legacy_header_and_body_bytes() {
        let root = test_root();
        let registry = root.join("book_docs.toml");
        fs::write(
            &registry,
            "[[document]]\nsource_markdown = 'docs/book/src/example.md'\ntitle = 'Example'\nbody_markdown = '# Body'\n",
        )
        .expect("write fixture registry");

        emit_legacy(&root, &registry, false).expect("render book page");
        let rendered =
            fs::read_to_string(root.join("docs/book/src/example.md")).expect("read rendered page");
        assert_eq!(
            rendered,
            "<!-- AUTO-GENERATED: DO NOT EDIT -->\n<!-- Source of truth: registry/book_docs.toml -->\n\n# Body\n"
        );
        fs::remove_dir_all(root).expect("remove fixture directory");
    }

    #[test]
    fn rejects_paths_outside_book_source() {
        let root = test_root();
        let registry = root.join("book_docs.toml");
        fs::write(
            &registry,
            "[[document]]\nsource_markdown = 'docs/book/src/../../outside.md'\nbody_markdown = '# Outside'\n",
        )
        .expect("write fixture registry");

        assert!(emit_legacy(&root, &registry, false).is_err());
        assert!(!root.join("docs/outside.md").exists());
        fs::remove_dir_all(root).expect("remove fixture directory");
    }
}
