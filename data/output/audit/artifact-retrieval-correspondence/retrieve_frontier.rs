//! Retain exact HTTP response observations for the finite historical identity frontier.
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path, process::Command};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
fn hash(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err("sha256sum failed".into());
    }
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .ok_or("empty digest")?
        .to_owned())
}
fn write(path: &Path, value: &Value) -> Result<()> {
    if path.exists() {
        return Err(format!("refusing overwrite {}", path.display()).into());
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn sanitize_headers(raw: &[u8]) -> Vec<u8> {
    let mut sanitized = Vec::new();
    let mut sensitive = false;
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        let ending: &[u8] = if line.ends_with(b"\r\n") {
            b"\r\n"
        } else if line.ends_with(b"\n") {
            b"\n"
        } else {
            b""
        };
        let content = &line[..line.len() - ending.len()];
        if content.starts_with(b" ") || content.starts_with(b"\t") {
            if sensitive {
                sanitized.extend_from_slice(b" [REDACTED]");
                sanitized.extend_from_slice(ending);
                continue;
            }
        } else {
            sensitive = false;
            if let Some(colon) = content.iter().position(|byte| *byte == b':') {
                let name = content[..colon].to_ascii_lowercase();
                sensitive = matches!(
                    name.as_slice(),
                    b"set-cookie"
                        | b"set-cookie2"
                        | b"cookie"
                        | b"authorization"
                        | b"proxy-authorization"
                        | b"authentication-info"
                        | b"proxy-authentication-info"
                );
                if sensitive {
                    sanitized.extend_from_slice(&content[..=colon]);
                    sanitized.extend_from_slice(b" [REDACTED]");
                    sanitized.extend_from_slice(ending);
                    continue;
                }
            }
        }
        sanitized.extend_from_slice(line);
    }
    sanitized
}
fn retain_sanitized_headers(root: &Path, headers: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let raw = fs::read(headers)?;
    let sanitized = sanitize_headers(&raw);
    if raw == sanitized {
        return Ok(());
    }
    let private = root.join(".cache/artifact-retrieval-private-headers");
    fs::create_dir_all(&private)?;
    fs::set_permissions(&private, fs::Permissions::from_mode(0o700))?;
    let original_hash = hash(headers)?;
    let name = headers
        .file_name()
        .ok_or("missing header filename")?
        .to_str()
        .ok_or("non-UTF8 header filename")?;
    let original = private.join(format!("{name}-{original_hash}.raw"));
    if original.exists() {
        if fs::read(&original)? != raw {
            return Err("private header archive differs".into());
        }
    } else {
        fs::write(&original, &raw)?;
        fs::set_permissions(&original, fs::Permissions::from_mode(0o600))?;
    }
    if hash(&original)? != original_hash {
        return Err("private header archive digest differs".into());
    }
    fs::write(headers, sanitized)?;
    let inventory = private.join(format!("{name}-{original_hash}.json"));
    if !inventory.exists() {
        write(
            &inventory,
            &json!({"header_path":headers.strip_prefix(root)?.to_str(),"original_path":original.strip_prefix(root)?.to_str(),"original_sha256":original_hash,"sanitized_sha256":hash(headers)?,"rule":"Redact cookie and authorization response fields, including folded continuation lines; preserve all other header bytes."}),
        )?;
        fs::set_permissions(&inventory, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let root = std::env::current_dir()?;
    let audit_path =
        "data/output/audit/artifact-remote-materializability/remote-identity-audit.toml";
    let directory = root.join("data/output/audit/artifact-retrieval-correspondence");
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments == ["--sanitize-headers"] {
        for entry in fs::read_dir(&directory)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or("header filename")?;
            if name.starts_with("headers-") && name.ends_with(".txt") {
                retain_sanitized_headers(&root, &path)?;
            }
        }
        return Ok(());
    }
    if !arguments.is_empty() {
        return Err("expected no arguments or --sanitize-headers".into());
    }
    let audit: toml::Value = toml::from_str(&fs::read_to_string(root.join(audit_path))?)?;
    let rows = audit["inventory_row"]
        .as_array()
        .ok_or("missing frontier")?;
    if rows.len() != 40 {
        return Err("frontier must contain exactly 40 rows".into());
    }
    let database = root.join("registry/canonical/control_plane.sqlite3");
    let output = Command::new("sqlite3")
        .args(["-readonly", "-json"])
        .arg(&database)
        .arg("SELECT id,key,canonical_functional_url FROM artifacts")
        .output()?;
    if !output.status.success() {
        return Err("canonical inventory read failed".into());
    }
    let artifacts: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    let artifacts: BTreeMap<_, _> = artifacts
        .into_iter()
        .map(|record| (record["id"].as_str().unwrap().to_owned(), record))
        .collect();
    let mut urls = BTreeMap::new();
    for row in rows {
        let url = if row["id"].as_str() == Some("ASOT-2652") {
            "https://arxiv.org/pdf/1602.02317v1.pdf"
        } else {
            row["url"].as_str().ok_or("URL")?
        };
        let index = urls.len();
        urls.entry(url.to_owned()).or_insert(index);
    }
    let plan: Vec<_> = urls
        .iter()
        .map(|(url, index)| json!({"url":url,"index":index}))
        .collect();
    write(
        &directory.join("retrieval-plan.json"),
        &json!({"audit_sha256":hash(&root.join(audit_path))?,"rows":40,"requests":plan,"method":"GET","timeout_seconds":45,"user_agent":"Mozilla/5.0 (compatible; Gororoba evidence retrieval)"}),
    )?;
    let mut receipts = BTreeMap::new();
    for (url, index) in &urls {
        let body = directory.join(format!("response-{index:02}.bin"));
        let headers = directory.join(format!("headers-{index:02}.txt"));
        let output = Command::new("curl")
            .args([
                "--silent",
                "--show-error",
                "--fail",
                "--location",
                "--max-time",
                "45",
                "--max-filesize",
                "33554432",
                "--user-agent",
                "Mozilla/5.0 (compatible; Gororoba evidence retrieval)",
                "--write-out",
                "%{json}",
                "--dump-header",
            ])
            .arg(&headers)
            .arg("--output")
            .arg(&body)
            .arg(url)
            .output()?;
        retain_sanitized_headers(&root, &headers)?;
        let metadata: Value = serde_json::from_slice(&output.stdout)?;
        let time = Command::new("date")
            .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
            .output()?;
        let completed = output.status.success();
        let bytes = fs::metadata(&body).map_or(0, |metadata| metadata.len());
        let digest = if body.exists() {
            Some(hash(&body)?)
        } else {
            None
        };
        let receipt = json!({"requested_url":url,"final_url":metadata["url_effective"],"method":"GET","http_status":metadata["http_code"],"completed":completed,"observed_at":String::from_utf8(time.stdout)?.trim(),"tool":"curl with Mozilla/5.0 user-agent; transport success never substitutes for expected-byte verification","body_sha256":digest,"body_bytes":bytes});
        let receipt_path = directory.join(format!("request-{index:02}.json"));
        write(&receipt_path, &receipt)?;
        fs::write(
            directory.join(format!("request-{index:02}.stderr")),
            output.stderr,
        )?;
        receipts.insert(url.clone(), (*index, receipt));
        eprintln!(
            "request {index}: HTTP {} completed={completed} bytes={bytes}",
            metadata["http_code"]
        );
    }
    let expected_hash = hash(&root.join(audit_path))?;
    let mut results = Vec::new();
    for row in rows {
        let id = row["id"].as_str().ok_or("ID")?;
        let key = row["key"].as_str().ok_or("key")?;
        let old_url = row["url"].as_str().ok_or("old URL")?;
        let url = if id == "ASOT-2652" {
            "https://arxiv.org/pdf/1602.02317v1.pdf"
        } else {
            old_url
        };
        let (index, receipt) = &receipts[url];
        let matches = receipt["completed"] == true
            && receipt["body_sha256"].as_str() == row["sha256"].as_str()
            && receipt["body_bytes"].as_u64()
                == row["byte_length"].as_integer().map(|value| value as u64);
        let compatible = artifacts.get(id).is_some_and(|record| {
            record["key"] == key && record["canonical_functional_url"] == old_url
        });
        let mut record = json!({"id":id,"key":key,"historical_url":old_url,"requested_url":url,"expected_sha256":row["sha256"].as_str(),"observed_sha256":receipt["body_sha256"],"digest_matches":matches,"canonical_prestate_matches":compatible,"http_status":receipt["http_status"],"document_identity":"unresolved"});
        if compatible {
            let request_path = format!(
                "data/output/audit/artifact-retrieval-correspondence/request-{index:02}.json"
            );
            let body_path = format!(
                "data/output/audit/artifact-retrieval-correspondence/response-{index:02}.bin"
            );
            let mut spec = json!({"schema_version":1,"observation_key":format!("exact-response-{id}-{}",hash(&root.join(&request_path))?),"actor":"codex","reason":"Reconcile exact response bytes with preserved historical inventory expectation","artifact_id":id,"artifact_key":key,"expected_canonical_url":old_url,"expected_sha256":row["sha256"].as_str(),"expected_bytes":row["byte_length"].as_integer(),"correct_canonical_url":id=="ASOT-2652" && matches,"document_identity":"unresolved","expectation_source":{"path":audit_path,"sha256":expected_hash},"request_evidence":{"path":request_path,"sha256":hash(&root.join(&request_path))?}});
            if root.join(&body_path).is_file() {
                spec["response"] = json!({"path":body_path,"sha256":hash(&root.join(&body_path))?});
            }
            let spec_path =
                format!("data/output/audit/artifact-retrieval-correspondence/{id}.toml");
            fs::write(root.join(&spec_path), toml::to_string(&spec)?)?;
            record["mutation_spec"] = json!(spec_path);
        } else {
            record["canonical_observation"] = artifacts.get(id).cloned().unwrap_or(Value::Null);
            record["unresolved_reason"] = json!(
                "Canonical ID/key/URL differs or is absent; retain compatibility observation for explicit identity reconciliation."
            );
        }
        results.push(record);
    }
    write(
        &directory.join("frontier-results.json"),
        &json!({"rows":results,"identity_boundary":"Matching expected bytes establishes response correspondence. Document identity remains separately unresolved; original hashes and failed responses remain retained."}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sensitive_fields_and_continuations_are_redacted_across_redirects() {
        let raw = b"HTTP/1.1 302 Found\r\nSeT-CoOkIe: secret\r\n\tfolded-secret\r\nLocation: https://example.org/\r\n\r\nHTTP/2 200\r\nAuthorization: Bearer secret\r\nContent-Length: 2\r\n";
        let expected = b"HTTP/1.1 302 Found\r\nSeT-CoOkIe: [REDACTED]\r\n [REDACTED]\r\nLocation: https://example.org/\r\n\r\nHTTP/2 200\r\nAuthorization: [REDACTED]\r\nContent-Length: 2\r\n";
        assert_eq!(sanitize_headers(raw), expected);
        assert_eq!(sanitize_headers(expected), expected);
        assert_eq!(
            sanitize_headers(b"Server: opaque\xff\n"),
            b"Server: opaque\xff\n"
        );
    }
}
