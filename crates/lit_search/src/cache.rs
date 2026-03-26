//! Local JSON cache for literature search results.

use crate::models::Paper;
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_TTL_SEC: u64 = 7 * 24 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchCachePayload {
    query: String,
    source: String,
    limit: usize,
    timestamp: u64,
    papers: Vec<Paper>,
}

fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn home_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".cache").join("lit_search").join("search"))
}

fn source_ttl_sec(source: &str) -> u64 {
    match source {
        "arxiv" => 24 * 60 * 60,
        "semantic_scholar" | "s2" | "openalex" => 3 * 24 * 60 * 60,
        _ => DEFAULT_TTL_SEC,
    }
}

fn search_cache_key(query: &str, source: &str, limit: usize) -> String {
    let mut hasher = Hasher::new();
    hasher.update(query.trim().to_ascii_lowercase().as_bytes());
    hasher.update(b"|");
    hasher.update(source.trim().to_ascii_lowercase().as_bytes());
    hasher.update(b"|");
    hasher.update(limit.to_string().as_bytes());
    hasher.finalize().to_hex()[..16].to_string()
}

pub fn get_cached_search(query: &str, source: &str, limit: usize) -> Option<Vec<Paper>> {
    let cache_dir = home_cache_dir()?;
    let _ = fs::create_dir_all(&cache_dir);
    let path = cache_dir.join(format!("{}.json", search_cache_key(query, source, limit)));
    let text = fs::read_to_string(path).ok()?;
    let payload: SearchCachePayload = serde_json::from_str(&text).ok()?;
    let age_sec = current_unix_ts().saturating_sub(payload.timestamp);
    if age_sec > source_ttl_sec(source) {
        return None;
    }
    Some(payload.papers)
}

pub fn put_cached_search(query: &str, source: &str, limit: usize, papers: &[Paper]) {
    let Some(cache_dir) = home_cache_dir() else {
        return;
    };
    let _ = fs::create_dir_all(&cache_dir);
    let path = cache_dir.join(format!("{}.json", search_cache_key(query, source, limit)));
    let payload = SearchCachePayload {
        query: query.to_string(),
        source: source.to_string(),
        limit,
        timestamp: current_unix_ts(),
        papers: papers.to_vec(),
    };
    if let Ok(text) = serde_json::to_string_pretty(&payload) {
        let _ = fs::write(path, text);
    }
}

#[cfg(test)]
mod tests {
    use super::search_cache_key;

    #[test]
    fn search_cache_key_is_case_insensitive() {
        assert_eq!(
            search_cache_key("Cayley Dickson", "openalex", 10),
            search_cache_key("cayley dickson", "OpenAlex", 10)
        );
    }
}
