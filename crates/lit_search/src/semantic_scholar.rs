//! Semantic Scholar helpers shared across live search and dataset access.

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::OnceLock;
use thiserror::Error;
use tokio::{
    sync::Mutex,
    time::{Duration, Instant, sleep},
};

const SEMANTIC_SCHOLAR_BASE_URL: &str = "https://api.semanticscholar.org";
const SEMANTIC_SCHOLAR_MIN_INTERVAL_MS: u64 = 1_100;

static SEMANTIC_SCHOLAR_GATE: OnceLock<Mutex<Instant>> = OnceLock::new();

#[derive(Debug, Error)]
pub enum SemanticScholarError {
    #[error("Semantic Scholar API key not set")]
    MissingApiKey,
    #[error("Semantic Scholar rate limited")]
    RateLimited,
    #[error("Semantic Scholar request forbidden")]
    Forbidden,
    #[error("Semantic Scholar unavailable: {0}")]
    Unavailable(String),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "README", default)]
    pub readme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRelease {
    pub release_id: String,
    #[serde(rename = "README", default)]
    pub readme: String,
    #[serde(default)]
    pub datasets: Vec<DatasetSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "README", default)]
    pub readme: String,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDiffEntry {
    pub from_release: String,
    pub to_release: String,
    #[serde(default)]
    pub update_files: Vec<String>,
    #[serde(default)]
    pub delete_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDiff {
    pub dataset: String,
    pub start_release: String,
    pub end_release: String,
    #[serde(default)]
    pub diffs: Vec<DatasetDiffEntry>,
}

pub async fn wait_for_semantic_scholar_slot() {
    let gate = SEMANTIC_SCHOLAR_GATE.get_or_init(|| Mutex::new(Instant::now()));
    let mut next_allowed = gate.lock().await;
    let now = Instant::now();
    if *next_allowed > now {
        sleep(*next_allowed - now).await;
    }
    *next_allowed = Instant::now() + Duration::from_millis(SEMANTIC_SCHOLAR_MIN_INTERVAL_MS);
}

pub struct SemanticScholarDatasetsClient {
    client: Client,
    api_key: String,
}

impl SemanticScholarDatasetsClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
        }
    }

    pub fn with_client(client: Client, api_key: impl Into<String>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
        }
    }

    pub async fn list_releases(&self) -> Result<Vec<String>, SemanticScholarError> {
        self.get_json("/datasets/v1/release/").await
    }

    pub async fn latest_release_id(&self) -> Result<String, SemanticScholarError> {
        let releases = self.list_releases().await?;
        releases
            .into_iter()
            .last()
            .ok_or_else(|| SemanticScholarError::Unavailable("no releases returned".into()))
    }

    pub async fn fetch_release(
        &self,
        release_id: &str,
    ) -> Result<DatasetRelease, SemanticScholarError> {
        self.get_json(&format!("/datasets/v1/release/{release_id}"))
            .await
    }

    pub async fn fetch_dataset_manifest(
        &self,
        release_id: &str,
        dataset_name: &str,
    ) -> Result<DatasetManifest, SemanticScholarError> {
        self.get_json(&format!(
            "/datasets/v1/release/{release_id}/dataset/{dataset_name}"
        ))
        .await
    }

    pub async fn fetch_dataset_diff(
        &self,
        start_release_id: &str,
        end_release_id: &str,
        dataset_name: &str,
    ) -> Result<DatasetDiff, SemanticScholarError> {
        self.get_json(&format!(
            "/datasets/v1/diffs/{start_release_id}/to/{end_release_id}/{dataset_name}"
        ))
        .await
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, SemanticScholarError> {
        if self.api_key.is_empty() {
            return Err(SemanticScholarError::MissingApiKey);
        }
        wait_for_semantic_scholar_slot().await;
        let resp = self
            .client
            .get(format!("{SEMANTIC_SCHOLAR_BASE_URL}{path}"))
            .header("x-api-key", &self.api_key)
            .send()
            .await?;
        let status = resp.status();
        if status == StatusCode::TOO_MANY_REQUESTS {
            return Err(SemanticScholarError::RateLimited);
        }
        if status == StatusCode::FORBIDDEN {
            return Err(SemanticScholarError::Forbidden);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SemanticScholarError::Unavailable(format!(
                "HTTP {}: {}",
                status.as_u16(),
                body
            )));
        }
        Ok(resp.json::<T>().await?)
    }
}
