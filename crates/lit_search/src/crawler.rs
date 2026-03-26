//! Web page → Markdown extraction crawler (Rust port).
//!
//! Primary engine: chromiumoxide (Headless Chrome) for JS-heavy sites.
//! Fallback: reqwest + scraper for lightweight HTML extraction.

use anyhow::Result;
use chromiumoxide::{
    browser::{Browser, BrowserConfig},
    handler::viewport::Viewport,
};
use futures::StreamExt;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlResult {
    pub url: String,
    pub markdown: String,
    pub title: String,
    pub success: bool,
    pub error: Option<String>,
    pub elapsed_seconds: f64,
}

pub struct WebCrawler {
    client: Client,
    timeout: Duration,
    max_content_length: usize,
}

impl Default for WebCrawler {
    fn default() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("ResearchClaw/0.5 (Academic Research Bot)")
                .build()
                .unwrap(),
            timeout: Duration::from_secs(30),
            max_content_length: 100_000,
        }
    }
}

impl WebCrawler {
    pub fn new(timeout_secs: u64, max_len: usize) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .user_agent("ResearchClaw/0.5 (Academic Research Bot)")
                .build()
                .unwrap(),
            timeout: Duration::from_secs(timeout_secs),
            max_content_length: max_len,
        }
    }

    /// Primary crawl entry point. Tries chromiumoxide first, then falls back to reqwest.
    pub async fn crawl(&self, url: &str) -> CrawlResult {
        let start = Instant::now();

        // Attempt chromiumoxide (JS rendering)
        match self.crawl_with_chrome(url).await {
            Ok(res) => res,
            Err(e) => {
                debug!(
                    "Chrome crawl failed for {}: {}, trying reqwest fallback",
                    url, e
                );
                match self.crawl_with_reqwest(url).await {
                    Ok(res) => res,
                    Err(e2) => CrawlResult {
                        url: url.to_string(),
                        markdown: String::new(),
                        title: String::new(),
                        success: false,
                        error: Some(format!("All backends failed: {}", e2)),
                        elapsed_seconds: start.elapsed().as_secs_f64(),
                    },
                }
            }
        }
    }

    async fn crawl_with_chrome(&self, url: &str) -> Result<CrawlResult> {
        let start = Instant::now();

        let (mut browser, mut handler) = Browser::launch(
            BrowserConfig::builder()
                .request_timeout(self.timeout)
                .viewport(Viewport::default())
                .build()
                .map_err(|e| anyhow::anyhow!(e))?,
        )
        .await?;

        // Spawn the handler in the background
        let handle = tokio::spawn(async move {
            while let Some(h) = handler.next().await {
                if h.is_err() {
                    break;
                }
            }
        });

        let page = browser.new_page(url).await?;

        // Wait for page load and some stability
        tokio::time::sleep(Duration::from_millis(2000)).await;

        let html = page.content().await?;
        let title = page.get_title().await?.unwrap_or_default();

        let markdown = self.html_to_markdown(&html);

        browser.close().await?;
        let _ = browser.wait().await;
        handle.abort();

        Ok(CrawlResult {
            url: url.to_string(),
            markdown: self.truncate(&markdown),
            title,
            success: true,
            error: None,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        })
    }

    async fn crawl_with_reqwest(&self, url: &str) -> Result<CrawlResult> {
        let start = Instant::now();
        let resp = self.client.get(url).send().await?;
        let html = resp.text().await?;

        let document = Html::parse_document(&html);
        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default();

        let markdown = self.html_to_markdown(&html);

        Ok(CrawlResult {
            url: url.to_string(),
            markdown: self.truncate(&markdown),
            title,
            success: !markdown.is_empty(),
            error: None,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        })
    }

    fn truncate(&self, s: &str) -> String {
        if s.len() > self.max_content_length {
            format!("{}\n\n[... truncated]", &s[..self.max_content_length])
        } else {
            s.to_string()
        }
    }

    /// Robust HTML → Markdown conversion.
    /// Removes nav, footer, header, scripts, styles.
    fn html_to_markdown(&self, html: &str) -> String {
        let document = Html::parse_document(html);

        // Remove unwanted elements
        // Scraper doesn't support easy removal, so we use a selective extraction approach

        let main_content_selectors = [
            "article",
            "main",
            "[role='main']",
            ".content",
            ".main-content",
            "#content",
        ];

        let mut content_html = String::new();
        let mut found_main = false;

        for sel in main_content_selectors {
            if let Ok(selector) = Selector::parse(sel)
                && let Some(element) = document.select(&selector).next()
            {
                content_html = element.html();
                found_main = true;
                break;
            }
        }

        if !found_main {
            content_html = html.to_string();
        }

        // Clean and convert via regex (ported/improved from python)
        let mut text = content_html;

        // Strip scripts and styles
        let re_script = regex::Regex::new(
            r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>",
        )
        .unwrap();
        text = re_script.replace_all(&text, "").to_string();

        // Headings
        let re_h1 = regex::Regex::new(r"(?i)<h1[^>]*>(.*?)</h1>").unwrap();
        text = re_h1.replace_all(&text, "\n# $1\n").to_string();
        let re_h2 = regex::Regex::new(r"(?i)<h2[^>]*>(.*?)</h2>").unwrap();
        text = re_h2.replace_all(&text, "\n## $1\n").to_string();
        let re_h3 = regex::Regex::new(r"(?i)<h3[^>]*>(.*?)</h3>").unwrap();
        text = re_h3.replace_all(&text, "\n### $1\n").to_string();

        // Lists
        let re_li = regex::Regex::new(r"(?i)<li[^>]*>(.*?)</li>").unwrap();
        text = re_li.replace_all(&text, "\n- $1").to_string();

        // Paragraphs and breaks
        let re_p = regex::Regex::new(r"(?i)<p[^>]*>(.*?)</p>").unwrap();
        text = re_p.replace_all(&text, "\n$1\n").to_string();
        let re_br = regex::Regex::new(r"(?i)<br\s*/?>").unwrap();
        text = re_br.replace_all(&text, "\n").to_string();

        // Links
        let re_a = regex::Regex::new(r#"(?i)<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)</a>"#).unwrap();
        text = re_a.replace_all(&text, "[$2]($1)").to_string();

        // Strip remaining tags
        let re_tags = regex::Regex::new(r"<[^>]+>").unwrap();
        text = re_tags.replace_all(&text, "").to_string();

        // Unescape common entities
        text = text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ");

        // Collapse whitespace
        let re_newlines = regex::Regex::new(r"\n{3,}").unwrap();
        text = re_newlines.replace_all(&text, "\n\n").to_string();
        let re_spaces = regex::Regex::new(r" {2,}").unwrap();
        text = re_spaces.replace_all(&text, " ").to_string();

        text.trim().to_string()
    }
}
