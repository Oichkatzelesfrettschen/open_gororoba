//! Literature-verification run persistence and query helpers.
//!
//! Methods added to `ProvenanceStore` via a second impl block:
//!   * `record_literature_verification_run`  -- atomic transaction
//!     that inserts the run row plus its results and similar-paper
//!     children
//!   * `recent_literature_verification_runs` -- N most-recent runs
//!     each enriched with their results + similar papers
//!   * `literature_verification_results_for_run` -- per-run results
//!   * `literature_novelty_similar_papers_for_run` -- per-run
//!     novelty similar-paper rows
//!
//! All four methods are pub on `ProvenanceStore` and re-exposed from
//! the parent module via the impl-block visibility rules. Access to
//! the private `self.conn: Connection` field works because this
//! submodule is a child of `lib.rs`, and Rust grants child modules
//! visibility into parent's private items.

use anyhow::Result;
use provenance_core::{
    LiteratureNoveltySimilarPaperRecord, LiteratureVerificationQueryResult,
    LiteratureVerificationResultRecord, LiteratureVerificationRunRecord,
};
use rusqlite::params;

use crate::ProvenanceStore;

impl ProvenanceStore {
    pub fn record_literature_verification_run(
        &mut self,
        run: &LiteratureVerificationRunRecord,
        results: &[LiteratureVerificationResultRecord],
        similar_papers: &[LiteratureNoveltySimilarPaperRecord],
    ) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO literature_verification_runs (
                input_path, topic, hypotheses_path, domains_json, search_queries_json,
                total_entries, verified_count, suspicious_count, hallucinated_count, skipped_count,
                integrity_score, novelty_score, novelty_assessment, recommendation,
                search_coverage, total_papers_retrieved, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                run.input_path,
                run.topic,
                run.hypotheses_path,
                serde_json::to_string(&run.domains)?,
                serde_json::to_string(&run.search_queries)?,
                run.total_entries as i64,
                run.verified_count as i64,
                run.suspicious_count as i64,
                run.hallucinated_count as i64,
                run.skipped_count as i64,
                run.integrity_score,
                run.novelty_score,
                run.novelty_assessment,
                run.recommendation,
                run.search_coverage,
                run.total_papers_retrieved.map(|value| value as i64),
                run.created_at,
            ],
        )?;
        let run_id = tx.last_insert_rowid();

        for result in results {
            tx.execute(
                "INSERT INTO literature_verification_results (
                    run_id, cite_key, title, status, confidence, method, details, doi, arxiv_id,
                    matched_paper_title, matched_paper_source, matched_paper_year,
                    matched_paper_url, relevance_score
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    run_id,
                    result.cite_key,
                    result.title,
                    result.status,
                    result.confidence,
                    result.method,
                    result.details,
                    result.doi,
                    result.arxiv_id,
                    result.matched_paper_title,
                    result.matched_paper_source,
                    result.matched_paper_year,
                    result.matched_paper_url,
                    result.relevance_score,
                ],
            )?;
        }

        for similar in similar_papers {
            tx.execute(
                "INSERT INTO literature_novelty_similar_papers (
                    run_id, title, paper_id, year, venue, citation_count, similarity, url, cite_key
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    run_id,
                    similar.title,
                    similar.paper_id,
                    similar.year,
                    similar.venue,
                    similar.citation_count,
                    similar.similarity,
                    similar.url,
                    similar.cite_key,
                ],
            )?;
        }

        tx.commit()?;
        Ok(run_id)
    }

    pub fn recent_literature_verification_runs(
        &self,
        limit: usize,
    ) -> Result<Vec<LiteratureVerificationQueryResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                id, input_path, topic, hypotheses_path, domains_json, search_queries_json,
                total_entries, verified_count, suspicious_count, hallucinated_count, skipped_count,
                integrity_score, novelty_score, novelty_assessment, recommendation,
                search_coverage, total_papers_retrieved, created_at
             FROM literature_verification_runs
             ORDER BY id DESC
             LIMIT ?1",
        )?;
        let run_rows = stmt.query_map(params![limit as i64], |row| {
            let domains_json: String = row.get(4)?;
            let search_queries_json: String = row.get(5)?;
            Ok(LiteratureVerificationRunRecord {
                id: row.get(0)?,
                input_path: row.get(1)?,
                topic: row.get(2)?,
                hypotheses_path: row.get(3)?,
                domains: serde_json::from_str(&domains_json).unwrap_or_default(),
                search_queries: serde_json::from_str(&search_queries_json).unwrap_or_default(),
                total_entries: row.get::<_, i64>(6)? as usize,
                verified_count: row.get::<_, i64>(7)? as usize,
                suspicious_count: row.get::<_, i64>(8)? as usize,
                hallucinated_count: row.get::<_, i64>(9)? as usize,
                skipped_count: row.get::<_, i64>(10)? as usize,
                integrity_score: row.get(11)?,
                novelty_score: row.get(12)?,
                novelty_assessment: row.get(13)?,
                recommendation: row.get(14)?,
                search_coverage: row.get(15)?,
                total_papers_retrieved: row.get::<_, Option<i64>>(16)?.map(|value| value as usize),
                created_at: row.get(17)?,
            })
        })?;

        let mut runs = Vec::new();
        for run in run_rows {
            let run = run?;
            let run_id = run.id.unwrap_or_default();
            runs.push(LiteratureVerificationQueryResult {
                results: self.literature_verification_results_for_run(run_id)?,
                similar_papers: self.literature_novelty_similar_papers_for_run(run_id)?,
                run,
            });
        }
        Ok(runs)
    }

    pub fn literature_verification_results_for_run(
        &self,
        run_id: i64,
    ) -> Result<Vec<LiteratureVerificationResultRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                id, run_id, cite_key, title, status, confidence, method, details, doi, arxiv_id,
                matched_paper_title, matched_paper_source, matched_paper_year,
                matched_paper_url, relevance_score
             FROM literature_verification_results
             WHERE run_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(LiteratureVerificationResultRecord {
                id: row.get(0)?,
                run_id: row.get(1)?,
                cite_key: row.get(2)?,
                title: row.get(3)?,
                status: row.get(4)?,
                confidence: row.get(5)?,
                method: row.get(6)?,
                details: row.get(7)?,
                doi: row.get(8)?,
                arxiv_id: row.get(9)?,
                matched_paper_title: row.get(10)?,
                matched_paper_source: row.get(11)?,
                matched_paper_year: row.get(12)?,
                matched_paper_url: row.get(13)?,
                relevance_score: row.get(14)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn literature_novelty_similar_papers_for_run(
        &self,
        run_id: i64,
    ) -> Result<Vec<LiteratureNoveltySimilarPaperRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                id, run_id, title, paper_id, year, venue, citation_count, similarity, url, cite_key
             FROM literature_novelty_similar_papers
             WHERE run_id = ?1
             ORDER BY similarity DESC, citation_count DESC, id ASC",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(LiteratureNoveltySimilarPaperRecord {
                id: row.get(0)?,
                run_id: row.get(1)?,
                title: row.get(2)?,
                paper_id: row.get(3)?,
                year: row.get(4)?,
                venue: row.get(5)?,
                citation_count: row.get(6)?,
                similarity: row.get(7)?,
                url: row.get(8)?,
                cite_key: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}
