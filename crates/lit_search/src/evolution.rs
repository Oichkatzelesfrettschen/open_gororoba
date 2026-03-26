//! Self-evolution system for the ResearchClaw pipeline (Rust port).
//!
//! Records lessons from each pipeline run (failures, slow stages, quality issues)
//! and injects them into future runs as prompt overlays.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LessonCategory {
    System,     // Environment / network / timeout
    Experiment, // Code validation, sandbox timeout
    Writing,    // Paper quality issues
    Analysis,   // Weak analysis, missing comparison
    Literature, // Search / verification failures
    Pipeline,   // Stage orchestration issues
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LessonEntry {
    pub stage_name: String,
    pub stage_num: u32,
    pub category: LessonCategory,
    pub severity: String, // "info", "warning", "error"
    pub description: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub run_id: String,
}

const HALF_LIFE_DAYS: f64 = 30.0;
const MAX_AGE_DAYS: f64 = 90.0;

/// JSONL-backed store for pipeline lessons.
pub struct EvolutionStore {
    path: PathBuf,
}

impl EvolutionStore {
    pub fn new(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let path = dir.join("lessons.jsonl");
        Ok(Self { path })
    }

    pub fn append(&self, lesson: &LessonEntry) -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(lesson)?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    pub fn append_many(&self, lessons: &[LessonEntry]) -> anyhow::Result<()> {
        if lessons.is_empty() {
            return Ok(());
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        for lesson in lessons {
            let line = serde_json::to_string(lesson)?;
            writeln!(file, "{}", line)?;
        }
        info!("Appended {} lessons to evolution store", lessons.len());
        Ok(())
    }

    pub fn load_all(&self) -> Vec<LessonEntry> {
        if !self.path.exists() {
            return Vec::new();
        }
        let file = match fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::new(file);
        reader
            .lines()
            .filter_map(|line| {
                let line = line.ok()?;
                serde_json::from_str(&line).ok()
            })
            .collect()
    }

    fn time_weight(timestamp: DateTime<Utc>) -> f64 {
        let age = Utc::now() - timestamp;
        let age_days = age.num_seconds() as f64 / 86400.0;
        if age_days > MAX_AGE_DAYS {
            return 0.0;
        }
        (-age_days * std::f64::consts::LN_2 / HALF_LIFE_DAYS).exp()
    }

    pub fn query_for_stage(&self, stage_name: &str, max_lessons: usize) -> Vec<LessonEntry> {
        let all = self.load_all();
        let mut scored: Vec<(f64, LessonEntry)> = all
            .into_iter()
            .filter_map(|l| {
                let mut weight = Self::time_weight(l.timestamp);
                if weight <= 0.0 {
                    return None;
                }
                // Boost direct stage matches
                if l.stage_name == stage_name {
                    weight *= 2.0;
                }
                // Boost errors
                if l.severity == "error" {
                    weight *= 1.5;
                }
                Some((weight, l))
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(max_lessons).map(|x| x.1).collect()
    }

    pub fn build_overlay(
        &self,
        stage_name: &str,
        max_lessons: usize,
        skills_dir: Option<&Path>,
    ) -> String {
        let mut parts = Vec::new();

        // Intra-run lessons
        let lessons = self.query_for_stage(stage_name, max_lessons);
        if !lessons.is_empty() {
            parts.push("## Lessons from Prior Runs".to_string());
            for (i, l) in lessons.iter().enumerate() {
                let icon = match l.severity.as_str() {
                    "error" => "❌",
                    "warning" => "⚠️",
                    "info" => "ℹ️",
                    _ => "•",
                };
                parts.push(format!(
                    "{}. {} [{:?}] {}",
                    i + 1,
                    icon,
                    l.category,
                    l.description
                ));
            }
            parts.push("\nUse these lessons to avoid repeating past mistakes.".to_string());
        }

        // Cross-run arc-* skills (ported from python version)
        if let Some(sd) = skills_dir
            && let Ok(entries) = fs::read_dir(sd)
        {
            let mut arc_skills = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir()
                    && path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.starts_with("arc-"))
                {
                    let skill_file = path.join("SKILL.md");
                    if let Ok(text) = fs::read_to_string(skill_file) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            arc_skills.push(trimmed.to_string());
                        }
                    }
                }
            }
            arc_skills.sort();
            if !arc_skills.is_empty() {
                parts.push("\n## Learned Skills from Prior Runs".to_string());
                for skill in arc_skills.iter().take(5) {
                    parts.push(skill.clone());
                }
                parts.push("\nApply these skills proactively to improve quality.".to_string());
            }
        }

        parts.join("\n")
    }
}
