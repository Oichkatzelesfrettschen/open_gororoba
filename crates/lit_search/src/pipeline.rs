//! Research pipeline stage contracts (Rust port).
//!
//! Encodes the 23-stage ResearchClaw methodology with strict I/O contracts,
//! "Definition of Done" (DoD) criteria, and error diagnostics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    // Phase A: Research Scoping
    TopicInit = 1,
    ProblemDecompose = 2,
    // Phase B: Literature Discovery
    SearchStrategy = 3,
    LiteratureCollect = 4,
    LiteratureScreen = 5,
    KnowledgeExtract = 6,
    // Phase C: Knowledge Synthesis
    Synthesis = 7,
    HypothesisGen = 8,
    // Phase D: Experiment Design
    ExperimentDesign = 9,
    CodeGeneration = 10,
    ResourcePlanning = 11,
    ExperimentRun = 12,
    // Phase E: Analysis & Refinement
    IterativeRefine = 13,
    ResultAnalysis = 14,
    ResearchDecision = 15,
    // Phase F: Dissemination
    PaperOutline = 16,
    PaperDraft = 17,
    PeerReview = 18,
    PaperRevision = 19,
    QualityGate = 20,
    KnowledgeArchive = 21,
    ExportPublish = 22,
    CitationVerify = 23,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageContract {
    pub stage: Stage,
    pub input_files: Vec<String>,
    pub output_files: Vec<String>,
    pub dod: String,
    pub error_code: String,
    pub max_retries: u32,
}

pub fn get_contracts() -> HashMap<Stage, StageContract> {
    let mut c = HashMap::new();

    c.insert(
        Stage::TopicInit,
        StageContract {
            stage: Stage::TopicInit,
            input_files: vec![],
            output_files: vec!["goal.md".into(), "hardware_profile.json".into()],
            dod: "SMART goal statement with topic, scope, and constraints".into(),
            error_code: "E01_INVALID_GOAL".into(),
            max_retries: 0,
        },
    );

    c.insert(
        Stage::ProblemDecompose,
        StageContract {
            stage: Stage::ProblemDecompose,
            input_files: vec!["goal.md".into()],
            output_files: vec!["problem_tree.md".into()],
            dod: ">=3 prioritized sub-questions identified".into(),
            error_code: "E02_DECOMP_FAIL".into(),
            max_retries: 1,
        },
    );

    c.insert(
        Stage::SearchStrategy,
        StageContract {
            stage: Stage::SearchStrategy,
            input_files: vec!["problem_tree.md".into()],
            output_files: vec![
                "search_plan.yaml".into(),
                "sources.json".into(),
                "queries.json".into(),
            ],
            dod: ">=2 search strategies defined with verified data sources".into(),
            error_code: "E03_STRATEGY_BAD".into(),
            max_retries: 1,
        },
    );

    c.insert(
        Stage::LiteratureCollect,
        StageContract {
            stage: Stage::LiteratureCollect,
            input_files: vec!["search_plan.yaml".into()],
            output_files: vec!["candidates.jsonl".into()],
            dod: ">=N candidate papers collected from specified sources".into(),
            error_code: "E04_COLLECT_EMPTY".into(),
            max_retries: 2,
        },
    );

    c.insert(
        Stage::LiteratureScreen,
        StageContract {
            stage: Stage::LiteratureScreen,
            input_files: vec!["candidates.jsonl".into()],
            output_files: vec!["shortlist.jsonl".into()],
            dod: "Relevance + quality dual screening completed and approved".into(),
            error_code: "E05_GATE_REJECT".into(),
            max_retries: 0,
        },
    );

    c.insert(
        Stage::KnowledgeExtract,
        StageContract {
            stage: Stage::KnowledgeExtract,
            input_files: vec!["shortlist.jsonl".into()],
            output_files: vec!["cards/".into()],
            dod: "Structured knowledge card per shortlisted paper".into(),
            error_code: "E06_EXTRACT_FAIL".into(),
            max_retries: 1,
        },
    );

    c.insert(
        Stage::Synthesis,
        StageContract {
            stage: Stage::Synthesis,
            input_files: vec!["cards/".into()],
            output_files: vec!["synthesis.md".into()],
            dod: "Topic clusters + >=2 research gaps identified".into(),
            error_code: "E07_SYNTHESIS_WEAK".into(),
            max_retries: 1,
        },
    );

    c.insert(
        Stage::HypothesisGen,
        StageContract {
            stage: Stage::HypothesisGen,
            input_files: vec!["synthesis.md".into()],
            output_files: vec!["hypotheses.md".into()],
            dod: ">=2 falsifiable research hypotheses".into(),
            error_code: "E08_HYP_INVALID".into(),
            max_retries: 1,
        },
    );

    // ... More stages could be added here following the same pattern ...

    c
}
