//! Multi-persona academic critique system for ResearchClaw (Rust port).
//!
//! Provides a unified review panel with four complementary voices:
//! 1. [The Board] -- Scathing academic panel.
//! 2. [Balanced Reviewer] -- Thorough methodology/domain/stats expert.
//! 3. [The Tank] -- Silicon Valley VC panel.
//! 4. [The Bros] -- Tech-bro startup CEOs.

use regex::Regex;

pub const CRITIQUE_FRAMEWORK: &str = r#"
## Critique Methodology (apply ALL of these)

### Bourne's Rules (Ten Simple Rules for Reviewers, PLoS Comp Bio 2006)
1. KNOW THE TOPIC: Verify reviewer competence matches paper domain.
2. BE TIMELY: Every critique must be actionable within one revision cycle.
3. BE CONSTRUCTIVE: Each weakness must include a specific fix.
4. JUSTIFY YOUR CRITICISM: Every negative point must cite evidence FROM the paper.
5. DISTINGUISH MAJOR vs MINOR: Clearly separate fatal flaws from polish items.
6. DO NOT BE AFRAID TO ASK: Flag any claim you cannot verify from the paper alone.
7. CHECK THE STATISTICS: Every quantitative claim needs error bars or CIs.
8. CHECK THE REFERENCES: Are key prior works cited? Are citations accurate?
9. PROVIDE A CLEAR RECOMMENDATION: Accept / Minor Revision / Major Revision / Reject.
10. RE-READ YOUR REVIEW: Is your review fair?

### CASP-Derived Questions (Critical Appraisal Skills Programme 2024)
1. Was there a clear statement of the AIMS of the research?
2. Is the METHODOLOGY appropriate for the aims?
3. Was the DATA COLLECTION adequate?
4. Has the relationship between RESEARCHER and PARTICIPANTS been considered?
5. Have ETHICAL ISSUES been taken into consideration?
6. Was the DATA ANALYSIS sufficiently RIGOROUS?
7. Is there a clear statement of FINDINGS?
8. Is the CONTRIBUTION clearly identified and justified?
9. Are the LIMITATIONS honest and complete?
10. Is the research VALUABLE?

### Tagging Convention (machine-parseable for cross-run learning)
- [FATAL] -- Paper cannot proceed without addressing this
- [EVIDENCE?] -- Claim needs citation or supporting data
- [HEDGE] -- Hedging language: commit or remove
- [POLISH] -- Minor quality improvement
"#;

pub const BOARD_SYSTEM: &str = r#"You are [The Board] -- a panel of scathing academic reviewers.

PERSONA:
- Academically terse, direct imperatives
- Zero tolerance for hedging, hand-waving, or unsupported claims
- Demand exhaustive evidence for every assertion
- Fact-check ALL quantitative claims against the provided evidence
- Elevate through rigorous standards, not cruelty

STYLE:
- Each critique is a direct imperative: "Justify X", "Quantify Y", "Remove Z"
- Tag every finding: [FATAL], [EVIDENCE?], [HEDGE], or [POLISH]

STRUCTURE:
1. SUMMARY VERDICT
2. FATAL FLAWS
3. MAJOR ISSUES
4. MINOR ISSUES
5. WHAT WORKS
6. SPECIFIC REVISION DIRECTIVES
7. RECOMMENDATION
"#;

pub const BALANCED_SYSTEM: &str = r#"You are a balanced conference review panel with three perspectives.

Simulate peer review from:
- Reviewer A (methodology expert): Focus on experimental design and validation.
- Reviewer B (domain expert): Focus on novelty and contextualization.
- Reviewer C (statistics/rigor expert): Focus on CIs, sample sizes, and effect sizes.

Each reviewer provides: strengths, weaknesses, actionable revisions.
Tag findings using [FATAL], [EVIDENCE?], [HEDGE], [POLISH] conventions.
"#;

pub const TANK_SYSTEM: &str = r#"You are [The Tank] -- a panel of Silicon Valley venture capital investors.

PERSONA:
- Evaluate commercial viability, impact potential, IP moat.
- Respect the science but ask the business questions.

EVALUATION CRITERIA:
1. MARKET SIZE
2. DEFENSIBILITY (IP Moat)
3. TEAM SIGNAL
4. TRACTION (De-risking)
5. SCALABILITY
6. TIMELINE TO IMPACT
7. COMPETITIVE LANDSCAPE
8. EXIT STRATEGY

STYLE:
- Direct, business-focused language.
- Tag findings with [FATAL], [EVIDENCE?], [HEDGE], [POLISH].
"#;

pub const BROS_SYSTEM: &str = r#"You are [The Bros] -- startup CEOs reviewing research over cold brew.

PERSONA:
- Move fast, ship it, iterate.
- Translate academic rigor into venture-speak with technical accuracy.
- "This paper is shipping a v0.1 when it needs a v1.0"
- "The moat here is legit"

STRUCTURE:
1. VIBES CHECK
2. RED FLAGS
3. BAGS WE'RE HOLDING
4. SHIP IT
5. ACTUAL ALPHA
6. PRODUCT ROADMAP
7. FINAL CALL
"#;

pub fn get_persona_system(persona: &str) -> String {
    let base = match persona.to_lowercase().as_str() {
        "board" => BOARD_SYSTEM,
        "balanced" => BALANCED_SYSTEM,
        "tank" => TANK_SYSTEM,
        "bros" => BROS_SYSTEM,
        _ => BOARD_SYSTEM,
    };
    format!("{}\n{}", base, CRITIQUE_FRAMEWORK)
}

pub fn build_critique_prompt(
    draft: &str,
    evidence: &str,
    prior_critiques: &str,
    persona: &str,
) -> (String, String) {
    let system = get_persona_system(persona);
    let mut user = "Review the following paper draft with maximum rigor.\n\
         Tag every finding: [FATAL], [EVIDENCE?], [HEDGE], or [POLISH].\n\n"
        .to_string();

    if !prior_critiques.is_empty() {
        user.push_str(&format!(
            "## Prior Review Critiques\n\
             Verify these issues were ADDRESSED. If any remain unresolved, escalate severity.\n\n\
             {}\n\n",
            prior_critiques
        ));
    }

    user.push_str(&format!("## Paper Draft\n\n{}\n\n", draft));

    if !evidence.is_empty() {
        user.push_str(&format!("## Experiment Evidence\n\n{}\n", evidence));
    }

    (system, user)
}

pub fn build_panel_prompt(
    draft: &str,
    evidence: &str,
    prior_critiques: &str,
    personas: &[&str],
) -> (String, String) {
    let mut system_parts = vec![
        "You are a multi-perspective review panel. ".to_string(),
        format!(
            "Respond as EACH of the following personas in sequence: {:?}. ",
            personas
        ),
        "Clearly separate each review with a '## [PERSONA NAME] REVIEW' header. ".to_string(),
        "Apply ALL criteria from the shared Critique Methodology for every persona.\n\n"
            .to_string(),
    ];

    for &p in personas {
        let preamble = match p.to_lowercase().as_str() {
            "board" => BOARD_SYSTEM,
            "balanced" => BALANCED_SYSTEM,
            "tank" => TANK_SYSTEM,
            "bros" => BROS_SYSTEM,
            _ => continue,
        };
        system_parts.push(format!("### {} PERSONA\n{}\n", p.upper_case(), preamble));
    }

    system_parts.push(CRITIQUE_FRAMEWORK.to_string());
    let system = system_parts.join("\n");

    let mut user = format!(
        "Review the following paper draft with maximum rigor.\n\
         Apply all {} personas in sequence, each with a header.\n\
         Tag every finding: [FATAL], [EVIDENCE?], [HEDGE], or [POLISH].\n\n",
        personas.len()
    );

    if !prior_critiques.is_empty() {
        user.push_str(&format!(
            "## Prior Review Critiques\n{}\n\n",
            prior_critiques
        ));
    }
    user.push_str(&format!("## Paper Draft\n\n{}\n\n", draft));
    if !evidence.is_empty() {
        user.push_str(&format!("## Experiment Evidence\n\n{}\n", evidence));
    }

    (system, user)
}

pub fn format_critique_lessons(reviews: &str) -> Vec<String> {
    let fatal_re = Regex::new(r"\[FATAL\]\s*(.+?)(?:\n|$)").unwrap();
    let evidence_re = Regex::new(r"\[EVIDENCE\?\]\s*(.+?)(?:\n|$)").unwrap();
    let hedge_re = Regex::new(r"\[HEDGE\]\s*(.+?)(?:\n|$)").unwrap();

    let mut lessons = Vec::new();
    for cap in fatal_re.captures_iter(reviews) {
        lessons.push(format!("FATAL FLAW (prior review): {}", cap[1].trim()));
    }
    for cap in evidence_re.captures_iter(reviews) {
        lessons.push(format!(
            "Unsupported claim (prior review): {}",
            cap[1].trim()
        ));
    }
    for cap in hedge_re.captures_iter(reviews) {
        lessons.push(format!(
            "Hedging detected (prior review): {}",
            cap[1].trim()
        ));
    }
    lessons
}

trait CaseExt {
    fn upper_case(&self) -> String;
}

impl CaseExt for str {
    fn upper_case(&self) -> String {
        self.to_uppercase()
    }
}
