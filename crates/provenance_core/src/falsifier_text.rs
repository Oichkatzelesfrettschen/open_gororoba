//! Strict compatibility parsing for textual and categorized claim falsifiers.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ClaimFalsifier {
    Legacy(String),
    Structured(FalsifierOutcomes),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "RawOutcomes")]
pub struct FalsifierOutcomes {
    pub verification_outcomes: Vec<String>,
    pub revision_outcomes: Vec<String>,
    pub abandonment_outcomes: Vec<String>,
    pub inconclusive_outcomes: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutcomes {
    verification_outcomes: Vec<String>,
    revision_outcomes: Vec<String>,
    abandonment_outcomes: Vec<String>,
    inconclusive_outcomes: Vec<String>,
}

impl TryFrom<RawOutcomes> for FalsifierOutcomes {
    type Error = String;
    fn try_from(raw: RawOutcomes) -> Result<Self, Self::Error> {
        let mut seen = BTreeSet::new();
        for (category, outcomes) in [
            ("verification_outcomes", &raw.verification_outcomes),
            ("revision_outcomes", &raw.revision_outcomes),
            ("abandonment_outcomes", &raw.abandonment_outcomes),
            ("inconclusive_outcomes", &raw.inconclusive_outcomes),
        ] {
            if outcomes.is_empty() {
                return Err(format!("{category} requires at least one outcome"));
            }
            for outcome in outcomes {
                if outcome.trim().is_empty() || !seen.insert(outcome.trim()) {
                    return Err(format!("{category} contains an empty or duplicate outcome"));
                }
            }
        }
        Ok(Self {
            verification_outcomes: raw.verification_outcomes,
            revision_outcomes: raw.revision_outcomes,
            abandonment_outcomes: raw.abandonment_outcomes,
            inconclusive_outcomes: raw.inconclusive_outcomes,
        })
    }
}

impl ClaimFalsifier {
    /// Produce a stable single-line display while retaining category labels and references.
    pub fn project(&self) -> String {
        match self {
            Self::Legacy(text) => text.clone(),
            Self::Structured(outcomes) => [
                ("Verification", &outcomes.verification_outcomes),
                ("Revision", &outcomes.revision_outcomes),
                ("Abandonment", &outcomes.abandonment_outcomes),
                ("Inconclusive", &outcomes.inconclusive_outcomes),
            ]
            .iter()
            .map(|(label, values)| {
                let text = values
                    .iter()
                    .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!("{label}: {text}")
            })
            .collect::<Vec<_>>()
            .join(". "),
        }
    }
}

pub fn deserialize_text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    ClaimFalsifier::deserialize(deserializer).map(|value| value.project())
}

pub fn deserialize_optional_text<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    Option::<ClaimFalsifier>::deserialize(deserializer)
        .map(|value| value.map(|value| value.project()))
}

/// Preserve absence separately from malformed data so callers can select an explicit fallback.
pub fn project_optional<'de, D: Deserializer<'de>>(
    value: Option<D>,
) -> Result<Option<String>, D::Error> {
    value.map(deserialize_text).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    const STRUCTURED: &str = "[what_would_verify_refute]\nverification_outcomes=['C-101 E-201 `verify.rs`']\nrevision_outcomes=['C-102 E-202 `revise.rs`']\nabandonment_outcomes=['C-103 E-203 `abandon.rs`']\ninconclusive_outcomes=['C-104 E-204 `inconclusive.rs`']\n";
    #[derive(Deserialize)]
    struct Required {
        #[serde(default, deserialize_with = "deserialize_text")]
        what_would_verify_refute: String,
    }
    #[derive(Deserialize)]
    struct Optional {
        #[serde(default, deserialize_with = "deserialize_optional_text")]
        what_would_verify_refute: Option<String>,
    }
    #[test]
    fn textual_and_structured_forms_preserve_every_category_reference() {
        let legacy: Required = toml::from_str("what_would_verify_refute='legacy C-100'").unwrap();
        assert_eq!(legacy.what_would_verify_refute, "legacy C-100");
        let required: Required = toml::from_str(STRUCTURED).unwrap();
        let optional: Optional = toml::from_str(STRUCTURED).unwrap();
        assert_eq!(
            optional.what_would_verify_refute.as_deref(),
            Some(required.what_would_verify_refute.as_str())
        );
        for reference in [
            "C-101",
            "E-201",
            "verify.rs",
            "C-102",
            "E-202",
            "revise.rs",
            "C-103",
            "E-203",
            "abandon.rs",
            "C-104",
            "E-204",
            "inconclusive.rs",
        ] {
            assert!(required.what_would_verify_refute.contains(reference));
        }
        assert!(
            toml::from_str::<Optional>("")
                .unwrap()
                .what_would_verify_refute
                .is_none()
        );
        let dynamic: toml::Value = toml::from_str(STRUCTURED).unwrap();
        assert_eq!(
            project_optional(dynamic.get("what_would_verify_refute").cloned()).unwrap(),
            Some(required.what_would_verify_refute)
        );
    }
    #[test]
    fn malformed_structures_fail_instead_of_becoming_empty_text() {
        for malformed in [
            "what_would_verify_refute=42".to_string(),
            STRUCTURED.replace("verification_outcomes=", "unknown_outcomes="),
            STRUCTURED.replace("['C-104 E-204 `inconclusive.rs`']", "[]"),
            STRUCTURED.replace("C-104 E-204 `inconclusive.rs`", "C-101 E-201 `verify.rs`"),
            STRUCTURED.replace("C-104 E-204 `inconclusive.rs`", " "),
            format!("{STRUCTURED}unknown_field='reject'\n"),
        ] {
            assert!(toml::from_str::<Required>(&malformed).is_err());
            let dynamic: toml::Value = toml::from_str(&malformed).unwrap();
            assert!(project_optional(dynamic.get("what_would_verify_refute").cloned()).is_err());
        }
    }
    #[test]
    fn untagged_serialization_retains_the_structured_table() {
        let document: toml::Value = toml::from_str(STRUCTURED).unwrap();
        let parsed: ClaimFalsifier = document["what_would_verify_refute"]
            .clone()
            .try_into()
            .unwrap();
        let serialized = toml::Value::try_from(&parsed).unwrap();
        assert_eq!(serialized, document["what_would_verify_refute"]);
    }
}
