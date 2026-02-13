//! Claims infrastructure: parsing, auditing, and verification.
//!
//! This module consolidates ~30 Python scripts into a unified Rust library
//! with two CLI entry points: `claims-audit` and `claims-verify`.
//!
//! # Architecture
//!
//! - `parser` -- Shared markdown table parser with escaped-pipe awareness
//! - `schema` -- Canonical tokens (status, task, domain) as static data
//! - `audit`  -- Read-only reports: ID inventory, status, staleness, etc.
//! - `verify` -- Verification checks that return pass/fail with failure messages

pub mod audit;
pub mod consolidate;
pub mod parser;
pub mod schema;
pub mod verify;
