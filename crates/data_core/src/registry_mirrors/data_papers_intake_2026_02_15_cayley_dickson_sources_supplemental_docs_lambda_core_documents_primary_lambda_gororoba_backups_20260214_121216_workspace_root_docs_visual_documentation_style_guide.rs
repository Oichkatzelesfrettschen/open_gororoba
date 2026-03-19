//! # Visual Documentation Style Guide (Dark Mode)
//!
//! ## Purpose
//! Define a consistent visual language for static dark-mode infographics used in this repository.
//!
//! ## Scope
//! - Applies to exported static graphics (PNG/SVG/PDF) used in docs and reports.
//! - Focuses on architecture maps, merge-state diagrams, reconciliation flows, and evidence dashboards.
//!
//! ## Canvas and Export
//! - Default canvas: `1920x1080` (16:9).
//! - Dense dashboard canvas: `2560x1440`.
//! - Print-friendly fallback: `A4 landscape` at `300 DPI`.
//! - Export formats:
//!   - primary: `PNG`
//!   - source-preserving: `SVG`
//!   - publication bundle: `PDF` (optional)
//!
//! ## Color System (Dark)
//! - `--bg-main: #0D1117`
//! - `--bg-panel: #161B22`
//! - `--bg-elevated: #1F2630`
//! - `--text-primary: #E6EDF3`
//! - `--text-muted: #9DA7B3`
//! - `--accent-info: #58A6FF`
//! - `--accent-success: #3FB950`
//! - `--accent-warn: #D29922`
//! - `--accent-critical: #F85149`
//! - `--line-subtle: #30363D`
//!
//! ## Typography
//! - Preferred families:
//!   - headings: `IBM Plex Sans`
//!   - body: `Source Sans 3`
//!   - code/paths: `JetBrains Mono`
//! - Minimum sizes:
//!   - title: `44px`
//!   - section heading: `28px`
//!   - body: `20px`
//!   - annotation/caption: `16px`
//!
//! ## Layout Rules
//! - Use a clear 12-column grid or equivalent alignment system.
//! - Keep one dominant message per visual.
//! - Group content as:
//!   1. state snapshot,
//!   2. evidence links,
//!   3. next actions.
//! - Keep whitespace intentional; avoid dense edge-to-edge blocks.
//!
//! ## Diagram Conventions
//! - Arrows:
//!   - solid: active data/workflow
//!   - dashed: archived/legacy path
//! - Node colors:
//!   - active lane: `accent-info`
//!   - verified/completed: `accent-success`
//!   - warning/open: `accent-warn`
//!   - blocked/failure: `accent-critical`
//! - Every lane diagram must include absolute or repo-relative paths.
//!
//! ## Accessibility
//! - Minimum contrast target: WCAG AA (`>= 4.5:1` for body text).
//! - Do not encode meaning with color alone; pair with labels/icons/patterns.
//! - Keep text left-aligned unless a centered title is visually necessary.
//!
//! ## Required Metadata Block
//! Each exported visual must include or be accompanied by:
//! - `title`
//! - `timestamp` (`YYYY-MM-DD HH:MM:SS`)
//! - `source_paths`
//! - `generator` (tool/script/manual)
//! - `evidence_links` (log/doc paths)
//!
//! ## Naming Convention
//! - Store under: `docs/visuals/`
//! - File name pattern:
//!   - `<YYYYMMDD>_<topic>_<variant>_dark.<ext>`
//! - Example:
//!   - `20260214_singular_merge_state_v1_dark.png`
//!   - `20260214_singular_merge_state_v1_dark.svg`
//!
//! ## Current Artifact
//! - Generated dark-mode singular merge state visual:
//!   - `docs/visuals/20260214_singular_merge_state_v1_dark.png`
//!   - `docs/visuals/20260214_singular_merge_state_v1_dark.svg`
//!   - `docs/visuals/20260214_singular_merge_state_v1_dark.md`
//!
//! ## Review Checklist
//! - Message is clear in 5 seconds.
//! - Paths are explicit and correct.
//! - Metadata block present.
//! - Color/contrast passes readability check.
//! - No stale lane references in active visuals.
//!
