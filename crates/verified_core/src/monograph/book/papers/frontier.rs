//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Formalization Frontier
//!
//! The paper corpus should not be routed uniformly. Different papers want
//! different first landings:
//!
//! | Paper shape | Prefer first | Reason |
//! |------------|--------------|--------|
//! | finite sign tables, multiplication identities, basis lemmas | Rocq | exact finite proofs are compact and stable |
//! | large enumerations, census tables, partner graphs, extraction pipelines | Rust | executable search and artifact generation are cheaper here |
//! | theorem plus concrete constructive witness | paired Rust + Rocq | Rust finds or enumerates, Rocq certifies |
//! | OCR cleanup, equation normalization, provenance bookkeeping | Rust | this is data-engineering work before proof work |
//!
//! ## Current routing map
//!
//! | Paper | First landing | Why |
//! |------|---------------|-----|
//! | Hurwitz (1898) | paired | finite theorem plus executable classification |
//! | Dickson (1919) | paired | constructive tower plus proof-friendly identities |
//! | Schafer (1945) | paired | explicit dim-16 witness plus theorem |
//! | Brown (1967) | paired | executable generalized CD plus formal division criterion |
//! | Brown (1972) | Rocq-heavy paired | theorems are algebraic, but witness search still benefits from Rust |
//! | Moreno (1997) | Rocq-first paired | operator and eigenspace statements are proof-centric |
//! | de Marrais (2000) | Rust-first paired | tables and circuits enumerate well, then admit exact spot proofs |
//! | Reggiani (2024) | Rust-first | partner graphs and annihilator geometry are computationally rich |
//! | Wilmot (2025) | paired | counting formulas plus structural retraction theorems |
//!
//! ## Next extraction order
//!
//! 1. Keep `./pre-moreno.md` authoritative for the chronological dependency chain
//!    from Hurwitz through Brown, so Moreno and later bridge work stays grounded
//!    in the exact earlier paper order.
//! 2. Isolate Schafer (1954) as a dedicated paper lane, since it sits directly
//!    between Schafer (1945) and Brown (1967) in the theorem chain.
//! 3. Add a dedicated Brown (1972) Rocq paper index, mirroring the new Moreno lane
//!    and clarifying where the dissertation theorems live in the proof tree.
//! 4. Finish the remaining Moreno abstract bridges, especially the arbitrary-`a`
//!    eigenspace/module discharge and the CD-specific cleanup around the 2.9 iff
//!    lane.
//! 5. Split Reggiani's standard-zero-divisor and partner-graph logic into its own
//!    paper crate instead of leaving it embedded in `algebra_analysis`.
//! 6. Use `docpipe` plus curated equation catalogs to normalize equations paper by
//!    paper before deciding whether each one belongs in Rust, Rocq, or both.
