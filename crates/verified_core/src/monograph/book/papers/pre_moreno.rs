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
//! # Pre-Moreno Distillation
//!
//! This page distills the paper chain we need before Moreno (1997), using the
//! off-repo chronology matrix and roadmap as reference surfaces:
//! `~/Documents/Projects/CayleyDickson/CHRONOLOGICAL_REFERENCE_MATRIX.md` and
//! `~/Documents/Projects/CayleyDickson/FORMALIZATION_ROADMAP.md`.
//!
//! ## Oldest papers already cached
//!
//! | Year | Paper | Role in this repo | Status |
//! |------|-------|-------------------|--------|
//! | 1770 | Euler four-square precursor | sums-of-squares ancestry | cached off-repo |
//! | 1818 | Degen eight-square precursor | eight-square ancestry | cached off-repo |
//! | 1835-1866 | Hamilton, Graves, Cayley, Cockle primary formation papers | quaternion/octonion origin story | cached off-repo |
//!
//! ## Required pre-Moreno chain
//!
//! | Year | Paper | Current landing | Gap |
//! |------|-------|-----------------|-----|
//! | 1898 | Hurwitz | `crates/hurwitz_1898/` + `proofs/theories/HurwitzTheorem.v` | complete the converse/classification lane |
//! | 1914 | Wedderburn | `proofs/theories/WedderburnPrimitive.v` | still a reference-stub lane, not a full paper extraction |
//! | 1919 | Dickson | `crates/dickson_1919/` + `proofs/theories/DicksonCDProcess.v` | paired lane is in place |
//! | 1921 | Dickson companion note | source on disk | not yet distilled into a dedicated repo lane |
//! | 1935-1942 | Zorn, Jacobson, Albert | sources on disk | still tracked as context, not yet broken into paper lanes |
//! | 1945 | Schafer | `crates/schafer_1945/` + `proofs/theories/SchaferDivAlg16.v` | paired lane is in place |
//! | 1951 | Freudenthal | support material only | exact original remains missing |
//! | 1954 | Schafer | source on disk | next dedicated pre-Moreno extraction target |
//! | 1958 | Jacobson | previews and support on disk | exact full text remains institution-locked |
//! | 1967 | Brown | `crates/brown_1967/` + `proofs/theories/BrownGeneralizedCD.v` | extend dim-32/dim-64 theorem surfacing |
//! | 1972 | Brown | `crates/brown_1972/` + Brown/Moreno Rocq companions | add a dedicated Brown paper index |
//!
//! ## Chronological gaps we still need to track
//!
//! - Exact-source gaps: Freudenthal (1951) and Jacobson (1958) remain the two
//!   important blocked pre-Moreno originals in the chain.
//! - Distillation gaps: Wedderburn (1914) is still a stub, Dickson (1921) is only
//!   on disk, Schafer (1954) lacks a dedicated paired lane, and Brown (1972)
//!   still needs a first-class paper index inside `proofs/theories/`.
//! - Context lanes on disk but not yet surfaced as papers: Zorn (1935), Jacobson
//!   (1939), Albert (1942), and Urbanik-Wright (1960).
//!
//! ## Next chronological extraction order
//!
//! 1. Finish the Hurwitz classification/converse lane so the oldest formal theorem
//!    chain is complete.
//! 2. Promote Schafer (1954) into a dedicated paper lane between Schafer (1945)
//!    and Brown (1967).
//! 3. Add a dedicated Brown (1972) paper index and map its theorem companions
//!    explicitly.
//! 4. Backfill Wedderburn (1914) and Dickson (1921) beyond stub/reference status.
//! 5. Continue Moreno only after those pre-Moreno surfaces stay readable and
//!    tracked in-repo.
