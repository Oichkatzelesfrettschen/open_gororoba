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
//! | 1898 | Hurwitz | `crates/hurwitz_1898/` + `proofs/theories/HurwitzTheorem.v` | tracked CD-tower fixed-point classification is formalized; full all-`n` matrix converse still open |
//! | 1914 | Wedderburn | `proofs/theories/WedderburnPrimitive.v` | module-dimension surface landed; determinant / cyclic-generation lane still open |
//! | 1919 | Dickson | `crates/dickson_1919/` + `proofs/theories/DicksonCDProcess.v` | paired lane is in place |
//! | 1921 | Dickson companion note | `proofs/theories/Dickson1921.v` | initial Rocq companion landed; broader general-field lane still open |
//! | 1935-1942 | Zorn, Jacobson, Albert | sources on disk | still tracked as context, not yet broken into paper lanes |
//! | 1945 | Schafer | `crates/schafer_1945/` + `proofs/theories/SchaferDivAlg16.v` | paired lane is in place |
//! | 1951 | Freudenthal | support material only | exact original remains missing |
//! | 1954 | Schafer | `proofs/theories/Schafer1954.v` | initial Rocq paper index landed; derivation-algebra lane still open |
//! | 1958 | Jacobson | previews and support on disk | exact full text remains institution-locked |
//! | 1967 | Brown | `crates/brown_1967/` + `proofs/theories/BrownGeneralizedCD.v` | extend dim-32/dim-64 theorem surfacing |
//! | 1972 | Brown | `crates/brown_1972/` + `proofs/theories/Brown1972.v` | paper index plus initial Chapter VII map landed; Chapters III-VI remain partial |
//!
//! ## Chronological gaps we still need to track
//!
//! - Exact-source gaps: Freudenthal (1951) and Jacobson (1958) remain the two
//!   important blocked pre-Moreno originals in the chain.
//! - Distillation gaps: Wedderburn (1914) still needs its determinant and
//!   cyclic-generation lane, Dickson (1921) still needs its broader
//!   general-field companion lane, Schafer (1954) still needs its
//!   derivation-algebra / type-G lane,
//!   and Brown (1972) still needs Chapters III-VI plus a Rocq Appendix C bridge.
//! - Context lanes on disk but not yet surfaced as papers: Zorn (1935), Jacobson
//!   (1939), Albert (1942), and Urbanik-Wright (1960).
//!
//! ## Next chronological extraction order
//!
//! 1. Extend Hurwitz from the tracked CD-tower fixed-point theorem to the fuller
//!    all-`n` converse/classification lane.
//! 2. Extend Wedderburn (1914) beyond the new module-dimension surface.
//! 3. Extend Dickson (1921) beyond the new companion surface to the broader
//!    general-field lane.
//! 4. Extend Schafer (1954) from the new paper index to the derivation-algebra
//!    and type-G theorem lane.
//! 5. Extend Brown (1972) beyond the current Chapter VII map.
//! 6. Continue Moreno only after those pre-Moreno surfaces stay readable and
//!    tracked in-repo.
