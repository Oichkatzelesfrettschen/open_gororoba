# sedenion_standard_model Commit Audit (2026-03-27)

## Summary

`docs/physics/sedenion_standard_model.md` is stale.

- Last touch to the document: `d28e79a2`
- Commits since that touch: `39`
- Touches to the document in that window: `0`

This audit classifies every commit in `d28e79a2..HEAD` by whether it should
change the public physics document, only change confidence/status wording, or
remain outside the document as archival / tooling work.

Classification labels:

- `direct-doc`: edit the physics document itself
- `indirect-status`: update confidence / control-language only if the document
  already discusses that lane
- `archive-only`: important for provenance or roadmap, but belongs in reports
- `not-relevant`: no public-doc delta needed

## Commit-by-commit classification

| Commit | Classification | Physics-doc action |
|--------|----------------|--------------------|
| `158c90e5` Add mixed mag+plasma embedding and block-size sensitivity sweep | `indirect-status` | note only in the control / 32D status appendix, not the main flavor narrative |
| `f913056a` Finish Schafer 1954 concrete theorem 4 lane | `direct-doc` | update formalization status: Schafer 1954 concrete theorem-4 lane is closed |
| `90de619b` Author POST_RC1_32D_EVIDENCE_NOTE.md | `direct-doc` | update 32D control-language and add report reference |
| `dc98ba81` Add leave-one-out invariance tests for 32D quench structure | `indirect-status` | tighten robustness wording for 32D controls |
| `dcb9abd4` Add regime-conditioned 32D quench maps (fast/slow wind, Br polarity) | `indirect-status` | tighten control-lane caveats, not the main flavor claims |
| `a9e1de95` Densified 32D diagnostic stack: 87 bins, 4-family null audit, spectral excess | `indirect-status` | update control-lane confidence wording |
| `e13f9cb2` Add provenance QA, algebra randomized tests, and 4-family null models | `indirect-status` | mention stronger null/provenance checks in appendix only |
| `2d9f2bba` Add 32D ablation artifacts and densified feature cube (1.15M rows) | `indirect-status` | mention densified 32D control stack in appendix only |
| `dbc5a56d` Update CD cache dossiers and acquisition sessions | `archive-only` | cite only in legacy-roadmap report |
| `1f229790` Advance Schafer 1954 y-square helper lemmas | `direct-doc` | part of Schafer formalization closure cluster |
| `6eeb2855` Extract Semantic Scholar source module and extend lit_search CLI | `not-relevant` | no standard-model delta |
| `f694446e` Densify heliosphere: full Voyager/Ulysses traverses + CUDA pipelining + predicated BC | `indirect-status` | only affects control-lane appendix wording |
| `5977f5f9` Add dimension-generic embedding infrastructure, fix sedenion SIMD bug, add 32D pathion SIMD | `direct-doc` | update SIMD / runtime status and 32D control wording |
| `e63b0243` Reduce Schafer converse to residual eq50 builder | `direct-doc` | part of Schafer formalization closure cluster |
| `856021dc` Return explicit HathiTrust unavailable error | `archive-only` | provenance tooling only |
| `b7bf6869` Narrow Schafer residual diagonal builder gap | `direct-doc` | part of Schafer formalization closure cluster |
| `ddcee7e3` Refresh Dickson cache outputs and search plumbing | `archive-only` | legacy roadmap only |
| `23a86069` Finish Schafer residual eq51 bridge fix | `direct-doc` | part of Schafer formalization closure cluster |
| `50cf06c7` Update acquisition caches and research artifacts | `archive-only` | legacy roadmap only |
| `a51405fd` Execute RC1 tightening plan: ablation studies, backend parity, evidence note | `direct-doc` | update control-lane confidence wording and cite RC1 evidence note |
| `61b45669` Add Schafer y-square helper lemmas | `direct-doc` | part of Schafer formalization closure cluster |
| `32a2b06d` Register dossier and acquisition markdown ownership | `archive-only` | registry hygiene only |
| `8a944554` Bridge Schafer theorem 4 through theorem 3 builder | `direct-doc` | part of Schafer formalization closure cluster |
| `47eceb18` Refine blocker dossier filtering and rerun-safe staging | `archive-only` | acquisition workflow only |
| `088442ed` Bridge Schafer theorem 3 coordinates into residual surface | `direct-doc` | part of Schafer formalization closure cluster |
| `f94e5e7a` Stabilize Schafer converse lane and data tooling | `direct-doc` | part of Schafer formalization closure cluster |
| `ac59151d` Land Schafer residual coordinates and repo updates | `direct-doc` | part of Schafer formalization closure cluster |
| `307f34c9` Refine Schafer 1954 residual converse lane | `direct-doc` | part of Schafer formalization closure cluster |
| `f53c8338` Strengthen Schafer 1954 concrete converse blocks | `direct-doc` | part of Schafer formalization closure cluster |
| `2991bc67` Refactor Schafer concrete block bridge | `direct-doc` | part of Schafer formalization closure cluster |
| `a3688b50` chore(rc1): documentation harmonization and backend parity logic fixes | `indirect-status` | append-only confidence note if needed; no new physics claim |
| `264556b5` Package Schafer 1954 converse surface | `direct-doc` | part of Schafer formalization closure cluster |
| `93a0ea29` Land Schafer 1954 theorem 3 and 4 surfaces | `direct-doc` | part of Schafer formalization closure cluster |
| `6f69a592` docs(rocq): land abstract derivation extension and restriction surfaces for Schafer 1954 | `direct-doc` | part of Schafer formalization closure cluster |
| `ff653036` Land Dickson reduction tranche and workspace fixes | `archive-only` | legacy roadmap only |
| `42be1920` Land Dickson obstruction surface | `archive-only` | legacy roadmap only |
| `bc44c34c` Sync Wedderburn handoff and workspace updates | `archive-only` | legacy roadmap only |
| `16aa0535` feat(researchclaw): port AutoResearchClaw to Rust and enhance physics pipeline | `not-relevant` | no standard-model delta |
| `dbe05ec5` Track .h5 and .cdf files with Git LFS | `not-relevant` | no standard-model delta |

## Required edits to the physics document

The public document needs four concrete changes.

1. Update the proof-status language.
   - Schafer 1954 is no longer an open theorem-4 blocker.
   - Brown 1972 Chapter III is now the next Rocq handoff.

2. Update the implementation-status language.
   - the live theorem count changed
   - the sedenion SIMD bug fix and 32D pathion SIMD support changed the
     runtime-status note

3. Update the 32D control-language.
   - the post-RC1 evidence note, null-audit tightening, and invariance tests
     should appear as a control-lane confidence update
   - they should not displace the main flavor-physics narrative

4. Add a compact appendix.
   - formalization status as of 2026-03-27
   - references to this audit and the pre-1954 roadmap

## Explicit non-edits

The following do **not** belong in the main physics doc except as cited support
in dedicated reports:

- acquisition session plumbing
- dossier ownership registration
- HathiTrust error handling
- Dickson/Freudenthal/Jacobson cache refreshes
- general literature tooling refactors
