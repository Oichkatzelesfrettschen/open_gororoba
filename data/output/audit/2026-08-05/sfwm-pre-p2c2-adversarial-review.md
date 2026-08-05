---
description: Pre-P2C2 SFWM source, material, and data-acquisition review
last_verified: 2026-08-05
evidence_class: bounded-source-and-data-acquisition-review
---

# Pre-P2C2 SFWM adversarial review

Declare the pre-P2C2 review complete only after the source packet, implementation boundary, material identity, missing-supplement status, and author-data request are explicit. Keep P2C2 unstarted while any load-bearing input remains unavailable.

## Review boundary

The review covers the primary Son-Chekhova source packet, the signed P2C1 source reconstruction, the Zelmon branch choice, the Wang chi(3) citation identity, the fused-silica percentage claim, and the request for author-controlled evidence. It does not fit experimental figures, digitize plotted counts, or modify `crates/optics_core`.

## Findings

| Severity | Finding | Grounding | Required disposition |
| --- | --- | --- | --- |
| SERIOUS | The reported fused-silica contribution is a percentage of total detected coincidences, not a standalone nonlinear amplitude. | The primary paper delegates the calculation to Supplement 1; the public PDF, source archive, and abstract page contain no supplement. | Keep C-838 Provisional. Require the supplement or an equivalent complete denominator model before fitting or adjudicating the percentage. |
| SERIOUS | The cited chi(3) source does not establish the material identity used in the thin pristine LN calculation. | The Wang record names CuZn alloy nanoparticle implanted LiNbO3; the source packet does not identify the sample, tensor component, wavelength, or conversion that yields 1.5e-20 m^2/V^2. | Keep the material assumption open. Do not turn the numeric ratio into a pristine-LN source claim. |
| SERIOUS | Experimental count, coincidence, detector, filter, collection, and background inputs are unavailable. | The primary source says the underlying data are not public and may be requested from the authors. | Retain a pending author request. Do not start P2C2 figure fitting. |
| MODERATE | The extraordinary Sellmeier branch is physically inferred but not named in the primary text. | The geometry is x-cut, propagation is along x, and the source selects z tensor components; Main.tex does not print ordinary or extraordinary. | Retain both branch labels and treat extraordinary as an explicit inference until the authors confirm the convention. |
| MODERATE | Absolute coherence lengths are more branch-sensitive than the dimensionless direct-to-cascaded ratio. | P2C1 gives 33.3 micrometres for the paper-input SFWM anchor and 25.2175 micrometres for the extraordinary branch, while the ratios are 0.0477442 and 0.0480109. | Report the separation. Do not use ratio stability to promote the absolute material model. |
| SLOPPY | The old near-match was not an independent validation. | The legacy susceptibility tuple and omitted Eq. 6 prefactor compensate in the old stress result. | Keep the legacy path as characterization and retain the corrected path as the oracle. |
| SOUND | The signed mismatch identity, complete Eq. 6 prefactor, complex amplitudes, independent fixtures, and frozen thickness grid survive the implementation attacks. | P2C1 retained component comparisons and an independent 80-digit mpmath fixture. | Preserve the source-reproduction result with its material and experimental limits. |

## Steelman

The source paper remains internally coherent at the source-model level. Its signed phase-matching relation, direct and cascaded amplitudes, source susceptibilities, and reported ratio can be reconstructed without refuting the paper. The corrected P2C1 calculation is therefore useful evidence for the bounded source-model claim. The missing supplement and unresolved Wang sample prevent a stronger claim about detected rates or pristine-LN material constants, not a refutation of the published source equations.

## Action taken

The request at `data/output/audit/2026-08-04/sfwm-author-data-request.txt` was sent on 2026-08-05 to the corresponding-author address displayed on the official arXiv PDF. It requests Supplement 1, signed mismatches, index and polarization conventions, the Figure 4 thickness grid, the substrate derivation, experimental data, and the sample and convention supporting chi(3) = 1.5e-20 m^2/V^2. The request remains pending.

## Entry decision

Declare the pre-P2C2 gate complete only after retaining this review, registering it with the canonical artifact path, and passing the canonical registry validation. Keep P2C2 unstarted. Permit P2C2 only after Supplement 1 or a bounded no-data decision, material-identity disposition, and a declared experimental-data input boundary are retained.

No P2C2 experimental-data conclusion is admissible from the current evidence.
