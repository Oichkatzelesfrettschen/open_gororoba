---
description: Bounded map of non-governance claims, insights, experiments, formal rows, and falsification work
last_verified: 2026-08-02
evidence_class: read-only-frontier-inventory
status: active
---

# Scientific frontier map and falsification queue

The registry contains 1,448 claims, 183 insights, and 232 experiments. The
current compatibility surface also records 496 binaries and 162 formal
theorem rows from `proofs/_RocqProject`. Four structural proof rows
(`C1635` through `C1638`) have no claim row yet and remain explicitly
allowlisted as unlinked formal surfaces. These counts describe the corpus, not
its scientific validity. A
registry status of `Verified` means that a registry row and a source anchor
exist; it does not prove that the associated numerical test is independent,
well-conditioned, or faithful to the cited paper.

The highest-value work outside registry policy is therefore validator
independence. Several claims have code and passing tests, but the test either
forces the residual to zero, copies a lower-level residual, uses a scalar
surrogate for a complex invariant, or omits the declared parameter sweep.

The rows below are bounded read-only handbacks. The full repository validation
proves the plumbing and test taxonomy, not these domain-specific claims. No
domain-specific agent build or scientific replay supports the observations
below. Each row names a falsifier and a retained next artifact.

The validation pipeline itself closes its P0 engineering row with the retained
report at `reports/validation/2026-08-02/144146/summary.md`. The next load-
bearing work therefore starts with independent Ward contractions, complex
TCMT flux residuals, and declared optics sweeps rather than another governance
rebuild.

## Priority queue

### P0: remove false confidence from load-bearing claims

| Registry row | Code surface | Current weakness | Falsifier or acceptance artifact |
| --- | --- | --- | --- |
| C-820, E-073 | `crates/gr_core/src/photon_graviton/irreducible.rs` | The Ward residual is multiplied by `on_shell_factor = 0`, so the test can pass without an independent contraction. | Implement an independent tensor contraction and retain on-shell and off-shell residuals. A nonzero residual after removing the forced zero falsifies the claim. |
| C-822, E-073 | `crates/gr_core/src/photon_graviton/ward.rs` | The full gravitational residual copies the irreducible residual instead of contracting all diagrams independently. | Contract irreducible, tadpole, and external-leg contributions separately, then retain the combined residual. Any mismatch above the declared tolerance falsifies the current claim. |
| C-866 | `crates/gr_core/src/photon_graviton_tcmt/amplitude_bridge.rs` and `tcmt_equations.rs` | Unitarity sets `R = 1 - T` for real scalar values instead of checking a complex S-matrix flux residual. | Add a complex transmission/reflection regression. A lossless flux residual above `1e-10` falsifies the claim. |
| C-864 | `crates/gr_core/src/photon_graviton_tcmt/amplitude_bridge.rs` | The three-diagram to TCMT parameter map uses floors and fixed decay ratios, so it is heuristic. | Extract parameters independently and require `epsilon < 0.1` while reproducing TCMT observables. |
| C-849, C-850, E-076 | `crates/optics_core/src/mie_cylinder.rs` | Single- and multi-channel Mie/TCMT agreement is provisional; the implementation lacks the declared one-percent and 20-point sweeps. | Retain lossless and lossy single-channel comparisons, then a 20-point `|l| <= 2` sweep. Any declared tolerance failure blocks promotion. |
| C-832, C-839, E-074 | `crates/optics_core/src/sfwm.rs` and `crates/optics_core/tests/test_sfwm_son_chekhova.rs` | The source supports direct SFWM over the cascaded SHG plus SPDC path at 10 um, with a reported ratio near 0.048, but the paper also says the total infrared signal can contain a dominant SPDC contribution. Paper-calibrated and Sellmeier-derived mismatch paths are mixed in the code. | Retain both calibration paths under one declared input set and compare rate ratios, fringe periods, and monotonicity separately. Keep the claim narrowly scoped to direct versus cascaded rates. |

P0 does not mean that the underlying physics is false. It means that the
current test can produce a passing result without measuring the claimed
quantity. This is the correct place to spend engineering time before adding
more theory or publishing a stronger conclusion.

### P1: bind formal, data, and numerical evidence to replayable artifacts

| Registry row | Code or evidence surface | Current boundary | Next artifact |
| --- | --- | --- | --- |
| `p1-rocq-project-completeness` | `proofs/_RocqProject`, `proofs/Makefile`, `rocq-project-audit` | Project parity and pinned `vos`/`vok` result are not in one dated record; axiom and parameter dispositions remain separate. | Retain project parity, both proof results, toolchain version, theorem count, and every assumption disposition. |
| `p1-formal-evidence-registry` | formal proof field schema and theorem mirrors | The theorem registry does not yet join theorem identity, assumptions, proof output, toolchain, and falsifier in one evidence record. | Add one typed theorem-evidence record shape and a retained proof output hash. |
| C-1635 through C-1638 | `proofs/verified/C1635_SedenionDriverSemantics.v` through `C1638_OctonionDowncastNoZeroDivisors.v` | The proof project contains four load-bearing structural theorem surfaces that are not yet represented as claim rows. The verifier admits them as explicitly unlinked structural rows; that admission is not claim validation. | Create claim-level evidence rows with source, assumptions, proof output hash, replay command, and a driver-facing falsifier before treating the surfaces as registry claims. |
| C-1538, C-1539 | `proofs/theories/C1538_MorZDSymmetry.v`, `C1539_MorSkewSymm.v` | Theoretical rows have no current Rocq result in the inspected surface. | Record premises, theorem status, kernel result, and a counterexample search boundary. |
| `p1-voyager-bartol-amda-comparator`, T-058, E-128 | `crates/gororoba_cli_data/src/bin/bartol_spdf_crossval.rs` | The retained comparison is Bartol versus SPDF, not the requested AMDA product. | Acquire and hash AMDA magnetometer data, record parser identity and units, then retain matched counts, correlations, RMSE, and offsets. |
| WS-OPTICS-GR-001 | `materials_core`, `gr_core`, `sign_imbalance`, `optics_core` | The workstream has implementation breadth, but each primary output still needs a claim-specific reproduction record. | Split each output into a source equation, parameter table, numerical tolerance, and retained comparison artifact. |
| I-137, I-140, I-141; E-218, E-224 | `lbm_3d`, `lbm_3d_cuda`, precision and stability lanes | Precision divergence, activated Smagorinsky feedback, and positivity thresholds lack matched CPU/GPU traces with identical inputs. | Retain f32/f64, BGK/MRT, positivity, Mach, mass, NaN, and `D_f` traces with hardware and feature metadata. |
| C-1101, I-104 | `crates/gr_core/src/forces/chingon_bivector_drag.rs` | The six-flyby 5/6 sign result and 128D partition are not represented by a complete retained case matrix. | Retain all six cases, block layout, signs, magnitudes, and the 42/42/43 partition output. |
| C-1120, C-1121 | `crates/gr_core/src/fractal_metric.rs` | Tests establish finite positive values at one point, not the stated magnitude mismatch or parameter domain. | Add an observed-versus-predicted numerical bound and a parameter/radius grid including boundary cases. |

### P2: improve coverage and reproducibility after P0/P1

| Registry row | Surface | Next falsifiable increment |
| --- | --- | --- |
| `p2-randomness-run-manifests` | Random and bootstrap call sites | Require generator, seed, toolchain, feature set, hardware, input hash, output hash, and legacy classification in every new run. |
| `p2-materials-source-contract` | `materials_core`, `materials_data` | Record generated-data boundaries, source hashes, table parity, and absent-path disposition. |
| I-102, I-103; E-084 | `spectral_dimension.rs` and orthoplex diffusion | Define one `d_s(t)` estimator across dimensions and domains, then retain cross-dimension curves and held-out tests. |
| C-1362 | `x87_primitives.rs`, `avx2_primitives.rs`, E-188 | Run the planned AVX/FMA comparison and replace the crossover heuristic with measured error and throughput bounds. |
| E-169, E-172, E-176 | Euclid and LBM GPU lanes | Retain paired timing, occupancy, registers, bandwidth, precision, mass, and `D_f` evidence rather than stdout-only results. |
| C-1117 | `crates/gr_core/src/nbody_integration.rs` | Add radius-response and trajectory-avoidance sweeps; current tests establish bounds but not physical avoidance. |

## Evidence contract

Every P0 and P1 row receives one evidence object with these fields:

| Field | Required content |
| --- | --- |
| Identity | Claim, insight, or experiment ID and the exact code owner. |
| Source | Primary paper, theorem, dataset, or specification with a stable identifier. |
| Inputs | File hashes, units, dimensions, parameter values, seed, feature flags, and backend. |
| Derivation | Equation or algorithm path, including coefficient conventions and normalization. |
| Independent check | A second implementation, reference value, conservation law, or formal kernel result that does not reuse the tested residual. |
| Falsifier | A numeric threshold, counterexample, missing-data condition, or failed proof obligation. |
| Output | Retained CSV, JSON, proof log, or figure with a content hash. |
| Status | `provisional`, `reproduced`, `falsified`, `blocked`, or `superseded`, with the registry row updated through SQLite. |

This contract separates three questions that the current registry sometimes
collapses: does code exist, does the test exercise the claimed mechanism, and
does the primary source comparison reproduce. A claim can pass the first and
fail the second without any contradiction. That distinction is the main
research-quality improvement in this queue.

## Primary-source intake

Three primary-source PDFs were acquired with an explicit Mozilla user agent
into `/var/tmp/open_gororoba-research-20260802/` for this bounded pass:

| Source | SHA-256 of bounded capture | Use |
| --- | --- | --- |
| arXiv:2601.23279 | `c647509b343b7348f8eff8b31a1a311f67e1e2bb442b53167ac89f902b8e0b90` | Photon-graviton mixing, tadpole, and Ward-identity source comparison. |
| arXiv:0909.3323 | `a355dc5a9358d05e6eeae3475c4722a37fb3d521fa457e6aac474d71a06d5c9a` | Ruan-Fan Fano and Mie/TCMT comparison. |
| arXiv:2601.23137 | `25c92caa576a805711fc0050291f8f15977147be927f32b4b8e3ae01112ec8e6` | Son-Chekhova SFWM rate and fringe comparison. |

Each PDF is rendered at bounded resolution and passed through Tesseract for
local OCR. The temporary capture is not a canonical source artifact. A
paper enters the repository evidence surface only after its existing paper
manifest, source hash, acquisition provenance, OCR or text extraction result,
and claim links are reconciled. Network access never becomes a CI dependency.

The replayable intake shape is:

```bash
wget --user-agent='Mozilla/5.0 (X11; Linux x86_64) open_gororoba-research/2026.08' \
  --continue --output-document=/var/tmp/source.pdf https://arxiv.org/pdf/<id>
pdftoppm -r 120 -png /var/tmp/source.pdf /var/tmp/source-page
tesseract /var/tmp/source-page-01.png stdout > /var/tmp/source-page-01.txt
sha256sum /var/tmp/source.pdf
```

The source text is an aid to inspection. Equations and coefficients are
checked against the PDF page image or an authoritative machine-readable source
before they enter a derivation record.

## Order of work

1. Falsify C-820 and C-822 with independent Ward contractions.
2. Replace C-866 scalar unitarity with a complex S-matrix residual.
3. Reproduce C-849 and C-850 against the Ruan-Fan source before promoting
  their provisional status.
4. Bind the Rocq and AMDA evidence boundaries to dated artifacts.
5. Run matched CPU/GPU LBM precision and stability sweeps.
6. Add randomness, materials, spectral-dimension, and performance manifests.

The queue remains intentionally finite. A new claim enters only with a named
source, code owner, falsifier, and output artifact. This prevents the registry
from growing faster than its ability to discriminate true, false, and merely
implemented statements.
