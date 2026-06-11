# RC1 Evidence Note

## Claim Envelope

**Primary conclusion**: Mission/product-respecting, time-sorted magnetic-Takens
embeddings produce structured deviations from null, and those deviations vary
strongly across heliocentric distance and latitude.

**Outer-limit caveat**: The 132.5-137.5 AU structure is resolved, but these bins
remain observationally sparse (mission_diversity = 1, Voyager-only).

## Frozen Anchor

- **Anchor commit**: `6f69a59228d8160429b92e2f9e4ff9d84f5673ba`
- **RC1 tightening commit**: See `git log --oneline` from this evidence note forward.
- **Generated**: 2026-03-27

## Artifact Manifest

| Path                                                     |       Bytes | LFS OID (sha256)  |
| -------------------------------------------------------- | ----------: | ----------------- |
| `data/output/heliosphere/full_feature_cube.csv`          |  26,862,534 | `96347a0c62b9...` |
| `data/output/heliosphere/takens_quench_scan.csv`         |       1,758 | `42fd5622981c...` |
| `data/output/heliosphere/associator_null_audit.json`     |      29,990 | `ff19dc85a713...` |
| `data/output/test_mms/mms/mms1_fgm_srvy_l2_2024_1_1.csv` | 113,394,600 | `47eb16fc1965...` |

Full OIDs in `docs/reports/rc1_manifest.txt`.

## Local Cleanliness

Working tree has tracked modifications to RC1-tightening binaries
(--exclude-mission flags, boxkite parity checker) and untracked ablation
outputs in `data/output/heliosphere/ablations/`. All other modifications are
from concurrent work (cd_cache, lit_search, Schafer proofs) and are not part
of the RC1 evidence scope.

---

## 1. Ulysses Ablation

**Method**: Run `heliosphere-quench-scan` on `full_feature_cube.csv` with and
without `--exclude-mission Ulysses`. Compare bin counts, sample counts,
latitude coverage, and mission_diversity.

### Results

| Metric                          | Baseline             | No Ulysses           | Delta  |
| ------------------------------- | -------------------- | -------------------- | ------ |
| Total bins                      | 23                   | 16                   | -7     |
| Total samples                   | 100,943              | 92,191               | -8,752 |
| Latitude bins with data         | 19 unique lat values | 12 unique lat values | -7     |
| Bins with mission_diversity > 1 | 9                    | 0                    | -9     |

**Lost bins** (all Ulysses-exclusive high-latitude coverage):

- lat = -75, -65, -55, 55, 65, 75, 85 deg (all at r = 2.5 AU)

**Unaffected bins** (Voyager-only outer heliosphere):

- 107.5 AU, 112.5 AU, 132.5 AU, 137.5 AU -- identical sample counts and
  associator values with and without Ulysses.

**Interpretation**: Ulysses is the sole provider of high-latitude coverage.
Removing it eliminates 7 bins entirely and reduces mission_diversity to 1
in all remaining inner-heliosphere bins. The outer heliosphere structure
(107-137 AU) is completely independent of Ulysses.

### Sensitivity: No Cassini vs No Outer2001

| Variant             | Bins | Samples | Interpretation                                                                              |
| ------------------- | ---- | ------- | ------------------------------------------------------------------------------------------- |
| No Cassini          | 21   | 93,204  | Loses 2 bins (7.5 AU region unaffected). Cassini contributes outer2001 window rows only.    |
| No outer2001 window | 14   | 75,692  | Loses 9 bins. Removes all outer2001-epoch data (Cassini + OMNI + Ulysses from that window). |

The outer2001 window ablation is strictly more aggressive than either
single-mission ablation, confirming that "Ulysses densification" and
"outer-window densification" are distinct effects: Ulysses provides
latitude coverage, while the outer2001 window provides temporal depth
at 2.5 AU in the equatorial plane.

---

## 2. OMNI Hygiene and Null Ablation

**Method**: Filter full_feature_cube.csv to OMNI-only rows, run
`heliosphere-associator-null-audit` with 20 null iterations.

### Row Counts

| Metric                                 | Value  |
| -------------------------------------- | ------ |
| OMNI total rows                        | 17,544 |
| Valid B-field rows (finite, b_mag > 0) | 17,542 |
| Filtered out (NaN/zero B)              | 2      |

### Null Family Comparison (magnetic-takens embedding)

| Statistic                         | Value |
| --------------------------------- | ----- |
| Base mean associator              | 3.205 |
| Base max associator               | 25.41 |
| Temporal-shuffle null mean        | 5.496 |
| Channel-permutation null mean     | 4.965 |
| Suppression ratio (base/temporal) | 0.583 |
| Suppression ratio (base/channel)  | 0.645 |

**Interpretation**: The OMNI magnetic-takens signal (mean = 3.205) sits
**below** both null families. Suppression ratios < 1.0 confirm the signal
is below the noise floor for OMNI data alone, as expected for a 1 AU
single-spacecraft dataset where the heliocentric distance gradient is absent.

Note: The legacy-raw and dynamic-bias-free embeddings show much larger
absolute values (10^7 to 10^10) but these are not normalized by local
field magnitude and are therefore not comparable across missions.

---

## 3. Backend Parity (Box-Kite Engine)

**Method**: Run `heliosphere-boxkite-alignment` on sorted `full_feature_cube.csv`
with `--backend cpu` and `--backend vulkan`. Compare row-by-row.

### Results

| Metric                              | Value    |
| ----------------------------------- | -------- |
| CPU rows                            | 100,973  |
| Vulkan rows                         | 100,973  |
| Orient mismatches (total)           | 55,148   |
| Orient mismatches (FP tie-breaking) | 55,148   |
| Orient mismatches (real)            | 0        |
| Alignment failures                  | 0        |
| Max absolute diff                   | 4.44e-16 |
| Max relative diff                   | 6.12e-16 |
| **Verdict**                         | **PASS** |

**Interpretation**: All 55,148 orientation index mismatches occur when
alignment scores are identical to within machine epsilon (< 1 ULP in f64).
Different instruction scheduling between CPU (Rayon) and Vulkan compute
shaders breaks ties in argmax differently, but the underlying alignment
values are bit-for-bit identical within FP precision.

The max absolute difference of 4.44e-16 corresponds to exactly one Unit
in the Last Place (ULP) for f64 values near 1.0, confirming numerical
equivalence across backends.

CUDA backend was not tested (requires GPU availability at runtime).

Full parity report: `data/output/heliosphere/ablations/boxkite_parity.json`

---

## 4. Frontier Selection

Deferred to post-RC1 sprint planning. Two candidates identified in the
original plan:

1. **Outer-Heliosphere Densification**: Add more missions/epochs to the
   sparse 107-137 AU bins.
2. **Formal/Literature Consolidation**: Use `lit_search` to ground the
   claim envelope in published work.

---

## Ablation Output Files

All ablation artifacts are in `data/output/heliosphere/ablations/`:

| File                           | Description                          |
| ------------------------------ | ------------------------------------ |
| `quench_scan_baseline.csv`     | Full-dataset quench scan (23 bins)   |
| `quench_scan_no_ulysses.csv`   | Ulysses-excluded (16 bins)           |
| `quench_scan_no_cassini.csv`   | Cassini-excluded (21 bins)           |
| `quench_scan_no_outer2001.csv` | outer2001 window excluded (14 bins)  |
| `omni_null_audit.json`         | OMNI-only null audit (20 iterations) |
| `boxkite_cpu.csv`              | CPU backend alignment scan           |
| `boxkite_vulkan.csv`           | Vulkan backend alignment scan        |
| `boxkite_parity.json`          | Backend parity comparison report     |
