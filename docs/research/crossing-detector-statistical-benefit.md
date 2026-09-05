# Crossing-detector statistical benefit and operational utility

The frozen-model archive experiment supports a small positive improvement in
pairwise ranking probability. The approximate 95% paired daily-file bootstrap
interval is [0.0010149049086891153, 0.0029188809429172872]. The estimand is the
minimum canonical-minus-geometric-baseline ROC-AUC increment across three fixed
calibration widths and the two final years. The interval conditions on the
fitted models and admitted archive; independence between daily clusters remains
unproven. The interval is neither an average across panels nor an external
transfer estimate.

The upper endpoint falls below the predeclared 0.005 discrimination target.
The original experiment therefore rejects its declared useful-increment
requirement. The target and verdict remain unchanged. The protocol identifies
0.005 as an investigator-defined threshold; physical or operational costs did
not determine that value. Failure against the target establishes failure of
that requirement, rather than universal practical uselessness.

ROC-AUC measures the probability that a positive sample outranks a negative
sample, with half credit for ties. A 0.2 percentage-point gain in that probability
does not imply 0.2% more detected physical crossings. The existing +/-120-second
sample labels, selected crossing days, and global ranking metric cannot supply
ordinary-day false-alert rates, event detection yield, or operational value.

Fourteen to sixteen of nineteen exact-support sign controls equal or exceed the
canonical point estimate in each final panel. Those controls preserve support,
normalization, temporal information and fitting budget; they represent generic
cubic constructions and need not define algebras. Their performance motivates
revision of an algebra-specific explanation. Paired uncertainty for the
mechanism comparisons remains uncomputed. The bounded statistical benefit and
an algebra-specific physical mechanism are separate claims.

The prospective extension in
`plans/crossing-detector-operational-utility.toml` separates catalog curation,
real-time alerts, and a cost-ratio sensitivity frontier. Operational value
requires measured benefits and costs for a defined decision unit. Frozen
archive results remain the historical record throughout the extension.

Sources: `data/output/audit/staples-causal-validation/protocol.toml`,
`data/output/audit/staples-causal-validation/findings.toml`, and
`data/output/audit/staples-causal-validation/research/adopted-calibration-findings.toml`.

## Unit-consistent breakeven boundaries

For sample decisions, the extension uses

\[
\Delta U_s = B_s\pi_s\Delta\mathrm{TPR}_s
             - C_s(1-\pi_s)\Delta\mathrm{FPR}_s - \Delta K_s.
\]

Inference seconds must be converted to the same utility units as benefits and
costs before entering \(\Delta K_s\). Raw seconds remain a separately reported
resource measurement. A program cannot add seconds to monetary or scientific
utility values.

For a common exposure \(T\), event accounting uses

\[
\Delta U_{e,\mathrm{total}}
= B_e N_{\mathrm{true}}\Delta\mathrm{POD}
- C_e\Delta F_{\mathrm{rate}}T - \Delta K_{e,\mathrm{total}}.
\]

Dividing every term by \(T\) gives utility per exposure hour:

\[
\Delta U_{e,\mathrm{hour}}
= B_e\lambda\Delta\mathrm{POD}
- C_e\Delta F_{\mathrm{rate}} - \Delta K_{e,\mathrm{hour}}.
\]

The total and hourly forms require distinct overhead denominators. Human review
and computation remain separately logged cost components; review included in a
false-alarm cost must not also enter overhead.

For either accounting unit, normalized utility has the affine form
\(\Delta U/B=a-rb-k\), where \(r=C/B\). The sample coefficients are
\(a=\pi_s\Delta\mathrm{TPR}_s\) and
\(b=(1-\pi_s)\Delta\mathrm{FPR}_s\). Event coefficients are
\(a=\Delta N_{\mathrm{matched}}/T\) and
\(b=\Delta N_{\mathrm{unmatched\ alerts}}/T\).
For \(b\ne0\), the breakeven ratio is \(r^*=(a-k)/b\).

| Change in false alarms | Positive-utility region for nonnegative cost ratios |
| --- | --- |
| \(b>0\) | \(0\le r<r^*\); the region is empty when \(r^*\le0\) |
| \(b<0\) | \(r>r^*\), restricted to \(r\ge0\) |
| \(b=0\) | Every ratio if \(a>k\); zero utility if \(a=k\); every ratio is unfavorable if \(a<k\) |

The prevalence-divided form requires \(0<\pi_s<1\) for both conditional
rates. Direct count differences remain meaningful on single-class exposure,
while the undefined TPR, FPR or POD remains unreported. Zero-event exposure can
measure alert cost but cannot establish event-detection probability.

The two-dimensional frontier is \(k=a-rb\). Plotting that boundary avoids
assigning fabricated mission costs. Paired draws propagate uncertainty in
counts, prevalence and measured overhead together. Positive lower utility
bounds define conditionally favorable regions; negative upper bounds favor the
baseline; overlap or missing uncertainty requires manual review. These labels
describe a decision model, rather than authorize spacecraft operations.

Independent adjudicators review the union of model candidates and catalog
events, plus continuous routine intervals, before comparison metrics are
computed. Adjudicators remain blinded to model identity. Reports retain strict
catalog matching and independently adjudicated reference results separately.
Unknown boundaries remain uncertain rather than being counted automatically as
false alarms. The independent continuous review supplies the denominator for
missed events and event prevalence.

## Replay the accounting instrument

`crossing-utility-frontier` consumes typed paired counts; it does not collect
human review observations or infer missing event labels. The retained
`mathematical-fixture.json` validates arithmetic only. Its events and costs are
synthetic, and its plot stays in Manual Review even when its point utility is
positive.

```bash
CARGO_TARGET_DIR="$(pwd)/.cache/gate-target" cargo run --profile validation \
  -p gororoba_cli_physics --bin crossing-utility-frontier -- \
  --input data/output/audit/crossing-detector-utility-instrument/mathematical-fixture.json \
  --out-dir .cache/crossing-utility-fixture-replay \
  --max-cost-ratio 4 --min-overhead-shift -2 --max-overhead-shift 3 --grid-size 5
```

The output directory must be new. `report.json` retains the request, its SHA256,
coefficients, signed breakeven region, grid and uncertainty limitations.
`frontier.csv` retains every plotted value; `frontier.svg` is a standalone plot.
The vertical axis adds a normalized overhead shift to the measured base overhead
and to every paired draw. A shift of zero preserves that base. Negative shifts
represent conditional savings, rather than measured savings by assertion.

Empirical requests declare `evidence_kind = "empirical"`, a `reference` of
`strict_catalog` or `adjudicated`, a nonempty measurement boundary, and source
receipts containing paths relative to the request file and expected SHA256
values. Digest admission establishes retained source identity; independent
adjudication and experimental validity remain separate obligations.
`utility.accounting.kind` is `sample` or `event`. Sample accounting requires
`decisions` and `positive_decisions`; event accounting requires `exposure_hours`
and `true_events`. Both carry `baseline` and `augmented` true/false-positive
counts over identical exposure. `additional_overhead` is a total cost across
that exposure; the instrument divides it by the positive
`benefit_per_true_detection` and the appropriate exposure denominator.

`paired_draws` is either absent/null or a list of paired accounting observations
with their own total overhead. Draws must preserve the same unit and shared
within-draw exposure. The producer must resample the complete comparison
jointly. The instrument checks numerical and dimensional validity; it cannot
recover the experimental pairing from aggregate counts. Missing draws retain
unknown uncertainty. The plot applies interval colors only to adjudicated
empirical inputs with at least 1000 draws. The draw-count floor is a computational
safeguard, rather than a guarantee of independent observations or interval
coverage. Pointwise grid colors cannot justify selecting a winner on final data.

E-284, E-285 and E-286 register the curation, alerting and conditional-frontier
measurement designs as planned. Their empirical run commands and outputs remain
unset until the respective activation manifests are sealed. The original
E-282/E-283 protocols, evidence and verdicts retain their historical identities.

Receipt paths must be normalized relative paths inside the request directory.
Absolute paths, parent components and resolved paths outside that directory are
rejected. Empirical inputs carry the source files beside the request or below
it, so a bundle can retain its own evidence identities.

The output publisher writes the CSV and SVG before committing `report.json` as
the completion marker. Ordinary write failures remove only the newly owned
bundle and permit retry. Existing output paths remain untouched. An interrupted
process can leave an incomplete directory; consumers require the completion
report before treating the bundle as a completed run.

The preservation query compares original canonical rows against a retained
parent snapshot. Replay from the repository root:

```bash
git show a19b7f3144b748c6530841e34ff9505197244ba6:registry/canonical/control_plane.sqlite3 > .cache/utility-before.sqlite3
sqlite3 -json registry/canonical/control_plane.sqlite3 < data/output/audit/crossing-detector-utility-instrument/preservation.sql
```
