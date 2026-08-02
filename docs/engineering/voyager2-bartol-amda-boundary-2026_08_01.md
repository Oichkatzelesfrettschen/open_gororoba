---
description: Bounded Voyager 2 Bartol to AMDA cross-validation evidence boundary
last_verified: 2026-08-01
evidence_class: partial-scientific-evidence
---

# Voyager 2 Bartol to AMDA evidence boundary

The local evidence does not close the requested Bartol versus AMDA magnetic
field comparison. The retained comparator output is a Bartol versus SPDF run,
not an AMDA run, and the AMDA-specific Voyager 2 files are absent from the
current checkout.

## Verified local result

The source record is
`docs/engineering/voyager2_v2_bartol_spdf_1990_1995_findings_2026_04_19.txt`.
It records 52,584 matched rows for 1990-1995. The B magnitude has zero finite
pairs, bulk speed has Pearson r 0.999414 with RMSE 1.647 km/s, and proton
density has Pearson r 0.269010 with RMSE 0.0509 cm^-3. The density result shows
an approximately tenfold provider discrepancy and needs a unit or column
mapping investigation.

The checkout has no `data/external/voyager2/` directory and no local
`*_amda_merged_hourly.asc` Voyager 2 captures. The registry still describes
the intended AMDA comparator and output, so the experiment remains partial.

## Boundary and falsifier

The claim is bounded to the SPDF proxy result. It does not establish Bartol to
AMDA agreement for B magnitude. A downloaded AMDA Voyager 2 magnetometer file
covering 1990-1995, its raw hash, a parser identity, and a replayed comparator
output falsify the blocked-data state. The missing-data state remains valid
until that evidence bundle enters the repository.

## Next action

Acquire the AMDA-specific Voyager 2 magnetometer product through the declared
source lane, retain the raw bytes and SHA-256, add the source manifest row, and
run `voyager2-bartol-amda-crossval`. Keep the SPDF result as a separate proxy
observation rather than renaming or overwriting its historical output.
