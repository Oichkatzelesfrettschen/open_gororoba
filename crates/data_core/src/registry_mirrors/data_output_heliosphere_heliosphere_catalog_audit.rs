//! # Heliosphere Catalog Audit
//!
//! Generated at `2026-03-10T19:50:56.729459655+00:00`.
//!
//! ## Datasets
//!
//! | Key | Role | Staged | Catalog | Acquisition | Contract | Satisfies Contract | Cadence | Start | End | Notes |
//! | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
//! | soho_celias_bundle | inner_boundary_primary | yes | known | staged | satisfied | yes | native_5min_bundle | 1996-01-20T20:18:00+00:00 | 2023-07-06T23:57:16+00:00 | Native 5-minute inner-boundary lane; hourly normalization is derived downstream. |
//! | sorce_daily | optional_secondary | yes | known | staged | satisfied | yes | daily | 2003-02-25T12:00:00+00:00 | 2020-02-25T12:00:00+00:00 | Legacy radiative context layer overlapping the late Cassini cruise and heliopause eras. |
//! | tsis1_daily | optional_secondary | yes | known | staged | satisfied | yes | daily | 2018-01-11T12:00:00+00:00 | 2026-02-18T12:00:00+00:00 | Daily thermodynamic context layer for 2018+ packs. |
//! | omni_1997_2004 | inner_boundary_primary | yes | known | staged | satisfied | yes | hourly | 1997-01-01T00:00:00+00:00 | 2004-12-31T23:00:00+00:00 | Bow-shock-propagated L1 hourly context for the full Cassini launch-to-insertion era, from canonical SPDF OMNI2 or governed AMDA HAPI fallback. Local source lineage: governed AMDA fallback only. |
//! | omni_1999_2004 | inner_boundary_primary | yes | known | staged | satisfied | yes | hourly | 1999-01-01T00:00:00+00:00 | 2004-12-31T23:00:00+00:00 | Bow-shock-propagated L1 hourly context for the late Cassini cruise era, from canonical SPDF OMNI2 or governed AMDA HAPI fallback. Local source lineage: governed AMDA fallback only. |
//! | omni_2005_2016 | inner_boundary_primary | yes | known | staged | satisfied | yes | hourly | 2005-01-01T00:00:00+00:00 | 2016-12-31T23:00:00+00:00 | Bow-shock-propagated L1 hourly context for the mid-mission continuous inner-boundary span, from governed AMDA HAPI fallback on this host. Local source lineage: governed AMDA fallback only. |
//! | omni_2017_2018 | inner_boundary_primary | yes | known | staged | satisfied | yes | hourly | 2017-01-01T00:00:00+00:00 | 2018-12-31T23:00:00+00:00 | Bow-shock-propagated L1 hourly context for aligned heliopause packs, from canonical SPDF OMNI2 or governed AMDA HAPI fallback. Local source lineage: governed AMDA fallback only. |
//! | omni_2019_2025 | inner_boundary_primary | yes | known | staged | satisfied | yes | hourly | 2019-01-01T00:00:00+00:00 | 2025-12-31T23:00:00+00:00 | Bow-shock-propagated L1 hourly context for the post-heliopause modern era, spanning the governed AMDA 2019 fallback and canonical SPDF 2020-2025 yearly ASCII. Local source lineage: canonical SPDF OMNI2 plus governed AMDA fallback. |
//! | omni_1997_2025 | inner_boundary_primary | yes | known | staged | satisfied | yes | hourly | 1997-01-01T00:00:00+00:00 | 2025-12-31T23:00:00+00:00 | Continuous governed OMNI hourly inner-boundary lane across the full locally staged 1997-2025 span, with AMDA fallback for 1997-2019 and canonical SPDF yearly ASCII for 2020-2025. Local source lineage: canonical SPDF OMNI2 plus governed AMDA fallback. |
//! | voyager1_1979_merged | optional_secondary | yes | known | staged | satisfied | yes | hourly | 1979-01-01T00:00:00+00:00 | 1979-12-31T23:00:00+00:00 | Merged plasma/magnetic/trajectory lane. |
//! | voyager2_1979_merged | outer_boundary_primary | yes | known | staged | satisfied | yes | hourly | 1979-01-01T00:00:00+00:00 | 1979-12-31T23:00:00+00:00 | Merged plasma/magnetic/trajectory lane. |
//! | voyager2_2017_2018_merged | outer_boundary_primary | yes | known | staged | satisfied | yes | hourly | 2017-01-01T00:00:00+00:00 | 2018-11-30T23:00:00+00:00 | Deep heliosheath / heliopause-era merged plasma and trajectory lane. |
//! | voyager2_jupiter_track | operational_validation | yes | known | staged | satisfied | yes | hourly_track | 1979-07-03T00:00:00+00:00 | 1979-07-06T00:00:00+00:00 | Operational fused telemetry-plus-position validation artifact. |
//! | gwosc_all_events | optional_secondary | yes | known | staged | satisfied | yes | event_catalog | 2015-09-14T09:50:45.400000000+00:00 | 2024-01-09T05:04:31.800000000+00:00 | Use this broader catalog for 2017 overlap, not GWTC-3 confident only. |
//! | fermi_gbm | optional_secondary | yes | known | staged | satisfied | yes | event_catalog | 2008-07-14T02:04:12.052999936+00:00 | 2026-02-23T04:11:09.811000064+00:00 | Gamma-ray transient context layer for 2012+ windows. |
//! | wow_1977 | context_only | yes | known | partial | blocked | no | single_event_artifact | - | - | Historical epoch/context anchor only; not a plasma boundary or chronology provider. |
//! | pioneer_annual_merged | optional_secondary | yes | known | partial | ready_governed_partial_annual_lane | no | hourly | 1987-01-01T00:00:00+00:00 | 1995-12-31T23:00:00+00:00 | Reachable UCLA PDS/PPI annual merged Pioneer lane is now partially staged locally. This improves scientific coverage beyond metadata-only state, but the local staging is still a subset of the full annual source family and does not yet satisfy the full annual provider contract. |
//! | pioneer10_jupiter_1973_encounter | optional_secondary | yes | known | staged | ready_governed_adjacent_lane | no | hourly_encounter | 1973-11-26T00:00:00+00:00 | 1973-12-31T23:00:00+00:00 | Reachable UCLA PDS/PPI Jupiter encounter subset preserving the original NSSDC merged Pioneer record format. Use as an encounter-window adjacent lane, not as a full annual sibling replacement. |
//! | pioneer11_jupiter_1974_encounter | optional_secondary | yes | known | staged | ready_governed_adjacent_lane | no | hourly_encounter | 1974-11-03T00:00:00+00:00 | 1974-12-31T23:00:00+00:00 | Reachable UCLA PDS/PPI Jupiter encounter subset preserving the original NSSDC merged Pioneer record format. Use as an encounter-window adjacent lane, not as a full annual sibling replacement. |
//! | pioneer11_saturn_1979_encounter | optional_secondary | yes | known | staged | ready_governed_adjacent_lane | no | hourly_encounter | 1979-07-31T00:00:00+00:00 | 1979-10-04T23:00:00+00:00 | Reachable UCLA PDS/PPI Saturn encounter subset preserving the original NSSDC merged Pioneer record format. This is a disjoint 1979 context lane, not a same-window Jupiter replacement. |
//! | cassini_cruise_1998_2004 | outer_boundary_primary | yes | known | staged | ready_governed_hybrid_lane | no | hourly | 1998-12-30T12:00:00+00:00 | 2004-07-03T23:00:00+00:00 | Governed Cassini cruise hourly lane derived from AMDA `cass-orb-cruise` (measured trajectory), `cass-mag-rtn60` (measured magnetic field), and `tao-cass-sw` (modeled solar-wind plasma). Full overlap begins in late 1998, so this supports a fully aligned late-cruise pack from 1999 onward rather than the full 1997 mission launch interval. |
//!
//! ## Packs
//!
//! | Pack | Status | Gap-Tolerant Keys | Target Start | Target End | Overlap Start | Overlap End | Missing Required |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | JUPITER_1979 | ready | - | - | - | 1979-07-03T00:00:00+00:00 | 1979-07-06T00:00:00+00:00 | - |
//! | PIONEER_10_JUPITER_1973 | ready | - | 1973-11-26T00:00:00Z | 1973-12-31T23:00:00Z | 1973-11-26T00:00:00+00:00 | 1973-12-31T23:00:00+00:00 | - |
//! | PIONEER_11_JUPITER_1974 | ready | - | 1974-11-03T00:00:00Z | 1974-12-31T23:00:00Z | 1974-11-03T00:00:00+00:00 | 1974-12-31T23:00:00+00:00 | - |
//! | PIONEER_11_SATURN_1979 | ready | - | 1979-07-31T00:00:00Z | 1979-10-04T00:00:00Z | 1979-07-31T00:00:00+00:00 | 1979-10-04T23:00:00+00:00 | - |
//! | HELIOPAUSE_2017_2018 | ready | - | 2017-01-01T00:00:00Z | 2018-11-30T23:00:00Z | 2017-01-01T00:00:00+00:00 | 2018-11-30T23:00:00+00:00 | - |
//! | CRUISE_1997_2004 | ready_gap_tolerant | cassini_cruise_1998_2004 | 1997-11-15T00:00:00Z | 2004-07-04T00:00:00Z | 1998-12-30T12:00:00+00:00 | 2004-07-03T23:00:00+00:00 | - |
//! | CRUISE_1999_2004 | ready | - | 1999-01-01T00:00:00Z | 2004-07-03T23:00:00Z | 1999-01-01T00:00:00+00:00 | 2004-07-03T23:00:00+00:00 | - |
//! | INNER_BOUNDARY_1997_2023 | ready | - | 1997-01-01T00:00:00Z | 2023-07-06T23:57:16Z | 1997-01-01T00:00:00+00:00 | 2023-07-06T23:57:16+00:00 | - |
//! | RADIATIVE_2003_2020 | ready | - | 2003-02-25T12:00:00Z | 2020-02-25T12:00:00Z | 2003-02-25T12:00:00+00:00 | 2020-02-25T12:00:00+00:00 | - |
//! | TSI_CROSSCAL_2018_2020 | ready | - | 2018-01-11T12:00:00Z | 2020-02-25T12:00:00Z | 2018-01-11T12:00:00+00:00 | 2020-02-25T12:00:00+00:00 | - |
//! | SOLAR_CYCLE24_2008_2019 | ready | - | 2008-01-01T00:00:00Z | 2019-12-31T23:00:00Z | 1997-01-01T00:00:00+00:00 | 2023-07-06T23:57:16+00:00 | - |
//! | POST_HELIOPAUSE_2019_2023 | ready | - | 2019-01-01T00:00:00Z | 2023-07-06T23:57:16Z | 2019-01-01T00:00:00+00:00 | 2023-07-06T23:57:16+00:00 | - |
//!
