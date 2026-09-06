# jq -e --slurpfile snapshot canonical-snapshot.json -f verify-pilot.jq findings.json
(["I-001","I-002","I-094","I-095","I-096","I-207","I-209","I-212"]) as $expected
| . as $findings
| ($snapshot[0]) as $canonical
| ([$findings.records[] | .id as $insight | .references[] | [$insight,.claim_id]]) as $actual_pairs
| ([$canonical.references[] | [.insight_id,.claim_id]]) as $canonical_pairs
| [
    (($findings.records | map(.id) | sort) == $expected),
    (($canonical.insights | map(.id) | sort) == $expected),
    (($actual_pairs | length) == 37),
    (($actual_pairs | unique | length) == 37),
    (($actual_pairs | sort) == ($canonical_pairs | sort)),
    (($actual_pairs | map(.[1]) | unique | length) == 26),
    ($canonical.references | all(.claim_status != null and .statement != null)),
    ($findings.records | all(
        (.disposition | length) > 0 and
        (.preserve | length) > 0 and
        (.falsifier | length) > 0 and
        (.successor | length) > 0 and
        (.independent_test | length) > 0 and
        (.residual | length) > 0 and
        (.atomic_propositions | length) > 0 and
        ((.atomic_propositions | map(.id) | unique | length) == (.atomic_propositions | length)) and
        (.atomic_propositions | all((.proposition | length) > 0 and (.assessment | length) > 0 and (.evidence | length) > 0)) and
        (.references | all((.role | length) > 0 and (.predicate | length) > 0 and (.boundary | length) > 0))
    ))
  ]
| if all then {status:"pass",insights:8,reference_pairs:37,distinct_claims:26}
  else error("Pilot identity, coverage or required evidence fields failed") end
