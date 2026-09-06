SELECT observation_key, source_key,
       json_extract(report_json,'$.corrects_observation_key') AS corrects_observation_key,
       json_extract(report_json,'$.spec.outcome') AS outcome,
       json_extract(report_json,'$.correspondence_basis') AS correspondence_basis,
       json_extract(report_json,'$.document_identity') AS document_identity,
       json_extract(report_json,'$.cryptographic_request_binding') AS cryptographic_request_binding
FROM source_observations AS observation
WHERE NOT EXISTS (
    SELECT 1 FROM source_observations AS correction
    WHERE json_extract(correction.report_json,'$.corrects_observation_key')=observation.observation_key
)
ORDER BY source_key, observation_key;
