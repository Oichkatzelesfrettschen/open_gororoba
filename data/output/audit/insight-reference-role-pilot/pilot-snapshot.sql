-- Run with sqlite3 -readonly against the pinned canonical database.
-- Plain output emits the single JSON object directly.
WITH RECURSIVE pilot AS (
    SELECT * FROM insights
    WHERE id IN ('I-001','I-002','I-094','I-095','I-096','I-207','I-209','I-212')
), related_claims(id) AS (
    SELECT reference.value FROM pilot, json_each(pilot.claim_refs_json) AS reference
    UNION
    SELECT relation.successor_claim_id FROM claim_relations AS relation
    JOIN related_claims ON relation.predecessor_claim_id = related_claims.id
)
SELECT json_object(
    'insights', json((SELECT json_group_array(json_object(
        'id', id, 'title', title, 'status', status,
        'claim_refs', json(claim_refs_json), 'status_note', status_note,
        'compat_toml_text', compat_toml_text)) FROM pilot)),
    'references', json((SELECT json_group_array(json_object(
        'insight_id', pilot.id, 'claim_id', reference.value,
        'claim_status', claim.status, 'statement', claim.statement,
        'where_stated', claim.where_stated, 'status_note', claim.status_note))
        FROM pilot JOIN json_each(pilot.claim_refs_json) AS reference
        LEFT JOIN claims AS claim ON claim.id = reference.value)),
    'counts', json_object('claims', (SELECT count(*) FROM claims),
        'insights', (SELECT count(*) FROM insights),
        'experiments', (SELECT count(*) FROM experiments_cp)),
    'relations', json((SELECT json_group_array(json_object(
        'predecessor', predecessor_claim_id, 'successor', successor_claim_id,
        'kind', relation_kind, 'event', transition_event_id))
        FROM claim_relations WHERE predecessor_claim_id IN (SELECT id FROM related_claims)))
) AS snapshot;
