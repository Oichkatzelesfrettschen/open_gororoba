-- 0018_theorem_identity_disambiguation: separate theorem identities from
-- legacy proof filenames and numeric claim identifiers.
--
-- A proof basename may carry a historical C#### prefix without proving the
-- canonical claim with that number. Stable theorem identities and explicit
-- claim links make that distinction queryable and enforceable.

CREATE TABLE theorem_identities (
    stable_id TEXT PRIMARY KEY CHECK (stable_id LIKE 'THM-%'),
    legacy_name TEXT NOT NULL UNIQUE,
    proof_path TEXT NOT NULL UNIQUE,
    identity_kind TEXT NOT NULL CHECK (
        identity_kind IN ('explicit_link', 'legacy_alias', 'unresolved')
    ),
    assumptions TEXT NOT NULL DEFAULT '',
    kernel_result TEXT NOT NULL DEFAULT '',
    replay_command TEXT NOT NULL DEFAULT '',
    falsifier TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '_RocqProject'
);

CREATE TABLE theorem_claim_links (
    theorem_stable_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    relation_kind TEXT NOT NULL CHECK (relation_kind = 'formal_proposition'),
    PRIMARY KEY (theorem_stable_id, claim_id),
    FOREIGN KEY (theorem_stable_id) REFERENCES theorem_identities(stable_id),
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);

CREATE INDEX theorem_claim_links_by_claim
    ON theorem_claim_links(claim_id);

CREATE TABLE theorem_identity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    binding_key TEXT NOT NULL UNIQUE,
    spec_sha256 TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    theorem_ids_json TEXT NOT NULL,
    claim_ids_json TEXT NOT NULL
);

CREATE TRIGGER theorem_identity_events_append_only_update
BEFORE UPDATE ON theorem_identity_events
BEGIN
    SELECT RAISE(ABORT, 'theorem identity events are append-only');
END;

CREATE TRIGGER theorem_identity_events_append_only_delete
BEFORE DELETE ON theorem_identity_events
BEGIN
    SELECT RAISE(ABORT, 'theorem identity events are append-only');
END;

INSERT INTO theorem_identities (
    stable_id, legacy_name, proof_path, identity_kind, source
)
SELECT
    'THM-LEGACY-' || theorems.id,
    theorems.id,
    theorems.proof_path,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM json_each(theorems.linked_claim_ids_json) AS linked
            JOIN claims ON claims.id = linked.value
            WHERE claims.formal_proof = theorems.proof_path
               OR instr(claims.where_stated, theorems.proof_path) > 0
               OR instr(COALESCE(claims.status_note, ''), theorems.proof_path) > 0
        ) THEN 'explicit_link'
        WHEN theorems.id IN (
            'C1007_CDPropertyLoss',
            'C1140b_PathionGap_6_10',
            'C1140c_PathionGap_11_15',
            'C1635_SedenionDriverSemantics',
            'C1636_Cariow2013SedenionSchedule',
            'C1637_R300SedenionZeroDivisor',
            'C1638_OctonionDowncastNoZeroDivisors',
            'C910_Right_e1',
            'C910_Right_e2',
            'C910_Right_e3',
            'C910_Right_e4',
            'C910_Right_e5',
            'C910_Right_e6',
            'C910_Right_e7',
            'C958_ZDGraphTopology',
            'C958b_ZDAdjacencyAnalytical',
            'C959_CHSHClassicalBound',
            'C993_CarlsonBranchFree',
            'C999_PathionEntropyBound',
            'C_ConjugateInvolution',
            'C_NormConjugate',
            'C_OctConjInvolution',
            'C_OverImbalancedSign',
            'C_QIBoundNegative',
            'C_QITauScaling',
            'C_SedConjInvolution',
            'C_TraceTracefreeVanishes',
            'C_WECImpliesNEC',
            'C_WarpEnergyNonpositive'
        ) THEN 'legacy_alias'
        ELSE 'unresolved'
    END,
    '_RocqProject'
FROM theorems;

INSERT INTO theorem_claim_links (theorem_stable_id, claim_id, relation_kind)
SELECT DISTINCT
    'THM-LEGACY-' || theorems.id,
    claims.id,
    'formal_proposition'
FROM theorems
JOIN json_each(theorems.linked_claim_ids_json) AS linked
JOIN claims ON claims.id = linked.value
WHERE claims.formal_proof = theorems.proof_path
   OR instr(claims.where_stated, theorems.proof_path) > 0
   OR instr(COALESCE(claims.status_note, ''), theorems.proof_path) > 0;

INSERT INTO source_of_truth_manifest (
    table_name, category, authoritative, legacy_toml_path,
    description, migration_status
) VALUES
    (
        'theorem_identities',
        'control_plane',
        1,
        'docs/THEOREMS.md',
        'Stable theorem identities and explicit legacy-name bindings',
        'migrated'
    ),
    (
        'theorem_claim_links',
        'control_plane',
        1,
        'docs/THEOREMS.md',
        'Explicit theorem-to-claim semantic relations',
        'migrated'
    ),
    (
        'theorem_identity_events',
        'control_plane',
        1,
        'docs/THEOREMS.md',
        'Append-only theorem identity binding operations',
        'migrated'
    );
