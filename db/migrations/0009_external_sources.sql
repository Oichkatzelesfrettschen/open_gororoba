CREATE TABLE external_source_contracts_meta (
    kind TEXT PRIMARY KEY,
    updated TEXT NOT NULL,
    authoritative INTEGER NOT NULL,
    policy_version TEXT NOT NULL
);
CREATE TABLE external_source_contracts (
    id TEXT PRIMARY KEY,
    path_glob TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    access_class TEXT NOT NULL,
    status TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    attempt_deadline_utc TEXT NOT NULL,
    resolution_deadline_utc TEXT NOT NULL,
    blocker_note TEXT NOT NULL
);
CREATE TABLE external_source_contract_values (
    contract_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    ord INTEGER NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(contract_id, relation, ord),
    FOREIGN KEY(contract_id) REFERENCES external_source_contracts(id) ON DELETE CASCADE
);
CREATE TABLE external_source_dossiers_meta (
    kind TEXT PRIMARY KEY,
    updated TEXT NOT NULL,
    authoritative INTEGER NOT NULL,
    source_markdown_glob TEXT NOT NULL,
    document_count INTEGER NOT NULL
);
CREATE TABLE external_source_dossiers (
    id TEXT PRIMARY KEY,
    source_markdown TEXT NOT NULL,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    status_token TEXT NOT NULL,
    content_kind TEXT NOT NULL,
    authority_level TEXT NOT NULL,
    verification_level TEXT NOT NULL,
    operational_role TEXT NOT NULL,
    source_lineage_summary TEXT NOT NULL,
    has_full_transcript INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    notes TEXT NOT NULL,
    body_markdown TEXT NOT NULL
);
CREATE TABLE external_source_dossier_values (
    dossier_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    ord INTEGER NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(dossier_id, relation, ord),
    FOREIGN KEY(dossier_id) REFERENCES external_source_dossiers(id) ON DELETE CASCADE
);
