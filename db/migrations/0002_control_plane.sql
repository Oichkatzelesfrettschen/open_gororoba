CREATE TABLE registry_snapshots (
    registry_kind TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_text TEXT NOT NULL,
    blake3_hex TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    where_stated TEXT NOT NULL,
    last_verified TEXT NOT NULL,
    formal_proof TEXT,
    status_note TEXT
);
CREATE TABLE insights (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    claim_refs_json TEXT NOT NULL
);
CREATE TABLE experiments_cp (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    binary_name TEXT,
    claim_refs_json TEXT NOT NULL
);
CREATE TABLE binaries_cp (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    experiment_id TEXT
);
CREATE TABLE theorems (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    proof_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    linked_claim_ids_json TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE control_plane_runs (
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    details_json TEXT NOT NULL
);
